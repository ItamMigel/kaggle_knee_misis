"""src/vol_cache.py — the on-disk volume cache: atomic per-study writer, parquet consolidation, mmap reader.

Layout under a cache root (exps/exp-0002/contracts.md, «Layout»):

    vol/<study>/<series>.npy     vol uint8 [Z, H, W], uncompressed np.save → np.load(mmap_mode='r')
    vol/<study>/_done.json       series/slice/dropped rows + series errors; written LAST, atomically
    series.parquet  slices.parquet  dropped.parquet      ← consolidate()

A study is done iff its `_done.json` exists. Meta values survive the JSON → parquet roundtrip
unchanged, so KneeVolCache.load() returns exactly what knee_volume.load_series_dir() returns.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.knee_volume import META_FIELDS, SeriesError, SeriesVolume

DONE_NAME = "_done.json"
TABLES = ("series", "slices", "dropped")

_KIND_ARROW = {"str": pa.string(), "int": pa.int64(), "float": pa.float64(), "bool": pa.bool_(),
               "floats": pa.list_(pa.float64()), "strs": pa.list_(pa.string())}
_ERROR_FIELDS = [("error_code", pa.string()), ("error_detail", pa.string())]
_CSV_FIELDS = [("Anatomical_Plane", pa.string()), ("Fluid_Sensitive", pa.int64()),
               ("Fat_Suppression", pa.int64()), ("plane_agree", pa.bool_())]
SCHEMAS = {
    "series": pa.schema([(k, _KIND_ARROW[kind]) for k, (kind, _) in META_FIELDS.items()]
                        + _ERROR_FIELDS + _CSV_FIELDS),
    "slices": pa.schema([("study", pa.string()), ("series", pa.string()), ("k", pa.int64()),
                         ("sop", pa.string()), ("file", pa.string()), ("instance_number", pa.int64()),
                         ("ipp_x", pa.float64()), ("ipp_y", pa.float64()), ("ipp_z", pa.float64()),
                         ("pos_mm", pa.float32()), ("transfer_syntax", pa.string())]),
    "dropped": pa.schema([("study", pa.string()), ("series", pa.string()), ("sop", pa.string()),
                          ("file", pa.string()), ("reason", pa.string()), ("detail", pa.string())]),
}


def _atomic_write(path: Path, data: bytes) -> None:
    """Write to a sibling tmp file, then os.replace: readers never see a partial file."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _sha256(data: bytes) -> str:
    """'sha256:<hex>', the format of utils/ctx.py."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def is_study_done(root, study: str) -> bool:
    """True iff the study's `_done.json` marker exists (all its files were written)."""
    return (Path(root) / "vol" / study / DONE_NAME).is_file()


def write_study(root, study: str, results: dict) -> list[dict]:
    """Write one study ({series: SeriesVolume | SeriesError}) atomically, `_done.json` last; → manifest rows."""
    root = Path(root)
    sdir = root / "vol" / study
    sdir.mkdir(parents=True, exist_ok=True)
    manifest, series_rows, slice_rows, dropped_rows, errors = [], [], [], [], []
    for series in sorted(results):
        r = results[series]
        if isinstance(r, SeriesError):
            errors.append({"study": study, "series": series, "error_code": r.code, "error_detail": r.detail})
            continue
        if not isinstance(r, SeriesVolume):
            raise TypeError(f"{study}/{series}: expected SeriesVolume or SeriesError, got {type(r).__name__}")
        if (r.meta["study"], r.meta["series"]) != (study, series):
            raise ValueError(f"DICOM tags say {r.meta['study']}/{r.meta['series']}, "
                             f"archive path says {study}/{series}")
        Z = r.meta["Z"]
        if (r.vol.dtype != np.uint8 or r.vol.shape != (Z, r.meta["H"], r.meta["W"])
                or r.pos_mm.dtype != np.float32 or r.pos_mm.shape != (Z,) or len(r.slices) != Z):
            raise ValueError(f"{study}/{series}: inconsistent SeriesVolume {r.vol.dtype}{r.vol.shape}, "
                             f"pos_mm {r.pos_mm.dtype}{r.pos_mm.shape}, {len(r.slices)} slice rows, meta Z={Z}")
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(r.vol), allow_pickle=False)
        data = buf.getvalue()
        rel = f"vol/{study}/{series}.npy"
        _atomic_write(root / rel, data)
        manifest.append({"file": rel, "bytes": len(data), "sha256": _sha256(data)})
        series_rows.append(r.meta)
        slice_rows += [{"study": study, "series": series, **row} for row in r.slices]
        dropped_rows += [{"study": study, "series": series, **row} for row in r.dropped]
    done = {"study": study, "series": series_rows, "slices": slice_rows, "dropped": dropped_rows,
            "errors": errors}
    data = json.dumps(done, sort_keys=True, allow_nan=False).encode()
    rel = f"vol/{study}/{DONE_NAME}"
    _atomic_write(root / rel, data)
    manifest.append({"file": rel, "bytes": len(data), "sha256": _sha256(data)})
    return manifest


def _write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    """Rows with exactly the schema's keys → parquet, atomically; a type mismatch raises in pyarrow."""
    for row in rows:
        if set(row) != set(schema.names):
            raise ValueError(f"{path.name}: row keys differ from the schema: {sorted(set(row) ^ set(schema.names))}")
    table = pa.Table.from_pylist(rows, schema=schema)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def consolidate(root, series_csv_path) -> dict[str, Path]:
    """All `_done.json` → series/slices/dropped.parquet in root (series joined with the *_series.csv)."""
    root = Path(root)
    dones = sorted((root / "vol").glob(f"*/{DONE_NAME}"))
    if not dones:
        raise FileNotFoundError(f"no vol/*/{DONE_NAME} under {root}: nothing to consolidate")
    csv = pd.read_csv(series_csv_path)
    if csv.SeriesInstanceUID.duplicated().any():
        raise ValueError(f"{series_csv_path}: duplicate SeriesInstanceUID")
    info = {s: (st, pl, fs, fat) for s, st, pl, fs, fat in zip(
        csv.SeriesInstanceUID, csv.StudyInstanceUID, csv.Anatomical_Plane, csv.Fluid_Sensitive, csv.Fat_Suppression)}

    series_rows, slice_rows, dropped_rows = [], [], []
    for p in dones:
        d = json.loads(p.read_text())
        if d["study"] != p.parent.name:
            raise ValueError(f"{p}: study {d['study']} != folder {p.parent.name}")
        for m in d["series"]:
            if set(m) != set(META_FIELDS):
                raise ValueError(f"{p}: series meta keys differ from META_FIELDS: {sorted(set(m) ^ set(META_FIELDS))}")
            series_rows.append({**m, "error_code": None, "error_detail": None})
        for e in d["errors"]:
            series_rows.append({**{k: None for k in META_FIELDS}, **e})
        slice_rows += d["slices"]
        dropped_rows += d["dropped"]

    for row in series_rows:
        if row["series"] not in info:
            raise KeyError(f"series {row['series']} (study {row['study']}) is not in {series_csv_path}")
        study, plane, fs, fat = info[row["series"]]
        if study != row["study"]:
            raise ValueError(f"series {row['series']}: study {row['study']} in cache, {study} in {series_csv_path}")
        row.update(Anatomical_Plane=str(plane), Fluid_Sensitive=int(fs), Fat_Suppression=int(fat),
                   plane_agree=None if row["plane_computed"] is None else row["plane_computed"] == plane)
    keys = [(r["study"], r["series"]) for r in series_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("a series appears twice across _done.json files")

    out = {}
    for name, rows in zip(TABLES, (series_rows, slice_rows, dropped_rows)):
        out[name] = root / f"{name}.parquet"
        _write_parquet(out[name], rows, SCHEMAS[name])
    return out


class KneeVolCache:
    """Reader of a consolidated cache root: .series/.slices/.dropped tables and per-series .load()."""

    def __init__(self, root):
        self.root = Path(root)
        missing = [n for n in TABLES if not (self.root / f"{n}.parquet").is_file()]
        if missing:
            raise FileNotFoundError(f"{self.root}: missing {missing}.parquet — run vol_cache.consolidate first")
        self._series_table = pq.read_table(self.root / "series.parquet")
        self.series = self._series_table.to_pandas()
        self.slices = pq.read_table(self.root / "slices.parquet").to_pandas()
        self.dropped = pq.read_table(self.root / "dropped.parquet").to_pandas()
        self._rows = None
        self._pos = None

    def _row(self, series: str) -> dict:
        if self._rows is None:
            self._rows = {r["series"]: r for r in self._series_table.to_pylist()}
        if series not in self._rows:
            raise KeyError(f"series {series} is not in {self.root / 'series.parquet'}")
        return self._rows[series]

    def meta(self, series: str) -> dict:
        """Meta of one series restricted to META_FIELDS (== load_series_dir meta); SeriesError if it failed."""
        row = self._row(series)
        if row["error_code"] is not None:
            raise SeriesError(row["error_code"], row["error_detail"])
        return {k: row[k] for k in META_FIELDS}

    def pos_mm(self, series: str) -> np.ndarray:
        """Slice positions (float32 [Z], increasing) of one series from slices.parquet."""
        if self._pos is None:
            s = self.slices.sort_values(["series", "k"], kind="stable")
            names = s.series.to_numpy()
            pos = s.pos_mm.to_numpy(dtype=np.float32)
            ks = s.k.to_numpy()
            starts = np.r_[0, np.nonzero(names[1:] != names[:-1])[0] + 1, len(names)]
            self._pos = {}
            for a, b in zip(starts[:-1], starts[1:]):
                if not np.array_equal(ks[a:b], np.arange(b - a)):
                    raise ValueError(f"series {names[a]}: slice indices are not 0..{b - a - 1}")
                self._pos[names[a]] = pos[a:b]
        if series not in self._pos:
            raise KeyError(f"series {series} has no rows in {self.root / 'slices.parquet'}")
        return self._pos[series].copy()

    def load(self, series: str, mmap: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
        """(vol uint8 [Z,H,W], pos_mm float32 [Z], meta) of one series; vol is memory-mapped by default."""
        meta = self.meta(series)
        vol = np.load(self.root / "vol" / meta["study"] / f"{series}.npy",
                      mmap_mode="r" if mmap else None, allow_pickle=False)
        pos = self.pos_mm(series)
        if vol.dtype != np.uint8 or vol.shape != (meta["Z"], meta["H"], meta["W"]) or pos.shape != (meta["Z"],):
            raise ValueError(f"series {series}: vol {vol.dtype}{vol.shape}, pos_mm {pos.shape}, "
                             f"meta Z,H,W = {meta['Z']},{meta['H']},{meta['W']}")
        return vol, pos, meta
