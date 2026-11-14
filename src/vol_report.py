"""src/vol_report.py — visual and statistical checks of a volume cache: PNG montages and the pilot report.

Montage tiles carry orientation letters derived from the stored geometry (meta row/col/normal
directions, LPS), not from the CANON table, so a look at the anatomy verifies the canonical
orientation: e.g. a sagittal tile must show the patella on the side labelled A.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

# LPS axis → (letter of the + direction, letter of the − direction)
_AXIS_LETTERS = {0: ("L", "R"), 1: ("P", "A"), 2: ("S", "I")}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def direction_letter(d) -> str:
    """Patient-direction letter (L/R/P/A/S/I) of the dominant LPS component of a direction vector."""
    d = np.asarray(d, dtype=float)
    i = int(np.abs(d).argmax())
    return _AXIS_LETTERS[i][0 if d[i] > 0 else 1]


def orientation_labels(meta: dict) -> dict:
    """Letters at the tile edges (top/bottom/left/right) and the slice-order ends (first/last)."""
    flip = {"L": "R", "R": "L", "P": "A", "A": "P", "S": "I", "I": "S"}
    row, col, nrm = (direction_letter(meta[k]) for k in ("row_dir", "col_dir", "normal_dir"))
    return {"top": flip[row], "bottom": row, "left": flip[col], "right": col, "first": flip[nrm], "last": nrm}


def _text(img, s, org, scale=0.4, color=(255, 255, 255)):
    """Text on a dark box (readable on any background)."""
    # a dark box behind the text: a thick outline would have a different advance width and drift
    (w, h), base = cv2.getTextSize(s, _FONT, scale, 1)
    cv2.rectangle(img, (org[0] - 1, org[1] - h - 2), (org[0] + w + 1, org[1] + base), (0, 0, 0), -1)
    cv2.putText(img, s, org, _FONT, scale, color, 1, cv2.LINE_AA)


def _tile(vol, pos_mm, meta, k, tile_px, labels, highlight):
    """One montage tile: slice k fitted into tile_px², fg bbox, edge letters, k and pos_mm."""
    img = cv2.cvtColor(np.ascontiguousarray(vol[k]), cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    s = tile_px / max(H, W)
    img = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((tile_px, tile_px, 3), np.uint8)
    oy, ox = (tile_px - img.shape[0]) // 2, (tile_px - img.shape[1]) // 2
    out[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    if meta["fg_r0"] is not None and meta["fg_z0"] <= k < meta["fg_z1"]:
        p0 = (ox + round(meta["fg_c0"] * s), oy + round(meta["fg_r0"] * s))
        p1 = (ox + round(meta["fg_c1"] * s) - 1, oy + round(meta["fg_r1"] * s) - 1)
        cv2.rectangle(out, p0, p1, (0, 220, 220), 1)
    c = (255, 255, 0)
    _text(out, labels["top"], (tile_px // 2 - 4, oy + 12), color=c)
    _text(out, labels["bottom"], (tile_px // 2 - 4, oy + img.shape[0] - 4), color=c)
    _text(out, labels["left"], (ox + 2, tile_px // 2), color=c)
    _text(out, labels["right"], (ox + img.shape[1] - 10, tile_px // 2), color=c)
    _text(out, f"k={k} {float(pos_mm[k]):.1f}mm", (ox + 2, oy + img.shape[0] - 18))
    if highlight:
        cv2.rectangle(out, (0, 0), (tile_px - 1, tile_px - 1), (0, 0, 255), 3)
    return out


def render_tiles(vol, pos_mm, meta: dict, ks, title_lines: list[str], tile_px: int = 224,
                 ncols: int = 6, highlight: set | None = None) -> np.ndarray:
    """BGR montage of slices `ks` with fg bbox, orientation letters and a text header."""
    labels = orientation_labels(meta)
    tiles = [_tile(vol, pos_mm, meta, k, tile_px, labels, k in (highlight or set())) for k in ks]
    ncols = min(ncols, len(tiles))
    nrows = math.ceil(len(tiles) / ncols)
    grid = np.zeros((nrows * tile_px, ncols * tile_px, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncols)
        grid[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px] = t
    lines = title_lines + [f"slice order: k=0 at {labels['first']} -> k={len(pos_mm) - 1} at {labels['last']}"]
    head = np.full((16 * len(lines) + 6, grid.shape[1], 3), 40, np.uint8)
    for i, line in enumerate(lines):
        _text(head, line, (4, 16 * (i + 1)), scale=0.45)
    return np.concatenate([head, grid], axis=0)


def series_title(meta: dict, row: dict | None = None) -> list[str]:
    """Header lines of a montage: ids, plane (computed vs csv), shape, laterality, fg flags."""
    plane = meta["plane_computed"] + (f" (csv {row['Anatomical_Plane']})" if row is not None else "")
    return [f"study ..{meta['study'][-12:]}  series ..{meta['series'][-12:]}  {plane}",
            f"ZxHxW {meta['Z']}x{meta['H']}x{meta['W']}  spacing {meta['slice_spacing_median']}  "
            f"lat tag={meta['lat_tag']} geom={meta['lat_geom']}  bilateral={meta['bilateral_suspect']}  "
            f"dropped={meta['n_dropped']}  gap_irregular={meta['gap_irregular']}"]


def write_png(path, img: np.ndarray) -> None:
    """cv2.imwrite that raises instead of returning False."""
    if not cv2.imwrite(str(path), img):
        raise OSError(f"cv2.imwrite failed: {path}")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame rows as JSON-ready dicts."""
    return [{k: _py(v) for k, v in r.items()} for r in df.to_dict("records")]


def _py(v):
    """numpy/pandas scalar or array → plain Python value; NaN/NA → None."""
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NA:
        return None
    return v


def _counts(s: pd.Series) -> dict:
    """Value counts with missing values counted as 'None'."""
    return {str(k): int(v) for k, v in s.astype(object).where(s.notna(), "None").value_counts().items()}


def _q(s: pd.Series) -> dict:
    """sum / mean / p50 / p90 / max of a numeric column."""
    return {"sum": _py(s.sum()), "mean": _py(s.mean()), "p50": _py(s.median()),
            "p90": _py(s.quantile(0.9)), "max": _py(s.max())}


def study_cache_bytes(root, study: str) -> int:
    """Bytes on disk of one study's files under <root>/vol/<study>/."""
    return sum(p.stat().st_size for p in (Path(root) / "vol" / study).iterdir() if p.is_file())


def pilot_report(cache, run_log: pd.DataFrame, throughput: dict, index: pd.DataFrame,
                 known_corrupt_sops: list[str], corrupt_pngs: dict[str, str], note: str) -> dict:
    """Pilot statistics of a cache + build run, with the full-pass size/time extrapolation."""
    se, sl, dr = cache.series, cache.slices, cache.dropped
    ok = se[se.error_code.isna()]

    ts_slices = sl.transfer_syntax.value_counts()
    ts = {"slices": {f"{pydicom.uid.UID(k).name} ({k})": int(v) for k, v in ts_slices.items()},
          "series": _counts(ok.transfer_syntaxes.map(lambda v: ",".join(v)))}

    plane = {"counts": _counts(se.plane_agree),
             "crosstab": {f"{a}->{b}": int(n) for (a, b), n in
                          ok.groupby(["plane_computed", "Anatomical_Plane"]).size().items()},
             "disagree": _records(ok.loc[ok.plane_agree == False, ["study", "series", "plane_computed",  # noqa: E712
                                                                   "Anatomical_Plane"]])}

    bil = ok[ok.bilateral_suspect == True]  # noqa: E712
    lat_cols = ["study", "series", "plane_computed", "lat_tag", "lat_geom", "fg_center_lps"]
    per_study = ok.groupby("study").agg(tags=("lat_tag", lambda s: sorted(set(s.dropna()))),
                                        geoms=("lat_geom", lambda s: sorted(set(s.dropna()))))
    single = per_study[(per_study.tags.map(len) == 1) & (per_study.geoms.map(len) == 1)]
    laterality = {
        "series": {"n": len(ok), "tag": int(ok.lat_tag.notna().sum()), "geom": int(ok.lat_geom.notna().sum()),
                   "agree": int((ok.lat_agree == True).sum()), "disagree": int((ok.lat_agree == False).sum()),  # noqa: E712
                   "pairs": {f"tag={a} geom={b}": int(n) for (a, b), n in
                             ok.fillna({"lat_tag": "None", "lat_geom": "None"})
                             .groupby(["lat_tag", "lat_geom"]).size().items()},
                   "disagree_rows": _records(ok.loc[ok.lat_agree == False, lat_cols])},  # noqa: E712
        "studies": {"n": len(per_study), "with_tag": int((per_study.tags.map(len) > 0).sum()),
                    "geom_resolved": int((per_study.geoms.map(len) > 0).sum()),
                    "tag_inconsistent": int((per_study.tags.map(len) > 1).sum()),
                    "geom_inconsistent": int((per_study.geoms.map(len) > 1).sum()),
                    "single_tag_and_geom": len(single),
                    "agree": int((single.tags.map(lambda t: t[0]) == single.geoms.map(lambda g: g[0])).sum())},
    }

    known = []
    for sop in known_corrupt_sops:
        srows, drows = sl[sl.sop == sop], dr[dr.sop == sop]
        series = srows.series.tolist() + drows.series.tolist()
        known.append({"sop": sop,
                      "status": "dropped" if len(drows) else "kept" if len(srows) else "not_in_cache",
                      "series_Z": [_py(se.loc[se.series == s, "Z"].iloc[0]) for s in series],
                      "slice_rows": _records(srows), "dropped_rows": _records(drows),
                      "png": corrupt_pngs.get(sop)})

    # per-study bytes and seconds of the build run + the extrapolation to the whole train archive
    stages = ["s_download", "s_inflate", "s_process", "s_write"]
    rl = run_log.copy()
    done = rl[rl.error.isna()].copy()
    done["cache_bytes"] = [study_cache_bytes(cache.root, s) for s in done.study]
    train = index[(index.split == "train_series") & index.study.notna()]
    g = train.groupby("study").agg(start=("lho", "min"), end=("next_lho", "max"))
    total_compressed = int((g.end - g.start).sum())
    pilot_compressed = int(done.compressed_bytes.sum())
    pilot_cache = int(done.cache_bytes.sum())
    n_series_pilot = int(done.n_series.sum())
    ratio = total_compressed / pilot_compressed
    stage_sum = float(done[stages].to_numpy().sum())
    extrap = {
        "train_compressed_bytes": total_compressed, "train_studies": len(g),
        "train_series": int(train.series.nunique()),
        "pilot_compressed_bytes": pilot_compressed, "pilot_cache_bytes": pilot_cache,
        "pilot_studies": len(done), "pilot_series": n_series_pilot,
        "ratio_compressed": ratio,
        "cache_gb_by_compressed": pilot_cache * ratio / 1e9,
        "cache_gb_by_series_count": pilot_cache / n_series_pilot * int(train.series.nunique()) / 1e9,
        "cache_to_compressed": pilot_cache / pilot_compressed,
        "wall_h_by_compressed": throughput["wall_s"] * ratio / 3600,
        "stage_h_by_compressed_per_worker": stage_sum * ratio / throughput["n_workers"] / 3600,
        "n_workers": throughput["n_workers"],
        "caveat": "wall time scales the pilot's wall clock (includes pool ramp-up/tail and the machine "
                  "load at that moment); the per-worker estimate assumes perfect parallelism",
    }

    return {
        "cache_root": str(cache.root),
        "n_series": len(se), "n_series_ok": len(ok), "n_slices": len(sl), "n_dropped": len(dr),
        "error_codes": _counts(se.error_code.dropna()),
        "errors": _records(se.loc[se.error_code.notna(), ["study", "series", "error_code", "error_detail"]]),
        "transfer_syntaxes": ts,
        "plane_agree": plane,
        "bilateral_suspect": {"count": len(bil), "rows": _records(bil[["study", "series", "plane_computed",
                                                                        "n_fg_components"]])},
        "laterality": laterality,
        "gap_irregular": int(ok.gap_irregular.sum()),
        "shape": {"Z": _q(ok.Z), "H": _q(ok.H), "W": _q(ok.W)},
        "dropped": {"reasons": _counts(dr.reason), "rows": _records(dr)},
        "known_reported_corrupt": {"note": note, "sops": known},
        "build_stats": {"throughput": throughput,
                      "per_study": _records(done[["study", "compressed_bytes", "cache_bytes", "n_members",
                                                  "n_series", "n_series_error", "attempts", *stages]]),
                      "failed_studies": _records(rl.loc[rl.error.notna(), ["study", "attempts", "error"]]),
                      "summary": {c: _q(done[c]) for c in ["compressed_bytes", "cache_bytes", *stages]}},
        "extrapolation": extrap,
    }
