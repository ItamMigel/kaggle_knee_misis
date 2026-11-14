"""src/knee_volume.py — the ONE preprocessing function: raw DICOM series → canonical uint8 volume.

The cache build (bytes inflated from the archive) and the inference kernels (files on the
Kaggle mount) call literally the same `preprocess_series`, so the cache and inference agree
by construction. Runs offline inside Kaggle kernels: numpy, pydicom, cv2 and stdlib only.

Contract (exps/exp-0002/contracts.md, section B), in short:
  * per slice: decode (all transfer syntaxes; a missing decoder raises, never skips),
    MONOCHROME1 → negate, RescaleSlope/Intercept; a slice that cannot be read, has no
    geometry, a wrong pixel byte count or fails to decode is dropped with a reason;
  * geometry: IOP/IPP/PixelSpacing required; slices off the majority (IOP tol 1e-3,
    Rows/Cols, PixelSpacing) are dropped; plane = dominant axis of cross(row, col);
  * order by IPP · canonical normal; duplicate positions keep the lowest InstanceNumber;
  * canonical in-plane orientation (CANON below, LPS) by transpose and flips only,
    no left/right knee mirroring; in-plane resample to `mm_per_px`, full FOV;
  * intensity: percentiles `clip_pct` over the whole resampled volume → uint8;
  * metadata: geometry, foreground bbox, bilateral suspicion, laterality (META_FIELDS).
The result does not depend on the order of the input files.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.multival import MultiValue
from pydicom.pixels import get_decoder

PLANES = ("Sagittal", "Coronal", "Axial")
# plane index (dominant axis of the slice normal, LPS) -> (slice-order sign, (col axis, sign), (row axis, sign))
# Sagittal: slices right->left, image columns anterior->posterior, rows superior->inferior
# Coronal:  slices anterior->posterior, columns patient-right->left, rows superior->inferior
# Axial:    slices superior->inferior, columns patient-right->left, rows anterior->posterior
CANON = {
    0: (+1, (1, +1), (2, -1)),
    1: (+1, (0, +1), (2, -1)),
    2: (-1, (0, +1), (1, +1)),
}

# Fixed parts of the contract (changing any of them means a new cache slug).
IOP_TOL = 1e-3          # max |ΔIOP| component inside the majority geometry group
PS_TOL_MM = 1e-4        # max |ΔPixelSpacing| inside the majority geometry group
DUP_TOL_MM = 0.01       # slices closer than this along the normal are duplicates
GAP_FACTOR = 1.5        # spacing > GAP_FACTOR × median → gap_irregular
FG_OPEN_PX = 5          # morphological opening of the Otsu mask
FG_MIN_AREA_FRAC = 0.01  # foreground components smaller than this share of H×W are noise
FG_Z_FRAC = 0.05        # slice is in the fg z-range if its fg count > this × per-slice max
BILAT_MIN_FRAC = 0.15   # bilateral: ≥2 components each ≥ this share of the fg area
LAT_GEOM_MM = 20.0      # |x| of the fg centre (LPS) beyond which laterality is inferred

PIXEL_TAGS = ("PixelData", "FloatPixelData", "DoubleFloatPixelData")

# Keys and value kinds of SeriesVolume.meta. vol_cache stores exactly these columns
# (plus join/error columns) and KneeVolCache.load() returns exactly these keys.
# kind: str | int | float | bool | floats (list[float]) | strs (list[str]); nullable flag.
META_FIELDS = {
    "study": ("str", False),                  # StudyInstanceUID tag
    "series": ("str", False),                 # SeriesInstanceUID tag
    "plane_computed": ("str", False),         # Sagittal / Coronal / Axial from the IOP normal
    "iop": ("floats", False),                 # median ImageOrientationPatient of the kept slices
    "row_dir": ("floats", False),             # LPS unit vector of increasing canonical row index
    "col_dir": ("floats", False),             # LPS unit vector of increasing canonical column index
    "normal_dir": ("floats", False),          # LPS unit vector of increasing slice index (pos_mm axis)
    "origin_lps": ("floats", False),          # LPS of the centre of stored pixel (0, 0) on slice 0
    "spacing_rc": ("floats", False),          # stored pixel pitch (row, col) in mm, ≈ mm_per_px
    "canon_transpose": ("bool", False),
    "canon_flip_rows": ("bool", False),
    "canon_flip_cols": ("bool", False),
    "frame_of_reference_uid": ("str", True),
    "orig_rows": ("int", False),
    "orig_cols": ("int", False),
    "pixel_spacing": ("floats", False),       # DICOM PixelSpacing (row, col) of the series
    "slice_thickness": ("float", True),
    "slice_spacing_median": ("float", True),  # None when Z == 1
    "gap_irregular": ("bool", False),
    "Z": ("int", False),
    "H": ("int", False),
    "W": ("int", False),
    "mm_per_px": ("float", False),
    "clip_pct": ("floats", False),
    "interp": ("str", False),                 # area (downsampling) / linear (upsampling)
    "lo": ("float", False),
    "hi": ("float", False),
    "n_files": ("int", False),
    "n_dropped": ("int", False),
    "transfer_syntaxes": ("strs", False),     # sorted unique UIDs of the kept slices
    "header_json": ("str", False),            # non-pixel header of slice 0 (file_meta + dataset)
    "fg_r0": ("int", True),                   # fg bbox in stored pixels; r1/c1/z1 exclusive
    "fg_r1": ("int", True),
    "fg_c0": ("int", True),
    "fg_c1": ("int", True),
    "fg_z0": ("int", True),
    "fg_z1": ("int", True),
    "n_fg_components": ("int", False),
    "bilateral_suspect": ("bool", True),      # None for sagittal or without foreground
    "lat_tag": ("str", True),                 # (0020,0060) Laterality or (0020,0062) ImageLaterality
    "lat_geom": ("str", True),                # 'L' / 'R' from fg_center_lps x, None if |x| ≤ 20 mm
    "lat_agree": ("bool", True),              # None unless both lat_tag and lat_geom are set
    "fg_center_lps": ("floats", True),
}

_KIND_TYPES = {"str": str, "int": int, "float": float, "bool": bool}
_DECODER_CHECKED: set[str] = set()


class SeriesError(Exception):
    """Series-level failure (no volume is produced): `code` is a short slug, `detail` free text."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    def __reduce__(self):
        return SeriesError, (self.code, self.detail)


@dataclass
class SeriesVolume:
    """Canonical volume of one series: vol uint8 [Z,H,W], pos_mm float32 [Z], slice/dropped rows, meta."""

    vol: np.ndarray
    pos_mm: np.ndarray
    slices: list[dict]
    dropped: list[dict]
    meta: dict


@dataclass
class _Slice:
    """One decoded slice with its geometry; `ds` keeps the header only."""

    file: str
    sop: str | None
    inst: int | None
    iop: np.ndarray
    ipp: np.ndarray
    ps: np.ndarray
    rows: int
    cols: int
    ts: str
    arr: np.ndarray
    slope: float
    intercept: float
    mono1: bool
    ds: pydicom.Dataset  # header only: pixel elements are deleted after decoding
    pos: float = 0.0


def read_series_dir(path) -> list[tuple[str, bytes]]:
    """Read every file of a series directory as (basename, bytes), sorted by name."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"series directory not found: {path}")
    return [(p.name, p.read_bytes()) for p in sorted(path.iterdir()) if p.is_file()]


def load_series_dir(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inference entry point: (vol, pos_mm, meta) of a series directory, same triple as KneeVolCache.load."""
    sv = preprocess_series(read_series_dir(path))
    return sv.vol, sv.pos_mm, sv.meta


def canon_px_to_lps(meta: dict, pos_mm: np.ndarray, k: float, r: float, c: float) -> np.ndarray:
    """LPS (mm) of stored pixel centre (slice k, row r, col c); fractional k interpolates pos_mm."""
    pos_mm = np.asarray(pos_mm, dtype=np.float64)
    if not 0 <= k <= len(pos_mm) - 1:
        raise ValueError(f"slice index {k} outside [0, {len(pos_mm) - 1}]")
    pos = float(np.interp(k, np.arange(len(pos_mm)), pos_mm))
    sy, sx = meta["spacing_rc"]
    return (np.asarray(meta["origin_lps"]) + (pos - pos_mm[0]) * np.asarray(meta["normal_dir"])
            + r * sy * np.asarray(meta["row_dir"]) + c * sx * np.asarray(meta["col_dir"]))


def _require_decoder(ts) -> None:
    """A transfer syntax without a working decoder is an environment problem: raise, never drop."""
    if ts in _DECODER_CHECKED:
        return
    dec = get_decoder(ts)  # NotImplementedError for a syntax pydicom cannot decode at all
    if not dec.is_available:
        raise RuntimeError(f"no pixel decoder plugin available for {ts.name} ({ts}); "
                           f"install gdcm / pylibjpeg in this environment")
    _DECODER_CHECKED.add(str(ts))


def _floats(ds, tag: str, n: int) -> np.ndarray | None:
    """Tag value as float64 [n]; None if the tag is absent, empty or has another multiplicity."""
    v = ds.get(tag)
    if v is None or v == "":
        return None
    vals = list(v) if isinstance(v, (list, tuple, MultiValue)) else [v]
    if len(vals) != n:
        return None
    return np.array([float(x) for x in vals], dtype=np.float64)


def _read_slice(name: str, data: bytes) -> tuple[_Slice | None, dict | None]:
    """Parse and decode one file → (_Slice, None) or (None, dropped row)."""
    def drop(reason, detail, sop=None):
        return None, {"sop": sop, "file": name, "reason": reason, "detail": detail}

    try:
        ds = pydicom.dcmread(io.BytesIO(data))
    except Exception as e:  # unreadable file → dropped slice, by contract
        return drop("read_error", f"{type(e).__name__}: {e}")
    sop = str(ds.SOPInstanceUID) if ds.get("SOPInstanceUID") else None
    ts = ds.file_meta.get("TransferSyntaxUID")
    if not ts:
        return drop("read_error", "no TransferSyntaxUID in file meta", sop)

    geo = {t: _floats(ds, t, n) for t, n in
           (("ImageOrientationPatient", 6), ("ImagePositionPatient", 3), ("PixelSpacing", 2))}
    missing = sorted(t for t, v in geo.items() if v is None)
    if missing:
        return drop("missing_geometry", "missing or malformed: " + ",".join(missing), sop)

    if "PixelData" not in ds:
        return drop("no_pixel_data", "no (7FE0,0010) PixelData", sop)
    absent = [t for t in ("Rows", "Columns", "BitsAllocated") if ds.get(t) in (None, "")]
    if absent:
        return drop("pixel_tags", "missing: " + ",".join(absent), sop)
    pi = ds.get("PhotometricInterpretation")
    if pi not in ("MONOCHROME1", "MONOCHROME2"):
        return drop("photometric", f"PhotometricInterpretation={pi}", sop)
    frames = int(ds.get("NumberOfFrames") or 1)
    if frames != 1:
        return drop("multiframe", f"NumberOfFrames={frames}", sop)
    rows, cols = int(ds.Rows), int(ds.Columns)

    _require_decoder(ts)
    if not ts.is_encapsulated:
        bits = int(ds.BitsAllocated)
        expected = (rows * cols * int(ds.get("SamplesPerPixel") or 1) * bits + 7) // 8
        actual = len(ds.PixelData)
        if not (actual == expected or (expected % 2 == 1 and actual == expected + 1)):
            return drop("pixel_bytes", f"{actual}/{expected}", sop)
    try:
        arr = ds.pixel_array
    except Exception as e:  # corrupt pixel payload → dropped slice, by contract
        return drop("decode_error", f"{type(e).__name__}: {e}", sop)
    if arr.shape != (rows, cols):
        return drop("pixel_shape", f"{arr.shape} != {(rows, cols)}", sop)
    for t in PIXEL_TAGS:
        if t in ds:
            del ds[t]

    inst = ds.get("InstanceNumber")
    return _Slice(
        file=name, sop=sop, inst=int(inst) if inst not in (None, "") else None,
        iop=geo["ImageOrientationPatient"], ipp=geo["ImagePositionPatient"], ps=geo["PixelSpacing"],
        rows=rows, cols=cols, ts=str(ts), arr=arr,
        slope=float(ds.RescaleSlope) if ds.get("RescaleSlope") not in (None, "") else 1.0,
        intercept=float(ds.RescaleIntercept) if ds.get("RescaleIntercept") not in (None, "") else 0.0,
        mono1=pi == "MONOCHROME1", ds=ds,
    ), None


def _same_geometry(a: _Slice, b: _Slice) -> bool:
    """Same Rows/Cols, IOP within IOP_TOL and PixelSpacing within PS_TOL_MM."""
    return (a.rows == b.rows and a.cols == b.cols
            and float(np.abs(a.iop - b.iop).max()) <= IOP_TOL
            and float(np.abs(a.ps - b.ps).max()) <= PS_TOL_MM)


def _geometry_desc(s: _Slice) -> dict:
    """Geometry of a slice for the detail of a geometry_mismatch / geometry_ambiguous record."""
    return {"iop": [round(float(x), 6) for x in s.iop], "rows": s.rows, "cols": s.cols,
            "ps": [round(float(x), 6) for x in s.ps]}


def _foreground(vol: np.ndarray) -> dict:
    """MIP → Otsu → opening → components > 1 % of H×W; bbox, z-range, component stats."""
    Z, H, W = vol.shape
    mip = vol.max(axis=0)
    t, mask = cv2.threshold(mip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FG_OPEN_PX, FG_OPEN_PX))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > FG_MIN_AREA_FRAC * H * W]
    if not keep:
        return {"keep": [], "stats": stats, "lab": lab}
    top = stats[keep, cv2.CC_STAT_TOP]
    left = stats[keep, cv2.CC_STAT_LEFT]
    r0, r1 = int(top.min()), int((top + stats[keep, cv2.CC_STAT_HEIGHT]).max())
    c0, c1 = int(left.min()), int((left + stats[keep, cv2.CC_STAT_WIDTH]).max())
    counts = (vol[:, r0:r1, c0:c1] > t).reshape(Z, -1).sum(axis=1)
    zs = np.nonzero(counts > FG_Z_FRAC * counts.max())[0]
    return {"keep": keep, "stats": stats, "lab": lab,
            "bbox": (r0, r1, c0, c1, int(zs[0]), int(zs[-1]) + 1)}


def _bilateral(fg: dict) -> bool:
    """≥2 fg components each ≥ BILAT_MIN_FRAC of the fg area whose column spans do not overlap."""
    keep, stats = fg["keep"], fg["stats"]
    area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in keep)
    big = [i for i in keep if stats[i, cv2.CC_STAT_AREA] >= BILAT_MIN_FRAC * area]
    spans = [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]))
             for i in big]
    return any(a1 <= b0 or b1 <= a0 for j, (a0, a1) in enumerate(spans) for (b0, b1) in spans[j + 1:])


def _header_json(ds: pydicom.Dataset) -> str:
    """Non-pixel header (file meta + dataset) as DICOM JSON with sorted keys."""
    return json.dumps({"file_meta": ds.file_meta.to_json_dict(), "dataset": ds.to_json_dict()},
                      sort_keys=True)


def _lat_tag(ds: pydicom.Dataset) -> str | None:
    """(0020,0060) Laterality, else (0020,0062) ImageLaterality, upper-cased; None if both are absent or empty."""
    for tag in ("Laterality", "ImageLaterality"):
        v = ds.get(tag)
        if v is not None and str(v).strip():
            return str(v).strip().upper()
    return None


def _check_meta(meta: dict) -> None:
    """Self-check: meta has exactly META_FIELDS with values of the declared kinds."""
    if set(meta) != set(META_FIELDS):
        raise RuntimeError(f"meta keys differ from META_FIELDS: {sorted(set(meta) ^ set(META_FIELDS))}")
    for k, (kind, nullable) in META_FIELDS.items():
        v = meta[k]
        if v is None:
            ok = nullable
        elif kind in ("floats", "strs"):
            et = float if kind == "floats" else str
            ok = type(v) is list and all(type(x) is et for x in v)
        else:
            ok = type(v) is _KIND_TYPES[kind]
        if not ok:
            raise RuntimeError(f"meta[{k!r}] = {v!r} is not of kind {kind} (nullable={nullable})")


def preprocess_series(files: list[tuple[str, bytes]], mm_per_px: float = 0.4,
                      clip_pct: tuple = (0.5, 99.5)) -> SeriesVolume:
    """Raw DICOM files (name, bytes) of one series → canonical SeriesVolume; raises SeriesError."""
    if not files:
        raise SeriesError("empty_series", "no files")
    named = sorted(((n.rsplit("/", 1)[-1], b) for n, b in files), key=lambda t: t[0])
    names = [n for n, _ in named]
    if len(set(names)) != len(names):
        dup = sorted(n for n, c in Counter(names).items() if c > 1)
        raise ValueError(f"duplicate file names in one series: {dup[:5]}")

    dropped, cands = [], []
    for name, data in named:
        s, d = _read_slice(name, data)
        if d is not None:
            dropped.append(d)
        else:
            cands.append(s)
    if not cands:
        reasons = Counter(d["reason"] for d in dropped)
        code = "missing_geometry" if set(reasons) == {"missing_geometry"} else "no_valid_slices"
        raise SeriesError(code, json.dumps(dict(sorted(reasons.items()))))

    # majority geometry group; representatives are taken in file-name order (input-order free)
    groups: list[list[_Slice]] = []
    for s in cands:
        for g in groups:
            if _same_geometry(g[0], s):
                g.append(s)
                break
        else:
            groups.append([s])
    groups.sort(key=len, reverse=True)
    if len(groups) > 1 and len(groups[0]) == len(groups[1]):
        raise SeriesError("geometry_ambiguous", json.dumps(
            [{"n": len(g), **_geometry_desc(g[0])} for g in groups]))
    main = groups[0]
    for g in groups[1:]:
        for s in g:
            dropped.append({"sop": s.sop, "file": s.file, "reason": "geometry_mismatch",
                            "detail": json.dumps({"slice": _geometry_desc(s), "majority": _geometry_desc(main[0])})})

    iop = np.median(np.stack([s.iop for s in main]), axis=0)
    ps = np.median(np.stack([s.ps for s in main]), axis=0)
    row = iop[:3] / np.linalg.norm(iop[:3])  # direction of increasing column index
    col = iop[3:] / np.linalg.norm(iop[3:])  # direction of increasing row index
    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)
    plane = int(np.abs(normal).argmax())
    order_sign, (cax, csgn), (rax, rsgn) = CANON[plane]
    n_dir = order_sign * np.sign(normal[plane]) * normal
    for s in main:
        s.pos = float(s.ipp @ n_dir)

    # order along the canonical normal; duplicate positions keep the lowest InstanceNumber
    def rank(s):
        return (s.inst if s.inst is not None else float("inf"), s.file)

    main.sort(key=lambda s: (s.pos, *rank(s)))
    kept, cluster = [], [main[0]]
    for s in main[1:] + [None]:
        if s is not None and s.pos - cluster[0].pos < DUP_TOL_MM:
            cluster.append(s)
            continue
        best = min(cluster, key=rank)
        kept.append(best)
        for o in cluster:
            if o is not best:
                identical = (o.arr.shape == best.arr.shape and np.array_equal(o.arr, best.arr)
                             and o.slope == best.slope and o.intercept == best.intercept)
                dropped.append({"sop": o.sop, "file": o.file, "reason": "duplicate_position",
                                "detail": json.dumps({"identical": bool(identical), "kept_sop": best.sop,
                                                      "kept_file": best.file})})
        cluster = [s]
    kept.sort(key=lambda s: s.pos)
    pos = np.array([s.pos for s in kept], dtype=np.float64)
    diffs = np.diff(pos)
    spacing = float(np.median(diffs)) if len(kept) > 1 else None
    gap_irregular = bool(len(kept) > 1 and (diffs > GAP_FACTOR * spacing).any())

    # canonical in-plane orientation (transpose and flips only)
    r_vec, c_vec, ps_rc = row.copy(), col.copy(), (float(ps[0]), float(ps[1]))
    h0, w0 = main[0].rows, main[0].cols
    transpose = bool(abs(r_vec[cax]) < abs(r_vec[rax]))
    if transpose:
        r_vec, c_vec, ps_rc, h0, w0 = c_vec, r_vec, ps_rc[::-1], w0, h0
    shift = np.zeros(3)
    flip_cols = bool(r_vec[cax] * csgn < 0)
    if flip_cols:
        shift += (w0 - 1) * ps_rc[1] * r_vec
        r_vec = -r_vec
    flip_rows = bool(c_vec[rax] * rsgn < 0)
    if flip_rows:
        shift += (h0 - 1) * ps_rc[0] * c_vec
        c_vec = -c_vec

    nh, nw = int(round(h0 * ps_rc[0] / mm_per_px)), int(round(w0 * ps_rc[1] / mm_per_px))
    if nh < 1 or nw < 1:
        raise SeriesError("tiny_image", f"{h0}x{w0} px at {ps_rc} mm → {nh}x{nw}")
    down = nh * nw < h0 * w0
    interp = cv2.INTER_AREA if down else cv2.INTER_LINEAR

    vol = np.empty((len(kept), nh, nw), dtype=np.float32)
    for k, s in enumerate(kept):
        img = s.arr.astype(np.float32) * s.slope + s.intercept
        if s.mono1:
            img = -img
        if transpose:
            img = img.T
        if flip_cols:
            img = img[:, ::-1]
        if flip_rows:
            img = img[::-1]
        vol[k] = cv2.resize(np.ascontiguousarray(img), (nw, nh), interpolation=interp)

    # one scratch copy for the selection; the rest in place (big allocations are slow)
    lo, hi = (float(v) for v in np.percentile(vol.reshape(-1).copy(), list(clip_pct), overwrite_input=True))
    if not hi > lo:
        raise SeriesError("flat_intensity", f"p{clip_pct[0]}={lo} p{clip_pct[1]}={hi}")
    vol -= lo
    vol /= hi - lo
    np.clip(vol, 0.0, 1.0, out=vol)
    vol *= 255.0
    np.rint(vol, out=vol)
    vol8 = vol.astype(np.uint8)
    del vol

    sy, sx = h0 * ps_rc[0] / nh, w0 * ps_rc[1] / nw
    origin = kept[0].ipp + shift + 0.5 * (sy - ps_rc[0]) * c_vec + 0.5 * (sx - ps_rc[1]) * r_vec
    pos32 = pos.astype(np.float32)
    head = kept[0].ds
    meta = {
        "study": str(head.StudyInstanceUID),
        "series": str(head.SeriesInstanceUID),
        "plane_computed": PLANES[plane],
        "iop": [float(x) for x in iop],
        "row_dir": [float(x) for x in c_vec],
        "col_dir": [float(x) for x in r_vec],
        "normal_dir": [float(x) for x in n_dir],
        "origin_lps": [float(x) for x in origin],
        "spacing_rc": [float(sy), float(sx)],
        "canon_transpose": transpose,
        "canon_flip_rows": flip_rows,
        "canon_flip_cols": flip_cols,
        "frame_of_reference_uid": str(head.FrameOfReferenceUID) if head.get("FrameOfReferenceUID") else None,
        "orig_rows": int(main[0].rows),
        "orig_cols": int(main[0].cols),
        "pixel_spacing": [float(x) for x in ps],
        "slice_thickness": float(head.SliceThickness) if head.get("SliceThickness") not in (None, "") else None,
        "slice_spacing_median": spacing,
        "gap_irregular": gap_irregular,
        "Z": len(kept), "H": nh, "W": nw,
        "mm_per_px": float(mm_per_px),
        "clip_pct": [float(x) for x in clip_pct],
        "interp": "area" if down else "linear",
        "lo": lo, "hi": hi,
        "n_files": len(files),
        "n_dropped": len(dropped),
        "transfer_syntaxes": sorted({s.ts for s in kept}),
        "header_json": _header_json(head),
        "lat_tag": _lat_tag(head),
    }

    fg = _foreground(vol8)
    meta["n_fg_components"] = len(fg["keep"])
    if fg["keep"]:
        r0, r1, c0, c1, z0, z1 = fg["bbox"]
        ys, xs = np.nonzero(np.isin(fg["lab"], fg["keep"]))
        center = canon_px_to_lps(meta, pos32, (z0 + z1 - 1) / 2, float(ys.mean()), float(xs.mean()))
        x = float(center[0])
        meta.update(fg_r0=r0, fg_r1=r1, fg_c0=c0, fg_c1=c1, fg_z0=z0, fg_z1=z1,
                    bilateral_suspect=None if plane == 0 else _bilateral(fg),
                    fg_center_lps=[float(v) for v in center],
                    lat_geom="L" if x > LAT_GEOM_MM else "R" if x < -LAT_GEOM_MM else None)
    else:
        meta.update(fg_r0=None, fg_r1=None, fg_c0=None, fg_c1=None, fg_z0=None, fg_z1=None,
                    bilateral_suspect=None, fg_center_lps=None, lat_geom=None)
    meta["lat_agree"] = (meta["lat_tag"] == meta["lat_geom"]
                         if meta["lat_tag"] is not None and meta["lat_geom"] is not None else None)
    _check_meta(meta)

    slices = [{"k": k, "sop": s.sop, "file": s.file, "instance_number": s.inst,
               "ipp_x": float(s.ipp[0]), "ipp_y": float(s.ipp[1]), "ipp_z": float(s.ipp[2]),
               "pos_mm": float(pos32[k]), "transfer_syntax": s.ts} for k, s in enumerate(kept)]
    dropped.sort(key=lambda d: (d["file"], d["reason"]))
    return SeriesVolume(vol=vol8, pos_mm=pos32, slices=slices, dropped=dropped, meta=meta)
