"""Joint-centred local views: study laterality, knee ROI per cached series, crop hook, canonical loader.

Works in exp-0002's cache frame: volume uint8 [Z,H,W] at a fixed mm/px, full FOV, in-plane
canonical in LPS (sagittal: slices R->L, cols A->P, rows S->I; coronal: slices A->P, cols R->L,
rows S->I; axial: slices S->I, cols R->L, rows A->P), no left/right mirroring.

Laterality canonicalisation (right knees made to look like left ones) lives in exactly two
places, shared by training and inference: `JointCrop` flips columns of coronal/axial crops,
`JointCache.load` reverses the slice order of sagittal volumes (and negates pos_mm).

Crop spec: a square of crop_mm_by_plane[plane] mm (src_px = round(crop_mm / mm_per_px) cache
pixels) around the ROI centre, area-resampled to out_px when src_px != out_px.

exp-0002 code is never imported here: `canon_px_to_lps(meta, pos_mm, k, r, c) -> LPS xyz` and the
KneeVolCache that `JointCache` wraps are injected.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# exp-0002 meta keys (= columns of its series.parquet). The ONLY place that names them:
# when exp-0002's code lands with other names, fix them here.
META_KEYS = {
    "study": "study",
    "series": "series",
    "plane": "plane_computed",
    "lat_tag": "lat_tag",
    "lat_geom": "lat_geom",
    "bilateral_suspect": "bilateral_suspect",
    "error_code": "error_code",  # series.parquet: not null = the series failed preprocessing, no volume
    "error_detail": "error_detail",
}
# error_code that JointCache gives the C3-degenerate series.
DEGENERATE_ERROR_CODE = "exp0005_degenerate_fg"
PLANES = ("Sagittal", "Coronal", "Axial")
SIDE_SOURCES = ("tag", "geom", "tag_conflict", "tag_geom_conflict", "geom_conflict", "unknown")
SIDES = ("L", "R")
# exp-0002 stores the raw, upper-cased tag. Values naming one knee -> its side (the full cache has
# R, L, RIGHT, LEFT); B = both and U = unpaired name no single knee: "no tag" for the side, reported in C3.
LAT_TAG_TO_SIDE = {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R"}
LAT_TAG_NO_SIDE = ("B", "U")
DEGENERATE_MODES = ("raise", "report", "image_centre")
METHODS = ("bbox", "bilateral_side", "bilateral_both", "degenerate", "image_centre")
# ROI dict fields. cy, cx: crop centre in pixels of the stored volume (None for "degenerate");
# src_px: side of the square source crop in cache pixels (the plane's crop_mm / mm_per_px);
# r0, r1, c0, c1: half-open bbox of the foreground region the centre came from (None when
# there is none); vol_h, vol_w: volume size the ROI belongs to (checked by the crop).
ROI_FIELDS = ("cy", "cx", "src_px", "method", "n_components", "side", "area_px",
              "vol_h", "vol_w", "r0", "r1", "c0", "c1")
# knee_rois.parquet = these id columns + ROI_FIELDS, one row per ROI.
ROI_TABLE_IDS = ("study", "series", "plane", "side_source", "roi_idx")
# degenerate.csv of the C3 run: series left out of knee_rois.parquet.
DEGENERATE_COLUMNS = ("study", "series", "plane", "side", "side_source", "n_components", "area_px",
                      "vol_h", "vol_w", "r0", "r1", "c0", "c1")


@dataclass(frozen=True)
class SeriesInfo:
    """The exp-0002 meta fields this module uses, read and validated in one place."""
    study: str
    series: str
    plane: str
    lat_tag: str | None  # 'L' / 'R' / None (LEFT/RIGHT normalised; B, U and missing -> None)
    lat_tag_raw: str | None  # the stored value: 'L' / 'LEFT' / 'R' / 'RIGHT' / 'B' / 'U' / None
    lat_geom: str | None
    bilateral_suspect: bool | None  # exp-0002: None for sagittal and for series without foreground


def _meta_value(meta, name):
    """meta[META_KEYS[name]]; a missing key raises with the available keys."""
    key = META_KEYS[name]
    if key not in meta:
        raise KeyError(f"exp-0002 meta has no {key!r} (needed as {name!r}); meta keys: "
                       f"{sorted(meta.keys())}. Fix META_KEYS in src/joint_view.py.")
    return meta[key]


def _missing(v):
    """True for None / NaN / pd.NA (how parquet and pandas give back a missing value)."""
    if v is None or v is pd.NA:
        return True
    return isinstance(v, (float, np.floating)) and bool(np.isnan(v))


def _side_value(v, name, series, allowed):
    """str value in `allowed` or None if missing; anything else raises."""
    if _missing(v):
        return None
    if v not in allowed:
        raise ValueError(f"series {series}: {name}={v!r}, expected one of {allowed} or missing")
    return str(v)


def series_info(meta) -> SeriesInfo:
    """Validated study/series/plane/laterality/bilateral fields of one exp-0002 meta (dict or row)."""
    series = str(_meta_value(meta, "series"))
    plane = _meta_value(meta, "plane")
    if plane not in PLANES:
        raise ValueError(f"series {series}: plane {plane!r} not in {PLANES}")
    bil = _meta_value(meta, "bilateral_suspect")
    if _missing(bil):
        bil = None
    elif isinstance(bil, (bool, np.bool_)):
        bil = bool(bil)
    else:
        raise ValueError(f"series {series}: bilateral_suspect={bil!r} is not a bool")
    tag = _side_value(_meta_value(meta, "lat_tag"), "lat_tag", series, tuple(LAT_TAG_TO_SIDE) + LAT_TAG_NO_SIDE)
    return SeriesInfo(
        study=str(_meta_value(meta, "study")), series=series, plane=str(plane),
        lat_tag=LAT_TAG_TO_SIDE.get(tag), lat_tag_raw=tag,
        lat_geom=_side_value(_meta_value(meta, "lat_geom"), "lat_geom", series, SIDES),
        bilateral_suspect=bil)


def study_side(metas) -> tuple[str | None, str]:
    """One laterality per study from its series' meta: (side, source), source in SIDE_SOURCES.

    Unique L/R lat_tag (LEFT/RIGHT normalised) -> it ("tag"), unless some series' lat_geom says the
    other side -> (None, "tag_geom_conflict"): a wrong side swaps the medial and lateral targets, so a
    disagreement leaves the study unmirrored (lead, 17.10). Several tags -> (None, "tag_conflict").
    No L/R tag (B/U count as none) -> the unanimous non-missing lat_geom ("geom"); several ->
    (None, "geom_conflict"); nothing -> (None, "unknown").
    """
    infos = [series_info(m) for m in metas]
    if not infos:
        raise ValueError("study_side: a study without series")
    studies = {i.study for i in infos}
    if len(studies) != 1:
        raise ValueError(f"study_side: series of several studies {sorted(studies)}")
    tags = {i.lat_tag for i in infos} - {None}
    geoms = {i.lat_geom for i in infos} - {None}
    if len(tags) == 1:
        if geoms - tags:
            return None, "tag_geom_conflict"
        return tags.pop(), "tag"
    if tags:
        return None, "tag_conflict"
    if len(geoms) == 1:
        return geoms.pop(), "geom"
    if geoms:
        return None, "geom_conflict"
    return None, "unknown"


def needs_column_flip(plane, side) -> bool:
    """Canonicalisation rule: coronal/axial crops of a right knee get their columns flipped."""
    return side == "R" and plane in ("Coronal", "Axial")


def needs_slice_reversal(plane, side) -> bool:
    """Canonicalisation rule: sagittal volumes of a right knee get their slice order reversed."""
    return side == "R" and plane == "Sagittal"


def foreground_components(vol, *, open_px, min_area_frac) -> list[dict]:
    """Foreground components of a volume, largest first: MIP over Z -> Otsu -> opening -> area > min_area_frac.

    exp-0002's fg-bbox recipe (its union bbox = the bbox of all returned components).
    Each: {area, r0, r1, c0, c1 (half-open bbox), cy, cx (centroid)} in stored-volume pixels.
    """
    if vol.dtype != np.uint8 or vol.ndim != 3 or len(vol) == 0:
        raise ValueError(f"expected a non-empty uint8 [Z,H,W] volume, got {vol.dtype} {vol.shape}")
    mip = np.ascontiguousarray(np.asarray(vol).max(axis=0))
    _, mask = cv2.threshold(mip, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mip.shape
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area > min_area_frac * h * w:
            comps.append(dict(area=area, r0=y, r1=y + ch, c0=x, c1=x + cw,
                              cy=float(cents[i][1]), cx=float(cents[i][0])))
    return sorted(comps, key=lambda c: -c["area"])


def src_px_by_plane(crop_mm_by_plane, mm_per_px) -> dict:
    """{plane: source crop side in cache pixels} = round(crop_mm / mm_per_px); every plane must be given."""
    if set(crop_mm_by_plane) != set(PLANES):
        raise ValueError(f"crop_mm_by_plane keys {sorted(crop_mm_by_plane)} != {sorted(PLANES)}")
    if not mm_per_px > 0 or not all(crop_mm_by_plane[p] > 0 for p in PLANES):
        raise ValueError(f"non-positive crop spec {crop_mm_by_plane} at {mm_per_px} mm/px")
    return {p: int(round(crop_mm_by_plane[p] / mm_per_px)) for p in PLANES}


def _box_roi(box, method, *, area, side, src_px, n_components, shape):
    """ROI dict centred on a half-open bbox {r0, r1, c0, c1}."""
    return dict(cy=(box["r0"] + box["r1"] - 1) // 2, cx=(box["c0"] + box["c1"] - 1) // 2,
                src_px=src_px, method=method, n_components=n_components, side=side,
                area_px=area, vol_h=shape[0], vol_w=shape[1],
                r0=box["r0"], r1=box["r1"], c0=box["c0"], c1=box["c1"])


def locate_knee(vol, pos_mm, meta, side, *, crop_mm_by_plane, canon_px_to_lps, mm_per_px, bilateral_min_mm,
                degenerate_min_mm, open_px, min_area_frac, on_degenerate="raise") -> list[dict]:
    """In-plane crop centre(s) of the knee in one cached series -> ROI dicts (ROI_FIELDS), largest region first.

    src_px of every ROI = round(crop_mm_by_plane[plane] / mm_per_px); centring does not depend on it.
    Coronal/axial with the two largest >= bilateral_min_mm wide components horizontally disjoint
    -> bilateral: side known -> the component whose centroid LPS x matches it (left = larger x),
    method "bilateral_side"; side None -> both, "bilateral_both". Otherwise the centre of the
    union bbox, "bbox". Degenerate (no component, or union bbox narrower than degenerate_min_mm
    in columns): on_degenerate "raise" -> ValueError; "report" -> one "degenerate" ROI with no
    centre; "image_centre" (inference only) -> the image centre, marked "image_centre".
    """
    info = series_info(meta)
    if on_degenerate not in DEGENERATE_MODES:
        raise ValueError(f"on_degenerate={on_degenerate!r} not in {DEGENERATE_MODES}")
    if side is not None and side not in SIDES:
        raise ValueError(f"side={side!r} not in {SIDES} or None")
    if len(pos_mm) != len(vol):
        raise ValueError(f"series {info.series}: {len(pos_mm)} positions for {len(vol)} slices")
    src_px = src_px_by_plane(crop_mm_by_plane, mm_per_px)[info.plane]
    comps = foreground_components(vol, open_px=open_px, min_area_frac=min_area_frac)
    shape = vol.shape[1:]
    common = dict(side=side, src_px=src_px, n_components=len(comps), shape=shape)
    area = sum(c["area"] for c in comps)
    union = None
    if comps:
        union = dict(r0=min(c["r0"] for c in comps), r1=max(c["r1"] for c in comps),
                     c0=min(c["c0"] for c in comps), c1=max(c["c1"] for c in comps))
    width_mm = 0.0 if union is None else (union["c1"] - union["c0"]) * mm_per_px

    if union is None or width_mm < degenerate_min_mm:
        detail = (f"series {info.series} ({info.plane}): degenerate foreground, {len(comps)} components, "
                  f"union bbox width {width_mm:.1f} mm < {degenerate_min_mm} mm")
        if on_degenerate == "raise":
            raise ValueError(detail + "; pass on_degenerate='report' or 'image_centre' to handle it")
        if on_degenerate == "report":
            roi = dict(cy=None, cx=None, src_px=src_px, method="degenerate", n_components=len(comps),
                       side=side, area_px=area, vol_h=shape[0], vol_w=shape[1],
                       r0=None, r1=None, c0=None, c1=None)
            if union is not None:
                roi.update(union)
            return [roi]
        full = dict(r0=0, r1=shape[0], c0=0, c1=shape[1])
        roi = _box_roi(full, "image_centre", area=area, **common)
        roi.update(r0=None, r1=None, c0=None, c1=None)
        return [roi]

    if info.plane in ("Coronal", "Axial"):
        wide = [c for c in comps if (c["c1"] - c["c0"]) * mm_per_px >= bilateral_min_mm][:2]
        if len(wide) == 2 and (wide[0]["c1"] <= wide[1]["c0"] or wide[1]["c1"] <= wide[0]["c0"]):
            if side is None:
                return [_box_roi(c, "bilateral_both", area=c["area"], **common) for c in wide]
            k = len(vol) // 2
            xs = [float(canon_px_to_lps(meta, pos_mm, k, c["cy"], c["cx"])[0]) for c in wide]
            if not np.isfinite(xs).all() or xs[0] == xs[1]:
                raise ValueError(f"series {info.series}: bilateral components with LPS x {xs}, cannot pick side {side}")
            pick = int(np.argmax(xs)) if side == "L" else int(np.argmin(xs))
            return [_box_roi(wide[pick], "bilateral_side", area=wide[pick]["area"], **common)]
    return [_box_roi(union, "bbox", area=area, **common)]


def crop_padded(slices, cy, cx, size) -> np.ndarray:
    """size x size crop of uint8 [m,H,W] centred at (cy, cx) at native scale; zeros outside the FOV."""
    m, h, w = slices.shape
    y0, x0 = int(cy) - size // 2, int(cx) - size // 2
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y0 + size, h), max(x0, 0), min(x0 + size, w)
    if sy1 <= sy0 or sx1 <= sx0:
        raise ValueError(f"crop at ({cy}, {cx}) size {size} does not overlap the {h}x{w} image")
    out = np.zeros((m, size, size), np.uint8)
    out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = slices[:, sy0:sy1, sx0:sx1]
    return out


def resize_slices(slices, size) -> np.ndarray:
    """uint8 [m,S,S] -> uint8 [m,size,size], cv2.INTER_AREA per slice."""
    return np.stack([cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in slices])


class JointCrop:
    """exp-0004 C10 crop_fn over precomputed ROIs: knee-centred src_px crop, resample to out_px, R coronal/axial flip."""

    def __init__(self, rois, sides, out_px, crop_mm_by_plane, mm_per_px, jitter_px):
        if jitter_px < 0:
            raise ValueError(f"jitter_px={jitter_px} < 0")
        self.src_px = src_px_by_plane(crop_mm_by_plane, mm_per_px)
        for series, lst in rois.items():
            methods = {r["method"] for r in lst}
            if not lst or len(methods) != 1:
                raise ValueError(f"series {series}: ROI list {lst} is empty or mixes methods")
            method = methods.pop()
            if method not in METHODS or method == "degenerate":
                raise ValueError(f"series {series}: method {method!r} cannot be cropped")
            if len(lst) != (2 if method == "bilateral_both" else 1):
                raise ValueError(f"series {series}: {len(lst)} ROIs for method {method}")
            if any(r["src_px"] not in self.src_px.values() for r in lst):
                raise ValueError(f"series {series}: ROI src_px {[r['src_px'] for r in lst]} not in the spec {self.src_px}")
        for study, s in sides.items():
            if s is not None and s not in SIDES:
                raise ValueError(f"study {study}: side {s!r} not in {SIDES} or None")
        self.rois, self.sides, self.out_px, self.jitter_px = rois, sides, out_px, jitter_px

    def __call__(self, slices, meta, train, rng):
        info = series_info(meta)
        if info.series not in self.rois:
            raise KeyError(f"series {info.series} has no ROI (degenerate or not located)")
        if info.study not in self.sides:
            raise KeyError(f"study {info.study} has no side entry")
        if slices.dtype != np.uint8 or slices.ndim != 3:
            raise ValueError(f"series {info.series}: expected uint8 [m,H,W], got {slices.dtype} {slices.shape}")
        lst = self.rois[info.series]
        roi = lst[int(rng.integers(len(lst)))] if train and len(lst) > 1 else lst[0]
        if roi["src_px"] != self.src_px[info.plane]:
            raise ValueError(f"series {info.series} ({info.plane}): ROI src_px {roi['src_px']} != spec "
                             f"{self.src_px[info.plane]} (ROIs made with another crop spec?)")
        if slices.shape[1:] != (roi["vol_h"], roi["vol_w"]):
            raise ValueError(f"series {info.series}: slices {slices.shape[1:]} vs ROI volume "
                             f"{(roi['vol_h'], roi['vol_w'])} (ROIs from another cache?)")
        cy, cx = roi["cy"], roi["cx"]
        if train and self.jitter_px:
            dy, dx = rng.integers(-self.jitter_px, self.jitter_px + 1, size=2)
            cy, cx = cy + int(dy), cx + int(dx)
        out = crop_padded(slices, cy, cx, roi["src_px"])
        if roi["src_px"] != self.out_px:
            out = resize_slices(out, self.out_px)
        if needs_column_flip(info.plane, self.sides[info.study]):
            out = np.ascontiguousarray(out[:, :, ::-1])
        return out


def make_joint_crop_fn(rois, sides, out_px, crop_mm_by_plane, mm_per_px, jitter_px) -> JointCrop:
    """crop_fn (exp-0004 C10) -> uint8 [m,out_px,out_px]; bilateral_both: random ROI in train, rois[0] at eval.

    jitter_px (train only) is in source cache pixels for every plane.
    """
    return JointCrop(rois, sides, out_px, crop_mm_by_plane, mm_per_px, jitter_px)


class JointCache:
    """exp-0002's KneeVolCache as exp-0004's run_folds sees it: C3-degenerate series as error rows, sagittal R-knee reversed.

    Implements only what exp-0004's reader25d/train25d touch (.series, .load, .meta, .pos_mm) and
    passes nothing else through, so a new dependency fails here loudly.
    - .series: a copy of cache.series whose degenerate series are error rows (error_code
      DEGENERATE_ERROR_CODE): cache_study_table gives them 0 slices, cache_depth_rows skips them;
      the row set stays the cache's (= the competition CSV series).
    - .load(series): cache.load plus the sagittal R-knee slice reversal (needs_slice_reversal): vol
      reversed (stays an mmap view), pos_mm negated and reversed (still increasing), meta unchanged,
      so no exp-0002 geometry helper may be applied to it. A degenerate series raises.
    - .meta / .pos_mm: the cache's own, original geometry and slice order. That is safe although .load
      reverses sagittal R stacks: joint_depth_fractions uses them to restrict axial stacks only (never
      reversed) and gives every other series (DEPTH_LO, DEPTH_HI), which is symmetric, so the window
      range is the same in either slice order. The symmetry is asserted here.
    """

    def __init__(self, cache, sides, degenerate):
        from src.reader25d import DEPTH_HI, DEPTH_LO  # exp-0004

        if abs(DEPTH_LO + DEPTH_HI - 1.0) > 1e-12:
            raise ValueError(f"reader25d depth range ({DEPTH_LO}, {DEPTH_HI}) is not symmetric: the sagittal R "
                             "reversal in .load would no longer match the .pos_mm-based depth fractions")
        st_col, s_col, e_col, d_col = (META_KEYS[k] for k in ("study", "series", "error_code", "error_detail"))
        series = cache.series.copy()
        missing = [c for c in (st_col, s_col, e_col, d_col) if c not in series.columns]
        if missing:
            raise KeyError(f"cache series table has no {missing}. Fix META_KEYS in src/joint_view.py.")
        uids = series[s_col].astype(str)
        if not uids.is_unique:
            raise ValueError("cache series table has duplicate series")
        degenerate = set(map(str, degenerate))
        unknown = degenerate - set(uids)
        if unknown:
            raise KeyError(f"{len(unknown)} degenerate series are not in the cache, e.g. {next(iter(unknown))}")
        is_deg = uids.isin(degenerate)
        failed = series.loc[is_deg & series[e_col].notna(), s_col]
        if len(failed):
            raise ValueError(f"{len(failed)} degenerate series already failed in the cache, e.g. {failed.iloc[0]}")
        series.loc[is_deg, e_col] = DEGENERATE_ERROR_CODE
        series.loc[is_deg, d_col] = ("exp-0005 C3: degenerate foreground (no component, or the union bbox narrower "
                                     "than degenerate_min_mm), so no knee ROI")
        for study, side in sides.items():
            if side is not None and side not in SIDES:
                raise ValueError(f"study {study}: side {side!r} not in {SIDES} or None")
        no_side = set(series.loc[series[e_col].isna(), st_col].astype(str)) - set(sides)
        if no_side:
            raise KeyError(f"{len(no_side)} studies with usable series have no side entry, e.g. {next(iter(no_side))}")
        self.series = series
        self._cache, self._sides, self._degenerate = cache, dict(sides), degenerate

    def load(self, series):
        """(vol, pos_mm, meta) of a usable series; sagittal R-knee stacks reversed, pos_mm still increasing."""
        if str(series) in self._degenerate:
            raise ValueError(f"series {series} is C3-degenerate: it has no knee ROI and must not be loaded")
        vol, pos_mm, meta = self._cache.load(series)
        info = series_info(meta)
        if info.series != str(series):
            raise ValueError(f"cache.load({series}) returned meta of series {info.series}")
        if not needs_slice_reversal(info.plane, self._sides[info.study]):
            return vol, pos_mm, meta
        pos_mm = np.asarray(pos_mm)
        if len(pos_mm) != len(vol) or not (np.diff(pos_mm) > 0).all():
            raise ValueError(f"series {series}: pos_mm is not strictly increasing over {len(vol)} slices")
        return vol[::-1], -pos_mm[::-1], meta

    def meta(self, series):
        """The cache's meta of a series (original geometry)."""
        return self._cache.meta(series)

    def pos_mm(self, series):
        """The cache's slice positions of a series (original slice order, see the class docstring)."""
        return self._cache.pos_mm(series)


def roi_rows(info: SeriesInfo, side_source, rois) -> list[dict]:
    """knee_rois.parquet rows of one series: ROI_TABLE_IDS + ROI_FIELDS, roi_idx in list order."""
    return [dict(study=info.study, series=info.series, plane=info.plane, side_source=side_source,
                 roi_idx=i, **{f: roi[f] for f in ROI_FIELDS}) for i, roi in enumerate(rois)]


def rois_from_table(df) -> tuple[dict, dict]:
    """knee_rois table -> ({series: [ROI]} ordered by roi_idx, {study: side}) for make_joint_crop_fn."""
    cols = set(ROI_TABLE_IDS) | set(ROI_FIELDS)
    if set(df.columns) != cols:
        raise ValueError(f"ROI table columns {sorted(df.columns)} != {sorted(cols)}")
    if (df["method"] == "degenerate").any():
        raise ValueError("ROI table holds degenerate rows")
    ints = ("cy", "cx", "src_px", "n_components", "area_px", "vol_h", "vol_w", "r0", "r1", "c0", "c1")
    rois, sides, idx = {}, {}, {}
    for row in df.sort_values(["series", "roi_idx"]).itertuples(index=False):
        roi = {f: getattr(row, f) for f in ROI_FIELDS}
        roi.update({f: int(roi[f]) for f in ints})
        roi["method"] = str(roi["method"])
        roi["side"] = None if _missing(roi["side"]) else str(roi["side"])
        rois.setdefault(str(row.series), []).append(roi)
        idx.setdefault(str(row.series), []).append(int(row.roi_idx))
        if sides.setdefault(str(row.study), roi["side"]) != roi["side"]:
            raise ValueError(f"study {row.study}: several sides in the ROI table")
    for series, lst in idx.items():
        if lst != list(range(len(lst))):
            raise ValueError(f"series {series}: roi_idx {lst} is not 0..{len(lst) - 1}")
    return rois, sides


def load_roi_run(roi_dir, *, cache_key, crop_mm_by_plane, mm_per_px, out_px, locate) -> tuple[dict, dict, set]:
    """C3 run dir -> (rois {series: [ROI]}, sides {study: side}, degenerate series); checks cache key, crop spec, locate params."""
    roi_dir = Path(roi_dir)
    stats = json.loads((roi_dir / "stats.json").read_text())
    want = {"cache_key": cache_key, "crop_mm_by_plane": dict(crop_mm_by_plane), "mm_per_px": mm_per_px,
            "out_px": out_px, "locate": dict(locate)}
    missing = [k for k in want if k not in stats]
    if missing:
        raise ValueError(f"ROI run {roi_dir.name}: stats.json has no {missing} (made before the per-plane crop spec?)")
    got = {k: stats[k] for k in want}
    if got != want:
        raise ValueError(f"ROI run {roi_dir.name} was made with {got}, expected {want}")
    df = pd.read_parquet(roi_dir / "knee_rois.parquet")
    src = df["plane"].map(src_px_by_plane(crop_mm_by_plane, mm_per_px))
    if (df["src_px"] != src).any():
        raise ValueError(f"ROI run {roi_dir.name}: {(df['src_px'] != src).sum()} ROIs with src_px off the spec")
    rois, sides = rois_from_table(df)
    deg = pd.read_csv(roi_dir / "degenerate.csv")
    if tuple(deg.columns) != DEGENERATE_COLUMNS:
        raise ValueError(f"{roi_dir / 'degenerate.csv'}: columns {list(deg.columns)} != {list(DEGENERATE_COLUMNS)}")
    degenerate = set(deg["series"].astype(str))
    if degenerate & set(rois):
        raise ValueError(f"ROI run {roi_dir.name}: series both located and degenerate")
    if len(rois) + len(degenerate) != stats["n_series"]:
        raise ValueError(f"ROI run {roi_dir.name}: {len(rois)} located + {len(degenerate)} degenerate "
                         f"!= {stats['n_series']} series")
    return rois, sides, degenerate


def check_table_rois(table, rois) -> None:
    """Raises unless every series kept in exp-0004's study table ({study: [[series of slot]]}) has ROIs."""
    missing = [s for slots in table.values() for cands in slots for s in cands if s not in rois]
    if missing:
        raise KeyError(f"{len(missing)} series of the study table have no ROI, e.g. {missing[0]}")
