"""Reporting for the knee-ROI pass (exp-0005 C3): method/side/laterality/bilateral stats and PNG montage tiles."""
from collections import Counter

import cv2
import numpy as np

from src.joint_view import METHODS, PLANES, SIDE_SOURCES, needs_column_flip, needs_slice_reversal

BILATERAL_METHODS = ("bilateral_side", "bilateral_both")
LAT_DISAGREEMENT_COLUMNS = ("study", "series", "plane", "lat_tag", "lat_geom", "study_side", "side_source")
BILATERAL_DISAGREEMENT_COLUMNS = ("study", "series", "plane", "our_method", "bilateral_suspect")


def wider_than_crop(infos, rois_by_series, crop_mm_by_plane, mm_per_px) -> dict:
    """{plane: {n, n_located, series}}: located series with a region wider (columns) than the plane's crop_mm."""
    out = {p: {"n": 0, "n_located": 0, "series": []} for p in PLANES}
    for i in infos:
        rois = rois_by_series[i.series]
        if rois[0]["method"] == "degenerate":
            continue
        out[i.plane]["n_located"] += 1
        if any((r["c1"] - r["c0"]) * mm_per_px > crop_mm_by_plane[i.plane] for r in rois):
            out[i.plane]["series"].append(i.series)
    for v in out.values():
        v["series"].sort()
        v["n"] = len(v["series"])
    return out


def method_counts(infos, rois_by_series) -> dict:
    """{plane: {method: n_series}} over the located series (method of the first ROI)."""
    out = {p: dict.fromkeys(METHODS, 0) for p in PLANES}
    for i in infos:
        out[i.plane][rois_by_series[i.series][0]["method"]] += 1
    return out


def side_stats(infos, sides) -> tuple[dict, list[dict]]:
    """Studies by side and side source, series by plane x canonicalisation, lat_tag vs lat_geom disagreements.

    sides: {study: (side, source)}. Returns (stats, rows of the series whose lat_tag and lat_geom
    are both present and differ).
    """
    by_study = {}
    for i in infos:
        by_study.setdefault(i.study, []).append(i)
    if set(by_study) != set(sides):
        raise ValueError("side_stats: series studies and side studies differ")
    side_names = {"L": "L_kept", "R": "R_mirrored", None: "unknown"}
    studies_by_side = dict.fromkeys(side_names.values(), 0)
    studies_by_source = dict.fromkeys(SIDE_SOURCES, 0)
    for side, source in sides.values():
        studies_by_side[side_names[side]] += 1
        studies_by_source[source] += 1

    canon = {p: {"kept_L": 0, "mirrored_R": 0, "unknown_side": 0} for p in PLANES}
    for i in infos:
        side = sides[i.study][0]
        mirrored = needs_column_flip(i.plane, side) or needs_slice_reversal(i.plane, side)
        if mirrored != (side == "R"):
            raise AssertionError(f"canonicalisation rule does not cover {i.plane} {side}")
        canon[i.plane]["mirrored_R" if mirrored else "kept_L" if side == "L" else "unknown_side"] += 1

    rows = []
    ser = {p: {"n_both": 0, "n_disagree": 0} for p in PLANES}
    for i in infos:
        if i.lat_tag is not None and i.lat_geom is not None:
            ser[i.plane]["n_both"] += 1
            if i.lat_tag != i.lat_geom:
                ser[i.plane]["n_disagree"] += 1
                side, source = sides[i.study]
                rows.append(dict(study=i.study, series=i.series, plane=i.plane, lat_tag=i.lat_tag,
                                 lat_geom=i.lat_geom, study_side=side, side_source=source))
    ser_studies = sorted({r["study"] for r in rows})

    st_both, st_dis = 0, []
    for study, lst in by_study.items():
        tags = {i.lat_tag for i in lst} - {None}
        geoms = {i.lat_geom for i in lst} - {None}
        if tags and geoms:
            st_both += 1
            if tags != geoms:
                st_dis.append(study)
    st_dis.sort()
    no_side = sorted((i for i in infos if i.lat_tag_raw is not None and i.lat_tag is None),
                     key=lambda i: (i.study, i.series))
    stats = {
        "n_studies": len(sides),
        "studies_by_side": studies_by_side,
        "studies_by_side_source": studies_by_source,
        "series_by_plane_canonicalisation": canon,
        "lat_tag_without_side": {  # ImageLaterality B / U: counted as no tag
            "n_series": len(no_side),
            "by_value": dict(Counter(i.lat_tag_raw for i in no_side)),
            "study_ids": sorted({i.study for i in no_side}),
        },
        "lat_tag_vs_geom": {
            "per_series": {
                "n_both": sum(v["n_both"] for v in ser.values()),
                "n_disagree": len(rows),
                "by_plane": ser,
                "study_ids": ser_studies,
            },
            "per_study": {
                "definition": "non-missing lat_tag set != non-missing lat_geom set over the study's series",
                "n_both": st_both,
                "n_disagree": len(st_dis),
                "by_side_source": dict(Counter(sides[s][1] for s in st_dis)),
                "study_ids": st_dis,
            },
        },
    }
    return stats, rows


def bilateral_agreement(infos, rois_by_series) -> tuple[dict, list[dict]]:
    """Coronal/axial: our bilateral detection vs exp-0002's bilateral_suspect (2x2 per plane, disagreeing series).

    Our degenerate series are left out; a located series whose bilateral_suspect is missing
    (exp-0002 saw no foreground) is counted as "theirs_missing" and listed.
    """
    out, rows, n_degenerate = {}, [], 0
    for plane in ("Coronal", "Axial"):
        t = {"both": 0, "ours_only": 0, "theirs_only": 0, "neither": 0, "theirs_missing": 0}
        for i in infos:
            if i.plane != plane:
                continue
            method = rois_by_series[i.series][0]["method"]
            if method == "degenerate":
                n_degenerate += 1
                continue
            ours, theirs = method in BILATERAL_METHODS, i.bilateral_suspect
            if theirs is None:
                t["theirs_missing"] += 1
                rows.append(dict(study=i.study, series=i.series, plane=plane, our_method=method,
                                 bilateral_suspect=None))
                continue
            t["both" if ours and theirs else "ours_only" if ours else "theirs_only" if theirs else "neither"] += 1
            if ours != theirs:
                rows.append(dict(study=i.study, series=i.series, plane=plane, our_method=method,
                                 bilateral_suspect=theirs))
        n = sum(t.values()) - t["theirs_missing"]
        t["agreement"] = (t["both"] + t["neither"]) / n if n else None
        out[plane] = t
    out["n_degenerate_excluded"] = n_degenerate
    return out, rows


_GREEN, _YELLOW, _CYAN, _RED, _WHITE = (0, 255, 0), (0, 255, 255), (255, 200, 0), (0, 0, 255), (255, 255, 255)


def _panel(img, rois, tile_px):
    """uint8 [H,W] -> BGR tile_px square, scaled to fit, with region bboxes and crop boxes drawn."""
    h, w = img.shape
    s = tile_px / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((tile_px, tile_px, 3), 40, np.uint8)
    out[:small.shape[0], :small.shape[1]] = small[..., None]
    for j, roi in enumerate(rois):
        if roi["r0"] is not None:
            cv2.rectangle(out, (int(roi["c0"] * s), int(roi["r0"] * s)),
                          (int((roi["c1"] - 1) * s), int((roi["r1"] - 1) * s)), _CYAN, 1)
        if roi["cy"] is not None:
            y0, x0 = roi["cy"] - roi["src_px"] // 2, roi["cx"] - roi["src_px"] // 2
            cv2.rectangle(out, (int(x0 * s), int(y0 * s)),
                          (int((x0 + roi["src_px"] - 1) * s), int((y0 + roi["src_px"] - 1) * s)),
                          _GREEN if j == 0 else _YELLOW, 2)
    return out


def roi_tile(vol, crop, rois, title_lines, *, tile_px) -> np.ndarray:
    """BGR tile: title | MIP and mid-slice with region bbox (cyan) and src_px crop box(es) (green = eval) | final crop."""
    vol = np.asarray(vol)
    left = _panel(np.ascontiguousarray(vol.max(axis=0)), rois, tile_px)
    mid = _panel(np.ascontiguousarray(vol[len(vol) // 2]), rois, tile_px)
    right = np.full((tile_px, tile_px, 3), 40, np.uint8)
    if crop is None:
        cv2.putText(right, "no crop: " + rois[0]["method"], (10, tile_px // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, _RED, 2, cv2.LINE_AA)
    else:
        right[:] = cv2.resize(crop, (tile_px, tile_px), interpolation=cv2.INTER_AREA)[..., None]
    title = np.zeros((18 * len(title_lines) + 6, 3 * tile_px, 3), np.uint8)
    for j, line in enumerate(title_lines):
        cv2.putText(title, line, (4, 16 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1, cv2.LINE_AA)
    return np.vstack([title, np.hstack([left, mid, right])])


def tile_grid(tiles, cols, *, gap_px) -> np.ndarray:
    """Equal-size BGR tiles -> one page image, row-major, `cols` per row, gap_px separators."""
    if not tiles:
        raise ValueError("tile_grid: no tiles")
    h, w = tiles[0].shape[:2]
    if any(t.shape != tiles[0].shape for t in tiles):
        raise ValueError("tile_grid: tiles differ in size")
    rows = -(-len(tiles) // cols)
    page = np.full((rows * (h + gap_px) + gap_px, cols * (w + gap_px) + gap_px, 3), 90, np.uint8)
    for j, t in enumerate(tiles):
        y, x = gap_px + (j // cols) * (h + gap_px), gap_px + (j % cols) * (w + gap_px)
        page[y:y + h, x:x + w] = t
    return page
