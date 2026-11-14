"""C3: knee ROI of every cached series -> knee_rois.parquet, montages/<plane>_*.png, stats.json.

study_side per study + locate_knee(on_degenerate="report") per series over exp-0002's cache,
CPU only, a pool of <= 16 processes. Run under nice:

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/locate_all.py

Crop spec and locate_knee parameters: src/joint_reader.py (one source of truth for C3, training
and inference; stats.json records them and load_roi_run checks them before training reads the ROIs).

Artifacts in ctx.output_dir: knee_rois.parquet (non-degenerate series, one row per ROI),
degenerate.csv, lat_disagreement.csv, bilateral_disagreement.csv, montages/ (+ index.csv),
stats.json. Visual gate: stats.json and the montage paths go to the lead before any training
reads the ROIs.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.joint_reader import CROP_MM_BY_PLANE, LOCATE, OUT_PX  # noqa: E402
from src.joint_view import (  # noqa: E402
    DEGENERATE_COLUMNS, PLANES, ROI_FIELDS, ROI_TABLE_IDS, JointCache, locate_knee, make_joint_crop_fn,
    roi_rows, series_info, study_side)
from src.joint_view_report import (  # noqa: E402
    BILATERAL_DISAGREEMENT_COLUMNS, BILATERAL_METHODS, LAT_DISAGREEMENT_COLUMNS, bilateral_agreement,
    method_counts, roi_tile, side_stats, tile_grid, wider_than_crop)
from utils.ctx import get_ctx  # noqa: E402

CACHE_KEY = "exp-0002/knee-vol04"  # the full cache (the pilot run used "exp-0002/knee-vol04-pilot")
SEED = 0
N_WORKERS = 16  # shared machine: <= 16 processes, under nice -n 19
N_RANDOM = 24  # random montage series per plane, plus every bilateral and degenerate one
WIDE_MM = 180.0  # montage group 'wide': located region at least this wide (a missed bilateral FOV?)
TILES_PER_PAGE, TILE_COLS, TILE_PX = 12, 3, 256

_CACHE = _CANON = _JOINT = None


def _init_worker(root, cache_cls, canon_fn):
    global _CACHE, _CANON
    cv2.setNumThreads(1)
    _CACHE, _CANON = cache_cls(root), canon_fn


def _init_tile_worker(root, cache_cls, sides, degenerate):
    global _CACHE, _JOINT
    cv2.setNumThreads(1)
    _CACHE = cache_cls(root)
    _JOINT = JointCache(_CACHE, sides, degenerate)


def _locate(task):
    series, side = task
    vol, pos_mm, meta = _CACHE.load(series)
    rois = locate_knee(vol, pos_mm, meta, side, crop_mm_by_plane=CROP_MM_BY_PLANE, canon_px_to_lps=_CANON,
                       on_degenerate="report", **LOCATE)
    return series, rois


def _tile(task):
    series, study, side, rois, title = task
    vol, _, meta = _CACHE.load(series)
    crop = None
    if rois[0]["method"] != "degenerate":
        # The canonicalised crop exactly as training sees it at eval: JointCache.load + crop_fn.
        cvol, _, cmeta = _JOINT.load(series)
        crop_fn = make_joint_crop_fn({series: rois}, {study: side}, OUT_PX, CROP_MM_BY_PLANE,
                                     LOCATE["mm_per_px"], jitter_px=0)
        crop = crop_fn(np.asarray(cvol[[len(cvol) // 2]]), cmeta, False, None)[0]
    return roi_tile(vol, crop, rois, title, tile_px=TILE_PX)


def run(cache_root, out_dir, n_workers, cache_cls, canon_fn):
    """The whole C3 pass; cache_cls = exp-0002's KneeVolCache, canon_fn = its canon_px_to_lps (injected)."""
    if n_workers > 16:
        raise ValueError(f"n_workers={n_workers}: the shared machine allows <= 16 processes")
    t0 = time.time()
    cache = cache_cls(cache_root)
    # exp-0002 keeps series that failed preprocessing as error rows (no volume, no meta): skip and count them.
    failed = cache.series[cache.series["error_code"].notna()]
    metas = cache.series[cache.series["error_code"].isna()].to_dict("records")
    infos = [series_info(m) for m in metas]
    if len({i.series for i in infos}) != len(infos):
        raise ValueError("duplicate series uids in the cache's series table")
    by_study = {}
    for m, i in zip(metas, infos):
        by_study.setdefault(i.study, []).append(m)
    sides = {study: study_side(ms) for study, ms in by_study.items()}
    print(f"[locate] {len(infos)} series, {len(sides)} studies; {n_workers} workers", flush=True)

    ctx_mp = mp.get_context("spawn")
    tasks = [(i.series, sides[i.study][0]) for i in infos]
    pool = dict(max_workers=n_workers, mp_context=ctx_mp, initializer=_init_worker,
                initargs=(str(cache_root), cache_cls, canon_fn))
    with ProcessPoolExecutor(**pool) as ex:
        rois_by_series = dict(ex.map(_locate, tasks, chunksize=16))
    t_locate = time.time() - t0

    rows, degenerate = [], []
    for i in infos:
        rois = rois_by_series[i.series]
        side, source = sides[i.study]
        if rois[0]["method"] == "degenerate":
            degenerate.append(dict(study=i.study, series=i.series, plane=i.plane, side=side, side_source=source,
                                   **{f: rois[0][f] for f in ("n_components", "area_px", "vol_h", "vol_w",
                                                              "r0", "r1", "c0", "c1")}))
        else:
            rows.extend(roi_rows(i, source, rois))
    table = pd.DataFrame(rows, columns=[*ROI_TABLE_IDS, *ROI_FIELDS])
    table.to_parquet(out_dir / "knee_rois.parquet", index=False)
    pd.DataFrame(degenerate, columns=DEGENERATE_COLUMNS).to_csv(out_dir / "degenerate.csv", index=False)

    side_st, lat_rows = side_stats(infos, sides)
    pd.DataFrame(lat_rows, columns=LAT_DISAGREEMENT_COLUMNS).to_csv(out_dir / "lat_disagreement.csv", index=False)
    bil_st, bil_rows = bilateral_agreement(infos, rois_by_series)
    pd.DataFrame(bil_rows, columns=BILATERAL_DISAGREEMENT_COLUMNS).to_csv(out_dir / "bilateral_disagreement.csv", index=False)

    wider = wider_than_crop(infos, rois_by_series, CROP_MM_BY_PLANE, LOCATE["mm_per_px"])

    # Montages: per plane N_RANDOM random series + every bilateral, degenerate, wide and wider-than-crop one.
    rng = np.random.default_rng(SEED)
    groups = []
    for plane in PLANES:
        pl = sorted((i for i in infos if i.plane == plane), key=lambda i: i.series)
        if not pl:
            continue
        pick = rng.choice(len(pl), size=min(N_RANDOM, len(pl)), replace=False)
        groups.append((plane, "random", [pl[j] for j in sorted(pick)]))
        for name, methods in (("bilateral", BILATERAL_METHODS), ("degenerate", ("degenerate",))):
            sel = [i for i in pl if rois_by_series[i.series][0]["method"] in methods]
            if sel:
                groups.append((plane, name, sel))
        wide = [i for i in pl if rois_by_series[i.series][0]["method"] == "bbox"
                and (rois_by_series[i.series][0]["c1"] - rois_by_series[i.series][0]["c0"]) * LOCATE["mm_per_px"] >= WIDE_MM]
        if wide:
            groups.append((plane, "wide", wide))
        # Sampled groups: on the full cache 'wider than crop' is the norm for coronal/axial (thousands of
        # series, 4.5 GB of pages in the first full-cache run, since deleted); all their ids stay in stats.json.
        sampled = (("wider_than_crop", set(wider[plane]["series"])),
                   ("exp0002_suspect_only", {i.series for i in pl if i.bilateral_suspect
                                             and rois_by_series[i.series][0]["method"] not in BILATERAL_METHODS}))
        for name, ids in sampled:
            sel = [i for i in pl if i.series in ids]
            if sel:
                pick = rng.choice(len(sel), size=min(N_RANDOM, len(sel)), replace=False)
                groups.append((plane, name, [sel[j] for j in sorted(pick)]))
    tile_tasks, tile_keys = [], []
    for plane, name, sel in groups:
        for i in sel:
            rois = rois_by_series[i.series]
            side, source = sides[i.study]
            title = [f"study  {i.study}", f"series {i.series}",
                     f"{plane} | {rois[0]['method']} x{len(rois)} | side {side} ({source}) | "
                     f"{rois[0]['n_components']} comp"]
            tile_tasks.append((i.series, i.study, side, rois, title))
            tile_keys.append((plane, name, i))
    tile_pool = dict(max_workers=n_workers, mp_context=ctx_mp, initializer=_init_tile_worker,
                     initargs=(str(cache_root), cache_cls, {st: sd for st, (sd, _) in sides.items()},
                               {d["series"] for d in degenerate}))
    with ProcessPoolExecutor(**tile_pool) as ex:
        tiles = list(ex.map(_tile, tile_tasks, chunksize=4))
    mont_dir = out_dir / "montages"
    mont_dir.mkdir()
    index, pages = [], []
    for plane, name, _ in groups:
        idx = [j for j, k in enumerate(tile_keys) if k[0] == plane and k[1] == name]
        for p in range(0, len(idx), TILES_PER_PAGE):
            chunk = idx[p:p + TILES_PER_PAGE]
            png = f"{plane}_{name}_{p // TILES_PER_PAGE:02d}.png"
            if not cv2.imwrite(str(mont_dir / png), tile_grid([tiles[j] for j in chunk], TILE_COLS, gap_px=6)):
                raise OSError(f"cv2.imwrite failed for {mont_dir / png}")
            pages.append(f"montages/{png}")
            for t, j in enumerate(chunk):
                i = tile_keys[j][2]
                index.append(dict(png=png, tile=t, group=name, study=i.study, series=i.series, plane=plane,
                                  method=rois_by_series[i.series][0]["method"]))
    pd.DataFrame(index).to_csv(mont_dir / "index.csv", index=False)

    counts = method_counts(infos, rois_by_series)
    stats = {
        "cache_key": CACHE_KEY, "crop_mm_by_plane": CROP_MM_BY_PLANE, "mm_per_px": LOCATE["mm_per_px"],
        "out_px": OUT_PX, "locate": LOCATE,
        "n_series": len(infos), "n_rois_rows": len(table), "n_degenerate": len(degenerate),
        "n_series_failed_in_cache": len(failed),
        "failed_in_cache_by_code": failed["error_code"].value_counts().to_dict(),
        "series_by_plane_method": counts,
        "wider_than_crop": wider,
        **side_st,
        "bilateral_vs_exp0002_suspect": bil_st,
        "montages": pages,
        "seconds": {"locate": round(t_locate, 1), "total": round(time.time() - t0, 1)},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for plane, c in counts.items():
        nz = {k: v for k, v in c.items() if v}
        print(f"[locate] {plane}: {nz}", flush=True)
    print(f"[locate] studies by side {side_st['studies_by_side']}, by source {side_st['studies_by_side_source']}")
    print(f"[locate] lat_tag != lat_geom: {side_st['lat_tag_vs_geom']['per_series']['n_disagree']} series, "
          f"{side_st['lat_tag_vs_geom']['per_study']['n_disagree']} studies")
    print(f"[locate] bilateral vs exp-0002: {bil_st}")
    print(f"[locate] wider than crop: {({p: (v['n'], v['n_located']) for p, v in wider.items()})}")
    print(f"[locate] done in {stats['seconds']['total']} s | {out_dir}", flush=True)


def main():
    ctx = get_ctx()
    from src.knee_volume import canon_px_to_lps  # exp-0002, reaches us via main
    from src.vol_cache import KneeVolCache  # exp-0002, reaches us via main

    run(ctx.mounts[CACHE_KEY], ctx.output_dir, N_WORKERS, KneeVolCache, canon_px_to_lps)


if __name__ == "__main__":
    main()
