"""exp-0005 member: the knee-view spec (one source of truth), knee ROIs at inference and predict_member (C5).

The view spec below is shared by the C3 pass (exps/exp-0005/locate_all.py), training
(exps/exp-0005/train.py, which checks the C3 run's stats.json against it via load_roi_run) and
inference (predict_member): a checkpoint's cfg does not record it, so it must not live anywhere else.

Inference repeats C3 on the kernel's KneeVolCache (exp-0004 infer25d.build_dir_cache over
test_series/): study_side per study, locate_knee per error-free series, with on_degenerate=
"image_centre" instead of C3's "report" (a test series cannot be left out), then the same
JointCache + JointCrop as training, without jitter, into exp-0004's predict_studies.
"""
import json
import time

import torch

from src.joint_view import (
    META_KEYS, JointCache, locate_knee, make_joint_crop_fn, series_info, study_side)
from src.joint_view_report import method_counts, side_stats

# Crop spec (lead's visual gate 17.10): a CROP_MM_BY_PLANE square around the ROI centre, resampled to
# OUT_PX = the model input (cfg crop_px). Sagittal 150 mm = 375 px -> 300 px (0.5 mm/px): the knee's AP
# extent is ~130-160 mm and 120 mm cut off the patella / posterior edge; coronal/axial 120 mm = 300 px as is.
CROP_MM_BY_PLANE = {"Sagittal": 150.0, "Coronal": 120.0, "Axial": 120.0}
MM_PER_PX = 0.4  # exp-0002 cache scale
OUT_PX = 300
# locate_knee parameters (= stats["locate"] of the C3 run training reads)
LOCATE = dict(
    mm_per_px=MM_PER_PX,
    # each of two disjoint components at least this wide -> bilateral FOV. 50 mm misread one fat-suppressed
    # knee split into 78 + 65 mm halves (pilot run 20261017t114420-435-816b); a whole knee is wider than 80 mm.
    bilateral_min_mm=80.0,
    degenerate_min_mm=30.0,  # union bbox narrower than this -> degenerate
    open_px=5,  # exp-0002 fg recipe: opening kernel
    min_area_frac=0.01,  # exp-0002 fg recipe: components above 1 % of the image area
)
# member settings (exp-0004 member contract: they live inside predict_member). k_eval must equal the
# checkpoints' cfg k_eval (= what their OOF was scored with); bs / workers as exp-0004's member (4 CPUs on Kaggle).
MEMBER_K_EVAL, MEMBER_BS, MEMBER_WORKERS = 16, 8, 4
# T4 has no bf16; precision_check.py decides fp16 vs fp32 (torch.float32 = no autocast).
MEMBER_AMP = torch.float16


def member_rois(studies, cache, canon_px_to_lps) -> tuple[dict, dict, dict]:
    """C3 at inference: study_side + locate_knee(on_degenerate="image_centre") over every error-free series of `studies`.

    cache: KneeVolCache (original geometry, as C3 read it); canon_px_to_lps: exp-0002's (injected).
    -> (rois {series: [ROI]}, sides {study: side} of the studies with an error-free series, counts):
    counts = n_studies, n_series, studies_without_usable_series, studies_by_side, studies_by_side_source
    (side_stats), series_by_plane_method (method_counts), seconds; for the kernel log.
    """
    studies = list(studies)
    if not studies or len(set(studies)) != len(studies):
        raise ValueError(f"member_rois: {len(studies)} studies, empty or with duplicates")
    st_col, s_col, e_col = (META_KEYS[k] for k in ("study", "series", "error_code"))
    tab = cache.series
    absent = set(studies) - set(tab[st_col].astype(str))
    if absent:
        raise KeyError(f"{len(absent)} studies are not in the cache, e.g. {next(iter(absent))}")
    ok = tab[tab[st_col].astype(str).isin(studies) & tab[e_col].isna()]
    series_of = {}
    for st, sr in zip(ok[st_col].astype(str), ok[s_col].astype(str)):
        series_of.setdefault(st, []).append(sr)
    t0 = time.time()
    rois, side_src, infos = {}, {}, []
    for study in studies:
        if study not in series_of:  # every series failed preprocessing: StudyDataset gives it an all-False mask
            continue
        metas = [cache.meta(sr) for sr in series_of[study]]
        infos.extend(series_info(m) for m in metas)
        side_src[study] = study_side(metas)
        for sr in series_of[study]:
            vol, pos_mm, meta = cache.load(sr)
            rois[sr] = locate_knee(vol, pos_mm, meta, side_src[study][0], crop_mm_by_plane=CROP_MM_BY_PLANE,
                                   canon_px_to_lps=canon_px_to_lps, on_degenerate="image_centre", **LOCATE)
    side_st, _ = side_stats(infos, side_src)
    counts = {"n_studies": len(studies), "n_series": len(rois),
              "studies_without_usable_series": len(studies) - len(side_src),
              "studies_by_side": side_st["studies_by_side"], "studies_by_side_source": side_st["studies_by_side_source"],
              "series_by_plane_method": method_counts(infos, rois), "seconds": round(time.time() - t0, 1)}
    return rois, {st: side for st, (side, _) in side_src.items()}, counts


def predict_member(studies, cache, comp_series, ckpt_paths, device):
    """exp-0005's member entry point (exp-0004 member contract): knee ROIs + canonicalisation into exp-0004's predict_studies.

    load_models -> member_rois (printed counts) -> JointCache(cache, sides, no degenerate series) ->
    make_joint_crop_fn(jitter 0) -> predict_studies(k_eval MEMBER_K_EVAL, MEMBER_BS, MEMBER_WORKERS,
    MEMBER_AMP); no TTA. cache: the kernel's KneeVolCache; every study in it with an error-free series
    must be among `studies` (JointCache needs its side). Returns probabilities (index StudyInstanceUID
    in the given order, columns LABELS).
    """
    from src.infer25d import load_models, predict_studies  # exp-0004, reaches this branch via main
    from src.knee_volume import canon_px_to_lps  # imports pydicom: after kernel_env.install_wheels in a kernel

    ckpt_paths = list(ckpt_paths)
    models = load_models(ckpt_paths, device)
    for path, (_, cfg) in zip(ckpt_paths, models, strict=True):
        if (cfg["crop_px"], cfg["k_eval"]) != (OUT_PX, MEMBER_K_EVAL):
            raise ValueError(f"{path}: cfg crop_px {cfg['crop_px']}, k_eval {cfg['k_eval']} != the member's "
                             f"OUT_PX {OUT_PX}, MEMBER_K_EVAL {MEMBER_K_EVAL}")
    rois, sides, counts = member_rois(studies, cache, canon_px_to_lps)
    print(f"[exp-0005 member] {json.dumps(counts)}", flush=True)
    joint = JointCache(cache, sides, set())
    crop_fn = make_joint_crop_fn(rois, sides, OUT_PX, CROP_MM_BY_PLANE, MM_PER_PX, jitter_px=0)
    return predict_studies(studies, joint, comp_series, models, crop_fn, MEMBER_K_EVAL, MEMBER_BS, MEMBER_WORKERS,
                           device, MEMBER_AMP)
