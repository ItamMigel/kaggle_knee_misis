"""C4: exp-0005 reader (EfficientNetV2-S on knee-centred crops) trained by exp-0004's run_folds.

Same pipeline as exps/exp-0004/train.py; the differences: our crop hook (C2 over the ROIs of the
C3 run ROI_RUN) and JointCache instead of the raw KneeVolCache (C3-degenerate series as error rows,
sagittal R-knee stacks reversed). The GPU is chosen from outside:

    CUDA_VISIBLE_DEVICES=1 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/train.py

Artifacts in ctx.output_dir (exp-0004 member contract): fold{k}.pt, oof_fold{k}.csv,
metrics_fold{k}.jsonl, depth_report.csv, oof.csv + scores.json over the folds that ran (run_folds
collects exactly those: SMOKE gives fold-0 scores with n_folds 1, not 5-fold ones), inputs.json.
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
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.joint_reader import CROP_MM_BY_PLANE, LOCATE, MM_PER_PX, OUT_PX  # noqa: E402
from src.joint_view import (  # noqa: E402
    DEGENERATE_ERROR_CODE, JointCache, check_table_rois, load_roi_run, make_joint_crop_fn)
from src.knee_labels import gold_labels, load_soft_targets  # noqa: E402
from src.reader25d import cache_study_table  # noqa: E402
from src.train25d import check_cfg, run_folds  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SEED = 1  # seed 0 = run 20261017t135225-664-aecb (resume_folds.py); seed 1 = the lead's second run (18.10)
COMP = "rsna-knee-abnormality-detection"
CACHE = "exp-0002/knee-vol04"  # the full cache; ROI_RUN must have been made on this same key
ROI_RUN = "20261017t124739-029-511f"  # C3 on exp-0002/knee-vol04 (lead-approved crop spec)
FOLDS_RUN = "20261017t111717-941-7125"  # exp-0004's frozen folds.csv
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
FOLDS = [0, 1, 2, 3, 4]
MEM_FRACTION = 0.65  # ~30 GB of the 46 GB L40, the lead's cap; CUDA_VISIBLE_DEVICES=1 makes it device 0 here
N_BOOT = 2000
JITTER_PX = 20  # +-8 mm crop-centre jitter in training (source cache pixels, every plane)
# Crop spec and locate params: src/joint_reader.py (shared with C3 and predict_member); load_roi_run checks
# that ROI_RUN was made with them. CFG crop_px = the model input = OUT_PX.
# every exp-0004 CFG_KEY explicit; ours: backbone, res/crop_px, fov, mean/std; the rest as exp-0004's
# train.py to start (k, bs, grad_ckpt, workers are set from the GPU smoke)
# 5-fold v1 (lead's GATEs 17.10): k 6, 10 epochs, eval_every 5 to fit <=10 h on GPU 1 under the shared
# machine's CPU load (smoke 20261017t125202-486-6244: 3.5-4.4 studies/s at k 8, 6 GB peak with grad_ckpt).
# Without grad_ckpt the run hits the 0.65 cap: bs 8 ~28.5 GB (run 20261017t133750-278-2a92), bs 6 26.7 GB plus
# fragmentation (run 20261017t134217-566-62c9) -> bs 6 with grad_ckpt on; the LR stays the recipe's.
CFG = dict(backbone="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, res=300, crop_px=OUT_PX, fov=1.0,
           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), gap_mm=3.0, k=6, k_eval=16,
           epochs=10, bs=6, lr=1e-4, lr_head=5e-4, wd=0.02, ema=0.998, grad_ckpt=True, eval_every=5,
           workers=7, seed=SEED, limit_train=0, max_steps_per_epoch=0, axial_half_mm=60, axial_max_extent_mm=150)
# SMOKE: fold 0 with capped steps, <= 1 h on one L40; run_folds then scores that one fold only.
SMOKE = False
SMOKE_FOLDS = [0]
SMOKE_KNOBS = dict(limit_train=0, max_steps_per_epoch=200, epochs=3, eval_every=1)
SIDE_NAMES = {"L": "L_kept", "R": "R_mirrored", None: "unknown"}


PEAK_LOG_AT_S = (180, 300, 600)  # lead's guard: peak memory early in fold 0 (~50 steps at bs 6 = ~1.5 min)


def log_peak_memory(at_s):
    """Prints the process' peak CUDA allocation at the given seconds after start (train_fold logs it only per epoch)."""
    t0 = time.time()
    for t in at_s:
        time.sleep(max(0.0, t0 + t - time.time()))
        gb = torch.cuda.max_memory_allocated(0) / 2**30
        print(json.dumps({"peak_mem_gb": round(gb, 2), "after_s": t}), flush=True)


def main():
    cfg = {**CFG, **SMOKE_KNOBS} if SMOKE else dict(CFG)
    folds = SMOKE_FOLDS if SMOKE else FOLDS
    check_cfg(cfg)
    torch.manual_seed(SEED)
    ctx = get_ctx()
    if ROI_RUN is None:
        raise ValueError("set ROI_RUN to the run_id of the C3 pass over CACHE")
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    threading.Thread(target=log_peak_memory, args=(PEAK_LOG_AT_S,), daemon=True).start()
    comp = ctx.mounts[COMP]
    train = pd.read_csv(comp / "train.csv")
    comp_series = pd.read_csv(comp / "train_series.csv")
    folds_df = pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv")
    targets = load_soft_targets(ctx.mounts[S4[0]] / S4[1], train.StudyInstanceUID)

    rois, sides, degenerate = load_roi_run(ctx.mounts[ROI_RUN], cache_key=CACHE, crop_mm_by_plane=CROP_MM_BY_PLANE,
                                           mm_per_px=MM_PER_PX, out_px=OUT_PX, locate=LOCATE)
    joint = JointCache(KneeVolCache(ctx.mounts[CACHE]), sides, degenerate)
    crop_fn = make_joint_crop_fn(rois, sides, OUT_PX, CROP_MM_BY_PLANE, MM_PER_PX, JITTER_PX)
    # the study table run_folds builds from the same inputs: every series it keeps must have a knee ROI
    table = cache_study_table(joint.series, comp_series, folds_df.StudyInstanceUID)
    check_table_rois(table, rois)

    err = joint.series.error_code
    empty = {s for s, slots in table.items() if not any(slots)}
    is_gold = folds_df.is_gold.astype(bool)
    inputs = {
        "cache": CACHE, "roi_run": ROI_RUN, "folds_run": FOLDS_RUN, "smoke": SMOKE, "folds": folds,
        "crop_mm_by_plane": CROP_MM_BY_PLANE, "mm_per_px": MM_PER_PX, "locate": LOCATE, "jitter_px": JITTER_PX, "cfg": cfg,
        "series": {"in_cache": len(err), "failed_in_cache": int((err.notna() & (err != DEGENERATE_ERROR_CODE)).sum()),
                   "degenerate": len(degenerate), "located": len(rois),
                   "in_study_table": sum(len(c) for slots in table.values() for c in slots)},
        "studies_by_side": dict(Counter("no_located_series" if s not in sides else SIDE_NAMES[sides[s]]
                                        for s in folds_df.StudyInstanceUID)),
        "studies_without_usable_series": {
            "gold": len(empty & set(folds_df.StudyInstanceUID[is_gold])),
            **{str(k): len(empty & set(folds_df.StudyInstanceUID[folds_df.fold == k])) for k in FOLDS}},
    }
    (ctx.output_dir / "inputs.json").write_text(json.dumps(inputs, indent=1))
    print(json.dumps({k: inputs[k] for k in ("series", "studies_by_side", "studies_without_usable_series")}),
          flush=True)

    oof, scores = run_folds(cfg, folds, joint, comp_series, folds_df, targets, gold_labels(train), crop_fn,
                            ctx.output_dir, "cuda", N_BOOT, SEED)
    print(json.dumps({"smoke": SMOKE, "n_folds": scores["n_folds"], "weak_oof": scores["weak_oof"]["macro"],
                      "weak_per_fold": scores["weak_per_fold"], "gold": scores["gold"]["macro"],
                      "gold_ci": scores["gold"]["bootstrap"]}))
    print(f"[train] {ctx.output_dir}")


if __name__ == "__main__":
    main()
