"""C5 precision check: one fold's checkpoint through exp-0004's predict_studies in bf16, fp16 and fp32.

    CUDA_VISIBLE_DEVICES=<free GPU> nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/precision_check.py

The first N_STUDIES val studies of oof_fold{FOLD}.csv of TRAIN_RUN, on the training eval path:
load_models(fold{FOLD}.pt), ROIs of ROI_RUN, JointCache, crop without jitter, the fold's cfg
k_eval / bs / workers. Per dtype: weak-label macro AUC (S4 > 0.5, metrics_auc.macro_auc as
collect_oof), max and mean |dp| vs bf16 and vs fp32, seconds, peak GPU memory; plus bf16 vs
sigmoid(oof_fold{FOLD} logits), i.e. train_fold's own bf16 eval (~0 expected; > SANITY_TOL raises
after precision.json is written). bf16 / fp16 run with torch's default TF32 flags (as train_fold's
eval); fp32 = no autocast with TF32 off, i.e. true fp32 as on a T4 (Turing has no TF32). Writes
precision.json.
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
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.infer25d import load_models, predict_studies  # noqa: E402  (exp-0004, reaches this branch via main)
from src.joint_reader import CROP_MM_BY_PLANE, LOCATE, MM_PER_PX, OUT_PX  # noqa: E402
from src.joint_view import JointCache, load_roi_run, make_joint_crop_fn  # noqa: E402
from src.knee_labels import LABELS, load_soft_targets  # noqa: E402
from src.metrics_auc import macro_auc  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

TRAIN_RUN = "20261017t135225-664-aecb"  # 5-fold v1 (resume_folds.py, grad_ckpt on)
FOLD = 0
N_STUDIES = 300
COMP = "rsna-knee-abnormality-detection"
CACHE = "exp-0002/knee-vol04"
ROI_RUN = "20261017t124739-029-511f"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}  # bf16 first: the reference
# max |dp| allowed between bf16 here and the run's own bf16 OOF: bf16 rounding may differ slightly between
# runs (cuDNN algorithm choice, the last partial batch); a wrong ROI or flip moves p far more.
SANITY_TOL = 1e-2


def dp(a, b):
    """{max, mean} of |a - b| over every study and label."""
    d = np.abs(a - b)
    return {"max": float(d.max()), "mean": float(d.mean())}


def main():
    if TRAIN_RUN is None:
        raise ValueError("set TRAIN_RUN to the train.py run to check")
    ctx = get_ctx()
    tf32_default = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    device = torch.device("cuda")
    run = ctx.mounts[TRAIN_RUN]
    models = load_models([run / f"fold{FOLD}.pt"], device)
    cfg = models[0][1]

    oof = pd.read_csv(run / f"oof_fold{FOLD}.csv", index_col="StudyInstanceUID")
    ids = list(oof.index[oof["set"] == "val"][:N_STUDIES])
    if len(ids) != N_STUDIES:
        raise ValueError(f"{TRAIN_RUN} fold {FOLD}: {len(ids)} val studies < N_STUDIES {N_STUDIES}")
    rois, sides, deg = load_roi_run(ctx.mounts[ROI_RUN], cache_key=CACHE, crop_mm_by_plane=CROP_MM_BY_PLANE,
                                    mm_per_px=MM_PER_PX, out_px=OUT_PX, locate=LOCATE)
    joint = JointCache(KneeVolCache(ctx.mounts[CACHE]), sides, deg)
    crop_fn = make_joint_crop_fn(rois, sides, OUT_PX, CROP_MM_BY_PLANE, MM_PER_PX, jitter_px=0)
    comp = ctx.mounts[COMP]
    comp_series = pd.read_csv(comp / "train_series.csv")
    train = pd.read_csv(comp / "train.csv")
    weak = (load_soft_targets(ctx.mounts[S4[0]] / S4[1], train.StudyInstanceUID).loc[ids, LABELS] > 0.5).astype(int)

    probs, per = {}, {}
    for name, dtype in DTYPES.items():
        tf32 = (False, False) if dtype == torch.float32 else tf32_default
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        p = predict_studies(ids, joint, comp_series, models, crop_fn, cfg["k_eval"], cfg["bs"], cfg["workers"], device,
                            dtype)
        if list(p.index) != ids or list(p.columns) != LABELS:
            raise ValueError(f"{name}: predict_studies returned another index or columns")
        probs[name] = p.to_numpy(np.float64)
        auc, each = macro_auc(weak.to_numpy(), probs[name])
        per[name] = {"weak_macro_auc": auc, "tf32_matmul_cudnn": list(tf32),
                     "weak_auc_per_class": {lab: (None if np.isnan(a) else a) for lab, a in zip(LABELS, each)},
                     "seconds": round(time.time() - t0, 1),
                     "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
        print(json.dumps({name: {k: per[name][k] for k in ("weak_macro_auc", "seconds", "peak_mem_gb")}}), flush=True)
    for name in DTYPES:
        per[name]["dp_vs_bf16"] = dp(probs[name], probs["bf16"])
        per[name]["dp_vs_fp32"] = dp(probs[name], probs["fp32"])
        per[name]["weak_auc_delta_vs_fp32"] = per[name]["weak_macro_auc"] - per["fp32"]["weak_macro_auc"]
    ref = 1 / (1 + np.exp(-oof.loc[ids, LABELS].to_numpy(np.float64)))
    sanity = dp(probs["bf16"], ref)
    res = {"train_run": TRAIN_RUN, "fold": FOLD, "n_studies": len(ids), "roi_run": ROI_RUN,
           "k_eval": cfg["k_eval"], "bs": cfg["bs"], "workers": cfg["workers"],
           "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
           "dtypes": per, "bf16_vs_oof_sigmoid": sanity, "sanity_tol": SANITY_TOL,
           "sanity_pass": sanity["max"] <= SANITY_TOL}
    (ctx.output_dir / "precision.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({n: {"weak_macro_auc": round(v["weak_macro_auc"], 5), "max_dp_vs_bf16": v["dp_vs_bf16"]["max"],
                          "max_dp_vs_fp32": v["dp_vs_fp32"]["max"]} for n, v in per.items()}), flush=True)
    print(json.dumps({"bf16_vs_oof_sigmoid": sanity}), flush=True)
    if sanity["max"] > SANITY_TOL:
        raise AssertionError(f"bf16 predict_studies differs from the run's OOF by {sanity['max']:.2e} > {SANITY_TOL}: "
                             "the eval path is not the training one")
    print(f"[precision] {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
