"""Inference dtype for the T4 kernel (D9): fp16 and fp32 vs bf16 on fold-0 OOF studies with the fold-0 checkpoint.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/dtype_check.py

The first N_STUDIES fold-0 val studies, fold0.pt of TRAIN_RUN, the kernel member's eval path (predict_studies:
center_crop, k_eval of the cfg) at batch BS under a hard GPU memory cap; bf16 (training eval), fp16 (T4) and
fp32 (no autocast). Reports weak-label (S4 > 0.5) macro AUC per dtype, deltas vs bf16, prob differences, and
bf16 vs the training's own OOF probabilities as a sanity check. Rule (lead, 17.10): fp32 in the kernel if
|AUC(fp16) - AUC(bf16)| > 0.001. Writes dtype_check.json and probs_<dtype>.csv.
PROBS_RUN set: no GPU, the probs_<dtype>.csv of that earlier run of this script are scored instead.
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

from src.infer25d import load_models, predict_studies  # noqa: E402
from src.knee_labels import LABELS, load_soft_targets  # noqa: E402
from src.metrics_auc import macro_auc  # noqa: E402
from src.reader25d import center_crop  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

CACHE = "exp-0002/knee-vol04"
FOLDS_RUN = "20261017t111717-941-7125"
TRAIN_RUN = "20261017t150919-625-2918"  # run A: fold0.pt
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
N_STUDIES = 300
BS, WORKERS = 2, 3
MEM_GB = 8.0  # lead's limit for running alongside the training on GPU 0
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
PROBS_RUN = "20261017t182108-948-e643"  # GPU part of this script done there (it then failed on a tuple bug); None = infer

ctx = get_ctx()
folds = pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv")
studies = folds.StudyInstanceUID[folds.fold == 0].iloc[:N_STUDIES].tolist()
y = (load_soft_targets(ctx.mounts[S4[0]] / S4[1], studies) > 0.5).astype(int).values

out, probs = {"n_studies": len(studies), "bs": BS, "probs_run": PROBS_RUN}, {}
if PROBS_RUN is None:
    torch.cuda.set_per_process_memory_fraction(MEM_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
    comp = ctx.mounts["rsna-knee-abnormality-detection"]
    models = load_models([ctx.mounts[TRAIN_RUN] / "fold0.pt"], "cuda")
    cfg = models[0][1]
    out["k_eval"] = cfg["k_eval"]
    cache = KneeVolCache(ctx.mounts[CACHE])
    comp_series = pd.read_csv(comp / "train_series.csv")
for name, dt in DTYPES.items():
    if PROBS_RUN is None:
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.time()
        p = predict_studies(studies, cache, comp_series, models, center_crop(cfg["crop_px"]), cfg["k_eval"], BS,
                            WORKERS, "cuda", dt)
        p.to_csv(ctx.output_dir / f"probs_{name}.csv")
        out[name] = {"seconds": round(time.time() - t0, 1),
                     "peak_gb": round(torch.cuda.max_memory_allocated(0) / 2**30, 2)}
    else:
        p = pd.read_csv(ctx.mounts[PROBS_RUN] / f"probs_{name}.csv", index_col=0)
        if p.index.tolist() != studies or p.columns.tolist() != LABELS:
            raise ValueError(f"{PROBS_RUN}/probs_{name}.csv: studies or columns differ from this script's")
        out[name] = {}
    probs[name] = p
    out[name]["macro_auc"], out[name]["per_class_auc"] = macro_auc(y, p.values)
    print(name, json.dumps(out[name]), flush=True)

base = probs["bf16"].values
for name in ("fp16", "fp32"):
    d = np.abs(probs[name].values - base)
    out[name].update(auc_delta_vs_bf16=out[name]["macro_auc"] - out["bf16"]["macro_auc"],
                     max_abs_prob_diff=float(d.max()), mean_abs_prob_diff=float(d.mean()))
oof = pd.read_csv(ctx.mounts[TRAIN_RUN] / "oof_fold0.csv", index_col="StudyInstanceUID")
oof = oof[oof["set"] == "val"].loc[studies, LABELS]
out["bf16_vs_training_oof_max_abs_prob_diff"] = float(np.abs(1 / (1 + np.exp(-oof.values)) - base).max())
out["rule"] = "fp32 if |auc_delta fp16| > 0.001"
out["choice"] = "fp32" if abs(out["fp16"]["auc_delta_vs_bf16"]) > 0.001 else "fp16"
(ctx.output_dir / "dtype_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)
print(f"[dtype_check] {ctx.output_dir}")
