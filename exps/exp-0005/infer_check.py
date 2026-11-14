"""C5 inference check on the competition's example test studies: exp-0004's build_dir_cache + exp-0005's predict_member.

    CUDA_VISIBLE_DEVICES=<free GPU> nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/infer_check.py

What the blend kernel does for this member: KneeVolCache of test_series.csv built from the raw
DICOM dirs into ctx.output_dir/cache (CACHE_WORKERS processes), then predict_member over test.csv's
studies with fold{k}.pt, k in FOLDS (every fold*.pt of TRAIN_RUN). Checks the output as the
submission format (StudyInstanceUID + the 12 LABELS of sample_submission.csv, every study once,
finite, in [0, 1]); writes pred.csv and timing.json (with predict_member's peak CUDA allocation:
the T4 has 15 GB); prints seconds per study for the cache build
and for predict_member (model loading, ROIs and prediction) separately, and both scaled to
N_HIDDEN studies (the local machine is not a T4, and 3 studies are dominated by fixed costs).
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

from src.infer25d import build_dir_cache  # noqa: E402  (exp-0004, reaches this branch via main)
from src.joint_reader import predict_member  # noqa: E402
from src.knee_labels import LABELS  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

COMP = "rsna-knee-abnormality-detection"
TRAIN_RUN = "20261017t135225-664-aecb"  # 5-fold v1 (resume_folds.py, grad_ckpt on)
FOLDS = [0, 1, 2, 3, 4]
CACHE_WORKERS = 4  # a Kaggle kernel has 4 CPUs
N_HIDDEN = 1300  # studies in the hidden test set (approx.)


def format_checks(pred, studies, sample_submission_cols):
    """Submission-format checks of a member's output: columns as sample_submission, the studies in order, finite, [0, 1]."""
    sub = pred.reset_index()
    p = sub[LABELS].to_numpy()
    return {"columns": list(sub.columns) == ["StudyInstanceUID", *LABELS] == list(sample_submission_cols),
            "studies": sub.StudyInstanceUID.tolist() == list(studies), "finite": bool(np.isfinite(p).all()),
            "in_0_1": bool(((p >= 0) & (p <= 1)).all())}


def main():
    ctx = get_ctx()
    comp, run = ctx.mounts[COMP], ctx.mounts[TRAIN_RUN]
    ckpts = [run / f"fold{k}.pt" for k in FOLDS]
    found = sorted(run.glob("fold*.pt"))
    if sorted(ckpts) != found:
        raise ValueError(f"{TRAIN_RUN}: fold checkpoints {[p.name for p in found]} != FOLDS {FOLDS}")
    studies = pd.read_csv(comp / "test.csv").StudyInstanceUID.tolist()
    comp_series = pd.read_csv(comp / "test_series.csv")

    t0 = time.time()
    cache = build_dir_cache(comp / "test_series", comp / "test_series.csv", ctx.output_dir / "cache", CACHE_WORKERS)
    t_cache = time.time() - t0
    t0 = time.time()
    pred = predict_member(studies, cache, comp_series, ckpts, "cuda")
    t_pred = time.time() - t0

    checks = format_checks(pred, studies, pd.read_csv(comp / "sample_submission.csv", nrows=0).columns)
    sub = pred.reset_index()
    sub.to_csv(ctx.output_dir / "pred.csv", index=False)
    n = len(studies)
    timing = {"train_run": TRAIN_RUN, "folds": FOLDS, "n_studies": n, "n_series": len(comp_series),
              "cache_workers": CACHE_WORKERS, "checks": checks,
              "cache_build_s": round(t_cache, 1), "cache_build_s_per_study": round(t_cache / n, 2),
              "predict_member_s": round(t_pred, 1), "predict_member_s_per_study": round(t_pred / n, 2),
              "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
              f"scaled_to_{N_HIDDEN}_studies_min": {"cache_build": round(t_cache / n * N_HIDDEN / 60, 1),
                                                    "predict_member": round(t_pred / n * N_HIDDEN / 60, 1)}}
    (ctx.output_dir / "timing.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing), flush=True)
    print(sub.round(4).to_string(), flush=True)
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(f"member output fails the submission format: {bad}")
    print(f"[infer_check] {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
