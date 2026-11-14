"""v1 2.5D reader: training of a subset of the frozen folds on the exp-0002 cache, then OOF and scores of that subset.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/train.py --folds 0,1,2
    CUDA_VISIBLE_DEVICES=1 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/train.py --folds 3

Folds run as separate runs so they can share out over both GPUs (lead, 17.10); the fold subset and the
GPU are recorded in inputs.json (the run record does not keep argv). Writes inputs.json, fold{k}.pt,
oof_fold{k}.csv, metrics_fold{k}.jsonl, oof.csv and scores.json to ctx.output_dir; the 5-fold OOF is
pooled from the runs afterwards.
Training targets = the mean of TARGET_SOURCES' soft labels over the weak studies. scores.json is always
against S4, so every run compares; when TARGET_SOURCES != [S4], run_folds' own scores against the training
targets go to scores_train_targets.json (score_against_s4). label_check.py checks the sources.
The smoke is the same pipeline with smoke constants: smoke_dinov3.py / smoke_in22k.py. Backbone = in22k by
the D1 rule: the dinov3 arm did not learn (val weak macro AUC 0.549 after 5 epochs vs in22k 0.751 after one).
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.knee_labels import gold_labels, load_soft_targets  # noqa: E402
from src.reader25d import center_crop  # noqa: E402
from src.train25d import collect_oof, run_folds  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

# seed 0 S4-only: runs 20261017t150919-625-2918 (folds 0-2), 20261017t214233-033-af68 (3), 20261018t000939-333-228f
# (4); seed 1 S4-only: 20261018t023618-204-1085 (folds 0-2; record failed = SIGINT after fold 2, lead 18.10) +
# 20261018t045524-429-281f (3-4); label-diverse seed 0 = TARGET_SOURCES [S4, PILKWANG] (lead 18.10)
SEED = 0
COMP = "rsna-knee-abnormality-detection"
CACHE = "exp-0002/knee-vol04"
FOLDS_RUN = "20261017t111717-941-7125"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
PILKWANG = ("pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv")  # NO 0.08 / UNK 0.28 / YES 0.68-0.94
# Training targets = the mean of these soft-label sources over the weak studies (one source = it as is; the
# S4-only runs above used [S4]). The OOF is always scored against S4 (scores.json), so every run compares;
# with other sources run_folds' own scores against the training targets go to scores_train_targets.json.
# [S4, PILKWANG] = the label-diverse member (lead, 18.10); label_check.py checks the sources.
TARGET_SOURCES = [S4, PILKWANG]
MEM_FRACTION = 0.65  # ~30 GB of the 46 GB L40: the card is shared (lead, 17.10)
N_BOOT = 2000
CFG = dict(backbone="convnext_tiny.fb_in22k_ft_in1k", pretrained=True, res=352, crop_px=384, fov=0.92,
           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), gap_mm=3.0, k=8, k_eval=16,
           epochs=14, bs=8, lr=1e-4, lr_head=5e-4, wd=0.02, ema=0.998, grad_ckpt=True, eval_every=2,
           workers=7, seed=SEED, limit_train=0, max_steps_per_epoch=0, axial_half_mm=60, axial_max_extent_mm=150)


def load_targets(ctx, sources, study_ids):
    """Mean of load_soft_targets over the (dataset, csv) sources on study_ids; each raises on a missing study or NaN."""
    parts = [load_soft_targets(ctx.mounts[ds] / name, study_ids) for ds, name in sources]
    return sum(parts) / len(parts)


def score_against_s4(ctx, out_dir, folds_df, folds, gold):
    """Moves run_folds' scores.json (vs the training targets) to scores_train_targets.json; rescores the OOF vs S4."""
    (out_dir / "scores.json").rename(out_dir / "scores_train_targets.json")
    is_gold = folds_df.is_gold.astype(bool)
    trained = folds_df[folds_df.fold.isin(folds) | is_gold]
    s4 = load_targets(ctx, [S4], trained.StudyInstanceUID[~is_gold])
    return collect_oof(out_dir, trained, s4, gold, N_BOOT, SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, help="comma-separated fold ids of the frozen folds, e.g. 0,1,2")
    folds = [int(f) for f in ap.parse_args().folds.split(",")]
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= {0, 1, 2, 3, 4}:
        raise ValueError(f"--folds must be distinct ids from 0..4, got {folds}")

    torch.manual_seed(SEED)
    ctx = get_ctx()
    (ctx.output_dir / "inputs.json").write_text(json.dumps(
        {"argv": sys.argv, "folds": folds, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
         "cache": CACHE, "folds_run": FOLDS_RUN, "target_sources": TARGET_SOURCES, "scored_against": S4,
         "mem_fraction": MEM_FRACTION, "cfg": CFG}, indent=1))
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    comp = ctx.mounts[COMP]
    train = pd.read_csv(comp / "train.csv")
    folds_df = pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv")
    targets = load_targets(ctx, TARGET_SOURCES, folds_df.StudyInstanceUID[~folds_df.is_gold.astype(bool)])
    oof, scores = run_folds(CFG, folds, KneeVolCache(ctx.mounts[CACHE]), pd.read_csv(comp / "train_series.csv"),
                            folds_df, targets, gold_labels(train), center_crop(CFG["crop_px"]), ctx.output_dir,
                            "cuda", N_BOOT, SEED)
    if TARGET_SOURCES != [S4]:
        oof, scores = score_against_s4(ctx, ctx.output_dir, folds_df, folds, gold_labels(train))
    print(json.dumps({"weak_oof": scores["weak_oof"]["macro"], "weak_per_fold": scores["weak_per_fold"],
                      "gold": scores["gold"]["macro"], "gold_ci": scores["gold"]["bootstrap"]}))
    print(f"[train] {ctx.output_dir}")


if __name__ == "__main__":
    main()
