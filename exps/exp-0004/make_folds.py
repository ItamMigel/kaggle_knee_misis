"""Freeze the CV folds of exp-0004 (contract D2), shared with exp-0005.

    ~/conda/envs/knee/bin/python exps/exp-0004/make_folds.py

Writes to ctx.output_dir:
  folds.csv      StudyInstanceUID, fold, is_gold (gold studies: fold -1)
  fold_stats.csv per fold: n studies and prevalence of S4 > 0.5 per label
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import pandas as pd  # noqa: E402

from src.knee_labels import gold_labels, load_soft_targets, make_folds, report_group  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

N_FOLDS = 5  # no seed: the folds depend on the report texts only (see make_folds)

ctx = get_ctx()

train = pd.read_csv(ctx.mounts["rsna-knee-abnormality-detection"] / "train.csv")
s4 = load_soft_targets(ctx.mounts["stevenleehans/rsna-knee-llm-report-labels"] / "llm_labels_v4_blend.csv",
                       train.StudyInstanceUID)
gold_ids = set(gold_labels(train).index)

folds = make_folds(train, N_FOLDS)

weak = folds[~folds.is_gold]
assert folds.StudyInstanceUID.is_unique and len(folds) == len(train)
assert set(weak.StudyInstanceUID) == set(train.StudyInstanceUID) - gold_ids, "weak studies != non-gold studies"
assert weak.fold.isin(range(N_FOLDS)).all(), "a weak study is outside folds 0..N_FOLDS-1"
assert set(folds.StudyInstanceUID[folds.is_gold]) == gold_ids, "is_gold != studies with all 12 labels"
assert (folds.fold[folds.is_gold] == -1).all(), "a gold study is in a CV fold"
groups = train.set_index("StudyInstanceUID").Report.map(report_group)
crossing = weak.groupby(weak.StudyInstanceUID.map(groups).values).fold.nunique()
assert (crossing == 1).all(), f"{(crossing > 1).sum()} report groups cross folds"

folds.to_csv(ctx.output_dir / "folds.csv", index=False)

hard = (s4 > 0.5).astype(float)
by_fold = folds.set_index("StudyInstanceUID").fold.loc[hard.index]
stats = hard.groupby(by_fold.values).mean()
stats.insert(0, "n", by_fold.value_counts())
stats.index.name = "fold"
stats.to_csv(ctx.output_dir / "fold_stats.csv")

print(stats.round(3).to_string())
print(f"[make_folds] {len(weak)} weak studies in {N_FOLDS} folds, {len(gold_ids)} gold | {ctx.output_dir}")
