"""Frozen gold58 diagnostic: macro/per-finding AUC with bootstrap CIs for our labels vs S4, paired deltas (local, CPU).

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0003/gold_diag.py

Run ONCE, after the policy is frozen and its commit is recorded in description.md. The only
script of exp-0003 that reads gold label values. Refuses to run unless the build_labels.py
run in LABELS_RUN was produced by FROZEN_SHA from a clean tree.

Per-target masking: each finding is scored on the gold studies where ours AND S4 have a
finite score (an invalid cell of ours costs only that finding); S4 is scored on exactly the
studies ours is, and the paired deltas use that mask. gold_auc.csv / gold_deltas.csv hold
that; gold_auc_s4_unmasked.csv is S4 alone on all gold studies (MACRO 0.8927 [0.8592, 0.9227]),
to show what the mask itself moves.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import os  # noqa: E402

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.report_labels.compare import gold_diagnostic, split_gold  # noqa: E402
from src.report_labels.rubric import TARGETS  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

LABELS_RUN = "20261017t191223-797-3f77"  # run id of the frozen build_labels.py run
FROZEN_SHA = "44ef751"  # policy-freeze commit recorded in description.md before this run
N_BOOT = 10_000
SEED = 20261013
REF = "S4"
COMP = "rsna-knee-abnormality-detection"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
UID = "StudyInstanceUID"

if not LABELS_RUN or len(FROZEN_SHA) < 7:
    raise SystemExit("set LABELS_RUN and FROZEN_SHA (>= 7 hex chars, as recorded in description.md)")

ctx = get_ctx()
labels_dir = ctx.mounts[LABELS_RUN]
code = json.loads((labels_dir / "record.json").read_text(encoding="utf-8"))["code"]
if code["dirty"] or not code["git_sha"].startswith(FROZEN_SHA):
    raise SystemExit(f"labels run {LABELS_RUN} is not the frozen policy: git_sha {code['git_sha']}, "
                     f"dirty {code['dirty']}; expected clean {FROZEN_SHA}")

train = pd.read_csv(ctx.mounts[COMP] / "train.csv")
gold_uids, _ = split_gold(train)
gold = train.loc[train[UID].isin(gold_uids), [UID, *TARGETS]]  # train.csv order fixes the bootstrap draws

ours = pd.read_csv(labels_dir / "labels_wide.csv")[[UID, *TARGETS]]
s4 = pd.read_csv(ctx.mounts[S4[0]] / S4[1])[[UID, *TARGETS]]
scores = {"ours": ours, REF: s4}
finite = {}
for name, df in scores.items():
    finite[name] = np.isfinite(df.set_index(UID).reindex(gold[UID])[list(TARGETS)].to_numpy(float))
    print(f"[gold_diag] {name}: complete on {int(finite[name].all(axis=1).sum())}/{len(gold)} gold studies")
if not finite[REF].all():
    raise SystemExit(f"{REF} lacks scores on some gold studies: the unmasked {REF} run would be masked too")
n_complete = int(np.logical_and.reduce(list(finite.values())).all(axis=1).sum())

metrics, deltas = gold_diagnostic(scores, gold, N_BOOT, SEED, REF)
metrics.to_csv(ctx.output_dir / "gold_auc.csv", index=False)
deltas.to_csv(ctx.output_dir / "gold_deltas.csv", index=False)
unmasked, _ = gold_diagnostic({REF: s4}, gold, N_BOOT, SEED, REF)
unmasked.to_csv(ctx.output_dir / "gold_auc_s4_unmasked.csv", index=False)


def auc_ci(df: pd.DataFrame, source: str) -> pd.Series:
    """'auc [low, high]' per target of one source."""
    d = df[df["source"] == source].set_index("target")
    return d.apply(lambda r: f"{r.auc:.4f} [{r.low:.4f}, {r.high:.4f}]", axis=1)


side = pd.DataFrame({"n_mask": metrics[metrics["source"] == REF].set_index("target")["n"],
                     f"{REF}_masked": auc_ci(metrics, REF), f"{REF}_unmasked": auc_ci(unmasked, REF),
                     "ours": auc_ci(metrics, "ours")})
print(f"[gold_diag] per-target masks: n studies kept per finding (MACRO: used by >= 1 finding); "
      f"{n_complete}/{len(gold)} gold studies complete in every finding and source")
print(side.to_string())
print(metrics.query("target == 'MACRO'").to_string(index=False, float_format="%.4f"))
print(deltas.to_string(index=False, float_format="%.4f"))
print(f"[gold_diag] done | {ctx.output_dir}")
