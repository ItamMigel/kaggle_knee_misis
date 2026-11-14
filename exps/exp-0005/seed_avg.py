"""Seed average of finished 5-fold exp-0005 runs: mean probability of their oof.csv, scored as collect_oof.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/seed_avg.py

RUNS must be trained on the same folds (inputs.json folds_run) and their oof.csv must hold the same
studies with the same set (val / gold). Weak-label macro AUC (S4 > 0.5) with per class, gold macro
AUC with a bootstrap CI (N_BOOT, BOOT_SEED as seed 0's collect_oof). Each run's own oof.csv is
rescored the same way and must match its scores.json, which checks this scoring path. Writes
oof.csv and scores.json to ctx.output_dir.
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
import math  # noqa: E402

import pandas as pd  # noqa: E402

from src.knee_labels import LABELS, gold_labels, load_soft_targets  # noqa: E402
from src.metrics_auc import bootstrap_macro_auc, macro_auc  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

RUNS = ["20261017t135225-664-aecb", "20261018t003924-017-1d75"]  # seed 0 (resume_folds.py), seed 1 (train.py SEED 1)
COMP = "rsna-knee-abnormality-detection"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
N_BOOT = 2000
BOOT_SEED = 0
TOL = 1e-9  # rescored run vs its scores.json


def auc_fields(y, p):
    """{"macro", "per_class"} of macro_auc, NaN classes as None."""
    m, each = macro_auc(y, p)
    return {"macro": m, "per_class": {c: (None if math.isnan(a) else a) for c, a in zip(LABELS, each)}}


def score(prob, sets, targets, gold):
    """(weak fields, gold fields + bootstrap) of probabilities indexed by study, split by sets."""
    val, g = prob[sets == "val"], prob[sets == "gold"]
    wy = (targets.loc[val.index] > 0.5).astype(int).to_numpy()
    gy = gold.loc[g.index].astype(int).to_numpy()
    return (auc_fields(wy, val.to_numpy()),
            {**auc_fields(gy, g.to_numpy()), "bootstrap": bootstrap_macro_auc(gy, g.to_numpy(), N_BOOT, BOOT_SEED),
             "n_boot": N_BOOT, "seed": BOOT_SEED, "n_gold": len(g)})


def main():
    if None in RUNS or len(set(RUNS)) != len(RUNS):
        raise ValueError(f"RUNS must be distinct finished runs, got {RUNS}")
    ctx = get_ctx()
    folds_runs = {r: json.loads((ctx.mounts[r] / "inputs.json").read_text())["folds_run"] for r in RUNS}
    if len(set(folds_runs.values())) != 1:
        raise ValueError(f"RUNS were trained on different folds: {folds_runs}")
    frames = {r: pd.read_csv(ctx.mounts[r] / "oof.csv", index_col="StudyInstanceUID").sort_index() for r in RUNS}
    sets = frames[RUNS[0]]["set"]
    for r, f in frames.items():
        if not f["set"].equals(sets):
            raise ValueError(f"{r}: oof.csv studies or sets differ from {RUNS[0]}'s")

    train = pd.read_csv(ctx.mounts[COMP] / "train.csv")
    targets = load_soft_targets(ctx.mounts[S4[0]] / S4[1], train.StudyInstanceUID)
    gold = gold_labels(train)

    per_run = {}
    for r, f in frames.items():
        weak, g = score(f[LABELS], sets, targets, gold)
        ref = json.loads((ctx.mounts[r] / "scores.json").read_text())
        d = max(abs(weak["macro"] - ref["weak_oof"]["macro"]), abs(g["macro"] - ref["gold"]["macro"]))
        if d > TOL:
            raise AssertionError(f"{r}: rescored oof.csv differs from its scores.json by {d}")
        per_run[r] = {"weak_oof": weak["macro"], "gold": g["macro"], "gold_ci": [g["bootstrap"]["lo"], g["bootstrap"]["hi"]]}

    avg = sum(f[LABELS] for f in frames.values()) / len(frames)
    weak, g = score(avg, sets, targets, gold)
    avg.insert(0, "set", sets)
    avg.to_csv(ctx.output_dir / "oof.csv")
    scores = {"runs": RUNS, "folds_run": folds_runs[RUNS[0]], "n_val": int((sets == "val").sum()),
              "weak_oof": weak, "gold": g, "per_run": per_run}
    (ctx.output_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps({"weak_oof": weak["macro"], "gold": g["macro"], "gold_ci": g["bootstrap"], "per_run": per_run}),
          flush=True)
    print(f"[seed_avg] {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
