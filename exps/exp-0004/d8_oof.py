"""D8 input: exp-0004's pooled 5-fold OOF, and the OOF rank blend of exp-0004 with exp-0005 plus their correlation.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/d8_oof.py

exp-0004's folds live in three runs (OWN_RUNS): their oof_fold{k}.csv are copied into ctx.output_dir/pooled and
pooled by train25d.collect_oof (C16, the training's own scoring) -> pooled/oof.csv + scores.json. exp-0005's
5-fold oof.csv (OTHER_RUN, same frozen folds: checked via inputs.json) is rescored the same way and must match
its scores.json. Then, per target, a rank blend w * pct(exp-0004) + (1 - w) * pct(exp-0005) over a grid of w,
on the weak val pool (S4 > 0.5) and on gold58 (fold-mean probabilities): macro AUC per w, Spearman per class
between the members, and a paired study-bootstrap of macro AUC(blend at w) - macro AUC(best single member).
Ranks are taken over the pooled OOF across folds (fold calibration differences are part of the estimate).
Writes d8.json to ctx.output_dir and prints a summary.
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
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.knee_labels import LABELS, gold_labels, load_soft_targets  # noqa: E402
from src.metrics_auc import macro_auc  # noqa: E402
from src.train25d import collect_oof  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

OWN_RUNS = {"20261017t150919-625-2918": [0, 1, 2], "20261017t214233-033-af68": [3], "20261018t000939-333-228f": [4]}
OTHER_RUN = "20261017t135225-664-aecb"  # exp-0005 seed 0, 5 folds (OOF weak 0.8785, gold58 0.900)
FOLDS_RUN = "20261017t111717-941-7125"
COMP = "rsna-knee-abnormality-detection"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
N_BOOT = 2000
BOOT_SEED = 0
W_GRID = [round(0.1 * i, 1) for i in range(11)]  # weight of exp-0004 in the two-member rank blend
TOL = 1e-9  # rescored exp-0005 oof.csv vs its scores.json

ctx = get_ctx()
for r in [*OWN_RUNS, OTHER_RUN]:
    fr = json.loads((ctx.mounts[r] / "inputs.json").read_text())["folds_run"]
    if fr != FOLDS_RUN:
        raise ValueError(f"{r} was trained on folds {fr}, not {FOLDS_RUN}")
train = pd.read_csv(ctx.mounts[COMP] / "train.csv")
targets = load_soft_targets(ctx.mounts[S4[0]] / S4[1], train.StudyInstanceUID)
gold = gold_labels(train)
folds_df = pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv")

# exp-0004: pool the 5 folds with the training's own C16
pooled = ctx.output_dir / "pooled"
pooled.mkdir()
for r, folds in OWN_RUNS.items():
    for k in folds:
        shutil.copy2(ctx.mounts[r] / f"oof_fold{k}.csv", pooled / f"oof_fold{k}.csv")
_, own_scores = collect_oof(pooled, folds_df, targets, gold, N_BOOT, BOOT_SEED)
frames = {"exp-0004": pd.read_csv(pooled / "oof.csv", index_col="StudyInstanceUID").sort_index(),
          "exp-0005": pd.read_csv(ctx.mounts[OTHER_RUN] / "oof.csv", index_col="StudyInstanceUID").sort_index()}
sets = frames["exp-0004"]["set"]
if not frames["exp-0005"]["set"].equals(sets):
    raise ValueError("exp-0005's oof.csv studies or sets differ from exp-0004's pooled oof.csv")
val_ids, gold_ids = sets.index[sets == "val"], sets.index[sets == "gold"]
wy = (targets.loc[val_ids] > 0.5).astype(int).to_numpy()
gy = gold.loc[gold_ids].astype(int).to_numpy()

ref5 = json.loads((ctx.mounts[OTHER_RUN] / "scores.json").read_text())
d5 = max(abs(macro_auc(wy, frames["exp-0005"].loc[val_ids, LABELS].to_numpy())[0] - ref5["weak_oof"]["macro"]),
         abs(macro_auc(gy, frames["exp-0005"].loc[gold_ids, LABELS].to_numpy())[0] - ref5["gold"]["macro"]))
if d5 > TOL:
    raise AssertionError(f"exp-0005: rescored oof.csv differs from its scores.json by {d5}")


def pct(ids):
    """{member: per-target pct rank over the given studies}."""
    return {m: f.loc[ids, LABELS].rank(pct=True).to_numpy() for m, f in frames.items()}


def blend(r, w):
    return w * r["exp-0004"] + (1 - w) * r["exp-0005"]


def per_class(y, p):
    return dict(zip(LABELS, (round(float(a), 4) for a in macro_auc(y, p)[1])))


out = {"own_runs": OWN_RUNS, "other_run": OTHER_RUN, "folds_run": FOLDS_RUN, "n_val": len(val_ids),
       "n_gold": len(gold_ids), "exp-0004_pooled": own_scores, "views": {}}
rng = np.random.default_rng(BOOT_SEED)
for view, ids, y in (("weak", val_ids, wy), ("gold", gold_ids, gy)):
    r = pct(ids)
    spearman = {c: round(float(np.corrcoef(r["exp-0004"][:, j], r["exp-0005"][:, j])[0, 1]), 4)
                for j, c in enumerate(LABELS)}
    grid = {w: macro_auc(y, blend(r, w))[0] for w in W_GRID}
    best_single = "exp-0004" if grid[1.0] >= grid[0.0] else "exp-0005"
    # paired study-bootstrap: the same resampled studies for the blend and the best single member
    idx = [rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT)]
    deltas = {}
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        d = []
        for ix in idx:
            yy = y[ix]
            if (yy.sum(0) == 0).all() or (yy.sum(0) == len(ix)).all():
                continue
            ps = r[best_single][ix]
            d.append(macro_auc(yy, blend(r, w)[ix])[0] - macro_auc(yy, ps)[0])
        d = np.array(d)
        deltas[w] = {"mean": round(float(d.mean()), 4), "lo": round(float(np.percentile(d, 2.5)), 4),
                     "hi": round(float(np.percentile(d, 97.5)), 4), "p_gt0": round(float((d > 0).mean()), 3),
                     "n_valid": len(d)}
    out["views"][view] = {"spearman_per_class": spearman, "spearman_mean": round(float(np.mean(list(spearman.values()))), 4),
                          "macro_by_w4": {str(w): round(float(a), 4) for w, a in grid.items()},
                          "best_w4": max(grid, key=grid.get), "best_single": best_single,
                          "delta_vs_best_single": {str(w): v for w, v in deltas.items()},
                          "per_class_w4_0": per_class(y, r["exp-0005"]), "per_class_w4_1": per_class(y, r["exp-0004"]),
                          "per_class_w4_0.5": per_class(y, blend(r, 0.5))}

(ctx.output_dir / "d8.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"exp-0004_pooled": {"weak": own_scores["weak_oof"]["macro"], "gold": own_scores["gold"]["macro"],
                                      "gold_ci": own_scores["gold"]["bootstrap"]},
                  **{v: {k: out["views"][v][k] for k in ("spearman_mean", "macro_by_w4", "best_w4", "best_single",
                                                         "delta_vs_best_single")} for v in out["views"]}}, indent=1))
print(f"[d8_oof] {ctx.output_dir}")
