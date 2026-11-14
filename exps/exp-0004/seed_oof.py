"""Seed averages of exp-0004 and exp-0005 and their two-member rank blends: which members fit the kernel's time (lead 18.10).

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/seed_oof.py

Inputs (all on the frozen folds FOLDS_RUN, OOF of 5 folds each): exp-0004 seed 0 = d8_oof run D8_RUN's pooled/oof.csv,
exp-0004 seed 1 = its folds from two runs (SEED1_RUNS: folds 0-2 on GPU 0, 3-4 on GPU 1) pooled here by
train25d.collect_oof into pooled_s1/ (as d8_oof.py pooled seed 0), exp-0005 seed 0 / seed 1 = OTHER_RUNS, whose
oof.csv are rescored here and must match their scores.json. A seed average is the mean probability of the two seeds (a member with 10 checkpoints).
Variants, per target on the weak val pool (S4 > 0.5) and on gold58:
  singles  e4s0, e4s1, e4avg, e5s0, e5s1, e5avg
  B  W4 * pct(e4s0)  + (1 - W4) * pct(e5s0)    the two-member safe probe (D8)
  D  W4 * pct(e4avg) + (1 - W4) * pct(e5s0)
  G  W4 * pct(e4s0)  + (1 - W4) * pct(e5avg)
  H  W4 * pct(e4avg) + (1 - W4) * pct(e5avg)
macro AUC of each, Spearman per class between the seeds of each model, paired study-bootstrap deltas (PAIRS), and the
own part's projected kernel time per blend from the measured steady T4 rates: 5-checkpoint members from the
two-member timing kernel (TIMING_RUN), 10-checkpoint (seed-average) members from the H timing kernel (TIMING_H_RUN);
cache s/series x 6500 series + per member s/study x 1300 + fixed, sequential on one T4, and for two T4s (one member
per GPU, perfect overlap: cache + the slower member + fixed; a lower bound until the split is timed). Ranks are pooled across folds, as in d8_oof.py. Writes seed_oof.json and prints a summary.
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

D8_RUN = "20261018t025721-274-46f2"  # its pooled/oof.csv = exp-0004 seed 0, 5 folds (collect_oof over 3 runs)
SEED0_RUNS = ["20261017t150919-625-2918", "20261017t214233-033-af68", "20261018t000939-333-228f"]  # folds check only
# exp-0004 train.py SEED 1: {run: folds used}; the GPU 0 run also trains folds 3-4 later (duplicates, not used)
SEED1_RUNS = {"20261018t023618-204-1085": [0, 1, 2], "20261018t045524-429-281f": [3, 4]}
OTHER_RUNS = {"e5s0": "20261017t135225-664-aecb", "e5s1": "20261018t003924-017-1d75"}  # exp-0005 seed 0 / seed 1
TIMING_RUN = "20261018t035044-226-b1c4"  # two-member timing kernel (T4), 5 checkpoints per member
TIMING_H_RUN = "20261018t052751-433-0d9c"  # H timing kernel (T4, sequential), 10 checkpoints per member
FOLDS_RUN = "20261017t111717-941-7125"
COMP = "rsna-knee-abnormality-detection"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
W4 = 0.6  # D8 (lead 18.10): exp-0004's share of the own weight
BLENDS = {"B": ("e4s0", "e5s0"), "D": ("e4avg", "e5s0"), "G": ("e4s0", "e5avg"), "H": ("e4avg", "e5avg")}
PAIRS = [("D", "B"), ("G", "B"), ("H", "B"), ("G", "D"), ("B", "e4s0"), ("e4avg", "e4s0"), ("e5avg", "e5s0")]
N_HIDDEN, SERIES_PER_STUDY = 1300, 5.0  # = the timing kernel's projection case
OWN_BUDGET_S, D7_LIMIT_S = 9000, 7200  # two-member budget and the lead's condition (lead 18.10)
N_BOOT = 2000
BOOT_SEED = 0
TOL = 1e-9  # rescored seed oof.csv vs its scores.json

ctx = get_ctx()
for r in [*SEED0_RUNS, *SEED1_RUNS, *OTHER_RUNS.values()]:
    fr = json.loads((ctx.mounts[r] / "inputs.json").read_text())["folds_run"]
    if fr != FOLDS_RUN:
        raise ValueError(f"{r} was trained on folds {fr}, not {FOLDS_RUN}")
train = pd.read_csv(ctx.mounts[COMP] / "train.csv")
targets = load_soft_targets(ctx.mounts[S4[0]] / S4[1], train.StudyInstanceUID)
gold = gold_labels(train)
for r in SEED1_RUNS:
    seed = json.loads((ctx.mounts[r] / "inputs.json").read_text())["cfg"]["seed"]
    if seed != 1:
        raise ValueError(f"{r} has cfg seed {seed}, want 1")
pooled_s1 = ctx.output_dir / "pooled_s1"
pooled_s1.mkdir()
for r, folds in SEED1_RUNS.items():
    for k in folds:
        shutil.copy2(ctx.mounts[r] / f"oof_fold{k}.csv", pooled_s1 / f"oof_fold{k}.csv")
_, s1_scores = collect_oof(pooled_s1, pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv"), targets, gold, N_BOOT, BOOT_SEED)


def read_oof(path):
    return pd.read_csv(path, index_col="StudyInstanceUID").sort_index()


frames = {"e4s0": read_oof(ctx.mounts[D8_RUN] / "pooled" / "oof.csv"), "e4s1": read_oof(pooled_s1 / "oof.csv"),
          **{k: read_oof(ctx.mounts[r] / "oof.csv") for k, r in OTHER_RUNS.items()}}
sets = frames["e4s0"]["set"]
for k, f in frames.items():
    if not f["set"].equals(sets):
        raise ValueError(f"{k}: oof.csv studies or sets differ from exp-0004 seed 0's pooled oof.csv")
val_ids, gold_ids = sets.index[sets == "val"], sets.index[sets == "gold"]
wy = (targets.loc[val_ids] > 0.5).astype(int).to_numpy()
gy = gold.loc[gold_ids].astype(int).to_numpy()
for k, r in OTHER_RUNS.items():
    ref = json.loads((ctx.mounts[r] / "scores.json").read_text())
    d = max(abs(macro_auc(wy, frames[k].loc[val_ids, LABELS].to_numpy())[0] - ref["weak_oof"]["macro"]),
            abs(macro_auc(gy, frames[k].loc[gold_ids, LABELS].to_numpy())[0] - ref["gold"]["macro"]))
    if d > TOL:
        raise AssertionError(f"{k} ({r}): rescored oof.csv differs from its scores.json by {d}")


def variants(ids):
    """{variant: per-target score array over `ids`}: probabilities for singles, rank blends for B/D/G/H."""
    p = {k: f.loc[ids, LABELS] for k, f in frames.items()}
    p["e4avg"], p["e5avg"] = (p["e4s0"] + p["e4s1"]) / 2, (p["e5s0"] + p["e5s1"]) / 2
    out = {k: v.to_numpy() for k, v in p.items()}
    pct = {k: v.rank(pct=True).to_numpy() for k, v in p.items()}
    for b, (m4, m5) in BLENDS.items():
        out[b] = W4 * pct[m4] + (1 - W4) * pct[m5]
    return out, pct


def timing_projection(run):
    return json.loads(next(line.split("TIMING_SUMMARY", 1)[1] for line in
                           (ctx.mounts[run] / "log.txt").read_text().splitlines() if "TIMING_SUMMARY" in line))["projection"]


pr5, pr10 = timing_projection(TIMING_RUN), timing_projection(TIMING_H_RUN)
rate = {f"{m}{tag}": pr["steady_member_s_per_study"][name] for m, name in (("e4", "exp-0004"), ("e5", "exp-0005"))
        for tag, pr in (("s0", pr5), ("avg", pr10))}  # s/study of a 5-ckpt (s0) and a 10-ckpt (avg) member
base_s = pr5["steady_cache_s_per_series"] * N_HIDDEN * SERIES_PER_STUDY + pr5["fixed"]["fixed_s"]
kernel_s = {}
for b, members in BLENDS.items():
    per = [rate[m] * N_HIDDEN for m in members]
    seq, par = base_s + sum(per), base_s + max(per)
    kernel_s[b] = {"one_t4_s": round(seq), "one_t4_le_d7": seq <= D7_LIMIT_S, "two_t4_lower_bound_s": round(par),
                   "two_t4_le_d7": par <= D7_LIMIT_S}

out = {"d8_run": D8_RUN, "seed1_runs": SEED1_RUNS, "e4s1_pooled": s1_scores, "other_runs": OTHER_RUNS, "timing_run": TIMING_RUN, "folds_run": FOLDS_RUN,
       "w4": W4, "blends": BLENDS, "n_val": len(val_ids), "n_gold": len(gold_ids),
       "kernel_time": {"basis": f"steady T4 rates of timing runs {TIMING_RUN} (5 ckpts) and {TIMING_H_RUN} (10 ckpts), "
                                f"{N_HIDDEN} studies x {SERIES_PER_STUDY} series; two T4s = perfect overlap (lower bound)",
                       "rates_s_per_study": rate, "cache_plus_fixed_s": round(base_s), **kernel_s},
       "views": {}}
rng = np.random.default_rng(BOOT_SEED)
for view, ids, y in (("weak", val_ids, wy), ("gold", gold_ids, gy)):
    v, pct = variants(ids)
    names = ["e4s0", "e4s1", "e4avg", "e5s0", "e5s1", "e5avg", *BLENDS]
    macro = {k: round(float(macro_auc(y, v[k])[0]), 4) for k in names}
    spearman = {m: round(float(np.mean([np.corrcoef(pct[f"{m}s0"][:, j], pct[f"{m}s1"][:, j])[0, 1]
                                        for j in range(len(LABELS))])), 4) for m in ("e4", "e5")}
    idx = [rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT)]
    deltas = {}
    for x, base in PAIRS:
        d = []
        for ix in idx:
            yy = y[ix]
            if (yy.sum(0) == 0).all() or (yy.sum(0) == len(ix)).all():
                continue
            d.append(macro_auc(yy, v[x][ix])[0] - macro_auc(yy, v[base][ix])[0])
        d = np.array(d)
        deltas[f"{x}-{base}"] = {"mean": round(float(d.mean()), 4), "lo": round(float(np.percentile(d, 2.5)), 4),
                                 "hi": round(float(np.percentile(d, 97.5)), 4), "p_gt0": round(float((d > 0).mean()), 3),
                                 "n_valid": len(d)}
    out["views"][view] = {"macro": macro, "spearman_seed0_seed1_mean": spearman, "deltas": deltas}

(ctx.output_dir / "seed_oof.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"kernel_time": out["kernel_time"], **out["views"]}, indent=1))
print(f"[seed_oof] {ctx.output_dir}")
