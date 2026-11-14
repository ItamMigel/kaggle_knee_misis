"""Label check of train.py's TARGET_SOURCES before the label-diverse run (CPU only), port of exp-0005's label_check.py.

    OMP_NUM_THREADS=2 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/label_check.py

Per source, on the weak studies of FOLDS_RUN and on the gold studies: coverage (missing studies) and
NaN per label in the raw CSV. Per label vs S4 on the weak studies: Pearson correlation, agreement at
0.5, prevalence > 0.5 (also of the mean = the training targets). Gold macro AUC of each source and of
the mean on the gold studies every source covers, with a paired bootstrap of mean - S4. Then the
exact training path: train.load_targets(TARGET_SOURCES) on the weak studies (raises on a missing
study or NaN), and train.score_against_s4 with each RESCORE_RUNS run's folds on a copy of its
oof_fold*.csv + scores.json (exercises the --folds subset), which must reproduce that S4-only run's
scores.json and keep the copied one as scores_train_targets.json. Writes label_check.json; raises if a
check fails.
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
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (train.py: TARGET_SOURCES, load_targets, score_against_s4)
from src.knee_labels import LABELS, gold_labels  # noqa: E402
from src.metrics_auc import macro_auc  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

# the seed-0 S4-only runs (boot seed 0 = train.SEED) with their --folds: S4 rescoring must reproduce each scores.json
RESCORE_RUNS = {"20261017t150919-625-2918": [0, 1, 2], "20261017t214233-033-af68": [3],
                "20261018t000939-333-228f": [4]}
N_BOOT = 2000
BOOT_SEED = 0
TOL = 1e-12


def raw_labels(ctx, source):
    """The source CSV's LABELS columns indexed by StudyInstanceUID (no clipping, NaN kept)."""
    ds, name = source
    df = pd.read_csv(ctx.mounts[ds] / name)
    if df.StudyInstanceUID.duplicated().any():
        raise ValueError(f"{ds}/{name}: duplicated StudyInstanceUID")
    return df.set_index("StudyInstanceUID")[LABELS].astype(float)


def paired_boot(y, a, b):
    """2.5 / 50 / 97.5 % of macro_auc(a) - macro_auc(b) over study resamples that keep every two-class label two-class."""
    rng = np.random.default_rng(BOOT_SEED)
    n, defined = len(y), (0 < y.sum(0)) & (y.sum(0) < len(y))
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        pos = y[i][:, defined].sum(0)
        if ((pos > 0) & (pos < n)).all():
            diffs.append(macro_auc(y[i], a[i])[0] - macro_auc(y[i], b[i])[0])
    return [float(x) for x in np.percentile(diffs, [2.5, 50, 97.5])]


def close(a, b):
    """Recursive equality of two JSON values, floats within TOL."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= TOL
    return a == b


def rescore_run(ctx, run, folds, folds_df, gold):
    """train.score_against_s4 on a copy of run's oof_fold*.csv + scores.json: does it reproduce the run's scores.json?"""
    src, dst = ctx.mounts[run], ctx.output_dir / "rescore_test" / run
    trained = json.loads((src / "inputs.json").read_text())["folds"]
    if trained != folds:
        raise ValueError(f"RESCORE_RUNS[{run!r}] = {folds}, but the run trained folds {trained}")
    dst.mkdir(parents=True)
    for f in [*src.glob("oof_fold*.csv"), src / "scores.json"]:
        shutil.copy2(f, dst / f.name)
    T.score_against_s4(ctx, dst, folds_df, folds, gold)
    ref = json.loads((src / "scores.json").read_text())
    return {"run": run, "folds": folds,
            "scores_reproduced": close(json.loads((dst / "scores.json").read_text()), ref),
            "train_targets_kept": json.loads((dst / "scores_train_targets.json").read_text()) == ref}


def main():
    ctx = get_ctx()
    if T.TARGET_SOURCES[0] != T.S4:
        raise ValueError(f"TARGET_SOURCES must start with S4 (the reference), got {T.TARGET_SOURCES}")
    folds_df = pd.read_csv(ctx.mounts[T.FOLDS_RUN] / "folds.csv")
    is_gold = folds_df.is_gold.astype(bool)
    weak_ids, gold_ids = pd.Index(folds_df.StudyInstanceUID[~is_gold]), pd.Index(folds_df.StudyInstanceUID[is_gold])
    gold = gold_labels(pd.read_csv(ctx.mounts[T.COMP] / "train.csv"))
    names = ["/".join(s) for s in T.TARGET_SOURCES]
    raw = {n: raw_labels(ctx, s) for n, s in zip(names, T.TARGET_SOURCES)}

    coverage = {n: {"weak_missing": int(len(weak_ids.difference(r.index))),
                    "gold_missing": int(len(gold_ids.difference(r.index))),
                    "gold_missing_ids": sorted(gold_ids.difference(r.index)),
                    "weak_nan_per_label": {c: int(v) for c, v in r.loc[weak_ids.intersection(r.index)].isna().sum().items()},
                    "values_top": {str(k): int(v) for k, v in r.stack().round(3).value_counts().head(8).items()}}
                for n, r in raw.items()}
    print(json.dumps({"coverage": coverage}), flush=True)

    targets = T.load_targets(ctx, T.TARGET_SOURCES, weak_ids)  # the training path: raises on missing / NaN
    s4 = raw[names[0]].loc[weak_ids]
    per_label = {}
    for c in LABELS:
        row = {"prev_s4": float((s4[c] > 0.5).mean()), "prev_mean": float((targets[c] > 0.5).mean())}
        for n in names[1:]:
            o = raw[n].loc[weak_ids, c]
            row[n] = {"corr": float(np.corrcoef(s4[c], o)[0, 1]), "agree_at_0.5": float(((s4[c] > 0.5) == (o > 0.5)).mean()),
                      "prev": float((o > 0.5).mean())}
        per_label[c] = row

    gids = gold_ids
    for r in raw.values():
        gids = gids.intersection(r.index)
    y = gold.loc[gids].astype(int).to_numpy()
    gsrc = {n: r.loc[gids].to_numpy() for n, r in raw.items()}
    gmean = sum(gsrc.values()) / len(gsrc)
    gold_auc = {n: macro_auc(y, p)[0] for n, p in gsrc.items()}
    gold_auc["mean"] = macro_auc(y, gmean)[0]
    gold_res = {"n_gold": len(gids), "macro_auc": gold_auc, "mean_minus_s4_paired_boot": paired_boot(y, gmean, gsrc[names[0]])}
    print(json.dumps({"gold": gold_res}), flush=True)

    rescore = [rescore_run(ctx, run, folds, folds_df, gold) for run, folds in RESCORE_RUNS.items()]

    res = {"target_sources": T.TARGET_SOURCES, "n_weak": len(weak_ids), "coverage": coverage, "per_label": per_label,
           "gold": gold_res, "rescore": rescore}
    (ctx.output_dir / "label_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"rescore": rescore}), flush=True)
    for c, row in per_label.items():
        print(f"{c:18s} prev S4 {row['prev_s4']:.3f} mean {row['prev_mean']:.3f} | "
              + " | ".join(f"corr {row[n]['corr']:.3f} agree {row[n]['agree_at_0.5']:.3f} prev {row[n]['prev']:.3f}"
                           for n in names[1:]), flush=True)
    bad = [n for n, c in coverage.items() if c["weak_missing"] or any(c["weak_nan_per_label"].values())]
    failed = [r["run"] for r in rescore if not (r["scores_reproduced"] and r["train_targets_kept"])]
    if bad or failed:
        raise AssertionError(f"label check failed: sources with weak gaps {bad}, rescore failed for {failed}: {rescore}")
    print(f"[label_check] {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
