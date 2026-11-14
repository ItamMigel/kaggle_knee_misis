"""Mirror TTA of the exp-0004 member on its OOF (lead 18.10): plain vs L-R mirror vs their mean, per seed and seed mean.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/tta_check.py
    OMP_NUM_THREADS=2 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/tta_check.py --smoke

exp-0004 keeps native knee sides, so the full anatomical mirror of a study along the patient L-R axis
(src/tta25d.MirrorCache) is in-distribution. For each seed of SEED_RUNS and each fold k: fold{k}.pt of its run on
fold k's val studies and on the gold studies (folds.csv order) through infer25d.predict_studies, exactly the
training eval path (center_crop(cfg crop_px), cfg k_eval, AMP bf16, the training eval loaders' BS / WORKERS; val
and gold in separate calls, as the training's two eval loaders, so the batches are the stored OOF's), on the
exp-0002 cache (plain) and on MirrorCache(cache) (mirror). Every call is timed (wall); probs_s{seed}_{view}_fold{k}.csv
(set + LABELS) and a progress.jsonl row are written per fold as it goes, so a stopped run keeps what it did; a
projection of the total is printed after the first fold.
Plain must reproduce the stored OOF: per fold vs the run's oof_fold{k}.csv (fails fast) and pooled vs REF_OOF (max
|prob diff| <= REF_TOL, weak macro AUC within REF_AUC_TOL), else it raises. Pooled as collect_oof (val rows from their
fold, gold = mean over the 5 folds) from the written CSVs. Variants: per seed plain, mirror, tta = (plain + mirror) / 2;
avg_<view> = the two seeds' mean (avg_plain = the H member as the kernel runs it). Macro AUC on the weak val pool
(S4 > 0.5) and on gold; paired study-bootstrap deltas of pairs() (as seed_oof.py: one set of resamples per pool for
every pair; a resample is skipped unless every label two-class in the pool stays two-class); time multipliers mirror /
plain and (plain + mirror) / plain. Writes tta_check.json and prints a summary.
--smoke: CPU, seed 0, fold 0, the first SMOKE_N val + SMOKE_N gold studies, loader workers SMOKE_WORKERS: the plain vs
reference diffs are reported, not checked, a score undefined on so few studies is null and there is no 2-seed mean.
PROBS_RUN set: no inference, that earlier run's probs_*.csv and progress.jsonl are scored instead.
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
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.infer25d import load_models, predict_studies  # noqa: E402
from src.knee_labels import LABELS, gold_labels, load_soft_targets  # noqa: E402
from src.metrics_auc import macro_auc  # noqa: E402
from src.reader25d import center_crop  # noqa: E402
from src.tta25d import MirrorCache  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

FOLDS_RUN = "20261017t111717-941-7125"  # make_folds.py: the frozen folds of every exp-0004 / exp-0005 run
CACHE = "exp-0002/knee-vol04"
COMP = "rsna-knee-abnormality-detection"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
# exp-0004 train.py S4-only runs, {seed: {run: folds used}} (as seed_oof.py): seed 0 = d8_oof.py's three runs; seed 1 =
# folds 0-2 of 20261018t023618-204-1085 (launched for 0-4, stopped after fold 2) + 20261018t045524-429-281f
SEED_RUNS = {0: {"20261017t150919-625-2918": [0, 1, 2], "20261017t214233-033-af68": [3],
                 "20261018t000939-333-228f": [4]},
             1: {"20261018t023618-204-1085": [0, 1, 2], "20261018t045524-429-281f": [3, 4]}}
# the stored pooled OOF probabilities of each seed (collect_oof over SEED_RUNS[seed]'s oof_fold*.csv)
REF_OOF = {0: ("20261018t025721-274-46f2", "pooled/oof.csv"),  # d8_oof.py
           1: ("20261018t085237-897-bf50", "pooled_s1/oof.csv")}  # seed_oof.py
BS, WORKERS = 8, 7  # train.py CFG bs / workers = the training eval's loaders: the same batches as the stored OOF
AMP = torch.bfloat16  # the training eval's dtype (train25d.predict_logits); the TTA delta is what is measured
MEM_FRACTION = 0.65  # cuda only: train.py's cap on the shared L40 (~30 GB); the eval peaked at ~16 GB inside training
N_BOOT = 2000
BOOT_SEED = 0
# plain vs the stored OOF: dtype_check 20261017t184108-754-3332 measured this bf16 eval path at bs 2 vs the training
# OOF (bs 8) at max |prob diff| 0.0059 over 300 studies; here the batches are the training's, so it should be far
# smaller. REF_TOL leaves room for cudnn's algorithm choice over ~4.4k rows; REF_AUC_TOL is the lead's AUC grain.
REF_TOL = 0.02
REF_AUC_TOL = 0.001
VIEWS = ("plain", "mirror")
SMOKE_RUNS = {0: {"20261017t150919-625-2918": [0]}}
SMOKE_N = 2
SMOKE_WORKERS = 0  # in-process loading: the smoke is one CPU process
PROBS_RUN = None  # a run of this script whose probs_*.csv + progress.jsonl are scored instead of inferring (no GPU)


def check_runs(ctx, seed_runs, full):
    """Raises unless each run was trained on FOLDS_RUN with its seed and has its listed folds; full: folds 0-4 each."""
    for seed, runs in seed_runs.items():
        for run, folds in runs.items():
            inp = json.loads((ctx.mounts[run] / "inputs.json").read_text())
            if inp["folds_run"] != FOLDS_RUN or inp["cfg"]["seed"] != seed or not set(folds) <= set(inp["folds"]):
                raise ValueError(f"{run}: folds_run {inp['folds_run']}, cfg seed {inp['cfg']['seed']}, folds "
                                 f"{inp['folds']}; want {FOLDS_RUN}, seed {seed}, folds {folds}")
        got = sorted(k for folds in runs.values() for k in folds)
        if full and got != [0, 1, 2, 3, 4]:
            raise ValueError(f"seed {seed}: SEED_RUNS cover folds {got}, not 0-4 once each")


def sigmoid_oof(path):
    """An oof_fold{k}.csv (set + LABELS logits) as probabilities, computed as collect_oof does."""
    df = pd.read_csv(path, index_col="StudyInstanceUID")
    return pd.concat([df[["set"]], 1 / (1 + np.exp(-df[LABELS]))], axis=1)


def predict_sets(cache, ids, comp_series, models, cfg, bs, workers, device):
    """predict_studies of each set's studies in its own call (training eval loaders) -> (set + LABELS, {set: s})."""
    parts, secs = [], {}
    for name, studies in ids.items():
        t0 = time.perf_counter()
        p = predict_studies(studies, cache, comp_series, models, center_crop(cfg["crop_px"]), cfg["k_eval"], bs,
                            workers, device, AMP)
        secs[name] = time.perf_counter() - t0
        p.insert(0, "set", name)
        parts.append(p)
    return pd.concat(parts), secs


def prob_diff(p, ref):
    """{n, max, mean} of |p - ref| over p's studies; ref must have every one of them in the same set."""
    r = ref.loc[p.index]
    if not r["set"].equals(p["set"]):
        raise ValueError("studies are in other sets than in the reference")
    d = np.abs(p[LABELS].to_numpy(float) - r[LABELS].to_numpy(float))
    return {"n": len(p), "max": float(d.max()), "mean": float(d.mean())}


def infer(ctx, jobs, folds_df, smoke, device, workers):
    """Plain and mirror probabilities of each (seed, run, fold) job -> probs CSVs + progress.jsonl rows (output_dir)."""
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    comp_series = pd.read_csv(ctx.mounts[COMP] / "train_series.csv")
    cache = KneeVolCache(ctx.mounts[CACHE])
    caches = {"plain": cache, "mirror": MirrorCache(cache)}
    gold_ids = folds_df.StudyInstanceUID[folds_df.is_gold.astype(bool)].tolist()
    progress = []
    for i, (seed, run, k) in enumerate(jobs):
        t0 = time.perf_counter()
        models = load_models([ctx.mounts[run] / f"fold{k}.pt"], device)
        cfg = models[0][1]
        want = json.loads((ctx.mounts[run] / "inputs.json").read_text())["cfg"]
        if json.loads(json.dumps(cfg)) != want:
            raise ValueError(f"{run}/fold{k}.pt: cfg {cfg} != the run's inputs.json cfg {want}")
        ids = {"val": folds_df.StudyInstanceUID[folds_df.fold == k].tolist(), "gold": gold_ids}
        if smoke:
            ids = {name: studies[:SMOKE_N] for name, studies in ids.items()}
        ref = sigmoid_oof(ctx.mounts[run] / f"oof_fold{k}.csv")
        row = {"seed": seed, "run": run, "fold": k, "n_val": len(ids["val"]), "n_gold": len(ids["gold"]),
               "load_s": round(time.perf_counter() - t0, 3)}
        for view, c in caches.items():
            p, secs = predict_sets(c, ids, comp_series, models, cfg, BS, workers, device)
            p.to_csv(ctx.output_dir / f"probs_s{seed}_{view}_fold{k}.csv")
            row[view] = {f"{name}_s": round(s, 3) for name, s in secs.items()}
            if view == "plain":
                row["plain_vs_oof_fold"] = prob_diff(p, ref)
                if not smoke and row["plain_vs_oof_fold"]["max"] > REF_TOL:
                    raise AssertionError(f"seed {seed} fold {k}: plain vs {run}/oof_fold{k}.csv "
                                         f"{row['plain_vs_oof_fold']} > REF_TOL {REF_TOL}: the eval path differs "
                                         "from the one of the stored OOF")
        with open(ctx.output_dir / "progress.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        progress.append(row)
        print("FOLD " + json.dumps(row), flush=True)
        if i == 0:
            first = time.perf_counter() - t0
            print("PROJECTION " + json.dumps({"first_fold_s": round(first, 1), "n_folds": len(jobs),
                                              "projected_total_s": round(first * len(jobs))}), flush=True)
        del models
        if device == "cuda":
            torch.cuda.empty_cache()
    return progress


def pool(parts):
    """collect_oof's pooling of {fold: set + LABELS probs}: val rows from their fold, gold = the mean over the folds."""
    gold_sets = {frozenset(p.index[p["set"] == "gold"]) for p in parts.values()}
    if len(gold_sets) != 1:
        raise ValueError("the folds' gold studies differ")
    val = pd.concat([p[p["set"] == "val"] for p in parts.values()])
    if val.index.has_duplicates:
        raise ValueError("a val study is in two folds")
    gold = pd.concat([p.loc[p["set"] == "gold", LABELS] for p in parts.values()]).groupby(level=0).mean()
    gold.insert(0, "set", "gold")
    return pd.concat([val, gold]).sort_index()


def defined_auc(y, p, smoke):
    """macro_auc(y, p)[0]; None in the smoke when every label is single-class there (outside the smoke that raises)."""
    if smoke and not ((0 < y.sum(0)) & (y.sum(0) < len(y))).any():
        return None
    return macro_auc(y, p)[0]


def variants(pooled, seeds, smoke):
    """({name: probs over the pool}, the pool's set column): s<seed>_<view>, s<seed>_tta, avg_<view> (not in smoke)."""
    base = pooled[(seeds[0], "plain")]
    v = {}
    for s in seeds:
        for view in VIEWS:
            f = pooled[(s, view)]
            if not f.index.equals(base.index) or not f["set"].equals(base["set"]):
                raise ValueError(f"seed {s} {view}: pooled studies or sets differ from seed {seeds[0]} plain")
            v[f"s{s}_{view}"] = f[LABELS]
        v[f"s{s}_tta"] = (v[f"s{s}_plain"] + v[f"s{s}_mirror"]) / 2
    if not smoke:
        for view in (*VIEWS, "tta"):
            v[f"avg_{view}"] = (v[f"s0_{view}"] + v[f"s1_{view}"]) / 2
    return v, base["set"]


def pairs(seeds, smoke):
    """[(variant, baseline)] of the paired deltas: tta / mirror vs plain per seed and for avg, s0_tta vs avg_plain."""
    out = [(f"s{s}_{x}", f"s{s}_plain") for s in seeds for x in ("tta", "mirror")]
    if not smoke:
        out += [("avg_tta", "avg_plain"), ("avg_mirror", "avg_plain"), ("s0_tta", "avg_plain")]
    return out


def pool_scores(v, ids, y, prs, rng, smoke):
    """Macro (+ per-class) AUC of every variant on one pool and the paired bootstrap deltas of prs, one resample set."""
    defined = (0 < y.sum(0)) & (y.sum(0) < len(y))
    if smoke and not defined.any():
        return {"n": len(ids), "undefined": "every label is single-class on the smoke's studies"}
    p = {k: f.loc[ids].to_numpy() for k, f in v.items()}
    aucs = {k: macro_auc(y, x) for k, x in p.items()}
    need = sorted({k for pr in prs for k in pr})
    d = {pr: [] for pr in prs}
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(ids), len(ids))
        pos = y[ix][:, defined].sum(0)
        if not ((pos > 0) & (pos < len(ix))).all():
            continue
        auc = {k: macro_auc(y[ix], p[k][ix])[0] for k in need}
        for a, b in prs:
            d[(a, b)].append(auc[a] - auc[b])
    deltas = {f"{a}-{b}": {"point": aucs[a][0] - aucs[b][0], "mean": float(np.mean(x)),
                           "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)),
                           "p_gt0": float(np.mean(np.array(x) > 0)), "n_valid": len(x)} for (a, b), x in d.items()}
    return {"n": len(ids), "macro": {k: a[0] for k, a in aucs.items()},
            "per_class": {k: {c: (None if math.isnan(x) else x) for c, x in zip(LABELS, a[1])}
                          for k, a in aucs.items()},
            "deltas": deltas}


def timing(progress, smoke):
    """Seconds of plain / mirror / model loads over the folds and the multipliers (steady = without the first fold)."""
    def tot(rows, view):
        return sum(r[view]["val_s"] + r[view]["gold_s"] for r in rows)

    def ratios(rows):
        plain, mirror = tot(rows, "plain"), tot(rows, "mirror")
        return {"plain_s": round(plain, 1), "mirror_s": round(mirror, 1), "mirror_over_plain": mirror / plain,
                "plain_plus_mirror_over_plain": (plain + mirror) / plain}

    out = {**ratios(progress), "load_s": round(sum(r["load_s"] for r in progress), 1)}
    if not smoke:
        out["steady_without_first_fold"] = ratios(progress[1:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, seed 0, fold 0, the first SMOKE_N val + gold studies")
    smoke = ap.parse_args().smoke
    seed_runs = SMOKE_RUNS if smoke else SEED_RUNS
    device, workers = ("cpu", SMOKE_WORKERS) if smoke else ("cuda", WORKERS)

    ctx = get_ctx()
    check_runs(ctx, seed_runs, not smoke)
    folds_df = pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv")
    jobs = [(seed, run, k) for seed, runs in seed_runs.items() for run, folds in runs.items() for k in folds]
    if PROBS_RUN is None:
        probs_dir, progress = ctx.output_dir, infer(ctx, jobs, folds_df, smoke, device, workers)
    else:
        probs_dir = ctx.mounts[PROBS_RUN]
        progress = [json.loads(line) for line in (probs_dir / "progress.jsonl").read_text().splitlines()]
        if [[r["seed"], r["run"], r["fold"]] for r in progress] != [list(j) for j in jobs]:
            raise ValueError(f"{PROBS_RUN}/progress.jsonl jobs differ from this mode's {jobs}")

    seeds = list(seed_runs)
    pooled = {(seed, view): pool({k: pd.read_csv(probs_dir / f"probs_s{seed}_{view}_fold{k}.csv",
                                                 index_col="StudyInstanceUID")
                                  for folds in runs.values() for k in folds})
              for seed, runs in seed_runs.items() for view in VIEWS}
    v, sets = variants(pooled, seeds, smoke)
    ids = {"weak": sets.index[sets == "val"], "gold": sets.index[sets == "gold"]}
    gold = gold_labels(pd.read_csv(ctx.mounts[COMP] / "train.csv"))
    y = {"weak": (load_soft_targets(ctx.mounts[S4[0]] / S4[1], ids["weak"]) > 0.5).astype(int).to_numpy(),
         "gold": gold.loc[ids["gold"]].astype(int).to_numpy()}

    ref_check, mirror_vs_plain = {}, {}
    for seed in seeds:
        ref = pd.read_csv(ctx.mounts[REF_OOF[seed][0]] / REF_OOF[seed][1], index_col="StudyInstanceUID").sort_index()
        plain = pooled[(seed, "plain")]
        if not smoke and not (plain.index.equals(ref.index) and plain["set"].equals(ref["set"])):
            raise ValueError(f"seed {seed}: pooled plain studies or sets differ from REF_OOF {REF_OOF[seed]}")
        auc = {f"{pool_name}_{name}": defined_auc(y[pool_name], f.loc[ids[pool_name], LABELS].to_numpy(), smoke)
               for pool_name in ("weak", "gold") for name, f in (("plain", plain), ("ref", ref))}
        ref_check[seed] = {"ref_oof": REF_OOF[seed], **prob_diff(plain, ref), **auc}
        mirror_vs_plain[seed] = prob_diff(pooled[(seed, "mirror")], plain)
        if not smoke and (ref_check[seed]["max"] > REF_TOL or abs(auc["weak_plain"] - auc["weak_ref"]) > REF_AUC_TOL):
            raise AssertionError(f"seed {seed}: pooled plain vs REF_OOF {ref_check[seed]} beyond REF_TOL {REF_TOL} / "
                                 f"REF_AUC_TOL {REF_AUC_TOL}: the eval path differs from the one of the stored OOF")

    rng = np.random.default_rng(BOOT_SEED)
    prs = pairs(seeds, smoke)
    scores = {name: pool_scores(v, ids[name], y[name], prs, rng, smoke) for name in ("weak", "gold")}
    out = {"smoke": smoke, "device": device, "amp": str(AMP).removeprefix("torch."), "bs": BS, "workers": workers,
           "probs_run": PROBS_RUN, "seed_runs": {str(s): r for s, r in seed_runs.items()},
           "ref_tol": REF_TOL, "ref_auc_tol": REF_AUC_TOL, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "timing": timing(progress, smoke), "progress": progress, "ref_check": ref_check,
           "mirror_vs_plain": mirror_vs_plain, "pools": scores}
    (ctx.output_dir / "tta_check.json").write_text(json.dumps(out, indent=1))

    print("TIMING " + json.dumps(out["timing"]))
    print("REF_CHECK " + json.dumps(ref_check))
    print("MIRROR_VS_PLAIN " + json.dumps(mirror_vs_plain))
    for name, sc in scores.items():
        if "undefined" in sc:
            print(f"{name}: n {sc['n']}, {sc['undefined']}")
            continue
        print(f"{name}: n {sc['n']} | " + " ".join(f"{k} {a:.4f}" for k, a in sc["macro"].items()))
        for pr, dl in sc["deltas"].items():
            print(f"  {pr:22s} {dl['point']:+.4f} [{dl['lo']:+.4f}, {dl['hi']:+.4f}] p>0 {dl['p_gt0']:.3f} "
                  f"(n_valid {dl['n_valid']})")
    print(f"[tta_check] {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
