"""Training, evaluation and OOF collection for the 2.5D study reader (src/reader25d.py)."""
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.knee_labels import LABELS
from src.metrics_auc import bootstrap_macro_auc, macro_auc
from src.reader25d import (DEPTH_HI, DEPTH_LO, KneeNet, StudyDataset, cache_depth_rows, cache_study_table,
                           depth_source_counts, joint_depth_fractions, seed_worker)

# every cfg key, no defaults: limit_train and max_steps_per_epoch are smoke knobs (0 = off);
# pretrained=False only for smoke runs that must not download weights;
# axial_half_mm / axial_max_extent_mm: joint_depth_fractions (both 0 = no restriction)
CFG_KEYS = ("backbone", "pretrained", "res", "crop_px", "fov", "mean", "std", "gap_mm", "k", "k_eval",
            "epochs", "bs", "lr", "lr_head", "wd", "ema", "grad_ckpt", "eval_every", "workers", "seed",
            "limit_train", "max_steps_per_epoch", "axial_half_mm", "axial_max_extent_mm")


def check_cfg(cfg):
    """Raises unless cfg has exactly the CFG_KEYS (a missing or unknown key is an error)."""
    missing, unknown = set(CFG_KEYS) - set(cfg), set(cfg) - set(CFG_KEYS)
    if missing or unknown:
        raise ValueError(f"cfg keys: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if cfg["eval_every"] < 1:
        raise ValueError(f"eval_every must be >= 1, got {cfg['eval_every']}")


def build_model(cfg, pretrained):
    """KneeNet from cfg (backbone, grad_ckpt), channels_last; `pretrained` is passed explicitly."""
    return KneeNet(cfg["backbone"], pretrained, grad_ckpt=cfg["grad_ckpt"]).to(memory_format=torch.channels_last)


def make_loader(dataset, cfg, train, device, persistent):
    """DataLoader over a StudyDataset: per-worker rng via seed_worker; shuffle + drop_last in training."""
    w = cfg["workers"]
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg["bs"], shuffle=train, drop_last=train, num_workers=w,
        pin_memory=device.type == "cuda", persistent_workers=persistent and w > 0,
        worker_init_fn=seed_worker if w > 0 else None)


@torch.no_grad()
def predict_logits(model, loader, cfg, device, amp_dtype):
    """C15: autocast(amp_dtype) eval logits [N, 12] of a model over a loader (views from cfg res/fov/mean/std)."""
    model.eval()
    out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.to(device, non_blocking=True), mask.to(device), pos.to(device)
        with torch.autocast(device.type, dtype=amp_dtype):
            logit = model(x, mask, pos, res=cfg["res"], train=False, fov=cfg["fov"], mean=cfg["mean"], std=cfg["std"])
        out.append(logit.float().cpu())
    return torch.cat(out).numpy()


def _rows(df, ids, what):
    """df.loc[ids, LABELS] as a numpy array; raises on missing studies or NaN."""
    missing = pd.Index(ids).difference(df.index)
    if len(missing):
        raise KeyError(f"{what}: {len(missing)} studies missing, e.g. {missing[0]}")
    v = df.loc[list(ids), LABELS].to_numpy()
    if np.isnan(v.astype(float)).any():
        raise ValueError(f"{what}: NaN labels")
    return v


def _auc_fields(y, p):
    """{"macro": float, "per_class": {label: auc or None}} of macro_auc."""
    m, each = macro_auc(y, p)
    return {"macro": m, "per_class": {c: (None if math.isnan(a) else a) for c, a in zip(LABELS, each)}}


def train_fold(cfg, fold, train_ids, val_ids, gold_ids, targets, gold, table, depth_frac, load_volume, crop_fn, out_dir,
               device):
    """C14: trains one fold, writes fold{k}.pt, oof_fold{k}.csv, metrics_fold{k}.jsonl; returns last metrics.

    targets: soft labels DataFrame (index StudyInstanceUID, LABELS) covering train and val ids; the
    val metric uses targets > 0.5. gold: official labels (index StudyInstanceUID, LABELS) covering
    gold_ids. val_ids may be empty (train on everything, evaluate on gold only). depth_frac: per-series
    window depth range for every StudyDataset (cache_depth_fractions).
    """
    check_cfg(cfg)
    device, out_dir = torch.device(device), Path(out_dir)
    train_ids, val_ids, gold_ids = list(train_ids), list(val_ids), list(gold_ids)
    for name, ids in (("train", train_ids), ("val", val_ids), ("gold", gold_ids)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}_ids contain duplicates")
    if set(train_ids) & (set(val_ids) | set(gold_ids)) or set(val_ids) & set(gold_ids):
        raise ValueError("train, val and gold ids must be pairwise disjoint")
    if not gold_ids:
        raise ValueError("gold_ids is empty")
    final_path = out_dir / f"fold{fold}.pt"
    if final_path.exists():
        raise FileExistsError(f"{final_path} exists: fold {fold} is already finished in this run dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["limit_train"]:
        train_ids = train_ids[:cfg["limit_train"]]
    torch.manual_seed(cfg["seed"])

    def say(msg):
        print(msg, flush=True)

    say(json.dumps({"fold": fold, "n_train": len(train_ids), "n_val": len(val_ids), "n_gold": len(gold_ids), **cfg}))
    y_train = _rows(targets, train_ids, "targets/train").astype(np.float32)
    val_y = (_rows(targets, val_ids, "targets/val") > 0.5).astype(int) if val_ids else None
    gold_y = _rows(gold, gold_ids, "gold").astype(int)

    def dataset(ids, y, k, train):
        return StudyDataset(ids, table, load_volume, y, k, train, crop_fn, cfg["gap_mm"], cfg["crop_px"], cfg["seed"],
                            depth_frac)

    train_dl = make_loader(dataset(train_ids, y_train, cfg["k"], True), cfg, True, device, persistent=True)
    if len(train_dl) == 0:
        raise ValueError(f"{len(train_ids)} training studies give no full batch of {cfg['bs']}")
    eval_sets = {"gold": gold_ids}
    if val_ids:
        eval_sets["val"] = val_ids
    # eval loaders are not persistent, so their workers live only while an evaluation runs
    eval_dl = {s: make_loader(dataset(ids, None, cfg["k_eval"], False), cfg, False, device, persistent=False)
               for s, ids in eval_sets.items()}

    model = build_model(cfg, cfg["pretrained"]).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_params = list(model.enc.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["lr"]}, {"params": head_params, "lr": cfg["lr_head"]}],
                            weight_decay=cfg["wd"])
    spe = min(len(train_dl), cfg["max_steps_per_epoch"]) if cfg["max_steps_per_epoch"] else len(train_dl)
    steps, warm = cfg["epochs"] * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    views = dict(res=cfg["res"], fov=cfg["fov"], mean=cfg["mean"], std=cfg["std"])
    cuda = device.type == "cuda"

    # a state file is written after every epoch so an interrupted run continues where it stopped
    state_path = out_dir / f"state_fold{fold}.pt"
    metrics_path = out_dir / f"metrics_fold{fold}.jsonl"
    step, start_ep, preds, row = 0, 0, None, None
    if state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        if st["cfg"] != cfg:
            raise ValueError(f"{state_path} was written with a different cfg; refusing to resume")
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, start_ep = st["step"], st["epoch"] + 1
        say(f"resumed after epoch {st['epoch']}")
    elif metrics_path.exists():
        raise FileExistsError(f"{metrics_path} exists without a state file to resume from")
    for ep in range(start_ep, cfg["epochs"]):
        model.train()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        t0, tot, n, n_studies, n_images, wait = time.time(), 0.0, 0, 0, 0, 0.0
        tw = time.time()
        for x, mask, pos, y in train_dl:
            wait += time.time() - tw
            x, mask, pos, y = x.to(device, non_blocking=True), mask.to(device), pos.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logit = model(x, mask, pos, train=True, **views)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            step += 1
            d = min(cfg["ema"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            li = loss.item()
            if not math.isfinite(li):
                raise FloatingPointError(f"non-finite loss {li} at epoch {ep} step {n}")
            tot += li
            n += 1
            n_studies += len(x)
            n_images += int(mask.sum()) * cfg["k"]
            if n % 100 == 0:
                print(f"  ep {ep} step {n}/{spe} loss {tot / n:.4f} {(time.time() - t0) / n:.2f}s/it", flush=True)
            if n == spe:
                break
            tw = time.time()
        sec = time.time() - t0
        row = {"epoch": ep, "step": step, "loss": tot / n, "seconds": sec, "data_wait_seconds": wait,
               "train_studies_per_s": n_studies / sec, "train_images_per_s": n_images / sec,
               "lr": sched.get_last_lr()[0], "lr_head": sched.get_last_lr()[1],
               "peak_mem_train_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None}
        msg = f"ep {ep:2d} loss {tot / n:.4f} time {sec:.0f}s {n_studies / sec:.2f} studies/s"
        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            te = time.time()
            preds = {s: predict_logits(ema, dl, cfg, device, torch.bfloat16) for s, dl in eval_dl.items()}
            row["eval_seconds"] = time.time() - te
            row["peak_mem_eval_gb"] = torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None
            row["gold"] = _auc_fields(gold_y, preds["gold"])
            msg += f" | gold {row['gold']['macro']:.4f}"
            if val_y is not None:
                row["val"] = _auc_fields(val_y, preds["val"])
                msg += f" | val {row['val']['macro']:.4f} [" + " ".join(
                    "nan" if a is None else f"{a:.3f}" for a in row["val"]["per_class"].values()) + "]"
        else:
            preds = None
            row.update(eval_seconds=None, peak_mem_eval_gb=None, gold=None)
            if val_y is not None:
                row["val"] = None
        say(msg)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "epoch": ep, "cfg": cfg}, str(state_path) + ".tmp")
        os.replace(str(state_path) + ".tmp", state_path)

    if preds is None:  # resumed after the last epoch had already finished
        preds = {s: predict_logits(ema, dl, cfg, device, torch.bfloat16) for s, dl in eval_dl.items()}
        row = {"epoch": cfg["epochs"] - 1, "gold": _auc_fields(gold_y, preds["gold"])}
        if val_y is not None:
            row["val"] = _auc_fields(val_y, preds["val"])
    torch.save({"model": ema.state_dict(), "cfg": cfg}, final_path)
    parts = []
    for s, ids in eval_sets.items():
        df = pd.DataFrame(preds[s], index=pd.Index(ids, name="StudyInstanceUID"), columns=LABELS)
        df.insert(0, "set", s)
        parts.append(df)
    pd.concat(parts).to_csv(out_dir / f"oof_fold{fold}.csv")
    say("gold per class: " + " ".join(f"{c[:6]}={'nan' if a is None else f'{a:.3f}'}"
                                      for c, a in row["gold"]["per_class"].items()))
    state_path.unlink(missing_ok=True)
    return row


def cache_depth_fractions(cfg, cache, studies):
    """(depth_frac, report) of joint_depth_fractions for `studies` of a KneeVolCache with the cfg axial keys."""
    check_cfg(cfg)
    return joint_depth_fractions(cache_depth_rows(cache, studies), cfg["axial_half_mm"], cfg["axial_max_extent_mm"],
                                 DEPTH_LO, DEPTH_HI)


def run_folds(cfg, folds, cache, comp_series, folds_df, targets, gold, crop_fn, out_dir, device, n_boot, seed):
    """Trains `folds` of folds_df on a KneeVolCache (C14 per fold) and collects their OOF (C16): a script's whole job.

    cache: KneeVolCache; comp_series: the competition train_series.csv; folds_df: the frozen
    folds table; targets: soft labels of every study; gold: official labels of the gold studies.
    Also writes depth_report.csv (the axial stacks restricted by cache_depth_fractions) to out_dir.
    """
    table = cache_study_table(cache.series, comp_series, folds_df.StudyInstanceUID)
    # lookups built once in the main process, so the forked loader workers share them
    cache.load(cache.series.series[cache.series.error_code.isna()].iloc[0])
    depth_frac, depth_report = cache_depth_fractions(cfg, cache, folds_df.StudyInstanceUID)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    depth_report.to_csv(Path(out_dir) / "depth_report.csv", index=False)
    print(json.dumps({"depth_restricted": len(depth_report), "sources": depth_source_counts(depth_report),
                      "kept_slices": depth_report.n_kept.describe().round(1).to_dict()}), flush=True)
    is_gold = folds_df.is_gold.astype(bool)
    gold_ids = folds_df.StudyInstanceUID[is_gold].tolist()
    for k in folds:
        tr = folds_df.StudyInstanceUID[~is_gold & (folds_df.fold != k)].tolist()
        va = folds_df.StudyInstanceUID[folds_df.fold == k].tolist()
        if not va:
            raise ValueError(f"fold {k} has no studies in folds_df")
        train_fold(cfg, k, tr, va, gold_ids, targets, gold, table, depth_frac, cache.load, crop_fn, out_dir, device)
    trained = folds_df[folds_df.fold.isin(folds) | is_gold]
    return collect_oof(out_dir, trained, targets, gold, n_boot, seed)


def collect_oof(run_dir, folds_df, targets, gold, n_boot, seed):
    """C16: pools val logits of oof_fold{k}.csv into probabilities, gold = fold mean; writes oof.csv and scores.json.

    folds_df (StudyInstanceUID, fold, is_gold) must describe exactly what was trained: every fold
    >= 0 in it needs its oof_fold{k}.csv, whose val rows equal that fold's studies and whose gold
    rows equal the is_gold studies. Scores (D4): pooled weak-label macro AUC (targets > 0.5) with
    per class and per fold, gold macro AUC of the fold-mean probability with a bootstrap CI.
    """
    run_dir = Path(run_dir)
    fd = folds_df.set_index("StudyInstanceUID")
    gold_set = set(fd.index[fd.is_gold.astype(bool)])
    if (fd.fold[fd.is_gold.astype(bool)] != -1).any() or (fd.fold[~fd.is_gold.astype(bool)] < 0).any():
        raise ValueError("folds_df: gold studies must have fold -1 and weak studies fold >= 0")
    folds = sorted(fd.fold[fd.fold >= 0].unique())
    val_parts, gold_parts, per_fold = [], [], {}
    for k in folds:
        path = run_dir / f"oof_fold{k}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (folds_df has fold {k})")
        df = pd.read_csv(path, index_col="StudyInstanceUID")
        if set(df.set.unique()) - {"val", "gold"}:
            raise ValueError(f"{path}: unexpected set values {sorted(set(df.set.unique()))}")
        prob = 1 / (1 + np.exp(-df[LABELS]))
        val, g = prob[df.set == "val"], prob[df.set == "gold"]
        if set(val.index) != set(fd.index[fd.fold == k]) or len(val) != (fd.fold == k).sum():
            raise ValueError(f"{path}: val rows differ from the studies of fold {k} in folds_df")
        if set(g.index) != gold_set or len(g) != len(gold_set):
            raise ValueError(f"{path}: gold rows differ from the is_gold studies in folds_df")
        per_fold[str(k)] = macro_auc(_rows(targets, val.index, "targets") > 0.5, val.to_numpy())[0]
        val_parts.append(val)
        gold_parts.append(g)
    val = pd.concat(val_parts)
    gmean = pd.concat(gold_parts).groupby(level=0).mean().loc[sorted(gold_set)]
    oof = pd.concat([val.assign(set="val"), gmean.assign(set="gold")])[["set"] + LABELS]
    oof.index.name = "StudyInstanceUID"
    oof.to_csv(run_dir / "oof.csv")

    wy = (_rows(targets, val.index, "targets") > 0.5).astype(int)
    gy = _rows(gold, gmean.index, "gold").astype(int)
    scores = {
        "n_folds": len(folds), "n_val": len(val), "n_gold": len(gmean),
        "weak_oof": _auc_fields(wy, val.to_numpy()),
        "weak_per_fold": per_fold,
        "gold": {**_auc_fields(gy, gmean.to_numpy()),
                 "bootstrap": bootstrap_macro_auc(gy, gmean.to_numpy(), n_boot, seed), "n_boot": n_boot, "seed": seed},
    }
    with open(run_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=1)
    return oof, scores
