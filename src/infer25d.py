"""Inference for the 2.5D reader: raw DICOM folders -> KneeVolCache, checkpoint loading, study probabilities.

The blend kernel preprocesses test_series/ once into a KneeVolCache (build_dir_cache) and every member
reads the same volumes, through the same study table, depth restriction and StudyDataset as training.
In a kernel import this module after kernel_env.install_wheels: it imports pydicom (via knee_volume).
build_dir_cache runs its pool in spawned processes: a forked child that calls cv2.setNumThreads after
the parent ran threaded cv2 hangs forever (inherited thread-pool state; orch-0005, 17.10).
"""
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.knee_labels import LABELS
from src.knee_volume import SeriesError, preprocess_series, read_series_dir
from src.reader25d import KneeNet, StudyDataset, cache_study_table, center_crop
from src.train25d import cache_depth_fractions, check_cfg, make_loader
from src.vol_cache import KneeVolCache, consolidate, is_study_done, write_study

# cfg keys that shape the data and the views: every model of one predict_studies call shares them
SHARED_KEYS = ("crop_px", "fov", "res", "mean", "std", "gap_mm", "axial_half_mm", "axial_max_extent_mm")
AMP_DTYPES = (torch.float16, torch.bfloat16, torch.float32)  # float32 = no autocast
# exp-0004's member settings (member contract: they live inside predict_member)
# bs 4: dtype_check (20261017t182108-948-e643) peaked at 4.3 GB for bs 2 in fp16, so bs 8 (~16 GB) would not fit a T4
MEMBER_K_EVAL, MEMBER_BS, MEMBER_WORKERS = 16, 4, 4
# build_dir_cache raises if no study finishes for this long (one study takes seconds): a hung pool dies loudly
POOL_RESULT_TIMEOUT_S = 600

_WORKER: dict = {}


def _init_worker(root):
    """Pool initializer: nice 19, one cv2 thread, the cache root."""
    os.nice(19 - os.nice(0))  # the shared machine's rule for CPU work
    cv2.setNumThreads(1)
    _WORKER.update(root=root)


def _build_task(task):
    """Pool task: preprocess every series dir of one study and write the study -> (study, n series errors)."""
    study, series_dirs = task
    results = {}
    for series, path in series_dirs:
        try:
            results[series] = preprocess_series(read_series_dir(path))
        except SeriesError as e:
            results[series] = e
    write_study(_WORKER["root"], study, results)
    return study, sum(isinstance(r, SeriesError) for r in results.values())


@contextlib.contextmanager
def _main_not_reimported():
    """Spawned children re-execute the parent's __main__ script (by __spec__ name or __file__) unless it
    has an `if __name__ == "__main__"` guard, which our scripts do not: hide both while the pool lives.
    The tasks only need src modules, which the children import from the parent's sys.path."""
    main = sys.modules["__main__"]
    saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if k in main.__dict__}
    main.__dict__.pop("__file__", None)
    main.__spec__ = None
    try:
        yield
    finally:
        main.__dict__.pop("__spec__", None)
        main.__dict__.update(saved)


def build_dir_cache(series_root, series_csv, out_root, workers):
    """KneeVolCache of every study of series_csv, preprocessed from <series_root>/<study>/<series>/ DICOM dirs.

    Same per-series path as the training cache (preprocess_series; a SeriesError becomes that series'
    error row), vol_cache.write_study per study in a spawn pool of `workers` processes, then
    consolidate. Studies already done under out_root are skipped; a missing series dir raises; no
    finished study for POOL_RESULT_TIMEOUT_S raises TimeoutError.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    series_root, out_root = Path(series_root), Path(out_root)
    csv = pd.read_csv(series_csv)
    if csv.SeriesInstanceUID.duplicated().any():
        raise ValueError(f"{series_csv}: duplicate SeriesInstanceUID")
    tasks = [(st, [(sr, series_root / st / sr) for sr in g.SeriesInstanceUID])
             for st, g in csv.groupby("StudyInstanceUID", sort=False) if not is_study_done(out_root, st)]
    n_studies = csv.StudyInstanceUID.nunique()
    print(f"[build_dir_cache] {n_studies} studies, {n_studies - len(tasks)} already done, {workers} workers", flush=True)
    if tasks:
        t0, n_err, step = time.time(), 0, max(1, len(tasks) // 20)
        with _main_not_reimported(), mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                                                  initargs=(out_root,)) as pool:
            it = pool.imap_unordered(_build_task, tasks, chunksize=1)
            for i in range(1, len(tasks) + 1):
                try:
                    _, err = it.next(timeout=POOL_RESULT_TIMEOUT_S)
                except mp.TimeoutError:
                    raise TimeoutError(f"build_dir_cache: no study finished in {POOL_RESULT_TIMEOUT_S}s "
                                       f"({i - 1}/{len(tasks)} done): the pool is hung") from None
                n_err += err
                if i % step == 0 or i == len(tasks):
                    print(f"[build_dir_cache] {i}/{len(tasks)} studies, {n_err} series errors, "
                          f"{time.time() - t0:.0f}s", flush=True)
    consolidate(out_root, series_csv)
    return KneeVolCache(out_root)


def _shared_cfg(models):
    """The SHARED_KEYS of the models' cfgs; raises if any model differs from the first in one of them."""
    if not models:
        raise ValueError("no models")

    def norm(v):
        return list(v) if isinstance(v, (list, tuple)) else v

    first = models[0][1]
    for i, (_, cfg) in enumerate(models[1:], 1):
        diff = {k: (first[k], cfg[k]) for k in SHARED_KEYS if norm(first[k]) != norm(cfg[k])}
        if diff:
            raise ValueError(f"model {i} differs from model 0 in shared cfg keys (model 0, model {i}): {diff}")
    return {k: first[k] for k in SHARED_KEYS}


def load_models(ckpt_paths, device):
    """C17: [(KneeNet, cfg)] from {"model", "cfg"} checkpoints: tagless backbone, pretrained=False, strict, eval.

    The backbone is built from cfg backbone without its pretrained tag, so no
    weights are fetched; all models must share the SHARED_KEYS of their cfgs.
    """
    ckpt_paths = [Path(p) for p in ckpt_paths]
    if not ckpt_paths:
        raise ValueError("no checkpoint paths")
    device = torch.device(device)
    models = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location="cpu", weights_only=True)
        if set(ck) != {"model", "cfg"}:
            raise ValueError(f"{p}: checkpoint keys {sorted(ck)} != ['cfg', 'model']")
        cfg = ck["cfg"]
        check_cfg(cfg)
        model = KneeNet(cfg["backbone"].split(".")[0], False, grad_ckpt=False)
        model.load_state_dict(ck["model"], strict=True)
        models.append((model.to(device).to(memory_format=torch.channels_last).eval(), cfg))
    _shared_cfg(models)
    return models


@torch.no_grad()
def predict_studies(studies, cache, comp_series, models, crop_fn, k_eval, bs, workers, device, amp_dtype):
    """C18: mean over models of sigmoid(logits) per study (index StudyInstanceUID in the given order, LABELS).

    The training eval path: cache_study_table, cache_depth_fractions and StudyDataset(train=False,
    k=k_eval) with the models' shared cfg; one pass over the loader, every model applied to each batch.
    comp_series: the competition *_series.csv of these studies; amp_dtype: torch.float16 or bfloat16
    (autocast), or torch.float32 (no autocast).
    Prints one PREDICT_TIMING JSON line per call (flushed; the probabilities do not depend on it): setup_s
    (table, depth fractions, dataset), data_wait_s (blocked in the loader: worker start, every next() incl.
    the last one and the workers' shutdown; first_batch_wait_s of it), compute_s (to device, the forwards,
    sigmoid, .cpu(); on cuda synchronized before each clock read, so the split is real), n_batches,
    n_studies, n_models, device, amp and on cuda torch's max allocated GB since its last peak reset.
    """
    t_call = time.perf_counter()
    studies = list(studies)
    if not studies:
        raise ValueError("no studies")
    if len(set(studies)) != len(studies):
        raise ValueError("studies contain duplicates")
    if amp_dtype not in AMP_DTYPES:
        raise ValueError(f"amp_dtype must be one of {AMP_DTYPES}, got {amp_dtype}")
    device = torch.device(device)
    cfg = _shared_cfg(models)
    table = cache_study_table(cache.series, comp_series, studies)
    depth_frac, _ = cache_depth_fractions(models[0][1], cache, studies)
    # eval draws nothing from the dataset rng (largest series, uniform centres) unless crop_fn does
    ds = StudyDataset(studies, table, cache.load, None, k_eval, False, crop_fn, cfg["gap_mm"], cfg["crop_px"],
                      models[0][1]["seed"], depth_frac)
    loader = make_loader(ds, {"bs": bs, "workers": workers}, False, device, persistent=False)
    views = dict(res=cfg["res"], fov=cfg["fov"], mean=cfg["mean"], std=cfg["std"])
    for model, _ in models:
        model.eval()
    out, waits, compute_s = [], [], 0.0
    t = t_loop = time.perf_counter()
    for x, mask, pos, _ in loader:
        t_batch = time.perf_counter()
        waits.append(t_batch - t)
        x, mask, pos = x.to(device, non_blocking=True), mask.to(device), pos.to(device)
        prob = torch.zeros(len(x), len(LABELS), device=device)
        for model, _ in models:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logit = model(x, mask, pos, train=False, **views)
            prob += torch.sigmoid(logit.float())
        out.append((prob / len(models)).cpu())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        compute_s += t - t_batch
    t_end = time.perf_counter()
    waits.append(t_end - t)  # the last next(): StopIteration and the workers' shutdown
    timing = {"n_studies": len(studies), "n_batches": len(out), "n_models": len(models), "device": str(device),
              "amp": str(amp_dtype).removeprefix("torch."), "bs": bs, "workers": workers, "k_eval": k_eval,
              "setup_s": round(t_loop - t_call, 3), "data_wait_s": round(sum(waits), 3),
              "first_batch_wait_s": round(waits[0], 3), "compute_s": round(compute_s, 3),
              "total_s": round(t_end - t_call, 3),
              "cuda_max_allocated_gb": (round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
                                        if device.type == "cuda" else None)}
    print("PREDICT_TIMING " + json.dumps(timing), flush=True)
    p = torch.cat(out).numpy()
    if not np.isfinite(p).all():
        raise FloatingPointError(f"non-finite probabilities for {int((~np.isfinite(p)).any(1).sum())} studies "
                                 f"(amp_dtype {amp_dtype})")
    return pd.DataFrame(p, index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)


def predict_member(studies, cache, comp_series, ckpt_paths, device):
    """exp-0004's member entry point: C17 + C18 with center_crop(cfg crop_px), k_eval 16, bs 4, 4 workers.

    amp: float16 on cuda (T4 has no bf16; fp16 vs bf16 OOF macro AUC +0.0004, dtype_check 20261017t184108-754-3332),
    bfloat16 on cpu. Returns probabilities (index
    StudyInstanceUID in the given order, columns LABELS).
    """
    device = torch.device(device)
    models = load_models(ckpt_paths, device)
    amp = torch.float16 if device.type == "cuda" else torch.bfloat16
    return predict_studies(studies, cache, comp_series, models, center_crop(models[0][1]["crop_px"]), MEMBER_K_EVAL,
                           MEMBER_BS, MEMBER_WORKERS, device, amp)
