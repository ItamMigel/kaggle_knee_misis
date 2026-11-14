"""Synthetic CPU smoke of C4 (train.py) inputs and model; no GPU, no training, no weight download.

C3 run on smoke_joint_view's fake cache -> load_roi_run -> JointCache -> exp-0004's
cache_study_table (degenerate series dropped) + cache_depth_fractions -> StudyDataset / make_loader
with our crop_fn and JointCache.load, i.e. what run_folds does before train_fold; then train.py's
CFG -> build_model (pretrained=False) -> KneeNet forward+backward on one study (2 slots, k=2,
300 px) with grad checkpointing on and off: same loss and gradients, blocks recomputed in backward.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/smoke_train_inputs.py
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import locate_all  # noqa: E402
import smoke_joint_view as sj  # noqa: E402
import train as c4  # noqa: E402
from src.joint_reader import CROP_MM_BY_PLANE, LOCATE, MM_PER_PX, OUT_PX  # noqa: E402
from src.joint_view import JointCache, check_table_rois, load_roi_run, make_joint_crop_fn  # noqa: E402
from src.reader25d import DEPTH_HI, DEPTH_LO, N_SLOTS, StudyDataset, cache_study_table  # noqa: E402
from src.knee_labels import LABELS, gold_labels, load_soft_targets  # noqa: E402
from src.train25d import build_model, cache_depth_fractions, check_cfg, collect_oof, make_loader  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SEED = 0
CPU_THREADS = 8  # shared machine


def expect_raise(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def check_inputs(out):
    """train.py's input path on the fake cache, then StudyDataset items and one DataLoader batch."""
    root, c3 = out / "fake_cache", out / "c3"
    sj.build_fake_cache(root)
    c3.mkdir()
    locate_all.run(root, c3, 2, sj.FakeCache, sj.canon_stub)
    crop_px = c4.CFG["crop_px"]
    assert crop_px == OUT_PX
    spec = dict(crop_mm_by_plane=CROP_MM_BY_PLANE, mm_per_px=MM_PER_PX, out_px=OUT_PX, locate=LOCATE)
    assert locate_all.CACHE_KEY == c4.CACHE
    expect_raise(ValueError, load_roi_run, c3, cache_key="exp-0002/knee-vol04-pilot", **spec)  # ROIs of another cache
    for bad in ({"crop_mm_by_plane": {**CROP_MM_BY_PLANE, "Sagittal": 120.0}}, {"out_px": 256}, {"mm_per_px": 0.5},
                {"locate": {**LOCATE, "bilateral_min_mm": 50.0}}, {"locate": {**LOCATE, "open_px": 3}}):
        expect_raise(ValueError, load_roi_run, c3, cache_key=locate_all.CACHE_KEY, **{**spec, **bad})
    rois, sides, degenerate = load_roi_run(c3, cache_key=locate_all.CACHE_KEY, **spec)
    assert degenerate == {"S5-cor", "S5-sag"} and len(rois) == 11, (degenerate, len(rois))

    cache = sj.FakeCache(root)
    joint = JointCache(cache, sides, degenerate)
    cs = cache.series  # the competition CSV has exactly the cache's series (cache_study_table checks it)
    comp_series = pd.DataFrame({"StudyInstanceUID": cs.study, "SeriesInstanceUID": cs.series,
                                "Fat_Suppression": cs.Fat_Suppression, "Anatomical_Plane": cs.Anatomical_Plane})
    studies = sorted(set(cs.study))
    kept = lambda t: {s for slots in t.values() for c in slots for s in c}  # noqa: E731
    raw = cache_study_table(cache.series, comp_series, studies)
    table = cache_study_table(joint.series, comp_series, studies)
    assert {"S5-cor", "S5-sag"} <= kept(raw) and kept(table) == set(rois), (kept(raw), kept(table))
    assert not any(table["S5"]) and sum(map(len, table["S1"])) == 3, table  # S1-failed and S5-* dropped
    check_table_rois(table, rois)
    expect_raise(KeyError, check_table_rois, table, {s: r for s, r in rois.items() if s != "S2-cor"})

    cfg = {**c4.CFG, "pretrained": False}
    check_cfg(cfg)
    depth_frac, depth_report = cache_depth_fractions(cfg, joint, studies)
    assert set(depth_frac) == set(rois) and set(depth_frac.values()) == {(DEPTH_LO, DEPTH_HI)}, depth_frac
    assert len(depth_report) == 0  # the fake axial stacks (21 mm) are below axial_max_extent_mm
    joint.load(joint.series.series[joint.series.error_code.isna()].iloc[0])  # run_folds' warm-up call

    crop_fn = make_joint_crop_fn(rois, sides, OUT_PX, CROP_MM_BY_PLANE, MM_PER_PX, c4.JITTER_PX)
    y = np.random.default_rng(SEED).random((len(studies), 12)).astype(np.float32)
    for train, k in ((True, 2), (False, 3)):
        ds = StudyDataset(studies, table, joint.load, y, k, train, crop_fn, c4.CFG["gap_mm"], crop_px, SEED,
                          depth_frac)
        for i, st in enumerate(studies):
            x, mask, pos, yy = ds[i]
            assert x.dtype == torch.uint8 and tuple(x.shape) == (N_SLOTS, k, 3, crop_px, crop_px), x.shape
            assert mask.tolist() == [bool(c) for c in table[st]], (st, mask)
            assert bool((x[~mask] == 0).all()) and (not mask.any() or float(x[mask].float().mean()) > 0), st
    cfg = {**c4.CFG, "workers": 2, "bs": 2}
    ds = StudyDataset(studies, table, joint.load, y, 2, True, crop_fn, cfg["gap_mm"], crop_px, SEED, depth_frac)
    x, mask, pos, yy = next(iter(make_loader(ds, cfg, True, torch.device("cpu"), persistent=False)))
    assert tuple(x.shape) == (2, N_SLOTS, 2, 3, crop_px, crop_px) and tuple(yy.shape) == (2, 12)
    return {"located": len(rois), "degenerate": sorted(degenerate), "studies": len(studies),
            "empty_studies": [s for s in studies if not any(table[s])], "depth_restricted": len(depth_report)}


def check_model():
    """KneeNet from train.py's CFG: pretrained_cfg, CPU fwd+bwd, grad checkpointing active and exact."""
    cfg = {**c4.CFG, "pretrained": False}
    check_cfg(cfg)
    torch.manual_seed(SEED)
    model = build_model(cfg, False)
    pc = model.enc.pretrained_cfg
    assert tuple(pc["mean"]) == cfg["mean"] and tuple(pc["std"]) == cfg["std"] and pc["input_size"][1] == cfg["res"], pc
    assert model.enc.grad_checkpointing is cfg["grad_ckpt"]  # train.py's switch reaches timm (v1: off)
    state = {k: v.clone() for k, v in model.state_dict().items()}

    gen = torch.Generator().manual_seed(SEED)
    x = torch.randint(0, 256, (1, N_SLOTS, 2, 3, cfg["crop_px"], cfg["crop_px"]), dtype=torch.uint8, generator=gen)
    mask = torch.zeros(1, N_SLOTS, dtype=torch.bool)
    mask[0, :2] = True
    pos = torch.rand(1, N_SLOTS, 2, generator=gen)
    y = torch.rand(1, 12, generator=gen)
    views = dict(res=cfg["res"], fov=cfg["fov"], mean=cfg["mean"], std=cfg["std"])

    def step(ckpt):
        m = build_model({**cfg, "grad_ckpt": ckpt}, False)
        m.load_state_dict(state)
        m.train()
        block = m.enc.blocks[3][0]
        bn = next(mod for mod in m.enc.blocks.modules() if isinstance(mod, torch.nn.BatchNorm2d))
        calls = []  # pre-hook: torch's non-reentrant checkpoint stops the recompute early, so a post-hook would not fire
        block.register_forward_pre_hook(lambda *_: calls.append(1))
        bn_before = int(bn.num_batches_tracked)
        torch.manual_seed(SEED)
        loss = F.binary_cross_entropy_with_logits(m(x, mask, pos, train=True, **views).float(), y)
        n_fwd = len(calls)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        return dict(loss=loss.item(), n_fwd=n_fwd, n_total=len(calls), bn_updates=int(bn.num_batches_tracked) - bn_before,
                    grads=grads, n_params=sum(1 for p in m.parameters() if p.requires_grad),
                    flag=m.enc.grad_checkpointing)

    on, off = step(True), step(False)
    assert on["flag"] and not off["flag"]
    assert on["n_fwd"] == 1 and on["n_total"] == 2, on["n_total"]  # recomputed in backward
    assert off["n_fwd"] == 1 and off["n_total"] == 1, off["n_total"]
    assert np.isfinite(on["loss"]) and abs(on["loss"] - off["loss"]) < 1e-6, (on["loss"], off["loss"])
    assert on["grads"].keys() == off["grads"].keys() and len(on["grads"]) == on["n_params"], \
        (len(on["grads"]), on["n_params"])
    max_diff = max(float((on["grads"][n] - off["grads"][n]).abs().max()) for n in on["grads"])
    assert max_diff < 1e-5, max_diff
    assert float(on["grads"]["enc.conv_stem.weight"].abs().sum()) > 0
    return {"loss": on["loss"], "grad_max_abs_diff_ckpt_vs_not": max_diff, "n_param_tensors_with_grad": len(on["grads"]),
            "block3_forward_calls": {"ckpt": on["n_total"], "no_ckpt": off["n_total"]},
            "bn_num_batches_tracked_per_step": {"ckpt": on["bn_updates"], "no_ckpt": off["bn_updates"]},
            "pretrained_cfg": {k: pc[k] for k in ("input_size", "test_input_size", "mean", "std")}}


def check_one_fold_scores(ctx):
    """What run_folds collects after SMOKE_FOLDS only: collect_oof on fold 0 + gold (random logits, real ids/labels)."""
    comp = ctx.mounts[c4.COMP]
    train = pd.read_csv(comp / "train.csv")
    folds_df = pd.read_csv(ctx.mounts[c4.FOLDS_RUN] / "folds.csv")
    targets = load_soft_targets(ctx.mounts[c4.S4[0]] / c4.S4[1], train.StudyInstanceUID)
    is_gold = folds_df.is_gold.astype(bool)
    trained = folds_df[folds_df.fold.isin(c4.SMOKE_FOLDS) | is_gold]  # run_folds' own selection
    ids = list(folds_df.StudyInstanceUID[folds_df.fold == 0]) + list(folds_df.StudyInstanceUID[is_gold])
    rng = np.random.default_rng(SEED)
    oof = pd.DataFrame(rng.normal(size=(len(ids), len(LABELS))), index=pd.Index(ids, name="StudyInstanceUID"),
                       columns=LABELS)
    oof.insert(0, "set", ["val"] * int((folds_df.fold == 0).sum()) + ["gold"] * int(is_gold.sum()))
    out = ctx.output_dir / "one_fold"
    out.mkdir()
    oof.to_csv(out / "oof_fold0.csv")
    _, scores = collect_oof(out, trained, targets, gold_labels(train), c4.N_BOOT, SEED)
    assert scores["n_folds"] == 1 and list(scores["weak_per_fold"]) == ["0"], scores["weak_per_fold"]
    assert scores["n_val"] == int((folds_df.fold == 0).sum()) and scores["n_gold"] == int(is_gold.sum()), scores
    return {k: scores[k] for k in ("n_folds", "n_val", "n_gold")} | {"files": sorted(p.name for p in out.iterdir())}


def main():
    ctx = get_ctx()
    torch.set_num_threads(CPU_THREADS)
    res = {"inputs": check_inputs(ctx.output_dir), "one_fold_scores": check_one_fold_scores(ctx), "model": check_model()}
    (ctx.output_dir / "smoke_train.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print(f"[smoke] C4 input and model checks passed | {ctx.output_dir}")


if __name__ == "__main__":
    main()
