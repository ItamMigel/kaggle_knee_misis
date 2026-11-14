"""Resume an interrupted train.py 5-fold run in a new run dir, with grad_ckpt on (the lead's OOM fallback, 17.10).

    CUDA_VISIBLE_DEVICES=1 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/resume_folds.py

train_fold resumes only from a state file in its own out_dir and refuses a state written with
another cfg; every launch gets a new run dir. So this script copies the RESUME_RUN artifacts into
ctx.output_dir: finished folds (fold{k}.pt, oof_fold{k}.csv, metrics_fold{k}.jsonl) as they are,
the interrupted fold's state_fold{k}.pt + metrics with the stored cfg's grad_ckpt set to True (the
only cfg change; grad_ckpt leaves the gradients bit-identical but BatchNorm running stats are
updated twice per step, i.e. effective momentum ~0.19 instead of 0.1, from the resume on). Then it
trains the remaining folds with train.py's setup and grad_ckpt on, and collects the 5-fold OOF.
inputs.json records what was copied and patched. DRY_RUN = True copies and checks without a GPU.
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

import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (train.py: constants and the training setup)
from src.joint_view import JointCache, load_roi_run, make_joint_crop_fn  # noqa: E402
from src.knee_labels import gold_labels, load_soft_targets  # noqa: E402
from src.train25d import check_cfg, collect_oof, run_folds  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

RESUME_RUN = "20261017t134217-566-62c9"  # OOM in fold 0 epoch 0 (bs 6, grad_ckpt off): no state yet, fold 0 restarts
DRY_RUN = False


def plan_resume(src, dst, folds, cfg_after):
    """Copies finished folds and the interrupted fold's state from src to dst; returns (remaining folds, report)."""
    finished = [k for k in folds if (src / f"fold{k}.pt").exists()]
    remaining = [k for k in folds if k not in finished]
    report = {"finished_copied": finished, "remaining": remaining, "state_patched": None}
    for k in finished:
        for name in (f"fold{k}.pt", f"oof_fold{k}.csv", f"metrics_fold{k}.jsonl"):
            shutil.copy2(src / name, dst / name)
    if remaining:
        k = remaining[0]
        state = src / f"state_fold{k}.pt"
        if state.exists():
            st = torch.load(state, map_location="cpu", weights_only=False)
            before = dict(st["cfg"])
            st["cfg"] = {**before, "grad_ckpt": True}
            if st["cfg"] != cfg_after:
                diff = {key: (st["cfg"].get(key), cfg_after.get(key)) for key in set(st["cfg"]) | set(cfg_after)
                        if st["cfg"].get(key) != cfg_after.get(key)}
                raise ValueError(f"{state}: cfg differs from the resume cfg beyond grad_ckpt: {diff}")
            torch.save(st, dst / f"state_fold{k}.pt")
            shutil.copy2(src / f"metrics_fold{k}.jsonl", dst / f"metrics_fold{k}.jsonl")
            report["state_patched"] = {"fold": k, "after_epoch": int(st["epoch"]),
                                       "grad_ckpt": [before["grad_ckpt"], True]}
        # no state file: fold k restarts from scratch in dst (its partial metrics are not copied)
    return remaining, report


def main():
    if RESUME_RUN is None:
        raise ValueError("set RESUME_RUN to the interrupted train.py run")
    ctx = get_ctx()
    src = ctx.mounts[RESUME_RUN]
    rec = json.loads((src / "record.json").read_text())
    if rec["script"] != "train.py":
        raise ValueError(f"{RESUME_RUN} is a {rec['script']} run, not train.py")
    src_inputs = json.loads((src / "inputs.json").read_text())
    cfg = {**T.CFG, "grad_ckpt": True}
    check_cfg(cfg)
    if {**src_inputs["cfg"], "grad_ckpt": True} != json.loads(json.dumps(cfg)):
        raise ValueError(f"{RESUME_RUN} was trained with another cfg than train.py's CFG (beyond grad_ckpt)")
    if src_inputs["roi_run"] != T.ROI_RUN or src_inputs["smoke"]:
        raise ValueError(f"{RESUME_RUN}: roi_run {src_inputs['roi_run']} / smoke {src_inputs['smoke']} do not match")

    remaining, report = plan_resume(src, ctx.output_dir, T.FOLDS, cfg)
    inputs = {"resumed_from": RESUME_RUN, "cfg": cfg, "roi_run": T.ROI_RUN, "folds_run": T.FOLDS_RUN,
              "dry_run": DRY_RUN, **report}
    (ctx.output_dir / "inputs.json").write_text(json.dumps(inputs, indent=1))
    print(json.dumps(inputs), flush=True)
    if DRY_RUN:
        return

    torch.manual_seed(T.SEED)
    torch.cuda.set_per_process_memory_fraction(T.MEM_FRACTION, 0)
    spec = dict(crop_mm_by_plane=T.CROP_MM_BY_PLANE, mm_per_px=T.MM_PER_PX)
    rois, sides, deg = load_roi_run(ctx.mounts[T.ROI_RUN], cache_key=T.CACHE, out_px=cfg["crop_px"], locate=T.LOCATE,
                                    **spec)
    joint = JointCache(KneeVolCache(ctx.mounts[T.CACHE]), sides, deg)
    crop_fn = make_joint_crop_fn(rois, sides, cfg["crop_px"], jitter_px=T.JITTER_PX, **spec)
    comp = ctx.mounts["rsna-knee-abnormality-detection"]
    train = pd.read_csv(comp / "train.csv")
    comp_series = pd.read_csv(comp / "train_series.csv")
    folds_df = pd.read_csv(ctx.mounts[T.FOLDS_RUN] / "folds.csv")
    targets = load_soft_targets(ctx.mounts[T.S4[0]] / T.S4[1], train.StudyInstanceUID)
    gold = gold_labels(train)
    if remaining:
        run_folds(cfg, remaining, joint, comp_series, folds_df, targets, gold, crop_fn, ctx.output_dir, "cuda",
                  T.N_BOOT, T.SEED)
    oof, scores = collect_oof(ctx.output_dir, folds_df, targets, gold, T.N_BOOT, T.SEED)
    print(json.dumps({"weak_oof": scores["weak_oof"]["macro"], "weak_per_fold": scores["weak_per_fold"],
                      "gold": scores["gold"]["macro"], "gold_ci": scores["gold"]["bootstrap"]}), flush=True)
    print(f"[resume] {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
