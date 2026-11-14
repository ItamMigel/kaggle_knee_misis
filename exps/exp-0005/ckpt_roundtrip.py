"""Checkpoint roundtrip of a train.py run: fold{k}.pt reloaded (pretrained=False, strict) reproduces its OOF logits.

    CUDA_VISIBLE_DEVICES=1 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/ckpt_roundtrip.py

Recomputes the logits of the first N_STUDIES validation studies of FOLD with the same eval path
(bf16, as train_fold) and compares them with oof_fold{FOLD}.csv; also reports fp16 vs bf16 (T4
inference runs fp16). Writes roundtrip.json to ctx.output_dir.
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

from src.joint_reader import CROP_MM_BY_PLANE, LOCATE, MM_PER_PX, OUT_PX  # noqa: E402
from src.joint_view import JointCache, load_roi_run, make_joint_crop_fn  # noqa: E402
from src.knee_labels import LABELS  # noqa: E402
from src.reader25d import StudyDataset, cache_study_table  # noqa: E402
from src.train25d import build_model, cache_depth_fractions, make_loader, predict_logits  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

TRAIN_RUN = "20261017t125202-486-6244"  # GPU smoke, fold 0
FOLD = 0
N_STUDIES = 8
ROI_RUN = "20261017t124739-029-511f"
CACHE = "exp-0002/knee-vol04"
FOLDS_RUN = "20261017t111717-941-7125"
TOL = 0.05  # max |Δlogit| allowed between the reloaded checkpoint and the run's OOF logits (bf16 eval)


def main():
    if TRAIN_RUN is None:
        raise ValueError("set TRAIN_RUN to the train.py run to check")
    ctx = get_ctx()
    run = ctx.mounts[TRAIN_RUN]
    ckpt = torch.load(run / f"fold{FOLD}.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device("cuda")
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    oof = pd.read_csv(run / f"oof_fold{FOLD}.csv", index_col=0)
    ids = list(oof.index[oof["set"] == "val"][:N_STUDIES])
    rois, sides, deg = load_roi_run(ctx.mounts[ROI_RUN], cache_key=CACHE, crop_mm_by_plane=CROP_MM_BY_PLANE,
                                    mm_per_px=MM_PER_PX, out_px=OUT_PX, locate=LOCATE)
    joint = JointCache(KneeVolCache(ctx.mounts[CACHE]), sides, deg)
    comp_series = pd.read_csv(ctx.mounts["rsna-knee-abnormality-detection"] / "train_series.csv")
    folds_df = pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv")
    table = cache_study_table(joint.series, comp_series, folds_df.StudyInstanceUID)
    depth_frac, _ = cache_depth_fractions(cfg, joint, ids)
    crop_fn = make_joint_crop_fn(rois, sides, OUT_PX, CROP_MM_BY_PLANE, MM_PER_PX, jitter_px=0)
    ds = StudyDataset(ids, table, joint.load, None, cfg["k_eval"], False, crop_fn, cfg["gap_mm"], cfg["crop_px"],
                      cfg["seed"], depth_frac)
    loader = make_loader(ds, cfg, False, device, persistent=False)

    got = predict_logits(model, loader, cfg, device, torch.bfloat16)  # as train_fold scores its OOF
    ref = oof.loc[ids, LABELS].to_numpy(np.float32)
    d_bf16 = float(np.abs(got - ref).max())
    model.eval()
    fp16 = []
    with torch.no_grad():
        for x, mask, pos, _ in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(x.to(device), mask.to(device), pos.to(device), res=cfg["res"], train=False,
                           fov=cfg["fov"], mean=cfg["mean"], std=cfg["std"])
            fp16.append(lo.float().cpu())
    d_fp16 = float(np.abs(torch.cat(fp16).numpy() - got).max())
    res = {"train_run": TRAIN_RUN, "fold": FOLD, "n_studies": len(ids), "max_abs_dlogit_vs_oof_bf16": d_bf16,
           "max_abs_dlogit_fp16_vs_bf16": d_fp16, "tol": TOL, "pass": d_bf16 <= TOL}
    (ctx.output_dir / "roundtrip.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    if d_bf16 > TOL:
        raise AssertionError(f"reloaded fold{FOLD}.pt differs from the run's OOF logits by {d_bf16:.4f} > {TOL}")


if __name__ == "__main__":
    main()
