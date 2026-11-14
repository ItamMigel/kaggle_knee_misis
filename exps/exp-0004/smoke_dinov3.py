"""Smoke of the v1 reader, pretraining arm `dinov3`: fold 0, 2000 training studies, 5 epochs, eval every epoch.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/smoke_dinov3.py

Same pipeline and settings as train.py except the smoke constants; the other arm is smoke_in22k.py.
Decision rule (D1): dinov3 stays unless its final val weak macro AUC is > 0.01 below in22k.
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

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.knee_labels import gold_labels, load_soft_targets  # noqa: E402
from src.reader25d import center_crop  # noqa: E402
from src.train25d import run_folds  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SEED = 0
CACHE = "exp-0002/knee-vol04"
FOLDS_RUN = "20261017t111717-941-7125"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
FOLDS = [0]
MEM_FRACTION = 0.65  # ~30 GB of the 46 GB L40: the card is shared (lead, 17.10)
N_BOOT = 2000
CFG = dict(backbone="convnext_tiny.dinov3_lvd1689m", pretrained=True, res=352, crop_px=384, fov=0.92,
           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), gap_mm=3.0, k=8, k_eval=16,
           epochs=5, bs=8, lr=1e-4, lr_head=5e-4, wd=0.02, ema=0.998, grad_ckpt=True, eval_every=1,
           workers=7, seed=SEED, limit_train=2000, max_steps_per_epoch=0, axial_half_mm=60, axial_max_extent_mm=150)

torch.manual_seed(SEED)
ctx = get_ctx()
torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
comp = ctx.mounts["rsna-knee-abnormality-detection"]
train = pd.read_csv(comp / "train.csv")
folds_df = pd.read_csv(ctx.mounts[FOLDS_RUN] / "folds.csv")
targets = load_soft_targets(ctx.mounts[S4[0]] / S4[1], train.StudyInstanceUID)
oof, scores = run_folds(CFG, FOLDS, KneeVolCache(ctx.mounts[CACHE]), pd.read_csv(comp / "train_series.csv"),
                        folds_df, targets, gold_labels(train), center_crop(CFG["crop_px"]), ctx.output_dir,
                        "cuda", N_BOOT, SEED)
print(json.dumps({"weak_oof": scores["weak_oof"]["macro"], "weak_per_fold": scores["weak_per_fold"],
                  "gold": scores["gold"]["macro"], "gold_ci": scores["gold"]["bootstrap"]}))
print(f"[smoke_dinov3] {ctx.output_dir}")
