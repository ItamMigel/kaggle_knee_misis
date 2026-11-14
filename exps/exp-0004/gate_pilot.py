"""Data gate of exp-0004 on the exp-0002 pilot cache: loader shapes and speed, window montage,
overfit one batch, checkpoint roundtrip.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/gate_pilot.py

Writes gate.json and montage_<study>.png (rows = 6 slots, columns = windows; a window's three
slices are the B, G, R channels, so misaligned neighbours show as colour fringes) to ctx.output_dir.
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
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.knee_labels import load_soft_targets  # noqa: E402
from src.reader25d import N_SLOTS, PLANES, StudyDataset, cache_study_table, center_crop, depth_source_counts  # noqa: E402
from src.train25d import build_model, cache_depth_fractions, check_cfg, make_loader  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SEED = 0
CACHE = "exp-0002/knee-vol04-pilot"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
MEM_FRACTION = 0.75  # the card is shared with a neighbour (lead, 17.10)
OVERFIT_STEPS = 200
N_MONTAGE = 2
TILE = 192
CFG = dict(backbone="convnext_tiny.dinov3_lvd1689m", pretrained=True, res=352, crop_px=384, fov=0.92,
           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), gap_mm=3.0, k=8, k_eval=16,
           epochs=1, bs=8, lr=1e-4, lr_head=5e-4, wd=0.02, ema=0.998, grad_ckpt=True, eval_every=1,
           workers=7, seed=SEED, limit_train=0, max_steps_per_epoch=0, axial_half_mm=60, axial_max_extent_mm=150)
SLOT_NAMES = [f"{p}{'' if i < 3 else ' noFS'}" for i, p in enumerate(PLANES * 2)]

check_cfg(CFG)
torch.manual_seed(SEED)
ctx = get_ctx()
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
views = dict(res=CFG["res"], fov=CFG["fov"], mean=CFG["mean"], std=CFG["std"])
report = {"cfg": CFG}

comp = ctx.mounts["rsna-knee-abnormality-detection"]
train = pd.read_csv(comp / "train.csv")
cache = KneeVolCache(ctx.mounts[CACHE])
studies = sorted(set(cache.series.study))
if not set(studies) <= set(train.StudyInstanceUID):
    raise ValueError("the pilot cache holds studies that are not in train.csv")
table = cache_study_table(cache.series, pd.read_csv(comp / "train_series.csv"), studies)
targets = load_soft_targets(ctx.mounts[S4[0]] / S4[1], studies)
fill = np.array([[len(c) > 0 for c in table[s]] for s in studies])
report["data"] = {
    "n_studies": len(studies), "n_series": len(cache.series),
    "n_series_failed": int(cache.series.error_code.notna().sum()),
    "error_codes": cache.series.error_code.value_counts().to_dict(),
    "n_studies_without_series": int((~fill.any(1)).sum()),
    "slot_fill": dict(zip(SLOT_NAMES, fill.mean(0).round(3).tolist())),
}
print(json.dumps(report["data"]), flush=True)

# lookups built once in the main process, so the forked loader workers share them
cache.load(cache.series.series[cache.series.error_code.isna()].iloc[0])
crop_fn = center_crop(CFG["crop_px"])
depth_frac, depth_report = cache_depth_fractions(CFG, cache, studies)
depth_report.to_csv(ctx.output_dir / "depth_report.csv", index=False)
report["depth"] = {"restricted": len(depth_report), "sources": depth_source_counts(depth_report)}
print(json.dumps(report["depth"]), flush=True)

# 1. loader: shapes and speed of a training pass (cold page cache, then warm)
ds = StudyDataset(studies, table, cache.load, targets.values, CFG["k"], True, crop_fn, CFG["gap_mm"],
                  CFG["crop_px"], SEED, depth_frac)
report["loader"] = []
for attempt in ("cold", "warm"):
    t, n = time.time(), 0
    for x, mask, pos, y in make_loader(ds, CFG, True, dev, persistent=False):
        assert x.dtype == torch.uint8 and tuple(x.shape[1:]) == (N_SLOTS, CFG["k"], 3, CFG["crop_px"], CFG["crop_px"])
        assert mask.dtype == torch.bool and tuple(mask.shape[1:]) == (N_SLOTS,)
        assert tuple(pos.shape[1:]) == (N_SLOTS, CFG["k"]) and float(pos.min()) >= 0 and float(pos.max()) <= 1
        assert tuple(y.shape[1:]) == (12,) and torch.isfinite(y).all()
        n += len(x)
    sec = time.time() - t
    report["loader"].append({"pass": attempt, "studies": n, "seconds": sec, "studies_per_s": n / sec})
    print(json.dumps(report["loader"][-1]), flush=True)

# 2. montage of the eval windows of a few studies
eval_ds = StudyDataset(studies, table, cache.load, targets.values, CFG["k"], False, crop_fn, CFG["gap_mm"],
                       CFG["crop_px"], SEED, depth_frac)
report["montage"] = []
for i in range(N_MONTAGE):
    x, mask, pos, _ = eval_ds[i]
    rows = []
    for s in range(N_SLOTS):
        tiles = []
        for j in range(CFG["k"]):
            win = np.ascontiguousarray(x[s, j].permute(1, 2, 0).numpy())
            tile = cv2.resize(win, (TILE, TILE), interpolation=cv2.INTER_AREA)
            if not mask[s]:
                tile[:] = 64
            elif j == 0:
                cv2.putText(tile, SLOT_NAMES[s], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, 1))
    path = ctx.output_dir / f"montage_{studies[i]}.png"
    cv2.imwrite(str(path), np.concatenate(rows, 0))
    report["montage"].append(str(path))

# 3. overfit one batch: 8 fixed studies, binarised S4 targets (soft targets cannot reach loss 0), no augmentation
xb, mb, pb, yb = (t.to(dev) for t in torch.utils.data.default_collate([eval_ds[i] for i in range(CFG["bs"])]))
yb = (yb > 0.5).float()
# reference: BCE of the best constant prediction (per-label mean of the batch)
const_bce = float(F.binary_cross_entropy(yb.mean(0, keepdim=True).clamp(1e-4, 1 - 1e-4).expand_as(yb), yb))
torch.cuda.reset_peak_memory_stats(dev)
model = build_model(CFG, CFG["pretrained"]).to(dev).train()
enc = list(model.enc.parameters())
head = [p for n_, p in model.named_parameters() if not n_.startswith("enc.")]
opt = torch.optim.AdamW([{"params": enc, "lr": CFG["lr"]}, {"params": head, "lr": CFG["lr_head"]}], weight_decay=CFG["wd"])
losses = []
for step in range(OVERFIT_STEPS):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = model(xb, mb, pb, train=False, **views)
    loss = F.binary_cross_entropy_with_logits(logit.float(), yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    opt.step()
    losses.append(loss.item())
report["overfit"] = {"steps": OVERFIT_STEPS, "const_bce": const_bce, "loss_first": losses[0], "loss_last": losses[-1],
                     "loss_every_20": losses[::20],
                     "peak_alloc_gb": torch.cuda.max_memory_allocated(dev) / 2**30,
                     "peak_reserved_gb": torch.cuda.max_memory_reserved(dev) / 2**30}
print(json.dumps(report["overfit"]), flush=True)

# 4. checkpoint roundtrip: the C14 checkpoint format reloaded into a fresh model gives the same logits
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    before = model(xb, mb, pb, train=False, **views).float()
ckpt = ctx.output_dir / "gate_ckpt.pt"
torch.save({"model": model.state_dict(), "cfg": CFG}, ckpt)
loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
fresh = build_model(loaded["cfg"], False).to(dev)
fresh.load_state_dict(loaded["model"])
fresh.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    after = fresh(xb, mb, pb, train=False, **views).float()
ckpt.unlink()
report["roundtrip"] = {"max_abs_logit_diff": float((before - after).abs().max())}
print(json.dumps(report["roundtrip"]), flush=True)

(ctx.output_dir / "gate.json").write_text(json.dumps(report, indent=1))
print(f"[gate_pilot] {ctx.output_dir}")
