"""GPU cost of the v1 reader on synthetic batches: train-step speed and peak memory per (bs, k, grad_ckpt).

    CUDA_VISIBLE_DEVICES=0 ~/conda/envs/knee/bin/python exps/exp-0004/bench.py

No data loader and no cache: random uint8 windows of the training shape, so the numbers are the
GPU side only. Writes bench.json to ctx.output_dir.
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.reader25d import N_SLOTS  # noqa: E402
from src.train25d import build_model  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SEED = 0
BACKBONE = "convnext_tiny.dinov3_lvd1689m"
RES, CROP, FOV = 352, 384, 0.92
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SLOTS_PRESENT = 5  # typical study; the worst case (6) is measured for the chosen setting below
# (bs, k, grad_ckpt)
TRAIN_SETTINGS = [(2, 12, False), (4, 12, False), (4, 12, True), (8, 12, True), (8, 8, False), (8, 8, True)]
WORST_CASE = [(4, 12, True), (8, 12, True)]
EVAL_SETTINGS = [(4, 16), (8, 16)]  # (bs, k_eval), no grad
WARMUP, STEPS = 3, 8
MEM_FRACTION = 0.75  # ~34.5 GB of the 46 GB L40: the card is shared with a neighbour (lead, 17.10)

torch.manual_seed(SEED)
ctx = get_ctx()
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
views =dict(res=RES, fov=FOV, mean=MEAN, std=STD)


def batch(bs, k, present):
    """Random uint8 windows [bs, 6, k, 3, CROP, CROP] with `present` slots per study."""
    x = torch.randint(0, 256, (bs, N_SLOTS, k, 3, CROP, CROP), dtype=torch.uint8).pin_memory()
    mask = torch.zeros(bs, N_SLOTS, dtype=torch.bool)
    mask[:, :present] = True
    pos = torch.rand(bs, N_SLOTS, k)
    return x, mask, pos


def train_cost(bs, k, ckpt, present):
    """Seconds per train step (fwd bf16 + bwd + AdamW) and peak GiB allocated/reserved; 'oom' on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        model = build_model({"backbone": BACKBONE, "grad_ckpt": ckpt}, False).to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x, mask, pos = batch(bs, k, present)
        y = torch.rand(bs, 12, device=dev)
        times = []
        for i in range(WARMUP + STEPS):
            torch.cuda.synchronize()
            t = time.time()
            xd, md, pd_ = x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xd, md, pd_, train=True, **views)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append(time.time() - t)
        sec = sum(times) / len(times)
        out = {"sec_per_step": sec, "studies_per_s": bs / sec, "images_per_s": bs * present * k / sec}
    except torch.cuda.OutOfMemoryError:
        out = {"oom": True}
    out.update(peak_alloc_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
               peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30)
    return out


@torch.no_grad()
def eval_cost(bs, k):
    """Seconds per eval batch (bf16, no grad) and peak GiB."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    model = build_model({"backbone": BACKBONE, "grad_ckpt": False}, False).to(dev).eval()
    x, mask, pos = batch(bs, k, SLOTS_PRESENT)
    times = []
    for i in range(WARMUP + STEPS):
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev), train=False, **views)
        torch.cuda.synchronize()
        if i >= WARMUP:
            times.append(time.time() - t)
    sec = sum(times) / len(times)
    return {"sec_per_batch": sec, "studies_per_s": bs / sec,
            "peak_alloc_gb": torch.cuda.max_memory_allocated(dev) / 2**30}


rows = []
for bs, k, ckpt in TRAIN_SETTINGS:
    r = {"mode": "train", "bs": bs, "k": k, "grad_ckpt": ckpt, "slots": SLOTS_PRESENT,
         **train_cost(bs, k, ckpt, SLOTS_PRESENT)}
    print(json.dumps(r), flush=True)
    rows.append(r)
for bs, k, ckpt in WORST_CASE:
    r = {"mode": "train", "bs": bs, "k": k, "grad_ckpt": ckpt, "slots": N_SLOTS, **train_cost(bs, k, ckpt, N_SLOTS)}
    print(json.dumps(r), flush=True)
    rows.append(r)
for bs, k in EVAL_SETTINGS:
    r = {"mode": "eval", "bs": bs, "k": k, **eval_cost(bs, k)}
    print(json.dumps(r), flush=True)
    rows.append(r)
(ctx.output_dir / "bench.json").write_text(json.dumps(rows, indent=1))
print(f"[bench] {len(rows)} settings | {ctx.output_dir}")
