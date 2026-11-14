"""Generate a submission from locally trained S6 models and a local slice cache."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="Deploy state_dict .pt files")
    parser.add_argument("--sample", type=Path, required=True, help="sample_submission.csv; fixes study order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="convnext_small")
    parser.add_argument("--crop", type=int, default=288)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "sources/B/runtime/heliosli"))
    import compact_train as ct
    index = json.loads((args.px / "index.json").read_text())
    if index.get("format") != "local-dicom-views-v1":
        raise ValueError("Expected a local cache")
    if not 0 < args.crop <= index["config"]["size"] or args.chunk < 1:
        raise ValueError("Invalid crop or chunk size")
    with args.sample.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if set(columns or []) != {"StudyInstanceUID", *ct.LABELS}:
            raise ValueError("Sample must contain StudyInstanceUID and the 12 target labels")
        uids = [row["StudyInstanceUID"] for row in reader]
    if not uids or len(uids) != len(set(uids)) or set(uids) - set(index["done"]):
        raise ValueError("Empty/duplicate sample studies or studies missing from the cache")
    if args.output.exists():
        raise FileExistsError(args.output)
    models = []
    for path in args.models:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = ct.Net(args.backbone, None, crop=args.crop,
                       spatial="sp_q" in state, extent="ext_u" in state)
        model.load_deploy_state_dict(state)
        models.append(model.to(args.device).eval())
    dataset = ct.Studies(str(args.px), uids, 64, "eval", 42, eval_stride=1,
                         slices=3, phys_mm=3.5, extent_meta=any(m.extent for m in models))
    predictions = []
    with torch.inference_mode():
        for i, uid in enumerate(uids):
            img, meta, mask, _ = ct.collate([dataset[i]])
            if img.shape[1] == 0:
                raise ValueError(f"No windows: {uid}")
            start = (img.shape[-1] - args.crop) // 2
            img = img[..., start:start + args.crop, start:start + args.crop].to(args.device)
            meta, mask = meta.to(args.device), mask.to(args.device)
            amp = torch.autocast("cuda", dtype=torch.float16) if args.device == "cuda" else contextlib.nullcontext()
            with amp:
                # Match compact-model aggregation: mean logits, then sigmoid.
                logits = torch.stack([m(img, meta, mask, chunk=args.chunk).float() for m in models]).mean(0)
                probability = logits.sigmoid()[0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise ValueError(f"Non-finite scores: {uid}")
            predictions.append({"StudyInstanceUID": uid, **dict(zip(ct.LABELS, map(float, probability)))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} studies to {args.output}; approximate local pipeline")


if __name__ == "__main__":
    main()
