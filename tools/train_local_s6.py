"""Run the packaged S6 trainer on a prepared local cache. Requires CUDA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--px", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", required=True, help="Local ImageNet safetensors/state_dict, or timm")
    parser.add_argument("--targets", type=Path,
                        default=ROOT / "sources/B/training/s6/data/targets_s6f0.csv")
    args, extra = parser.parse_known_args()
    cache_path = args.px / "index.json"
    cache = json.loads(cache_path.read_text())
    if cache.get("format") != "local-dicom-views-v1" or not cache.get("done"):
        raise ValueError("Expected a completed local cache")
    if cache["config"]["size"] < 288:
        raise ValueError("Default training crop 288 requires a cache of size >= 288")
    source_dir = ROOT / "sources/B/training/s6"
    runtime = ROOT / "sources/B/runtime/heliosli"
    sys.path.insert(0, str(source_dir))
    import s6common as common
    common.HELIOS_SRC = str(runtime)
    common.CODE = str(ROOT / "sources/B/runtime/cloud_code")
    common.install_paths()
    common.verify_copies()  # Keep the original six-file checksum guard intact.
    spec = importlib.util.spec_from_file_location("train_s6f0_all", source_dir / "train_s6f0_all.py")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"approximate_preprocessing": True,
               "cache_index_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
               "preprocessing_config": cache["config"]}
    receipt_path = args.out / "local_preprocessing.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Output belongs to a different cache; choose a new --out")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    # Defaults describe a fold-0 experiment, not an all-data reproduction.
    # Extra trainer options can override recipe defaults; --px is fixed last.
    sys.argv = [str(source_dir / "train_s6f0_all.py"),
                "--backbone", "convnext_small", "--arms", "E_prior", "--seeds", "1", "--seed-base", "42",
                "--folds", "5", "--only-folds", "0", "--epochs", "16", "--batch", "2",
                "--train-windows", "64", "--crop", "288", "--slices", "3", "--phys-mm", "3.5",
                "--spatial-attn", "--grad-ckpt", "--lr", "1e-4", "--head-lr", "1e-3", "--wd", ".05",
                "--aug", "crop", "--train-sampling", "uniform", "--ema", "0", "--workers", "2",
                "--no-final-predict", "--i-have-user-authorisation-to-finetune", *extra,
                "--px", str(args.px.resolve()), "--targets", str(args.targets.resolve()),
                "--out", str(args.out.resolve()), "--init", args.init]
    print("Local approximate preprocessing; historical leaderboard scores do not apply.")
    trainer.main()


if __name__ == "__main__":
    main()
