"""Optional CPU smoke: real tensor/model path and trainer target preflight, no fitting."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ("torch", "timm", "pydicom", "cv2")),
                     "Full model smoke needs torch, timm and preprocessing dependencies")
class LocalModelSmoke(unittest.TestCase):
    def test_submission_and_training_target_preflight(self):
        import torch
        torch.set_num_threads(2)
        from test_local_preprocessing import STUDY, write_study
        from src.local_preprocessing import Config, prepare_cache
        sys.path.insert(0, str(ROOT / "sources/B/runtime/heliosli"))
        import compact_train as ct
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cache = root / "dicoms", root / "cache"
            second = STUDY + ".2"
            write_study(data)
            write_study(data, second)
            prepare_cache(data, cache, Config(size=32, mm_per_px=1))
            sample = root / "sample.csv"
            columns = ["StudyInstanceUID", *ct.LABELS]
            with sample.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for uid in (second, STUDY):
                    writer.writerow({"StudyInstanceUID": uid, **{k: 0 for k in ct.LABELS}})
            # A small backbone exercises the same spatial and study attention heads.
            # These random weights are a plumbing check, never a trained solution.
            model = ct.Net("resnet18", None, crop=32, spatial=True).eval()
            weights, output = root / "random.pt", root / "submission.csv"
            torch.save(model.deploy_state_dict(), weights)
            args = ["infer_local_s6.py", "--px", str(cache), "--models", str(weights),
                    "--sample", str(sample), "--output", str(output), "--backbone", "resnet18",
                    "--crop", "32", "--chunk", "8", "--device", "cpu"]
            with patch.object(sys, "argv", args):
                module("infer_local_s6").main()
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["StudyInstanceUID"] for r in rows], [second, STUDY])
            self.assertTrue(all(0 <= float(r[k]) <= 1 for r in rows for k in ct.LABELS))

            # --check-targets exits before loading images or weights. Build a small
            # index with real table membership to check wrapper paths and hash guards.
            target_path = ROOT / "sources/B/training/s6/data/targets_s6f0.csv"
            with target_path.open(newline="") as stream:
                targets = list(csv.DictReader(stream))
            ids = [r["StudyInstanceUID"] for r in targets if r["is_gold"] == "0"
                   and all(r[k + "|prior"] for k in ct.LABELS)][:20]
            gold = [r["StudyInstanceUID"] for r in targets if r["is_gold"] == "1"][:1]
            metadata_only = root / "preflight-cache"
            metadata_only.mkdir()
            (metadata_only / "index.json").write_text(json.dumps({
                "format": "local-dicom-views-v1", "config": {"size": 320},
                "done": {uid: [0, 1, 1] for uid in ids + gold}}))
            args = ["train_local_s6.py", "--px", str(metadata_only), "--out", str(root / "train"),
                    "--init", "unused-during-preflight.safetensors", "--check-targets"]
            with patch.object(sys, "argv", args):
                with self.assertRaises(SystemExit) as completed:
                    module("train_local_s6").main()
                self.assertEqual(completed.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
