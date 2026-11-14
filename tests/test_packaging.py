"""Offline checks for the generated Kaggle input and kernel directories."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare", ROOT / "tools/prepare_kaggle.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
ctx_spec = importlib.util.spec_from_file_location("utils.ctx", ROOT / "utils/ctx.py")
import sys
ctx = importlib.util.module_from_spec(ctx_spec)
sys.modules[ctx_spec.name] = ctx
ctx_spec.loader.exec_module(ctx)


class PackagingTest(unittest.TestCase):
    def test_both_weights_are_self_consistent(self):
        for weight in (20, 30):
            with self.subTest(weight=weight), tempfile.TemporaryDirectory() as tmp:
                out = prepare.build("example-user", weight, Path(tmp) / "bundle",
                                    "nartaa/rsna-knee-s6fall-weights")
                sources, kernel = out / "sources", out / "kernel"
                stamp = json.loads((sources / "_stamp.json").read_text())
                self.assertEqual(stamp["src_hash"], ctx.compute_src_hash(sources))
                metadata = json.loads((kernel / "kernel-metadata.json").read_text())
                self.assertTrue((kernel / metadata["code_file"]).is_file())
                self.assertEqual(len(metadata["dataset_sources"]), 6)
                self.assertFalse(metadata["enable_internet"])
                self.assertEqual((kernel / metadata["code_file"]).read_bytes(),
                                 (ROOT / "exps/exp-0001" / metadata["code_file"]).read_bytes())
                text = (sources / "src/public_fork_s6fall/pipeline.py").read_text()
                self.assertIn('WEIGHTS_MOUNT = "nartaa/rsna-knee-s6fall-weights"', text)
                self.assertFalse(any(p.suffix == ".pt" for p in out.rglob("*")))

    def test_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                prepare.build("example-user", 20, Path(tmp), "nartaa/rsna-knee-s6fall-weights")
            self.assertEqual(sentinel.read_text(), "keep")

    def test_rejects_invalid_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare.build("../other", 20, Path(tmp) / "out", "nartaa/rsna-knee-s6fall-weights")
            self.assertFalse((Path(tmp) / "out").exists())


if __name__ == "__main__":
    unittest.main()
