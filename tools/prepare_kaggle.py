"""Build a local sources dataset and kernel folder; never upload or run them."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
INPUTS = [
    "goodpjw2008/rsna-knee-2-5d-convnext-reader",
    "heliosli/rsna-knee-r6-frozen-fe055406-w1",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
]


def payload_hash(root: Path) -> str:
    files = [p for folder in ("src", "utils") for p in (root / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
             and p.name not in {"_stamp.json", ".DS_Store"}]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def build(owner: str, weight: int, output: Path, s6_dataset: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle username")
    if weight not in (20, 30):
        raise ValueError("Choose weight 20 or 30")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/rsna-knee-s6fall-weights", s6_dataset):
        raise ValueError("The S6 dataset must keep the rsna-knee-s6fall-weights slug")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose another directory")
    output.mkdir(parents=True)
    sources, kernel = output / "sources", output / "kernel"
    sources.mkdir()
    kernel.mkdir()
    for folder in ("src", "utils"):
        shutil.copytree(ROOT / folder, sources / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    runner = sources / "src/public_fork_s6fall/pipeline.py"
    text = runner.read_text()
    old = 'WEIGHTS_MOUNT = "bebra3222/rsna-knee-s6fall-weights"'
    if text.count(old) != 1:
        raise ValueError("S6 mount declaration changed; inspect the runner")
    runner.write_text(text.replace(old, f"WEIGHTS_MOUNT = {json.dumps(s6_dataset)}"))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    source_id = f"{owner}/rsna-knee-misis-sources-w{weight}"
    dump(sources / "_stamp.json", {"git_sha": commit, "src_hash": payload_hash(sources)})
    dump(sources / "dataset-metadata.json", {
        "id": source_id, "title": f"RSNA Knee MISIS sources w{weight}",
        "licenses": [{"name": "other"}],
    })
    filename = f"submit_a_s6fall_w{weight}.py"
    shutil.copyfile(ROOT / "exps/exp-0001" / filename, kernel / filename)
    dump(kernel / "kernel-metadata.json", {
        "id": f"{owner}/rsna-knee-misis-w{weight}",
        "title": f"RSNA Knee MISIS w{weight}", "code_file": filename,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4", "docker_image": DOCKER,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [source_id, *INPUTS, s6_dataset], "kernel_sources": [],
    })
    return output


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--weight", type=int, choices=(20, 30), default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--s6-dataset", default="nartaa/rsna-knee-s6fall-weights")
    args = parser.parse_args()
    output = args.output or ROOT / f"build/kaggle-w{args.weight}"
    result = build(args.owner, args.weight, output.resolve(), args.s6_dataset)
    print(f"Prepared: {result}")
    print("sources/: dataset files; kernel/: entry point and Kaggle metadata")
    print("No uploads, GPU runs or submissions were performed.")


if __name__ == "__main__":
    main()
