"""Verify packaged files and reproduce the saved rank blends without ML packages."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"20261019t052412-099-f1c6": 0.2, "20261019t122344-261-6000": 0.3}


def read_csv(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["StudyInstanceUID"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate study identifiers: {path}")
    return result


def percentile_ranks(rows: dict, label: str) -> dict:
    values = {uid: float(row[label]) for uid, row in rows.items()}
    return {uid: (sum(x < value for x in values.values())
                  + (sum(x == value for x in values.values()) + 1) / 2) / len(values)
            for uid, value in values.items()}


def verify(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    compiled = 0
    for folder in ("src", "utils", "exps", "tools", "sources", "kaggle", "tests"):
        for path in (root / folder).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
            compiled += 1
    for folder in ("public_fork", "public_fork_s6fall"):
        base = root / "src" / folder / "vendored"
        for name, expected in json.loads((base / "sha256.json").read_text()).items():
            content = (base / (name + ".txt")).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Embedded source checksum mismatch: {folder}/{name}")
            ast.parse(content, filename=name)
            compiled += 1
    notebook = json.loads((root / "sources/A/notebook.ipynb").read_text())
    if notebook.get("nbformat") != 4 or not any(c["cell_type"] == "code" for c in notebook["cells"]):
        raise ValueError("Component A notebook is not a complete nbformat-4 notebook")
    ast.parse((root / "sources/B/notebook.py").read_text())
    training = root / "sources/B/training"
    training_files = 0
    for line in (training / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if hashlib.sha256((training / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Training source checksum mismatch: {name}")
        training_files += 1
    runtime = json.loads((root / "sources/B/code_manifest.json").read_text())["members"]
    available, external = [], []
    for name, info in runtime.items():
        path = root / "sources/B/runtime" / name
        if not path.is_file():
            external.append(name)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Runtime source checksum mismatch: {name}")
        available.append(name)
    blends = {}
    for rid, weight in RUNS.items():
        base = root / "results" / rid
        a = read_csv(base / "members/final_prob.csv")
        b = read_csv(base / "s6fall/s6fall_prob.csv")
        final = read_csv(base / "submission.csv")
        if set(a) != set(b) or set(a) != set(final):
            raise ValueError(f"Study identifiers differ in {rid}")
        labels = set(next(iter(final.values()))) - {"StudyInstanceUID"}
        if len(labels) != 12:
            raise ValueError(f"Expected 12 findings in {rid}")
        errors = []
        for label in labels:
            ar, br = percentile_ranks(a, label), percentile_ranks(b, label)
            for uid in final:
                expected = (1 - weight) * ar[uid] + weight * br[uid]
                errors.append(abs(expected - float(final[uid][label])))
        if max(errors) > 1e-12:
            raise ValueError(f"Rank blend mismatch: {rid}, {max(errors)}")
        blends[rid] = {"visible_studies": len(final), "max_abs_error": max(errors)}
    return {"checked_files": len(manifest["files"]), "parsed_python_files": compiled,
            "training_source_files": training_files, "available_runtime_files": len(available),
            "external_runtime_files_not_bundled": len(external),
            "saved_validation_blends": blends, "gpu_inference_run": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
