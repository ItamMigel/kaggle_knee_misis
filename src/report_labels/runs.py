"""extract.py run folders: the gapless raw.part-NNNN.jsonl sequence and per-report token statistics.

A run folder (fetched with `kaggle/fetch.py --from-log`, or a decode_log.py output subfolder)
holds selection.csv, raw.part-NNNN.jsonl from 0000 on, and stats.json if the kernel finished.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

UID = "StudyInstanceUID"
PART_RE = re.compile(r"raw\.part-(\d{4})\.jsonl")


def collect_parts(run_dir: Path, *, allow_unfinished: bool) -> list[Path]:
    """Ordered raw parts of one extract.py run, after loud checks: gapless 0000.., unique uids from selection.csv, stats.json agrees.

    Without stats.json the kernel stopped early (fail-fast, error, kill): refused unless
    allow_unfinished, which returns the parts it did emit.
    """
    run_dir = Path(run_dir)
    sel_path = run_dir / "selection.csv"
    if not sel_path.is_file():
        raise ValueError(f"{run_dir}: no selection.csv — not an extract.py run folder?")
    with sel_path.open(encoding="utf-8", newline="") as f:
        selected = [row[UID] for row in csv.DictReader(f)]
    if len(set(selected)) != len(selected):
        raise ValueError(f"{run_dir}: selection.csv has duplicated {UID}")

    found = sorted(run_dir.glob("raw.part-*.jsonl"))
    odd = [p.name for p in found if not PART_RE.fullmatch(p.name)]
    if odd:
        raise ValueError(f"{run_dir}: unexpected part file names {odd}")
    parts = sorted((int(PART_RE.fullmatch(p.name).group(1)), p) for p in found)
    if not parts:
        raise ValueError(f"{run_dir}: no raw.part-NNNN.jsonl")
    indices = [i for i, _ in parts]
    if indices != list(range(len(indices))):
        gaps = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise ValueError(f"{run_dir}: raw parts not gapless from 0000: have {indices}, missing {gaps}")

    uids, sizes = [], []
    for _, p in parts:
        lines = p.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError(f"{p}: empty part or blank line")
        uids += [json.loads(line)[UID] for line in lines]
        sizes.append(len(lines))
    dup = sorted({u for u, n in Counter(uids).items() if n > 1})
    if dup:
        raise ValueError(f"{run_dir}: {len(dup)} studies in more than one raw line, e.g. {dup[0]}")
    foreign = set(uids) - set(selected)
    if foreign:
        raise ValueError(f"{run_dir}: {len(foreign)} raw studies are not in selection.csv, e.g. {sorted(foreign)[0]}")

    stats_path = run_dir / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        bs = stats["batch_size"]
        problems = [msg for bad, msg in [
            (stats["n_done"] != len(uids), f"n_done {stats['n_done']} != {len(uids)} raw lines"),
            (stats["n_batches"] != len(parts), f"n_batches {stats['n_batches']} != {len(parts)} parts"),
            (stats["n_total"] != len(selected), f"n_total {stats['n_total']} != {len(selected)} selected"),
            (stats["n_done"] + stats["n_skipped"] != stats["n_total"], "n_done + n_skipped != n_total"),
            (set(stats["skipped_uids"]) != set(selected) - set(uids), "skipped_uids != selected - extracted"),
            (any(n != bs for n in sizes[:-1]) or sizes[-1] > bs, f"part sizes {sizes} vs batch_size {bs}"),
        ] if bad]
        if problems:
            raise ValueError(f"{run_dir}: stats.json disagrees with the raw parts: {problems}")
    elif not allow_unfinished:
        raise ValueError(f"{run_dir}: no stats.json — the kernel stopped before the end (fail-fast, error or "
                         f"kill); it emitted {len(uids)}/{len(selected)} studies in {len(parts)} parts. "
                         f"Pass allow_unfinished=True to use them.")
    else:
        print(f"[collect_parts] {run_dir.name}: UNFINISHED (no stats.json): {len(uids)}/{len(selected)} "
              f"studies in {len(parts)} parts", flush=True)
    return [p for _, p in parts]


def token_summary(raw_files: list[Path]) -> dict:
    """Prompt/output tokens per report (mean, p50, p90, max) and finish_reason counts per `model/pass`, from raw lines."""
    groups: dict[str, dict] = {}
    for path in raw_files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            g = groups.setdefault(f"{rec['model']}/{rec['pass']}",
                                  {"prompt": [], "output": [], "finish": Counter()})
            g["prompt"].append(int(rec["prompt_tokens"]))
            g["output"].append(int(rec["output_tokens"]))
            g["finish"][rec["finish_reason"]] += 1
    if not groups:
        raise ValueError(f"no raw lines in {raw_files}")

    def dist(x: list[int]) -> dict:
        a = np.asarray(x, dtype=float)
        return {"mean": round(float(a.mean()), 1), "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)), "max": int(a.max())}

    return {k: {"n": len(g["output"]), "output_tokens": dist(g["output"]), "prompt_tokens": dist(g["prompt"]),
                "finish_reasons": dict(g["finish"])} for k, g in groups.items()}
