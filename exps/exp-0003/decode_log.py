"""exp-0003 decode_log.py — local: rebuild extract.py kernel outputs from their fetched logs.

    ~/conda/envs/knee/bin/python exps/exp-0003/decode_log.py

Before: `uv run --project kaggle python kaggle/fetch.py --kernel <owner/slug> --log-only`
for every run in KERNEL_RUNS (puts log.txt into exps/exp-0003/runs/<run_id>/).
Per run: every emitted file → <output_dir>/<run_id>/, the raw.part-NNNN.jsonl
in order → raw.jsonl (checked by runs.collect_parts: gapless parts, unique studies,
all from selection.csv, stats.json consistent), decode_summary.json with what came through.
Kernels fetched with `kaggle/fetch.py --from-log` need no decoding: build_labels.py reads
their run folders directly. probe.bin is the one file allowed to
arrive cut: its received chunks are the measured log limit.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

from src.log_transport import log_inventory, write_decoded  # noqa: E402
from src.report_labels.runs import collect_parts  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

KERNEL_RUNS: list[str] = ["20261017t113644-153-79d4"]   # run_ids of extract.py kernels, log.txt fetched with --log-only
PROBE_NAME = "probe.bin"
ALLOW_UNFINISHED = True  # decode_log is also for diagnosing kernels that stopped early

ctx = get_ctx()
if not KERNEL_RUNS:
    raise SystemExit("KERNEL_RUNS is empty: list the extract.py run_ids to decode")

summary = {}
for run_id in KERNEL_RUNS:
    log_path = ctx.mounts[run_id] / "log.txt"
    text = log_path.read_text(encoding="utf-8")
    inventory = log_inventory(text)
    print(f"[decode] {run_id}: {len(text):,} log chars, {len(inventory)} emitted files")

    probe = inventory.get(PROBE_NAME)
    probe_cut = probe is not None and bool(probe["missing"] or probe["bad_crc"] or not probe["ended"])
    if probe_cut:
        print(f"[decode]   {PROBE_NAME} CUT: {probe['received']}/{probe['chunks']} chunks received, "
              f"END {'seen' if probe['ended'] else 'missing'}, crc failures {len(probe['bad_crc'])} "
              f"→ log limit measured, file left out")
    out_dir = ctx.output_dir / run_id
    written = write_decoded(log_path, out_dir, ignore=(PROBE_NAME,) if probe_cut else ())
    for path in sorted(written):
        print(f"[decode]   {path.name:28s} {path.stat().st_size:>11,} B  "
              f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}… ok")

    names = {p.name for p in written}
    parts = collect_parts(out_dir, allow_unfinished=ALLOW_UNFINISHED)
    lines = []
    for part in parts:
        lines += part.read_text(encoding="utf-8").splitlines()
    with (out_dir / "selection.csv").open(encoding="utf-8", newline="") as f:
        selected = [row["StudyInstanceUID"] for row in csv.DictReader(f)]
    (out_dir / "raw.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    finished = "stats.json" in names
    summary[run_id] = {
        "finished": finished, "n_selected": len(selected), "n_raw": len(lines),
        "n_not_extracted": len(selected) - len(lines), "n_parts": len(parts),
        "files": sorted(names),
        "probe": probe and {"bytes": probe["bytes"], "chunks": probe["chunks"],
                            "received": probe["received"], "n_bad_crc": len(probe["bad_crc"]),
                            "ended": probe["ended"], "cut": probe_cut},
    }
    print(f"[decode]   raw.jsonl: {len(lines)}/{len(selected)} selected studies from {len(parts)} parts"
          f"{'' if finished else ' (UNFINISHED)'}")

(ctx.output_dir / "decode_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"[decode] готово | {ctx.output_dir}")
