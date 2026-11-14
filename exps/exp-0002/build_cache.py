"""Build the knee volume cache from the archive in one streaming pass (contract D).

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0002/build_cache.py

Each study is one HTTP Range read; its series go through knee_volume.preprocess_series
in memory and land in <kaggle_cache>/datasets/exp-0002/<SLUG>/ (key
ctx.mounts["exp-0002/<SLUG>"] in every worktree), not in ctx.output_dir. Raw DICOM is
never written. Resumable: studies with vol/<study>/_done.json are skipped, so a rerun
of the same script finishes what failed. <SLUG>/_provenance.json pins src_hash and
params: a run with other code or params refuses to touch the cache.

ctx.output_dir gets progress.jsonl (one line per finished study, appended live),
run_log.parquet, manifest.parquet (file, bytes, sha256 of every file written),
throughput.json and copies of the consolidated series/slices/dropped tables.
"""

# --- bootstrap: find utils/. Constant text, the same in every script ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- end of bootstrap ---

import os  # noqa: E402

# One thread per worker process; must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import inspect  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.knee_volume import preprocess_series  # noqa: E402
from src.stream_cache import build_studies, open_provenance  # noqa: E402
from src.vol_cache import consolidate, is_study_done  # noqa: E402
from src.zip_range import ArchiveReader  # noqa: E402
from utils.ctx import compute_src_hash, get_ctx, local_paths  # noqa: E402

COMP = "rsna-knee-abnormality-detection"
INDEX_RUN = "20261017t111437-152-050c"  # index_archive.py
SLUG = "knee-vol04"  # pilot: knee-vol04-pilot, full: knee-vol04
STUDIES = "all"  # "pilot" | "all" | a tuple of StudyInstanceUIDs
N_PILOT = 48  # random train studies of the pilot, plus CORRUPT_SOP_STUDIES
SEED = 0
CORRUPT_SOP_STUDIES = (
    "1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035",
    "1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680",
)
N_WORKERS = 16
STUDY_ATTEMPTS = 2  # whole-study retries on ArchiveError (read_range retries on its own)
MAX_TASKS_PER_CHILD = 25  # recycle workers against decoder memory growth

ctx = get_ctx()
out = ctx.output_dir

index = pd.read_parquet(ctx.mounts[INDEX_RUN] / "archive_index.parquet")
train = index[(index.split == "train_series") & index.study.notna()]
all_studies = sorted(train.study.unique())
if STUDIES == "pilot":
    others = [s for s in all_studies if s not in CORRUPT_SOP_STUDIES]
    picked = np.random.default_rng(SEED).choice(others, N_PILOT, replace=False)
    selected = sorted(picked.tolist()) + list(CORRUPT_SOP_STUDIES)
elif STUDIES == "all":
    selected = all_studies
elif isinstance(STUDIES, tuple):
    selected = list(STUDIES)
else:
    raise ValueError(f"STUDIES must be 'pilot', 'all' or a tuple of study UIDs, got {STUDIES!r}")
unknown = sorted(set(selected) - set(all_studies))
if unknown:
    raise RuntimeError(f"{len(unknown)} selected studies are not train studies of the index: "
                       f"{unknown[:5]}")

root = Path(local_paths(ctx.sources_dir)["kaggle_cache"]).expanduser().resolve() \
    / "datasets" / "exp-0002" / SLUG
params = {
    "archive": COMP,
    "index_run": INDEX_RUN,
    # build_cache calls preprocess_series(files) exactly as load_series_dir does at
    # inference; record the defaults that call actually used.
    "preprocess_series": {name: p.default
                          for name, p in inspect.signature(preprocess_series).parameters.items()
                          if p.default is not inspect.Parameter.empty},
}
open_provenance(root, SLUG, params, compute_src_hash(ctx.sources_dir), out)

todo = [s for s in selected if not is_study_done(root, s)]
print(f"[build_cache] {SLUG}: {len(selected)} studies selected, {len(selected) - len(todo)} "
      f"already done, {len(todo)} to build with {N_WORKERS} workers → {root}", flush=True)
study_rows = dict(tuple(train[train.study.isin(todo)].groupby("study", sort=False)))

t0 = time.time()
run_log, manifest = build_studies(ArchiveReader(COMP), root, study_rows, N_WORKERS,
                                  STUDY_ATTEMPTS, MAX_TASKS_PER_CHILD, out / "progress.jsonl")
wall = time.time() - t0
run_log.to_parquet(out / "run_log.parquet", index=False)
manifest.to_parquet(out / "manifest.parquet", index=False)

for table in consolidate(root, ctx.mounts[COMP] / "train_series.csv").values():
    shutil.copy2(table, out / table.name)

ok = run_log[run_log.error.isna()]
failed = run_log[run_log.error.notna()]
throughput = {
    "studies": STUDIES if isinstance(STUDIES, str) else "list",
    "n_selected": len(selected),
    "n_skipped_done": len(selected) - len(todo),
    "n_processed": len(run_log),
    "n_ok": len(ok),
    "n_failed": len(failed),
    "n_series": int(ok.n_series.sum()),
    "n_series_error": int(ok.n_series_error.sum()),
    "compressed_bytes": int(run_log.compressed_bytes.sum()),
    "volume_bytes": int(manifest.bytes.sum()),
    "wall_s": round(wall, 1),
    "compressed_mb_per_s": round(run_log.compressed_bytes.sum() / 1e6 / wall, 1),
    "studies_per_hour": round(len(run_log) / wall * 3600, 1),
    "n_workers": N_WORKERS,
    "sum_s": {c: round(float(run_log[c].sum()), 1)
              for c in ("s_download", "s_inflate", "s_process", "s_write")},
}
(out / "throughput.json").write_text(json.dumps(throughput, indent=2) + "\n", encoding="utf-8")
print(json.dumps(throughput, indent=2))

if len(failed):
    # Everything above is written; the record still says "failed" so the gap is not missed.
    raise RuntimeError(f"{len(failed)} studies failed after retries and are not done (see "
                       f"run_log.parquet, e.g. {failed.error.iloc[0]}). Rerun this script to "
                       "resume them.")
print(f"[build_cache] {len(ok)} studies built | {root} | {out}")
