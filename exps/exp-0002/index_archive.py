"""Index the competition's download-all zip: one row per member with its byte range.

    ~/conda/envs/knee/bin/python exps/exp-0002/index_archive.py

Reads only the archive tail and the ZIP64 central directory (~233 MB, ~10 s).
Writes ctx.output_dir/archive_index.parquet (columns of zip_range.INDEX_COLUMNS)
and index_stats.json. The other exp-0002 scripts read the index through
ctx.mounts[INDEX_RUN] with this run's id.
"""

# --- bootstrap: find utils/. Constant text, the same in every script ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- end of bootstrap ---

import json  # noqa: E402
import time  # noqa: E402

from src.zip_range import ArchiveReader, read_central_directory  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SLUG = "rsna-knee-abnormality-detection"

ctx = get_ctx()

t0 = time.time()
reader = ArchiveReader(SLUG)
archive_bytes = reader.total_size()
index = read_central_directory(reader)
seconds = time.time() - t0
index.to_parquet(ctx.output_dir / "archive_index.parquet", index=False)

# Archive order: a study (series) is one contiguous byte range iff it starts one run.
index_runs = index.assign(
    study_start=(index.study != index.study.shift()) & index.study.notna(),
    series_start=(index.series != index.series.shift()) & index.series.notna(),
)
members = index_runs[index_runs.study.notna()]
stats = {
    "slug": SLUG,
    "archive_bytes": archive_bytes,
    "central_directory_offset": int(index.next_lho.iloc[-1]),
    "n_members": len(index),
    "methods": {str(k): int(v) for k, v in index.method.value_counts().items()},
    "unknown_layout": index.loc[index.study.isna(), "name"].tolist(),
    "seconds": round(seconds, 1),
    "splits": {},
}
for split, g in members.groupby("split"):
    stats["splits"][split] = {
        "n_members": len(g),
        "n_studies": int(g.study.nunique()),
        "n_series": int(g.series.nunique()),
        "study_runs": int(g.study_start.sum()),
        "series_runs": int(g.series_start.sum()),
        "compressed_bytes": int(g.csize.sum()),
        "uncompressed_bytes": int(g.usize.sum()),
    }
(ctx.output_dir / "index_stats.json").write_text(json.dumps(stats, indent=2) + "\n",
                                                 encoding="utf-8")
print(json.dumps({k: v for k, v in stats.items() if k != "unknown_layout"}, indent=2))
print(f"[index_archive] {len(index)} members | {ctx.output_dir}")
