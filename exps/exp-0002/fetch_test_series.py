"""Put the example test_series/ DICOMs into the local competition mount, Kaggle layout.

    ~/conda/envs/knee/bin/python exps/exp-0002/fetch_test_series.py

The 557 `test_series/` members of the archive (3 studies, ~290 MB compressed) →
<kaggle_cache>/competitions/<SLUG>/test_series/<study>/<series>/<sop>.dcm, so
inference scripts run locally on the same paths as in a Kaggle kernel. Writes are
atomic; a file that is already there must equal the archive member byte for byte,
otherwise the script stops and overwrites nothing. ctx.output_dir gets
test_series_files.parquet (name, bytes, sha256, status written/present).
"""

# --- bootstrap: find utils/. Constant text, the same in every script ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- end of bootstrap ---

import hashlib  # noqa: E402
import os  # noqa: E402

import pandas as pd  # noqa: E402

from src.zip_range import ArchiveReader, iter_members, member_range  # noqa: E402
from utils.ctx import get_ctx, local_paths  # noqa: E402

SLUG = "rsna-knee-abnormality-detection"
INDEX_RUN = "20261017t111437-152-050c"  # index_archive.py
N_TEST_MEMBERS = 557

ctx = get_ctx()

index = pd.read_parquet(ctx.mounts[INDEX_RUN] / "archive_index.parquet")
test = index[(index.split == "test_series") & index.study.notna()]
if len(test) != N_TEST_MEMBERS:
    raise RuntimeError(f"index {INDEX_RUN} has {len(test)} test_series members, expected "
                       f"{N_TEST_MEMBERS}: the archive changed, recheck before writing the mount")
comp_dir = Path(local_paths(ctx.sources_dir)["kaggle_cache"]).expanduser().resolve() \
    / "competitions" / SLUG
if not comp_dir.is_dir():
    raise RuntimeError(f"{comp_dir} does not exist: pull the competition csv files first")

reader = ArchiveReader(SLUG)
out = []
for study, rows in test.groupby("study", sort=False):
    start, end = member_range(rows)
    buf = reader.read_range(start, end)
    for name, raw in iter_members(buf, start, rows):
        dest = comp_dir / name
        if dest.exists():
            if dest.read_bytes() != raw:
                raise RuntimeError(f"{dest} exists and differs from archive member {name}; "
                                   "nothing overwritten. Move it away and rerun.")
            status = "present"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
            tmp.write_bytes(raw)
            os.replace(tmp, dest)
            status = "written"
        out.append({"name": name, "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(), "status": status})
    print(f"[fetch_test_series] {study}: {len(rows)} files, {(end - start) / 1e6:.0f} MB read")

files = pd.DataFrame(out)
files.to_parquet(ctx.output_dir / "test_series_files.parquet", index=False)
print(f"[fetch_test_series] {files.status.value_counts().to_dict()} → {comp_dir / 'test_series'}")
