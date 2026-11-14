"""Parity check (contract D): the cache path and the inference path give the same triple on the test studies.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0002/check_parity.py

(a) cache path: the 3 example test studies are read from the archive with the build code
    itself (stream_cache.build_study: one Range read → iter_members → preprocess_series →
    write_study, cv2 on 1 thread as in build_cache's workers) into ctx.output_dir/parity_cache,
    consolidated, and read back with KneeVolCache.load;
(b) inference path: the same series from the competition mount's test_series/ (placed by
    fetch_test_series.py) → knee_volume.load_series_dir, cv2 on its default thread count, as in
    a Kaggle kernel.
Per series: input files equal the archive members (size + CRC32 from the index), vol and
pos_mm byte-identical, meta equal (values and types), slices/dropped rows equal. Writes
ctx.output_dir/parity.json and fails if any series differs.
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
import zlib  # noqa: E402

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.knee_volume import META_FIELDS, SeriesError, load_series_dir, preprocess_series, read_series_dir  # noqa: E402
from src.stream_cache import build_study  # noqa: E402
from src.vol_cache import KneeVolCache, consolidate  # noqa: E402
from src.zip_range import ArchiveReader  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

COMP = "rsna-knee-abnormality-detection"
INDEX_RUN = "20261017t111437-152-050c"  # index_archive.py
N_TEST_SERIES = 15
ATTEMPTS = 3  # whole-study retries on ArchiveError, as in build_cache

ctx = get_ctx()
comp = ctx.mounts[COMP]
index = pd.read_parquet(ctx.mounts[INDEX_RUN] / "archive_index.parquet")
test = index[(index.split == "test_series") & index.study.notna()].sort_values("lho")
csv = pd.read_csv(comp / "test_series.csv")
if set(csv.SeriesInstanceUID) != set(test.series) or len(csv) != N_TEST_SERIES:
    raise RuntimeError(f"test_series.csv ({len(csv)} series) and the archive index "
                       f"({test.series.nunique()} test series) disagree; expected {N_TEST_SERIES}")

# (a) cache path
root = ctx.output_dir / "parity_cache"
reader = ArchiveReader(COMP)
cv2.setNumThreads(1)
for study, rows in test.groupby("study", sort=True):
    log, _ = build_study(reader, root, study, rows, ATTEMPTS)
    if log["error"] is not None:
        raise RuntimeError(f"{study}: {log['error']}")
    print(f"[parity] cached {study}: {log['compressed_bytes'] / 1e6:.1f} MB, "
          f"{log['n_series']} series, {log['n_series_error']} errors", flush=True)
consolidate(root, comp / "test_series.csv")
cache = KneeVolCache(root)
cached_slices = pq.read_table(root / "slices.parquet").to_pylist()
cached_dropped = pq.read_table(root / "dropped.parquet").to_pylist()


def rows_of(table, series):
    return [{k: v for k, v in r.items() if k not in ("study", "series")} for r in table if r["series"] == series]


def triple(load):
    try:
        return load(), None
    except SeriesError as e:
        return None, [e.code, e.detail]


# (b) inference path, compared series by series
cv2.setNumThreads(-1)  # OpenCV default, as in a kernel
report = []
for r in csv.sort_values(["StudyInstanceUID", "SeriesInstanceUID"]).itertuples(index=False):
    study, series = r.StudyInstanceUID, r.SeriesInstanceUID
    d = comp / "test_series" / study / series
    members = test[test.series == series]
    expected = {n.rsplit("/", 1)[1]: (usize, crc) for n, usize, crc in zip(members.name, members.usize, members.crc32)}
    files = read_series_dir(d)
    inputs_identical = (sorted(expected) == [n for n, _ in files]
                        and all((len(b), zlib.crc32(b)) == expected[n] for n, b in files))

    a, err_a = triple(lambda: cache.load(series, mmap=True))
    b, err_b = triple(lambda: load_series_dir(d))
    res = {"study": study, "series": series, "inputs_identical": inputs_identical,
           "error_cache": err_a, "error_inference": err_b}
    if a is not None and b is not None:
        (va, pa_, ma), (vb, pb, mb) = a, b
        sv = preprocess_series(files)
        res.update(
            vol_identical=va.dtype == vb.dtype and va.shape == vb.shape and va.tobytes() == vb.tobytes(),
            pos_identical=pa_.dtype == pb.dtype and pa_.tobytes() == pb.tobytes(),
            meta_diff=[k for k in META_FIELDS if ma[k] != mb[k] or type(ma[k]) is not type(mb[k])],
            slices_equal=rows_of(cached_slices, series) == sv.slices,
            dropped_equal=rows_of(cached_dropped, series) == sv.dropped,
            shape=list(va.shape), plane=ma["plane_computed"], n_dropped=ma["n_dropped"],
        )
        res["ok"] = (inputs_identical and res["vol_identical"] and res["pos_identical"] and not res["meta_diff"]
                     and res["slices_equal"] and res["dropped_equal"])
    else:
        res["ok"] = inputs_identical and err_a == err_b
    report.append(res)
    print(f"[parity] {series[-12:]}: {'ok' if res['ok'] else 'MISMATCH'} {res.get('shape', err_a)}", flush=True)

summary = {"ok": all(r["ok"] for r in report), "n_series": len(report), "n_ok": sum(r["ok"] for r in report),
           "cv2_threads": {"cache": 1, "inference": cv2.getNumThreads()}, "index_run": INDEX_RUN,
           "series": report}
(ctx.output_dir / "parity.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
if not summary["ok"]:
    raise RuntimeError(f"parity FAILED for {summary['n_series'] - summary['n_ok']} series; see parity.json")
print(f"[parity] ok: {summary['n_ok']}/{summary['n_series']} series identical | {ctx.output_dir / 'parity.json'}")
