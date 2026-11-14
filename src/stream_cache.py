"""src/stream_cache.py — build the volume cache from the archive, one Range read per study.

Per study: one read_range over its contiguous members → iter_members (CRC-checked
inflate) → group by series → knee_volume.preprocess_series → vol_cache.write_study.
Raw DICOM never touches the disk. Studies run in a fork process pool; the parent
only collects run-log and manifest rows and appends them to progress.jsonl as they
arrive, so even a killed run leaves the hashes of everything it wrote.

Failure policy: knee_volume.SeriesError → recorded for that series (write_study
keeps it in _done.json); zip_range.ArchiveError (network after retries, size/CRC
mismatch) → the whole study is retried, then becomes an error row and stays not
done, so a resume redoes it; anything else is a bug and stops the run.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import pandas as pd

from src.knee_volume import SeriesError, preprocess_series
from src.vol_cache import write_study
from src.zip_range import ArchiveError, ArchiveReader, iter_members, member_range

PROVENANCE_NAME = "_provenance.json"
MANIFEST_COLUMNS = ["file", "bytes", "sha256"]  # vol_cache.write_study rows
RUN_LOG_COLUMNS = ["study", "compressed_bytes", "n_members", "n_series", "n_series_error",
                   "attempts", "s_download", "s_inflate", "s_process", "s_write", "error"]


def open_provenance(root: Path, slug: str, params: dict, src_hash: str, output_dir: Path) -> dict:
    """Create or extend <root>/_provenance.json with this run; a different src_hash/params → raise."""
    root = Path(root)
    path = root / PROVENANCE_NAME
    params = json.loads(json.dumps(params))  # tuples → lists, as they read back from the file
    if path.exists():
        prov = json.loads(path.read_text(encoding="utf-8"))
        for field, mine in (("slug", slug), ("src_hash", src_hash), ("params", params)):
            if prov[field] != mine:
                raise RuntimeError(
                    f"{path}: {field} is {prov[field]!r}, this run has {mine!r}. One cache never "
                    "mixes code or parameter versions: build into a new slug (e.g. "
                    f"{slug}-v2), or restore the code that produced this one."
                )
    else:
        root.mkdir(parents=True, exist_ok=True)
        prov = {"slug": slug, "params": params, "src_hash": src_hash, "runs": []}
    prov["runs"].append(str(Path(output_dir).resolve()))
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return prov


def build_study(reader: ArchiveReader, root: Path, study: str, rows: pd.DataFrame,
                attempts: int) -> tuple[dict, list[dict]]:
    """Download, inflate, preprocess and write one study → (run-log row, manifest rows)."""
    start, end = member_range(rows)
    log = {"study": study, "compressed_bytes": end - start, "n_members": len(rows),
           "n_series": int(rows.series.nunique()), "n_series_error": 0, "attempts": 0,
           "s_download": 0.0, "s_inflate": 0.0, "s_process": 0.0, "s_write": 0.0, "error": None}
    for attempt in range(1, attempts + 1):
        log.update(attempts=attempt, s_download=0.0, s_inflate=0.0, s_process=0.0, error=None)
        try:
            t = time.perf_counter()
            buf = reader.read_range(start, end)
            log["s_download"] = time.perf_counter() - t
            results = {}
            for series, srows in rows.groupby("series", sort=False):
                t = time.perf_counter()
                files = [(name.rsplit("/", 1)[1], raw)  # '<sop>.dcm', as read_series_dir
                         for name, raw in iter_members(buf, start, srows)]
                log["s_inflate"] += time.perf_counter() - t
                t = time.perf_counter()
                try:
                    results[series] = preprocess_series(files)
                except SeriesError as e:
                    results[series] = e
                log["s_process"] += time.perf_counter() - t
                del files
            break
        except ArchiveError as e:
            log["error"] = f"ArchiveError: {e}"
            print(f"[stream_cache] {study}: attempt {attempt}/{attempts}: {e}", flush=True)
    if log["error"] is not None:
        return log, []
    del buf
    log["n_series_error"] = sum(isinstance(r, SeriesError) for r in results.values())
    t = time.perf_counter()
    manifest = write_study(root, study, results)
    log["s_write"] = time.perf_counter() - t
    return log, manifest


_WORKER: dict = {}


def _init_worker(reader: ArchiveReader, root: Path, attempts: int) -> None:
    """Pool initializer: nice 19, one cv2 thread, the shared reader/root/attempts."""
    os.nice(19 - os.nice(0))  # the shared machine's rule for CPU work
    cv2.setNumThreads(1)
    _WORKER.update(reader=reader, root=root, attempts=attempts)


def _run_task(task: tuple[str, pd.DataFrame]) -> tuple[dict, list[dict]]:
    """Pool task: build_study on (study, its index rows)."""
    study, rows = task
    return build_study(_WORKER["reader"], _WORKER["root"], study, rows, _WORKER["attempts"])


def build_studies(reader: ArchiveReader, root: Path, study_rows: dict[str, pd.DataFrame],
                  n_workers: int, attempts: int, max_tasks_per_child: int,
                  progress_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run build_study over a fork pool (largest studies first) → (run_log, manifest) DataFrames."""
    sizes = {}
    for study, rows in study_rows.items():  # fail on a non-contiguous study before any work
        start, end = member_range(rows)
        sizes[study] = end - start
    tasks = sorted(study_rows.items(), key=lambda kv: -sizes[kv[0]])
    total_bytes = sum(sizes.values())
    logs, manifest = [], []
    t0 = time.time()
    done_bytes = 0
    with open(progress_path, "a", encoding="utf-8") as progress, \
            mp.get_context("fork").Pool(n_workers, initializer=_init_worker,
                                        initargs=(reader, Path(root), attempts),
                                        maxtasksperchild=max_tasks_per_child) as pool:
        for log, written in pool.imap_unordered(_run_task, tasks, chunksize=1):
            logs.append(log)
            manifest.extend(written)
            progress.write(json.dumps({"log": log, "manifest": written}, default=str) + "\n")
            progress.flush()
            done_bytes += log["compressed_bytes"]
            wall = time.time() - t0
            rate = done_bytes / wall
            failed = sum(lg["error"] is not None for lg in logs)
            print(f"[stream_cache] {len(logs)}/{len(tasks)} studies, {failed} failed | "
                  f"{done_bytes / 1e9:.1f}/{total_bytes / 1e9:.1f} GB, {rate / 1e6:.0f} MB/s | "
                  f"eta {(total_bytes - done_bytes) / max(rate, 1) / 60:.0f} min", flush=True)
    odd = [m for m in manifest if sorted(m) != sorted(MANIFEST_COLUMNS)]
    if odd:
        raise RuntimeError(f"write_study returned manifest rows with other keys: {odd[0]}")
    return (pd.DataFrame(logs, columns=RUN_LOG_COLUMNS),
            pd.DataFrame(manifest, columns=MANIFEST_COLUMNS))
