"""Repro of the fork-pool hang in build_dir_cache (orch-0005, 17.10) and proof that the spawn pool and the
stage watchdog fix it. CPU only.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/hang_repro.py

1. The parent runs threaded cv2 (CV2_THREADS threads), as a member's ROI code does in the blend kernel.
2. fork pools from that parent, one trivial task each, FORK_WAIT_S to answer:
   (a) initializer = the old build_dir_cache one (_init_worker: nice + cv2.setNumThreads(1)) -> expected hung;
   (b) initializer = nice only (control) -> expected ok.
3. build_dir_cache (spawn pool) on the visible test_series from the same parent: every series must match the
   parity reference fingerprints of REF_RUN byte for byte, and the children must not re-run this script.
4. A child python runs kernel_env.stage("demo", 3) around a 60 s sleep: it must exit with code 1 after ~3 s
   with faulthandler's "Timeout" traceback.
Every step runs under kernel_env.stage, so a hang of the repro itself dies loudly too. Writes hang_repro.json.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.infer25d import _init_worker, build_dir_cache  # noqa: E402
from src.kernel_env import stage, volume_fingerprint  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

REF_RUN = "20261017t120738-561-0468"  # parity_ref.py: fingerprints of the visible test series
CV2_THREADS = 8
FORK_WAIT_S = 60
WORKERS = 2


def _nice_only():
    os.nice(19 - os.nice(0))


ctx = get_ctx()
comp = ctx.mounts["rsna-knee-abnormality-detection"]
ref = [r for r in json.loads((ctx.mounts[REF_RUN] / "parity_ref.json").read_text())["series"] if r["split"] == "test"]
out = {"cv2": cv2.__version__, "cv2_threads_parent": None}

with stage("parent threaded cv2", 300):
    cv2.setNumThreads(CV2_THREADS)
    a = np.random.default_rng(0).integers(0, 256, (4096, 4096), dtype=np.uint8)
    for _ in range(20):
        cv2.resize(a, (3000, 3000), interpolation=cv2.INTER_LINEAR)
    out["cv2_threads_parent"] = cv2.getNumThreads()


def fork_probe(initializer, initargs):
    pool = mp.get_context("fork").Pool(WORKERS, initializer=initializer, initargs=initargs)
    t0 = time.time()
    try:
        pool.apply_async(os.getpid).get(timeout=FORK_WAIT_S)
        res = f"ok in {time.time() - t0:.1f}s"
    except mp.TimeoutError:
        res = f"hung (no answer in {FORK_WAIT_S}s)"
    pool.terminate()
    pool.join()
    return res


with stage("fork probes", 2 * FORK_WAIT_S + 120):
    out["fork_with_cv2_setNumThreads"] = fork_probe(_init_worker, (ctx.output_dir / "cache_fork",))
    out["fork_nice_only_control"] = fork_probe(_nice_only, ())
print(json.dumps(out), flush=True)

with stage("spawn build_dir_cache", 900):
    runs_before = set(os.listdir(ctx.output_dir.parent))
    t0 = time.time()
    cache = build_dir_cache(comp / "test_series", comp / "test_series.csv", ctx.output_dir / "cache_spawn", WORKERS)
    sec = time.time() - t0
    new_runs = sorted(set(os.listdir(ctx.output_dir.parent)) - runs_before)
    match = []
    for r in ref:
        vol, pos, meta = cache.load(r["series"], mmap=False)
        match.append(volume_fingerprint(vol, pos, meta) == r["fp"])
    out["spawn"] = {"seconds": round(sec, 1), "series": len(ref), "byte_identical": int(sum(match)),
                    "new_run_dirs_during_build": new_runs}

with stage("watchdog demo", 120):
    code = ("import sys, time; sys.path.insert(0, %r); from src.kernel_env import stage\n"
            "with stage('demo', 3):\n    time.sleep(60)\n" % str(ctx.sources_dir))
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    out["watchdog"] = {"returncode": p.returncode, "seconds": round(time.time() - t0, 1),
                       "timeout_traceback": "Timeout" in p.stderr, "stdout": p.stdout.strip().splitlines(),
                       "stderr_head": p.stderr.strip().splitlines()[:3]}

(ctx.output_dir / "hang_repro.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)
ok = (out["fork_with_cv2_setNumThreads"].startswith("hung") and out["fork_nice_only_control"].startswith("ok")
      and out["spawn"]["byte_identical"] == len(ref) and not out["spawn"]["new_run_dirs_during_build"]
      and out["watchdog"]["returncode"] == 1 and out["watchdog"]["timeout_traceback"])
print(f"[hang_repro] {'PASS' if ok else 'FAIL'} {ctx.output_dir}")
