"""Blend kernel (exps/exp-0004/blend_contracts.md, B1-B6): our part in a child process, the public pipeline, the rank blend.

blend_main (B6) is the kernel body. The parent runs our part first in a fresh interpreter (own_part_subprocess ->
own_part_main): our pinned wheels first on its PYTHONPATH, the test cache built once, parity, every member's
probabilities, the member check, <out_dir>/<member>_prob.csv and <out_dir>/report.json. Then the public
pipeline, then rank_blend mixes our members into its submission; with cfg own_only the public pipeline is skipped
and rank_own makes the submission from our members alone. Module-level imports are stdlib only: the
parent imports this module before the public pipeline installs its wheels, and numpy/pandas/torch/pydicom are
imported by the functions that use them.
"""
import contextlib
import gc
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SPEC_KEYS = {"series_root", "series_csv", "cache_dir", "out_dir", "device", "workers", "chunk_studies", "budgets",
             "members", "member_gpus", "parity_expected", "member_expected", "member_tol"}
MEMBER_KEYS = {"name", "module", "ckpt_paths"}
MEMBER_POLL_S = 0.2  # parallel members: how often own_part_main checks its member children
LOG_TAIL_LINES = 40  # of a failed member child's log, in the raised error
PR_SET_PDEATHSIG = 1  # linux/prctl.h
# budgets keys besides one "member:<name>" per member; "cache" and "member:<name>" are per chunk, the rest one-off
STEPS = ("env", "cache", "parity", "member_check")
REPORT_NAME = "report.json"
RANGE_EPS = 1e-9  # float rounding of the weighted rank sum above 1
CFG_KEYS = {"visible_studies", "wheels", "pins", "site_dir", "own_spec", "own_budget_s", "weights", "ckpt_sha256",
            "n_hidden_studies", "hidden_fallback", "own_only"}
OWN_ONLY_KEYS = {"test_csv", "sample_submission"}
# own-only (lead 18.10): a member that cannot use more than this share of the test studies raises instead of
# giving them rank 0.5 (in the hidden rerun a submission error, not a silently degraded score)
MAX_UNUSABLE_FRAC = 0.01
RECEIPT_NAME = "exp0004_blend_receipt.json"  # run_public writes its own blend_receipt.json into the same dir


def _kill_group(pgid):
    """SIGKILL every process of process group `pgid`; True if the group still had any."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return True


def own_part_subprocess(spec, spec_path, sources_dir, site_dir, budget_s):
    """B1: own_part_main(spec) in a fresh interpreter (PYTHONPATH site_dir:sources_dir, own session) -> its report.json.

    Writes the JSON-serialisable `spec` to spec_path; the child's stdout/stderr go to this process's (the kernel
    log). Past budget_s the whole process group (the child and its pool / loader workers) is killed and
    TimeoutError raised; a non-zero exit kills what is left of the group and raises RuntimeError. The wheels
    are the caller's job: kernel_env.install_wheels(..., target=site_dir) first; site_dir None = sources only.
    """
    if not budget_s > 0:
        raise ValueError(f"budget_s must be > 0, got {budget_s}")
    spec_path = Path(spec_path)
    report_path = Path(spec["out_dir"]) / REPORT_NAME
    if report_path.exists():
        raise FileExistsError(f"{report_path} exists before the child ran: it would be returned as this run's report")
    spec_path.write_text(json.dumps(spec, indent=1))
    path = [str(sources_dir)] if site_dir is None else [str(site_dir), str(sources_dir)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path)}
    code = f"from src.blend_kernel import own_part_main; own_part_main({str(spec_path)!r})"
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-u", "-c", code], env=env, start_new_session=True)
    print(f"[own] child pid {proc.pid} = its process group, PYTHONPATH {env['PYTHONPATH']}, budget {budget_s:.0f}s",
          flush=True)
    try:
        rc = proc.wait(timeout=budget_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        proc.wait()
        raise TimeoutError(f"own part exceeded its budget of {budget_s:.0f}s: process group {proc.pid} killed") from None
    left = _kill_group(proc.pid)
    if rc != 0:
        raise RuntimeError(f"own part exited with code {rc} after {time.time() - t0:.0f}s (its traceback is in the "
                           f"log above){'; killed processes left in its group' if left else ''}")
    if left:
        print(f"[own] child exited 0 but left processes in group {proc.pid}: killed them", flush=True)
    return json.loads(report_path.read_text())


def _check_spec(spec):
    """Raises unless the spec has exactly SPEC_KEYS, well-formed unique members and one budget / reference per member."""
    if set(spec) != SPEC_KEYS:
        raise ValueError(f"spec keys: missing {sorted(SPEC_KEYS - set(spec))}, unexpected {sorted(set(spec) - SPEC_KEYS)}")
    if not spec["members"]:
        raise ValueError("spec has no members")
    for m in spec["members"]:
        if set(m) != MEMBER_KEYS:
            raise ValueError(f"member {m}: keys must be {sorted(MEMBER_KEYS)}")
        if not m["name"] or "/" in m["name"] or not m["ckpt_paths"]:
            raise ValueError(f"member {m}: needs a file-name-safe name and at least one checkpoint")
    names = [m["name"] for m in spec["members"]]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate member names: {names}")
    want = set(STEPS) | {f"member:{n}" for n in names}
    if set(spec["budgets"]) != want:
        raise ValueError(f"budgets keys: missing {sorted(want - set(spec['budgets']))}, "
                         f"unexpected {sorted(set(spec['budgets']) - want)}")
    if set(spec["member_expected"]) != set(names):
        raise ValueError(f"member_expected must name exactly the members {names}, got {sorted(spec['member_expected'])}")
    if isinstance(spec["chunk_studies"], bool) or not isinstance(spec["chunk_studies"], int) or spec["chunk_studies"] < 1:
        raise ValueError(f"chunk_studies must be an int > 0, got {spec['chunk_studies']!r}")
    series = [e["series"] for e in spec["parity_expected"]]
    if not series or len(set(series)) != len(series):
        raise ValueError(f"parity_expected must name at least one series, each once ({len(series)} rows)")
    gpus = spec["member_gpus"]
    if gpus is not None:
        if not isinstance(gpus, dict) or set(gpus) != set(names):
            raise ValueError(f"member_gpus must be null (sequential) or name exactly the members {names}, got {gpus!r}")
        if not all(type(g) is int and g >= 0 for g in gpus.values()) or len(set(gpus.values())) != len(gpus):
            raise ValueError(f"member_gpus must give every member its own GPU index (distinct ints >= 0), got {gpus}")
        if spec["device"] not in ("cuda", "cpu"):
            raise ValueError(f"member_gpus needs device 'cuda' (each member child sees its one GPU as cuda) or 'cpu' "
                             f"(smoke: the children see no GPU), got {spec['device']!r}")


@contextlib.contextmanager
def _step(name, budget_s, seconds):
    """kernel_env.stage 'own:<name>' with budget_s; records the step's seconds under `name`."""
    from src.kernel_env import stage

    t0 = time.time()
    with stage(f"own:{name}", budget_s):
        yield
    seconds[name] = round(time.time() - t0, 1)


def _cuda_torch():
    """torch if it is imported and has initialised CUDA in this process, else None (nothing to free or measure)."""
    torch = sys.modules.get("torch")
    return torch if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized() else None


def _member_gpu_release():
    """After a member: its peak reserved CUDA memory in GB (None without CUDA), then gc + empty_cache, so the next
    member starts from an empty allocator pool instead of this one's reserve (a T4 has 15 GB; exp-0004 alone
    reserved ~14 GB in the timing kernel 20261018t031157-487-5678); peak stats reset for the next member."""
    torch = _cuda_torch()
    if torch is None:
        return None
    peak = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return peak


def _dir_bytes(path):
    """Total size in bytes of the files under `path` (0 if it does not exist)."""
    path = Path(path)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def csv_studies(comp_series):
    """Studies of a competition *_series.csv DataFrame in order of first appearance: the order own_part_main predicts."""
    return list(dict.fromkeys(comp_series.StudyInstanceUID))


def _check_member_probs(name, p, chunk, labels):
    """Raises unless a member's predict_member result has index = the chunk's studies in order and columns = labels."""
    if list(p.index) != chunk or list(p.columns) != labels:
        raise ValueError(f"member {name}: predict_member must return index = the {len(chunk)} studies "
                         f"in order and columns = LABELS, got {p.shape} {list(p.columns)}")


def _member_gpu_env(member_gpus, device):
    """{member: CUDA_VISIBLE_DEVICES for its child}: its GPU among this process's visible ones; "" on cpu (no GPU)."""
    if device == "cpu":
        return {n: "" for n in member_gpus}
    import torch

    n_visible = torch.cuda.device_count()
    ids = ([s.strip() for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if s.strip()]
           if "CUDA_VISIBLE_DEVICES" in os.environ else [str(k) for k in range(n_visible)])
    if len(ids) != n_visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']!r} lists {len(ids)} GPUs, torch "
                           f"sees {n_visible}")
    over = {n: g for n, g in member_gpus.items() if g >= n_visible}
    if over:
        raise ValueError(f"member_gpus {over}: only {n_visible} GPUs are visible")
    return {n: ids[g] for n, g in member_gpus.items()}


def _die_with_parent(parent_pid):
    """prctl(PR_SET_PDEATHSIG, SIGKILL): the OS kills this process when its parent dies; raises if it already has."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG, SIGKILL) failed")
    if os.getppid() != parent_pid:
        raise RuntimeError(f"parent {parent_pid} died before PR_SET_PDEATHSIG was set (ppid now {os.getppid()})")


def member_child_main(job_path):
    """Parallel B2's member child: predict_member on the chunk's re-opened KneeVolCache -> its prob pickle + stats json.

    Started by _run_members_parallel in its own session with CUDA_VISIBLE_DEVICES = its GPU (the member sees it
    as cuda) and stdout/stderr in its log file; dies with own_part_main (PR_SET_PDEATHSIG), so a killed own part
    leaves no member on a GPU. The job json: name, module, ckpt_paths, device, studies, chunk_dir, rows_pkl (the
    chunk's series rows as own_part_main has them), prob_pkl, stats_json (peak reserved CUDA GB), parent_pid.
    """
    job = json.loads(Path(job_path).read_text())
    _die_with_parent(job["parent_pid"])
    import pandas as pd

    from src.vol_cache import KneeVolCache

    cache, rows = KneeVolCache(job["chunk_dir"]), pd.read_pickle(job["rows_pkl"])
    p = importlib.import_module(job["module"]).predict_member(job["studies"], cache, rows, job["ckpt_paths"],
                                                              job["device"])
    p.to_pickle(job["prob_pkl"])  # lossless: the parent's probabilities are the child's bits
    torch = _cuda_torch()
    peak = None if torch is None else round(torch.cuda.max_memory_reserved() / 1e9, 2)
    Path(job["stats_json"]).write_text(json.dumps({"gpu_peak_reserved_gb": peak}))


def _member_log(chunk_dir, name):
    """Path of member `name`'s child log in a chunk dir."""
    return Path(chunk_dir) / f"{name}.log"


def _run_members_parallel(i, chunk, chunk_dir, rows, spec, gpu_env, seconds):
    """Parallel B2, one chunk: every member at once in its own child (member_child_main) -> ({name: probs}, {name: GB}).

    Each child: a fresh interpreter in its own session (killed with its process group), env CUDA_VISIBLE_DEVICES
    gpu_env[name], device spec["device"], stdout/stderr -> chunk_dir/<name>.log. Past budgets["member:<name>"]
    TimeoutError names the member and chunk; a non-zero exit raises RuntimeError with its log's tail; either way
    every child's group is killed. Then (also on failure) the logs are printed in member order, each line
    prefixed "[member <name> chunk <i>] ". seconds: "member:<name>[i]" = that child's wall (launch -> exit seen),
    "members[i]" = launch -> all done. GB: each child's peak reserved CUDA memory (None on cpu).
    """
    import pandas as pd

    names, budgets = [m["name"] for m in spec["members"]], spec["budgets"]
    rows_pkl = chunk_dir / "chunk_rows.pkl"
    rows.to_pickle(rows_pkl)
    print(f"[members] chunk {i}: {names} in parallel, CUDA_VISIBLE_DEVICES {gpu_env}", flush=True)
    procs, t_start, walls = {}, {}, {}
    t0 = time.time()
    try:
        for m in spec["members"]:
            name = m["name"]
            job = {"name": name, "module": m["module"], "ckpt_paths": m["ckpt_paths"], "device": spec["device"],
                   "studies": chunk, "chunk_dir": str(chunk_dir), "rows_pkl": str(rows_pkl),
                   "prob_pkl": str(chunk_dir / f"{name}_prob.pkl"), "stats_json": str(chunk_dir / f"{name}_stats.json"),
                   "parent_pid": os.getpid()}
            job_path = chunk_dir / f"{name}_job.json"
            job_path.write_text(json.dumps(job))
            code = f"from src.blend_kernel import member_child_main; member_child_main({str(job_path)!r})"
            with open(_member_log(chunk_dir, name), "wb") as log:
                procs[name] = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=log, stderr=subprocess.STDOUT,
                                               env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_env[name]},
                                               start_new_session=True)
            t_start[name] = time.time()
        while True:
            for name, proc in procs.items():
                if name in walls:
                    continue
                rc, elapsed = proc.poll(), time.time() - t_start[name]
                if rc is None:
                    if elapsed > budgets[f"member:{name}"]:
                        raise TimeoutError(f"member {name} chunk {i} exceeded its budget of "
                                           f"{budgets[f'member:{name}']:.0f}s: its process group and the other "
                                           "members' are killed")
                    continue
                walls[name] = round(elapsed, 1)
                if rc != 0:
                    tail = _member_log(chunk_dir, name).read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
                    raise RuntimeError(f"member {name} chunk {i} exited with code {rc} after {elapsed:.0f}s (the other "
                                       f"members are killed); its log's last {len(tail)} lines:\n" + "\n".join(tail))
            if len(walls) == len(procs):
                seconds[f"members[{i}]"] = round(time.time() - t0, 1)
                break
            time.sleep(MEMBER_POLL_S)
    finally:
        for proc in procs.values():
            _kill_group(proc.pid)
            proc.wait()
        for name in procs:
            for line in _member_log(chunk_dir, name).read_text(errors="replace").splitlines():
                print(f"[member {name} chunk {i}] {line}")
        sys.stdout.flush()
    probs, peaks = {}, {}
    for name in names:
        seconds[f"member:{name}[{i}]"] = walls[name]
        probs[name] = pd.read_pickle(chunk_dir / f"{name}_prob.pkl")
        peaks[name] = json.loads((chunk_dir / f"{name}_stats.json").read_text())["gpu_peak_reserved_gb"]
    print(f"[members] chunk {i}: done in {seconds[f'members[{i}]']:.0f}s, per member {walls}", flush=True)
    return probs, peaks


def own_part_main(spec_path):
    """B2: the child's whole job: env report; per chunk of studies cache -> members -> rmtree; parity (B4), B5, report.json.

    Spec (JSON; every key required, see SPEC_KEYS): DICOM root + its *_series.csv, cache and output dirs,
    device, cache workers, chunk_studies, budgets {step or "member:<name>": seconds; cache and members per
    chunk}, members [{name, module, ckpt_paths}] (module exposes predict_member(studies, cache, comp_series,
    ckpt_paths, device)), parity_expected [{series, fp}], member_expected {name: {study: [12 probs]}} ({} = no
    reference), member_tol. The studies (csv_studies order) go in consecutive chunks of chunk_studies: each
    chunk gets its own cache from its own series rows, every member predicts it from that cache, then the
    chunk cache is deleted, so the disk peak is one chunk's cache. member_gpus null: the members run in turn
    in this process; {member: gpu}: all at once, each in its own child on its own GPU (_run_members_parallel),
    with the same per-chunk budgets; chunk rows carry members_wall_seconds (the members' wall per chunk) and
    member_gpus either way. `unusable` in the report: studies whose reader study table is empty (no
    error-free series with >= 3 slices); members still predict them, rank_blend masks them.
    """
    from collections import Counter

    import pandas as pd

    from src.infer25d import build_dir_cache
    from src.kernel_env import env_report
    from src.knee_labels import LABELS
    from src.reader25d import cache_study_table

    t_start = time.time()
    spec = json.loads(Path(spec_path).read_text())
    _check_spec(spec)
    out_dir, cache_root = Path(spec["out_dir"]), Path(spec["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets, seconds = spec["budgets"], {}
    names = [m["name"] for m in spec["members"]]

    with _step("env", budgets["env"], seconds):
        env = env_report()
        print("OWN_ENV " + json.dumps(env), flush=True)

    comp_series = pd.read_csv(spec["series_csv"])
    studies = csv_studies(comp_series)
    if not studies:
        raise ValueError(f"{spec['series_csv']} has no studies")
    size = spec["chunk_studies"]
    chunks = [studies[i:i + size] for i in range(0, len(studies), size)]
    gpus = spec["member_gpus"]
    gpu_env = None if gpus is None else _member_gpu_env(gpus, spec["device"])
    expected_series = [e["series"] for e in spec["parity_expected"]]
    fps, unusable, errors, parts, chunk_rows = {}, [], Counter(), {n: [] for n in names}, []
    for i, chunk in enumerate(chunks, 1):
        chunk_dir = cache_root / f"chunk{i}"
        gpu_peak = {}
        rows = comp_series[comp_series.StudyInstanceUID.isin(chunk)].reset_index(drop=True)  # original row order
        with _step(f"cache[{i}]", budgets["cache"], seconds):
            chunk_dir.mkdir(parents=True, exist_ok=False)
            chunk_csv = chunk_dir / "chunk_series.csv"
            rows.to_csv(chunk_csv, index=False)
            cache = build_dir_cache(spec["series_root"], chunk_csv, chunk_dir, spec["workers"])
            fps.update(parity_fingerprints(cache, expected_series))
            table = cache_study_table(cache.series, rows, chunk)
            unusable += [st for st in chunk if not any(table[st])]
            errors.update(cache.series.error_code.dropna())
        n_bytes = _dir_bytes(chunk_dir)  # before the members: parallel ones write their job, log and prob files here
        if gpus is None:
            t_members = time.time()
            for m in spec["members"]:
                name = m["name"]
                with _step(f"member:{name}[{i}]", budgets[f"member:{name}"], seconds):
                    p = importlib.import_module(m["module"]).predict_member(chunk, cache, rows, m["ckpt_paths"],
                                                                            spec["device"])
                    _check_member_probs(name, p, chunk, LABELS)
                parts[name].append(p)
                gpu_peak[name] = _member_gpu_release()
            seconds[f"members[{i}]"] = round(time.time() - t_members, 1)
        else:
            got, gpu_peak = _run_members_parallel(i, chunk, chunk_dir, rows, spec, gpu_env, seconds)
            for name in names:
                _check_member_probs(name, got[name], chunk, LABELS)
                parts[name].append(got[name])
        del cache
        shutil.rmtree(chunk_dir)
        if chunk_dir.exists():
            raise RuntimeError(f"{chunk_dir} still exists after rmtree")
        row = {"chunk": i, "studies": len(chunk), "series": len(rows), "cache_seconds": seconds[f"cache[{i}]"],
               "cache_bytes": n_bytes, "member_seconds": {n: seconds[f"member:{n}[{i}]"] for n in names},
               "members_wall_seconds": seconds[f"members[{i}]"], "member_gpus": gpus,
               "member_gpu_peak_reserved_gb": gpu_peak,
               "fs_free_gb": round(shutil.disk_usage(cache_root).free / 1e9, 2), "removed": True}
        chunk_rows.append(row)
        print(f"OWN_CHUNK {i}/{len(chunks)} " + json.dumps(row), flush=True)

    with _step("parity", budgets["parity"], seconds):
        parity = judge_parity(spec["parity_expected"], fps)
        print("OWN_PARITY " + json.dumps(parity), flush=True)
    probs, members = {}, {}
    for name in names:
        p = pd.concat(parts[name])
        if list(p.index) != studies:
            raise ValueError(f"member {name}: concatenated chunks are not the {len(studies)} studies in order")
        path = out_dir / f"{name}_prob.csv"
        p.to_csv(path)
        probs[name] = p
        member_s = sum(r["member_seconds"][name] for r in chunk_rows)
        members[name] = {"prob_csv": str(path), "rows": len(p), "nan": int(p.isna().to_numpy().sum()),
                         "ckpts": len(next(m for m in spec["members"] if m["name"] == name)["ckpt_paths"]),
                         "seconds": round(member_s, 1), "s_per_study": round(member_s / len(p), 3)}
        print(f"OWN_MEMBER {name} " + json.dumps(members[name]), flush=True)
    with _step("member_check", budgets["member_check"], seconds):
        check = member_check(probs, spec["member_expected"], spec["member_tol"])
        print("OWN_MEMBER_CHECK " + json.dumps(check), flush=True)

    cache_s = sum(r["cache_seconds"] for r in chunk_rows)
    cache_bytes = sum(r["cache_bytes"] for r in chunk_rows)
    report = {"env": env, "studies": len(studies), "series": len(comp_series), "chunk_studies": size,
              "n_chunks": len(chunks), "chunks": chunk_rows,
              "cache": {"dir": str(cache_root), "seconds": round(cache_s, 1),
                        "s_per_series": round(cache_s / len(comp_series), 3), "bytes": cache_bytes,
                        "bytes_per_series": round(cache_bytes / len(comp_series)), "series_errors": sum(errors.values()),
                        "error_counts": dict(errors)},
              "cache_peak_bytes": max(r["cache_bytes"] for r in chunk_rows),
              "unusable": unusable, "parity": parity, "members": members, "member_check": check,
              "seconds": seconds, "total_seconds": round(time.time() - t_start, 1)}
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=1))
    print("OWN_REPORT " + json.dumps({k: v for k, v in report.items() if k not in ("env", "chunks")}), flush=True)


def _pct(v):
    """Percentile rank (ties averaged, pandas rank pct=True) of a 1-d array (exp-0001's blend_stack pct)."""
    import numpy as np
    import pandas as pd

    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).values


def _pct_on(v, ok):
    """Percentile rank among the rows in `ok` only; the other rows are NaN (exp-0001's blend_stack pct_on)."""
    import numpy as np

    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = _pct(np.asarray(v, dtype=np.float64)[ok])
    return r


def rank_blend(public, own, weights, unusable, labels):
    """B3: per target w_pub * rank(public) + sum_m w_m * rank_m; a study member m cannot score gives w_m to public.

    public: DataFrame StudyInstanceUID + labels (the public pipeline's final submission); own: {name: DataFrame
    indexed by StudyInstanceUID, columns labels}; weights: {name: w > 0}, sum < 1, w_pub = 1 - sum; unusable:
    {name: set of study ids}. Ranks: percentile, ties averaged; public over all studies, each member over its
    usable studies only. Returns (DataFrame with public's rows and column order, values in (0, 1], receipt).
    """
    import numpy as np
    import pandas as pd

    labels = list(labels)
    if sorted(public.columns) != sorted(["StudyInstanceUID", *labels]):
        raise ValueError(f"public columns {list(public.columns)} != StudyInstanceUID + {labels}")
    ids = public.StudyInstanceUID.tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("public has duplicate StudyInstanceUID")
    if public[labels].isna().to_numpy().any():
        raise ValueError("public has NaN")
    if not own:
        raise ValueError("no own members")
    if not set(own) == set(weights) == set(unusable):
        raise ValueError(f"own {sorted(own)}, weights {sorted(weights)} and unusable {sorted(unusable)} must name "
                         "the same members")
    if not all(np.isfinite(w) and w > 0 for w in weights.values()) or not sum(weights.values()) < 1:
        raise ValueError(f"weights must be > 0 with sum < 1, got {weights}")
    w_pub = 1.0 - sum(weights.values())
    frames, usable = _member_frames(own, ids, labels, unusable, "public")

    out = np.zeros((len(ids), len(labels)))
    rho = {name: {} for name in own}
    for j, c in enumerate(labels):
        r_pub = _pct(public[c].values)
        b = w_pub * r_pub
        for name, w in weights.items():
            ok = usable[name]
            b = b + w * np.where(ok, _pct_on(frames[name][c].values, ok), r_pub)
            r = pd.Series(public[c].values[ok]).corr(pd.Series(frames[name][c].values[ok]), method="spearman")
            rho[name][c] = round(float(r), 4) if np.isfinite(r) else None
        out[:, j] = b
    if not ((out > 0) & (out <= 1 + RANGE_EPS)).all():
        raise ValueError(f"blended ranks outside (0, 1]: min {out.min()}, max {out.max()}")
    blend = pd.DataFrame(np.minimum(out, 1.0), columns=labels)
    blend.insert(0, "StudyInstanceUID", ids)
    blend = blend[list(public.columns)]
    n_unusable = {name: int((~ok).sum()) for name, ok in usable.items()}
    if any(n_unusable.values()):
        print(f"OWN_UNUSABLE {json.dumps(n_unusable)} of {len(ids)} studies: those members' weight went to public",
              flush=True)
    receipt = {"rows": len(ids), "weights": {"public": w_pub, **weights}, "unusable": n_unusable,
               "spearman_vs_public": rho,
               "rank_basis": "percentile ranks, ties averaged: public over all studies, each member over its usable "
                             "studies; an unusable study's member weight goes to public"}
    return blend, receipt


def _member_frames(own, ids, labels, unusable, ref):
    """({name: member probs at rows ids, columns labels}, {name: bool mask of its usable rows}); raises on a mismatch with `ref`."""
    import numpy as np

    frames, usable = {}, {}
    for name, df in own.items():
        if df.index.duplicated().any() or set(df.index) != set(ids):
            raise ValueError(f"member {name}: studies differ from {ref}'s ({len(df)} rows, {len(ids)} in {ref})")
        if sorted(df.columns) != sorted(labels):
            raise ValueError(f"member {name}: columns {list(df.columns)} != {labels}")
        frames[name] = df.loc[ids, labels]
        if frames[name].isna().to_numpy().any():
            raise ValueError(f"member {name} has NaN")
        extra = set(unusable[name]) - set(ids)
        if extra:
            raise ValueError(f"member {name}: {len(extra)} unusable studies are not in {ref}, e.g. {next(iter(extra))}")
        usable[name] = np.array([u not in unusable[name] for u in ids])
    return frames, usable


def rank_own(ids, cols, own, weights, unusable, max_unusable_frac):
    """Own-only submission: per target sum_m w_m * rank_m over our members alone (no public); weights sum to 1.

    ids: the submission's StudyInstanceUID order (test.csv); cols: its columns (sample_submission.csv:
    StudyInstanceUID + labels); own, unusable as rank_blend. Ranks: percentile, ties averaged, each member over its
    usable studies; a study member m cannot score gets rank 0.5 from m (there is no public to hand its weight to).
    A member with more than max_unusable_frac of the studies unusable raises (lead 18.10). Returns (DataFrame:
    columns cols, rows ids, values in (0, 1]; receipt).
    """
    import numpy as np
    import pandas as pd

    cols = list(cols)
    if cols[0] != "StudyInstanceUID" or len(set(cols)) != len(cols):
        raise ValueError(f"submission columns must be StudyInstanceUID + unique labels, got {cols}")
    labels = cols[1:]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"ids must be non-empty and unique ({len(ids)} given, {len(set(ids))} unique)")
    if not own:
        raise ValueError("no own members")
    if not set(own) == set(weights) == set(unusable):
        raise ValueError(f"own {sorted(own)}, weights {sorted(weights)} and unusable {sorted(unusable)} must name "
                         "the same members")
    if not all(np.isfinite(w) and 0 < w <= 1 for w in weights.values()) or abs(sum(weights.values()) - 1) > RANGE_EPS:
        raise ValueError(f"own-only weights must be in (0, 1] and sum to 1, got {weights}")
    frames, usable = _member_frames(own, ids, labels, unusable, "test.csv")
    n_unusable = {name: int((~ok).sum()) for name, ok in usable.items()}
    over = {name: n for name, n in n_unusable.items() if n > max_unusable_frac * len(ids)}
    if over:
        raise RuntimeError(f"own-only: unusable studies {over} of {len(ids)} exceed {max_unusable_frac:.0%} in a member: "
                           "rank 0.5 for that many would degrade the score silently")

    out = np.zeros((len(ids), len(labels)))
    for j, c in enumerate(labels):
        for name, w in weights.items():
            ok = usable[name]
            out[:, j] += w * np.where(ok, _pct_on(frames[name][c].values, ok), 0.5)
    if not ((out > 0) & (out <= 1 + RANGE_EPS)).all():
        raise ValueError(f"own-only ranks outside (0, 1]: min {out.min()}, max {out.max()}")
    sub = pd.DataFrame(np.minimum(out, 1.0), columns=labels)
    sub.insert(0, "StudyInstanceUID", ids)
    if any(n_unusable.values()):
        print(f"OWN_UNUSABLE {json.dumps(n_unusable)} of {len(ids)} studies: rank 0.5 from those members", flush=True)
    names = list(own)
    rho = {f"{a}~{b}": round(float(np.mean([pd.Series(frames[a][c].values).corr(pd.Series(frames[b][c].values),
                                                                                  method="spearman")
                                              for c in labels])), 4)
           for i, a in enumerate(names) for b in names[i + 1:]} if len(ids) > 2 else {}
    receipt = {"rows": len(ids), "weights": dict(weights), "unusable": n_unusable,
               "max_unusable_frac": max_unusable_frac, "spearman_between_members_mean": rho,
               "rank_basis": "percentile ranks, ties averaged, each member over its usable studies; an unusable "
                             "study gets rank 0.5 from that member (own-only: no public)"}
    return sub, receipt


def parity_fingerprints(cache, series):
    """{series: volume_fingerprint} of those `series` that are in a KneeVolCache; a failed one -> {"error": its code}."""
    from src.kernel_env import volume_fingerprint
    from src.knee_volume import SeriesError

    present = set(cache.series.series)
    got = {}
    for s in series:
        if s in present:
            try:
                got[s] = volume_fingerprint(*cache.load(s, mmap=False))
            except SeriesError as err:
                got[s] = {"error": err.code}
    return got


def judge_parity(expected, got):
    """B4's verdict on expected [{series, fp}] given the fingerprints `got` {series: fp} of those seen; mismatch raises.

    None of them seen (hidden rerun) -> {"status": "absent", "n_expected": n}; only some -> raises; all seen
    and equal (parity_ref's format: a failed series is {"error": code}) -> {"status": "ok", "n": n}.
    """
    series = [e["series"] for e in expected]
    if not series:
        raise ValueError("no expected series")
    if len(set(series)) != len(series):
        raise ValueError("expected series contain duplicates")
    if set(got) - set(series):
        raise ValueError(f"fingerprints of {len(set(got) - set(series))} series that are not expected")
    missing = [s for s in series if s not in got]
    if len(missing) == len(series):
        return {"status": "absent", "n_expected": len(series)}
    if missing:
        raise ValueError(f"{len(missing)} of {len(series)} expected series are not in the cache, e.g. {missing[0]}: "
                         "the visible test is either complete or absent")
    bad = [{"series": e["series"], "want": e["fp"], "got": got[e["series"]]} for e in expected
           if got[e["series"]] != e["fp"]]
    if bad:
        raise RuntimeError(f"PARITY FAILED: {len(bad)}/{len(series)} series differ from the reference: {bad}")
    return {"status": "ok", "n": len(series)}


def visible_parity(cache, expected):
    """B4 on one KneeVolCache: judge_parity of the expected series' parity_fingerprints in it."""
    return judge_parity(expected, parity_fingerprints(cache, [e["series"] for e in expected]))


def member_check(probs, expected, tol):
    """B5: per member max |prob - reference| over the reference studies; any member above tol raises.

    probs: {name: DataFrame index StudyInstanceUID, columns LABELS}; expected: {name: {study: [12 probs in
    LABELS order]}}. Per member: empty reference -> "no reference"; none of its studies in probs (hidden
    rerun) -> "absent"; only some -> raises; else {"status": "ok", "n", "max_abs_diff"}.
    """
    import numpy as np

    from src.knee_labels import LABELS

    if set(probs) != set(expected):
        raise ValueError(f"members with probabilities {sorted(probs)} != members with a reference entry {sorted(expected)}")
    if not tol >= 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    out = {}
    for name, ref in expected.items():
        if not ref:
            out[name] = {"status": "no reference"}
            continue
        p = probs[name]
        missing = [s for s in ref if s not in p.index]
        if len(missing) == len(ref):
            out[name] = {"status": "absent", "n_expected": len(ref)}
            continue
        if missing:
            raise ValueError(f"member {name}: {len(missing)} of {len(ref)} reference studies have no probabilities, "
                             f"e.g. {missing[0]}")
        want = np.array([ref[s] for s in ref], dtype=np.float64)
        got = p.loc[list(ref), LABELS].to_numpy(dtype=np.float64)
        if want.shape != got.shape:
            raise ValueError(f"member {name}: reference shape {want.shape} != probabilities {got.shape}")
        if not np.isfinite(got).all():
            raise ValueError(f"member {name}: non-finite probabilities on reference studies")
        out[name] = {"status": "ok", "n": len(ref), "max_abs_diff": float(np.abs(got - want).max())}
    bad = {n: r for n, r in out.items() if r["status"] == "ok" and r["max_abs_diff"] > tol}
    if bad:
        raise RuntimeError(f"MEMBER CHECK FAILED (tol {tol}): {bad}")
    return out


def _file_sha256(path):
    """sha256 hex digest of a file, read in 16 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _disk_report(tag, path):
    """Prints (EXP0004_DISK) and returns free / total GB of the filesystem holding `path` and the size of `path`."""
    path = Path(path)
    du = shutil.disk_usage(next(p for p in (path, *path.parents) if p.exists()))
    size = _dir_bytes(path)
    rec = {"tag": tag, "path": str(path), "fs_free_gb": round(du.free / 1e9, 2), "fs_total_gb": round(du.total / 1e9, 2),
           "size_gb": round(size / 1e9, 4), "size_bytes": size}
    print("EXP0004_DISK " + json.dumps(rec), flush=True)
    return rec


def _check_submission(sub, final_sub):
    """Raises unless `sub` has final_sub's columns, studies and order, no NaN and values in [0, 1]."""
    import numpy as np

    if list(sub.columns) != list(final_sub.columns):
        raise ValueError(f"submission columns {list(sub.columns)} != the public's {list(final_sub.columns)}")
    if sub.StudyInstanceUID.tolist() != final_sub.StudyInstanceUID.tolist():
        raise ValueError("submission studies or their order differ from the public's")
    v = sub.drop(columns="StudyInstanceUID").to_numpy(dtype=np.float64)
    if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
        raise ValueError(f"submission has NaN or values outside [0, 1]: min {np.nanmin(v)}, max {np.nanmax(v)}")


def _projection(report, own_wall_s, fs_free_gb, n_hidden, budgets, own_budget_s):
    """Hidden-test extrapolation of the own part from the visible run's report: per-series / per-study rates, per chunk."""
    import math

    series_per_study = report["series"] / report["studies"]
    n_series = n_hidden * series_per_study
    chunk = min(report["chunk_studies"], n_hidden)
    s_series = report["cache"]["seconds"] / report["series"]
    member_s_study = {m: r["seconds"] / r["rows"] for m, r in report["members"].items()}
    # the members' wall per chunk: their sum when sequential, the slowest (plus contention) when parallel
    members_wall = sum(r["members_wall_seconds"] for r in report["chunks"])
    cache_s = s_series * n_series
    member_s = {m: v * n_hidden for m, v in member_s_study.items()}
    members_wall_s = members_wall / report["studies"] * n_hidden
    chunk_cache_s = s_series * series_per_study * chunk
    chunk_member_s = {m: v * chunk for m, v in member_s_study.items()}
    peak_gb = report["cache"]["bytes_per_series"] * series_per_study * chunk / 1e9
    fixed_s = own_wall_s - report["cache"]["seconds"] - members_wall
    total_s = cache_s + members_wall_s + fixed_s
    return {"basis": f"EXTRAPOLATED from the {report['studies']} visible studies (cache per series, members per study; "
                     "checkpoint loading and warm-up inflate the member rate); total = cache + the members' wall "
                     "(parallel members overlap) + fixed",
            "n_hidden_studies": n_hidden, "series_per_study": round(series_per_study, 2), "n_series": round(n_series),
            "chunk_studies": report["chunk_studies"], "n_chunks": math.ceil(n_hidden / report["chunk_studies"]),
            "member_gpus": report["chunks"][0]["member_gpus"], "cache_seconds": round(cache_s),
            "member_seconds": {m: round(v) for m, v in member_s.items()}, "members_wall_seconds": round(members_wall_s),
            "chunk_cache_seconds": round(chunk_cache_s), "chunk_cache_budget_s": budgets["cache"],
            "chunk_member_seconds": {m: round(v) for m, v in chunk_member_s.items()},
            "chunk_member_budget_s": {m: budgets[f"member:{m}"] for m in chunk_member_s},
            "cache_peak_gb": round(peak_gb, 2), "fs_free_gb_before": fs_free_gb,
            "fixed_seconds": round(fixed_s), "own_total_seconds": round(total_s), "own_budget_s": own_budget_s,
            "fits": {"own_budget": total_s < own_budget_s, "chunk_cache_budget": chunk_cache_s < budgets["cache"],
                     "chunk_member_budgets": all(v < budgets[f"member:{m}"] for m, v in chunk_member_s.items()),
                     "disk": peak_gb < fs_free_gb}}


def blend_main(ctx, t0, cfg, public_fn):
    """B6: the blend kernel's body: guards, own part (B1) in a child, public_fn(ctx, t0), rank blend (B3) -> submission.

    cfg (every key of CFG_KEYS required): visible_studies, wheels + pins + site_dir (install_wheels target),
    own_spec (B2's spec), own_budget_s (B1), weights {member: w}, ckpt_sha256 {path: sha256} of every checkpoint
    of own_spec, n_hidden_studies (validation-mode projection), hidden_fallback (bool), own_only (None = the blend
    with public_fn; {test_csv, sample_submission} = own-only: public_fn must be None and is skipped, weights sum to
    1, hidden_fallback must be False, the submission is rank_own over test.csv's studies, which must be the
    series_csv's). Mode: validation iff the
    studies of own_spec series_csv are visible_studies, else hidden. A failure of the own part or of the blend
    re-raises, except in hidden mode with hidden_fallback (D6 amended: LB probes run with False, so a probe score
    is always our blend; final candidates with True): then it prints EXP0004_FALLBACK and the submission is the
    public one. Guard failures
    (cwd/src, checkpoint sha256, cfg) raise in both modes. Writes submission.csv (atomically, over the public's),
    own_members/<member>_prob.csv and RECEIPT_NAME into ctx.output_dir; prints EXP0004_SUMMARY.
    """
    import pandas as pd

    from src.kernel_env import install_wheels

    if set(cfg) != CFG_KEYS:
        raise ValueError(f"cfg keys: missing {sorted(CFG_KEYS - set(cfg))}, unexpected {sorted(set(cfg) - CFG_KEYS)}")
    spec, weights, own_only = cfg["own_spec"], cfg["weights"], cfg["own_only"]
    _check_spec(spec)
    names = [m["name"] for m in spec["members"]]
    if set(weights) != set(names):
        raise ValueError(f"weights {sorted(weights)} must name exactly the members {names}")
    # rank_blend / rank_own check them too, but only after the own part and the public pipeline (hours)
    if own_only is None:
        if public_fn is None:
            raise ValueError("public_fn is None without cfg own_only: the blend needs the public pipeline")
        if not all(type(w) in (int, float) and 0 < w < 1 for w in weights.values()) or not sum(weights.values()) < 1:
            raise ValueError(f"weights must be numbers in (0, 1) with sum < 1 (public gets 1 - sum), got {weights}")
    else:
        if not isinstance(own_only, dict) or set(own_only) != OWN_ONLY_KEYS:
            raise ValueError(f"own_only must be None or a dict with keys {sorted(OWN_ONLY_KEYS)}, got {own_only!r}")
        if public_fn is not None:
            raise ValueError("own_only: public_fn must be None (the public pipeline is skipped)")
        if cfg["hidden_fallback"] is not False:
            raise ValueError("own_only needs hidden_fallback False: there is no public submission to fall back to")
        if (not all(type(w) in (int, float) and 0 < w <= 1 for w in weights.values())
                or abs(sum(weights.values()) - 1) > RANGE_EPS):
            raise ValueError(f"own_only: weights must be numbers in (0, 1] summing to 1, got {weights}")
    if (Path.cwd() / "src").exists():
        raise RuntimeError(f"{Path.cwd() / 'src'} exists: the own part's python -c puts the cwd first on sys.path and "
                           "would import that src instead of ctx.sources_dir's")
    ckpts = {str(p) for m in spec["members"] for p in m["ckpt_paths"]}
    if ckpts != set(cfg["ckpt_sha256"]):
        raise ValueError(f"ckpt_sha256 must name exactly the spec's checkpoints: missing {sorted(ckpts - set(cfg['ckpt_sha256']))}, "
                         f"unexpected {sorted(set(cfg['ckpt_sha256']) - ckpts)}")
    for path, want in cfg["ckpt_sha256"].items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"checkpoint {path} does not exist")
        got = _file_sha256(path)
        if got != want:
            raise ValueError(f"checkpoint {path}: sha256 {got} != expected {want} (wrong weights dataset version?)")
    print(f"EXP0004_CKPTS sha256 ok for {len(ckpts)} checkpoints", flush=True)
    if not isinstance(cfg["hidden_fallback"], bool):
        raise ValueError(f"hidden_fallback must be a bool, got {cfg['hidden_fallback']!r}")

    studies = csv_studies(pd.read_csv(spec["series_csv"]))
    mode = "validation" if set(studies) == set(cfg["visible_studies"]) else "hidden"
    fallback_ok = mode == "hidden" and cfg["hidden_fallback"]
    if own_only is not None:
        test_ids = pd.read_csv(own_only["test_csv"], dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
        sub_cols = list(pd.read_csv(own_only["sample_submission"], nrows=0).columns)
        if sorted(test_ids) != sorted(studies):
            raise ValueError(f"own_only: the {len(test_ids)} studies of {own_only['test_csv']} != the {len(studies)} "
                             f"of {spec['series_csv']}: a test study without series would get no prediction")
    print(f"EXP0004_MODE {mode} n_studies={len(studies)} hidden_fallback={cfg['hidden_fallback']} "
          f"own_only={own_only is not None}", flush=True)

    # ---- own part (D6): a failure re-raises unless fallback_ok, then the public submission stands ----
    own, unusable, report, fallbacks = {}, set(), None, {"own": None, "blend": None}
    cache_dir, out_dir = Path(spec["cache_dir"]), Path(spec["out_dir"])
    disk = {"before": _disk_report("before own", cache_dir)}
    t = time.time()
    try:
        install_wheels(cfg["wheels"], cfg["pins"], target=cfg["site_dir"])
        out_dir.mkdir(parents=True, exist_ok=False)
        report = own_part_subprocess(spec, out_dir / "spec.json", ctx.sources_dir, cfg["site_dir"], cfg["own_budget_s"])
        for name in names:
            own[name] = pd.read_csv(out_dir / f"{name}_prob.csv", index_col="StudyInstanceUID",
                                    dtype={"StudyInstanceUID": str})
        unusable = set(report["unusable"])
    except Exception as e:
        if not fallback_ok:
            raise
        own = {}
        fallbacks["own"] = repr(e)
        print(f"!!! EXP0004_FALLBACK own {e!r} — submission = A's final !!!", flush=True)
    finally:
        disk["after"] = _disk_report("after own", cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        print(f"EXP0004_DISK cache dir {cache_dir} removed", flush=True)
    own_wall_s = time.time() - t
    sub_path = ctx.output_dir / "submission.csv"

    if own_only is not None:
        # ---- own-only: no public pipeline; the own part above re-raised any failure (hidden_fallback False) ----
        final_sub, public_timings, part_b = None, None, None
        submission, receipt = rank_own(test_ids, sub_cols, own, weights, {m: unusable for m in own}, MAX_UNUSABLE_FRAC)
        tmp = sub_path.with_name(sub_path.name + ".exp0004.tmp")
        submission.to_csv(tmp, index=False)
        os.replace(tmp, sub_path)
    else:
        # ---- the public pipeline, untouched ----
        final_sub, _, _, public_timings, part_b = public_fn(ctx, t0)  # (final_sub, members, masks, timings, part_b)
        submission, receipt = final_sub, None

    # ---- blend (only with own members) ----
    if own_only is None and own:
        tmp = sub_path.with_name(sub_path.name + ".exp0004.tmp")
        try:
            from src.knee_labels import LABELS

            blend, receipt = rank_blend(final_sub, own, weights, {m: unusable for m in own}, LABELS)
            _check_submission(blend, final_sub)
            blend.to_csv(tmp, index=False)
            os.replace(tmp, sub_path)
            submission = blend
        except Exception as e:
            if not fallback_ok:
                raise
            receipt = None
            fallbacks["blend"] = repr(e)
            if tmp.exists():
                tmp.unlink()
            print(f"!!! EXP0004_FALLBACK blend {e!r} — submission = A's final !!!", flush=True)
    if own_only is None and submission is final_sub and not pd.read_csv(sub_path, dtype={"StudyInstanceUID": str}).equals(final_sub):
        raise RuntimeError(f"{sub_path} is not the public's final submission returned by public_fn")

    # ---- members, receipt, summary ----
    om = ctx.output_dir / "own_members"
    om.mkdir(exist_ok=False)
    for name, df in own.items():
        df.to_csv(om / f"{name}_prob.csv")
    fallbacks["part_b"] = part_b
    (ctx.output_dir / RECEIPT_NAME).write_text(json.dumps(
        {"mode": mode, "hidden_fallback": cfg["hidden_fallback"], "own_only": own_only, "weights": weights, "blend": receipt, "fallbacks": fallbacks, "own_wall_seconds": round(own_wall_s, 1),
         "disk": disk, "own_report": report}, indent=1))
    summary = {"mode": mode, "hidden_fallback": cfg["hidden_fallback"], "own_only": own_only is not None, "n_studies": len(studies), "rows": len(submission),
               "blended": own_only is None and submission is not final_sub,
               "parity": report and report["parity"], "member_check": report and report["member_check"],
               "own_timings": report and {"own_wall_seconds": round(own_wall_s, 1), "steps": report["seconds"],
                                          "n_chunks": report["n_chunks"], "cache_peak_bytes": report["cache_peak_bytes"],
                                          "cache_s_per_series": report["cache"]["s_per_series"],
                                          "member_s_per_study": {m: r["s_per_study"] for m, r in report["members"].items()}},
               "public_timings": public_timings, "fallbacks": fallbacks, "weights": weights,
               "unusable": receipt and receipt["unusable"], "spearman_vs_public": receipt and receipt.get("spearman_vs_public"),
               "spearman_between_members_mean": receipt and receipt.get("spearman_between_members_mean"),
               "submission_head": submission.head(5).to_dict(orient="records")}
    if mode == "validation":
        summary["projection_hidden"] = _projection(report, own_wall_s, disk["before"]["fs_free_gb"],
                                                   cfg["n_hidden_studies"], spec["budgets"], cfg["own_budget_s"])
    print("EXP0004_SUMMARY " + json.dumps(summary), flush=True)
    return submission
