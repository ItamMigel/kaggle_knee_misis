"""exp-0004 timing kernel: our own part (variant H: exp-0004, exp-0005 with 10 checkpoints each) as blend_main runs it.

    uv run --project kaggle python kaggle/run_kernel.py --exp exp-0004 --script timing_own --accelerator t4 \
      --docker-image gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f \
      --ds bebra3222/rsna-knee-wheels-cp313,bebra3222/rsna-knee-exp-0004-w-8f67ee3d,bebra3222/rsna-knee-exp-0005-w-109a48bc,bebra3222/rsna-knee-exp-0005-w-e2a42938

Why (lead 18.10): the validation kernel's 3-study projection is per-run overhead (pool spawn, checkpoint loads,
CUDA warm-up) spread over 3 studies. Here our part runs on N_SAMPLE train studies in 2 chunks of CHUNK_STUDIES,
exactly as blend_main runs it: checkpoint sha256 and cwd guards, install_wheels into SITE_DIR,
own_part_subprocess -> own_part_main (per chunk: cache -> every member of MEMBER_MODULES at once, each in its own
child on its own GPU (MEMBER_GPUS; None = in turn in one process) -> rmtree; parity; member check), cache dir
removed in finally. Chunk 1 = warm-up, chunk 2 = steady. Phase 6 (lead 18.10): H in turn on one T4 took 7911 s >
D7's 7200 with both members GPU-bound, so the members run in parallel on the two T4s; the projection's members
part is the measured members' wall per chunk (launch -> both done), not the slowest member. Phase 5 (lead 18.10):
variant H = both members as 2-seed averages, 10 checkpoints each (CKPTS). predict_studies makes one loader pass
for all of a member's models, so 10 checkpoints may cost well under 2x of 5: its PREDICT_TIMING line per call
splits loader wait from compute; fd 1 is teed while the child runs, so those lines (prefixed "[member <name>
chunk <i>] " when parallel) land per member per chunk in the summary, with the steady chunk's `bound` (loader
| gpu) per member. A daemon thread samples system RAM and GPU memory every SAMPLE_EVERY_S while the child runs.
Then the first file's transfer syntax per series and the series/study of the sample and of the visible test.
Prints one TIMING_SUMMARY JSON line (fetch --from-log is the only channel: the kernel's outputs cannot be
downloaded) and writes it as timing.json: per chunk rates
(cache, each member, the members' wall, each member's predict_studies split), resource peaks, the hidden-test
projection from the steady chunk (per member, the members' wall, total, and each member alone in turn) and the
lead's D7 criteria for the two-member kernel (D7_*).
Does NOT measure: A's part (run_public is not called: no public datasets, no --submit-file); decoding of
compressed DICOM if the hidden test has any (train is Explicit VR LE only; the visible test's syntaxes are
reported); B4 / B5 themselves (train series are not the visible test's -> parity "absent"; no reference for
train studies -> member check "no reference"); exp-0004's seed 1 (not trained yet: its 5 seed-0 files stand in,
see CKPTS).
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.ctx import get_ctx  # noqa: E402

ctx = get_ctx()

from src.blend_kernel import _check_spec, _disk_report, _file_sha256, own_part_subprocess  # noqa: E402
from src.kernel_env import install_wheels  # noqa: E402

# our members, predicted in this order in every chunk (= submit_blend.py's)
MEMBER_MODULES = {"exp-0004": "src.infer25d", "exp-0005": "src.joint_reader"}
# weights datasets (fold{k}.pt + SHA256SUMS.txt each) and their SHA256SUMS.txt; <sha8> = sha256(SHA256SUMS.txt)[:8].
# exp-0004 seed 0: fold0..4.pt of runs 20261017t150919-625-2918 (0-2), 20261017t214233-033-af68 (3),
# 20261018t000939-333-228f (4) (= submit_blend.py's); exp-0005 seed 0: run 20261017t135225-664-aecb, git 8fb6668e
# (= submit_blend.py's); exp-0005 seed 1: run 20261018t003924-017-1d75 (orch-0005's SHA256SUMS.txt)
DS4_S0 = "bebra3222/rsna-knee-exp-0004-w-8f67ee3d"
DS5_S0, DS5_S1 = "bebra3222/rsna-knee-exp-0005-w-109a48bc", "bebra3222/rsna-knee-exp-0005-w-e2a42938"
SHA256SUMS = {
    DS4_S0: {"fold0.pt": "c7071ece3734bc4f6c92a9b23b1d88051cc04e8c9ccbccceda0776af2f45d5c7",
             "fold1.pt": "d87264cb6444f059ea33ac6454e0e9f44fc3231dc11f9aa2ec36771036ab3f06",
             "fold2.pt": "aa469496d2f941bbae4d0dfb63a6a1295d943527fe2eca21080b3d21cc60a3ec",
             "fold3.pt": "d5f2323130dfdfcaaae4d0fd8cc72e5cb063cf6f5dc483dc2f27536b29168ece",
             "fold4.pt": "5a5875680775c44741844d83803506c4d4b49b66512ab2e000bc59fac6beaa4e"},
    DS5_S0: {"fold0.pt": "e0100e381741e7a5457d492957f68d7cbe9b7b6b137a09f33613c954ae39db9a",
             "fold1.pt": "4e949a1272a98c2ccb8cb46908a2cf55f261a28d873ca6049b94f9fdece549f4",
             "fold2.pt": "857caf4b91ec8c8b2f3e921a9f3f3fdab9cf4aedef774af74dae430ff7953ad4",
             "fold3.pt": "9763c9122d1268bdcb9bf3e2c299853de79e663d4439c9c332b69a7bf9ff2f9c",
             "fold4.pt": "24fa76d42f58807dba5f63f1e8834e4738d76a686836bf93f43d30c6c3e346a2"},
    DS5_S1: {"fold0.pt": "2d1a63a9f48c93f23128a5b57c5aa91b9abe106b836e17c5e7d355bc72de5b6c",
             "fold1.pt": "15cab94ae909b0bbf809c78098a673b1ebf089f8f40c46563eb94a52d325dcd0",
             "fold2.pt": "7bb73c97e298eea65dc2b982817eac471275ff59f243e250b61f52da54d007b1",
             "fold3.pt": "e0e93cd89f7fe2712c531901deee3d91ccb4512d3a200799af9331053a553260",
             "fold4.pt": "f7e5575a8c12f160391c742655daa046b0ffec73ef7a73c1e9ba47ca69b2f0e6"},
}
# {member: [(weights dataset, file)]}: variant H, each member = its 2-seed average, 10 checkpoints. exp-0004's
# seed 1 is not trained yet: its 5 seed-0 files are listed twice. TIMING ONLY: compute stand-ins for seed 1 (same
# architecture and cfg, so the same load and forward cost; every chunk reloads every file anyway, so a repeat
# being in the page cache changes nothing after the sha256 pass). exp-0005: seed 0 + seed 1, 10 distinct files.
CKPTS = {"exp-0004": [(DS4_S0, f"fold{k}.pt") for k in range(5)] * 2,
         "exp-0005": [(ds, f"fold{k}.pt") for ds in (DS5_S0, DS5_S1) for k in range(5)]}
MEMBER_GPUS = {"exp-0004": 0, "exp-0005": 1}  # = submit_blend.py's (T4 x2, one GPU per member); None = in turn
# = submit_blend.py's: the 15 visible TEST series. Train series never match them -> B4 "absent" (parity_ref's
# train rows would be a partial hit, which raises)
PARITY_EXPECTED = json.loads(r"""[{"series":"1.2.826.0.1.3680043.8.498.11580656442259111255675562605155903947","fp":{"shape":[34,375,375],"sha_vol":"c4457295bc2f1860","sha_pos":"67e4e2ab1bb79d11","sum":368441360,"lo":2.9319445800781248,"hi":2090.8626831054685}},{"series":"1.2.826.0.1.3680043.8.498.17811502614030631664517622518906646132","fp":{"shape":[30,375,375],"sha_vol":"7fd5a12a4cd8e25c","sha_pos":"93a2cb77c1a92d6d","sum":345524607,"lo":3.41015625,"hi":1376.033203125}},{"series":"1.2.826.0.1.3680043.8.498.30565395595045942404081022062489758495","fp":{"shape":[30,375,375],"sha_vol":"24b43404ae4ef526","sha_pos":"08f7395a2a3c8bce","sum":375753594,"lo":6.120652856826782,"hi":2226.6173461914054}},{"series":"1.2.826.0.1.3680043.8.498.32856494541816845805947850304653662081","fp":{"shape":[30,375,375],"sha_vol":"8a6cc2e34f6dd0dc","sha_pos":"c01ac7bc9fdbffc7","sum":287809716,"lo":1.0344692993164062,"hi":458.2434936523432}},{"series":"1.2.826.0.1.3680043.8.498.44334485654554877495800965096767429689","fp":{"shape":[160,400,800],"sha_vol":"6b12ba8e8a21658d","sha_pos":"bb9012c3d0d0979c","sum":1386525482,"lo":0.0,"hi":253.765625}},{"series":"1.2.826.0.1.3680043.8.498.10543708582570884867685900269888836671","fp":{"shape":[24,450,450],"sha_vol":"96be98367570256c","sha_pos":"a29d3c2566ca5038","sum":277759769,"lo":-17.5576171875,"hi":5909.636879882809}},{"series":"1.2.826.0.1.3680043.8.498.10818107180155454609278933308651876447","fp":{"shape":[33,375,375],"sha_vol":"df2fe1cfe24e0f29","sha_pos":"e618015f99f75b3f","sum":185178554,"lo":-0.7348284912109375,"hi":3889.8841894531233}},{"series":"1.2.826.0.1.3680043.8.498.34851887739080902195133258975769749352","fp":{"shape":[24,450,450],"sha_vol":"f90a7fbcb0ddd208","sha_pos":"a29d3c2566ca5038","sum":205284587,"lo":-6.3341677856445315,"hi":4368.035205078124}},{"series":"1.2.826.0.1.3680043.8.498.49057071886338907232646925625200929717","fp":{"shape":[24,450,450],"sha_vol":"869dd8d40815d947","sha_pos":"a29d3c2566ca5038","sum":298966235,"lo":-7.626600444316865,"hi":7718.3652783203115}},{"series":"1.2.826.0.1.3680043.8.498.49397095733611495516760005725956804191","fp":{"shape":[30,500,400],"sha_vol":"6711492cb4d55a67","sha_pos":"d2d78c39ac9f84f4","sum":265144398,"lo":0.0,"hi":3983.4005566406213}},{"series":"1.2.826.0.1.3680043.8.498.10927312244766790279817592893017048392","fp":{"shape":[24,425,425],"sha_vol":"8225b92ab15a7083","sha_pos":"8ca850c6aaa26d99","sum":484910035,"lo":0.0,"hi":1861.0731622314443}},{"series":"1.2.826.0.1.3680043.8.498.12207733879434320501816063789123909030","fp":{"shape":[24,425,425],"sha_vol":"592fac6f7d8434a1","sha_pos":"8ca850c6aaa26d99","sum":333085423,"lo":0.0,"hi":752.7015048217768}},{"series":"1.2.826.0.1.3680043.8.498.15858543841290540180647614348005539746","fp":{"shape":[40,425,425],"sha_vol":"9bf42f0f9321a962","sha_pos":"858413852242ebc1","sum":350165666,"lo":0.0,"hi":1141.4560821533196}},{"series":"1.2.826.0.1.3680043.8.498.19236602294493686533219656323303746544","fp":{"shape":[25,450,450],"sha_vol":"3a7ff6c6fb31fad0","sha_pos":"0d2200bb0a7599df","sum":391749146,"lo":0.0,"hi":1213.1723657226562}},{"series":"1.2.826.0.1.3680043.8.498.66069370990398014317043644514003034312","fp":{"shape":[25,450,450],"sha_vol":"355eb5ee637200af","sha_pos":"0d2200bb0a7599df","sum":333545594,"lo":0.0,"hi":678.3313427734365}}]""")
MEMBER_EXPECTED = {m: {} for m in MEMBER_MODULES}  # no reference for train studies -> B5 "no reference"
MEMBER_TOL = 0.02  # = submit_blend.py's (unused without a reference)
WHEELS_DS = "bebra3222/rsna-knee-wheels-cp313"  # = submit_blend.py's
PINS = {"pydicom": "3.0.2", "pylibjpeg": "2.1.0", "pylibjpeg-libjpeg": "2.4.0", "pylibjpeg-openjpeg": "2.6.0",
        "python-gdcm": "3.2.6"}  # = submit_blend.py's
SITE_DIR = Path("/kaggle/tmp/exp0004_site")  # = submit_blend.py's
CACHE_DIR = Path("/kaggle/tmp/exp0004_cache")  # = submit_blend.py's
DEVICE, WORKERS = "cuda", 4  # = submit_blend.py's
CHUNK_STUDIES = 150  # = submit_blend.py's
BUDGETS = {"env": 300, "cache": 900, "parity": 300, "member:exp-0004": 900, "member:exp-0005": 900,
           "member_check": 120}  # = submit_blend.py's
N_SAMPLE, SAMPLE_SEED = 300, 0  # 2 chunks: chunk 1 warm-up, chunk 2 steady
N_HIDDEN = 1300  # = submit_blend.py's N_HIDDEN_STUDIES
N_CHUNKS = math.ceil(N_SAMPLE / CHUNK_STUDIES)
# every step of the child has its own watchdog, so the child cannot outlive their sum (2 x (900 + 900 + 900) + 300
# + 300 + 120 = 6120 s) by more than its work between steps (child start, imports, csv, rmtree: seconds); + 300 s
# so a step's watchdog, not this outer budget, names a slow step. Parallel members run their watchdogs side by
# side, so the sum bounds that mode too
OWN_BUDGET_S = (N_CHUNKS * (BUDGETS["cache"] + sum(BUDGETS[f"member:{m}"] for m in MEMBER_MODULES))
                + BUDGETS["env"] + BUDGETS["parity"] + BUDGETS["member_check"] + 300)
# lead's criteria (18.10), for the visible test's series/study, projected for N_HIDDEN studies. D7, two-member
# kernel: own total <= 80% of its OWN_BUDGET_S 9000, per-chunk cache <= 70% of the 900 s cache watchdog, each
# member's per-chunk time <= 70% of its watchdog
D7_OWN_S, D7_CHUNK_CACHE_S, D7_CHUNK_MEMBER_FRAC = 7200, 630, 0.7
SAMPLE_EVERY_S = 2
# src.infer25d.predict_studies' line, one per member call; own_part_main prefixes a parallel member's log lines
PREDICT_RE = re.compile(rb"^(?:\[member (\S+) chunk (\d+)\] )?PREDICT_TIMING (.*)$")
PREDICT_KEYS = ("n_batches", "setup_s", "data_wait_s", "first_batch_wait_s", "compute_s", "total_s",
                "cuda_max_allocated_gb")
TEE_JOIN_S = 60  # after the child's group is gone the pipe hits EOF at once; longer = something else holds it


def ram_used_gb():
    """System RAM in use, GB: /proc/meminfo MemTotal - MemAvailable (and MemTotal)."""
    lines = Path("/proc/meminfo").read_text().splitlines()
    kb = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in lines)}
    return (kb["MemTotal"] - kb["MemAvailable"]) * 1024 / 1e9, kb["MemTotal"] * 1024 / 1e9


def gpu_mem_used_mib():
    """{gpu index: memory.used MiB} from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=30).stdout
    return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}


def sample_resources(stop, peaks):
    """Until `stop` is set, every SAMPLE_EVERY_S: running peaks of RAM used and per-GPU memory into `peaks`.

    The first error is stored as peaks["error"] and ends the loop; the main thread raises it after the summary.
    """
    try:
        while not stop.is_set():
            peaks["ram_used_gb"] = max(peaks["ram_used_gb"], ram_used_gb()[0])
            for i, m in gpu_mem_used_mib().items():
                peaks["gpu_mem_used_mib"][i] = max(peaks["gpu_mem_used_mib"][i], m)
            peaks["samples"] += 1
            stop.wait(SAMPLE_EVERY_S)
    except Exception as e:
        peaks["error"] = repr(e)


@contextlib.contextmanager
def tee_stdout():
    """fd 1 (this process's stdout, inherited by the own-part child) through a pipe for the block; a thread copies
    every byte to the real stdout (the kernel log) and keeps it. Yields the list of kept byte chunks."""
    sys.stdout.flush()
    real = os.dup(1)
    r, w = os.pipe()
    os.dup2(w, 1)
    os.close(w)
    kept, errors = [], []

    def pump():
        try:
            while data := os.read(r, 1 << 16):
                kept.append(data)
                view = memoryview(data)
                while view:
                    view = view[os.write(real, view):]
        except BaseException as e:
            errors.append(e)
            raise

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        yield kept
    finally:
        sys.stdout.flush()
        os.dup2(real, 1)  # the real stdout back on fd 1; drops this process's write end of the pipe
        th.join(TEE_JOIN_S)
        if th.is_alive():
            raise RuntimeError(f"stdout tee: the pipe is still open {TEE_JOIN_S}s after the block (a process outside "
                               "the child's group holds it)")
        os.close(r)
        os.close(real)
        if errors:
            raise RuntimeError("stdout tee failed (the child's output after the failure is not in the log)") \
                from errors[0]


def first_file_syntaxes(series_root, series_df):
    """Counts by transfer syntax name of each series' first file (sorted names, as read_series_dir); header only."""
    import pydicom

    counts = Counter()
    for st, se in series_df[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(index=False):
        files = sorted(p for p in (series_root / st / se).iterdir() if p.is_file())
        if not files:
            counts["<no files>"] += 1
            continue
        counts[pydicom.dcmread(files[0], stop_before_pixels=True).file_meta.TransferSyntaxUID.name] += 1
    return dict(counts)


def series_stats(series_df):
    """Series per study (mean, median, p90, max) and series per Anatomical_Plane of a *_series.csv frame."""
    per = series_df.groupby("StudyInstanceUID").size()
    return {"studies": int(per.size), "series": len(series_df),
            "series_per_study": {"mean": round(float(per.mean()), 3), "median": float(per.median()),
                                 "p90": float(per.quantile(0.9)), "max": int(per.max())},
            "planes": {k: int(v) for k, v in series_df.Anatomical_Plane.value_counts().items()}}


def project(cache_s_per_series, member_s_per_study, members_wall_s_per_study, fixed_s, warmup_s, series_per_study):
    """N_HIDDEN-study extrapolation at the steady rates and the given series/study, per-chunk times vs the watchdogs.

    member_s_per_study: {member: s}; warmup_s: {"members": the members' wall, member: s} chunk-1 warm-up excess.
    own_total_s = cache + the members' wall + fixed_s (parallel: the members overlap; in turn: their sum);
    alone_own_total_s {member: cache + that member + fixed_s with the members' warm-up replaced by that member's}
    (that member alone in turn; the first member's warm-up also holds the child's CUDA init when in turn).
    """
    n_series = N_HIDDEN * series_per_study
    cache_s = n_series * cache_s_per_series
    member_s = {m: N_HIDDEN * v for m, v in member_s_per_study.items()}
    members_wall_s = N_HIDDEN * members_wall_s_per_study
    alone = {m: cache_s + member_s[m] + fixed_s - warmup_s["members"] + warmup_s[m] for m in member_s}
    return {"series_per_study": round(series_per_study, 3), "n_series": round(n_series),
            "n_chunks": math.ceil(N_HIDDEN / CHUNK_STUDIES), "cache_s": round(cache_s),
            "member_s": {m: round(v) for m, v in member_s.items()}, "members_wall_s": round(members_wall_s),
            "fixed_s": round(fixed_s), "own_total_s": round(cache_s + members_wall_s + fixed_s),
            "alone_own_total_s": {m: round(v) for m, v in alone.items()},
            "chunk_cache_s": round(CHUNK_STUDIES * series_per_study * cache_s_per_series),
            "chunk_cache_budget_s": BUDGETS["cache"],
            "chunk_member_s": {m: round(CHUNK_STUDIES * v) for m, v in member_s_per_study.items()},
            "chunk_members_wall_s": round(CHUNK_STUDIES * members_wall_s_per_study),
            "chunk_member_budget_s": {m: BUDGETS[f"member:{m}"] for m in member_s_per_study}}


if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
    raise RuntimeError("timing_own.py runs only inside a Kaggle kernel (CUDA, /kaggle/working). Launch it with the "
                       "kaggle skill, command in the docstring.")
if N_CHUNKS != 2:
    raise ValueError(f"N_SAMPLE {N_SAMPLE} / CHUNK_STUDIES {CHUNK_STUDIES} gives {N_CHUNKS} chunks, want 2 "
                     "(warm-up + steady)")
if MEMBER_GPUS is not None and set(MEMBER_GPUS) != set(MEMBER_MODULES):
    raise ValueError(f"MEMBER_GPUS must be None or name exactly the members {list(MEMBER_MODULES)}, got {MEMBER_GPUS}")
if set(CKPTS) != set(MEMBER_MODULES) or not all(CKPTS.values()):
    raise ValueError(f"CKPTS must name exactly the members {list(MEMBER_MODULES)}, each with checkpoints, got "
                     f"{ {m: len(v) for m, v in CKPTS.items()} }")
if (Path.cwd() / "src").exists():
    raise RuntimeError(f"{Path.cwd() / 'src'} exists: the own part's python -c puts the cwd first on sys.path and "
                       "would import that src instead of ctx.sources_dir's")
# {member: [(path, sha256)]}, repeats kept: the member loads every listed file
ckpts = {m: [(str(ctx.mounts[ds] / name), SHA256SUMS[ds][name]) for ds, name in CKPTS[m]] for m in MEMBER_MODULES}
for path, want in ((p, s) for m in ckpts for p, s in ckpts[m]):  # every listed file, repeats too
    got = _file_sha256(path)
    if got != want:
        raise ValueError(f"checkpoint {path}: sha256 {got} != expected {want} (wrong weights dataset version?)")
n_files = {m: len({p for p, _ in v}) for m, v in ckpts.items()}
print(f"TIMING_CKPTS sha256 ok for {sum(map(len, ckpts.values()))} listed checkpoints "
      f"{ {m: len(v) for m, v in ckpts.items()} } ({n_files} distinct files)", flush=True)

comp = ctx.mounts["rsna-knee-abnormality-detection"]
train_series = pd.read_csv(comp / "train_series.csv")
pick = set(np.random.default_rng(SAMPLE_SEED).choice(sorted(train_series.StudyInstanceUID.unique()), N_SAMPLE,
                                                     replace=False))
sample = train_series[train_series.StudyInstanceUID.isin(pick)]  # original row order
sample_csv = ctx.output_dir / "sample_series.csv"
sample.to_csv(sample_csv, index=False)
test_series = pd.read_csv(comp / "test_series.csv")
out_dir = ctx.output_dir / "own"
members = [{"name": m, "module": module, "ckpt_paths": [p for p, _ in ckpts[m]]}
           for m, module in MEMBER_MODULES.items()]
spec = {"series_root": str(comp / "train_series"), "series_csv": str(sample_csv), "cache_dir": str(CACHE_DIR),
        "out_dir": str(out_dir), "device": DEVICE, "workers": WORKERS, "chunk_studies": CHUNK_STUDIES,
        "budgets": BUDGETS, "members": members, "member_gpus": MEMBER_GPUS, "parity_expected": PARITY_EXPECTED,
        "member_expected": MEMBER_EXPECTED, "member_tol": MEMBER_TOL}
_check_spec(spec)
print(f"TIMING_SAMPLE {sample.StudyInstanceUID.nunique()} studies, {len(sample)} series -> {sample_csv}", flush=True)

# ---- our part, as blend_main runs it ----
disk = {"before": _disk_report("before own", CACHE_DIR)}
t = time.time()
install_wheels([str(w) for w in sorted(ctx.mounts[WHEELS_DS].glob("*.whl")) if not w.name.startswith("opencv")],
               PINS, target=SITE_DIR)
wheels_s = time.time() - t
out_dir.mkdir(parents=True, exist_ok=False)
ram_before, ram_total = ram_used_gb()
gpu_before = gpu_mem_used_mib()
peaks = {"ram_used_gb": ram_before, "gpu_mem_used_mib": dict(gpu_before), "samples": 0, "error": None}
maxrss_before_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # pip's, so far
stop = threading.Event()
sampler = threading.Thread(target=sample_resources, args=(stop, peaks), daemon=True)
sampler.start()
t_child = time.time()
try:
    with tee_stdout() as child_out:
        report = own_part_subprocess(spec, out_dir / "spec.json", ctx.sources_dir, SITE_DIR, OWN_BUDGET_S)
finally:
    child_s = time.time() - t_child
    stop.set()
    sampler.join()
    disk["after"] = _disk_report("after own", CACHE_DIR)
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    print(f"TIMING_DISK cache dir {CACHE_DIR} removed", flush=True)
own_wall_s = time.time() - t
maxrss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if (report["n_chunks"] != 2 or report["parity"]["status"] != "absent"
        or any(report["member_check"][m]["status"] != "no reference" for m in MEMBER_MODULES)):
    raise RuntimeError(f"unexpected own report: {report['n_chunks']} chunks, parity {report['parity']}, "
                       f"member check {report['member_check']}")
# the child's predict_studies lines, in own_part_main's order: per chunk, every member in MEMBER_MODULES order
# (parallel: prefixed with the member and chunk, printed in that order after both are done)
lines = [mt for mt in map(PREDICT_RE.match, b"".join(child_out).split(b"\n")) if mt]
calls = [(r, m) for r in report["chunks"] for m in MEMBER_MODULES]
if len(lines) != len(calls):
    raise RuntimeError(f"{len(lines)} PREDICT_TIMING lines in the child's stdout, want {len(calls)} (chunks x members)")
predict = {}
for (r, m), mt in zip(calls, lines, strict=True):
    tag = None if mt[1] is None else (mt[1].decode(), int(mt[2]))
    want = None if MEMBER_GPUS is None else (m, r["chunk"])
    if tag != want:
        raise RuntimeError(f"PREDICT_TIMING line for chunk {r['chunk']} {m} is tagged {tag}, want {want}")
    predict[r["chunk"], m] = json.loads(mt[3].decode())
for (i, m), line in predict.items():
    r = report["chunks"][i - 1]
    if (line["n_studies"], line["n_models"]) != (r["studies"], len(ckpts[m])):
        raise RuntimeError(f"PREDICT_TIMING line for chunk {i} {m} has {line['n_studies']} studies, {line['n_models']} "
                           f"models, want {r['studies']}, {len(ckpts[m])}: out of order?")

# ---- transfer syntax (pydicom from SITE_DIR, as the child imports it) and series/study ----
sys.path.insert(0, str(SITE_DIR))
import pydicom  # noqa: E402

if not Path(pydicom.__file__).is_relative_to(SITE_DIR) or pydicom.__version__ != PINS["pydicom"]:
    raise RuntimeError(f"pydicom {pydicom.__version__} from {pydicom.__file__}, want {PINS['pydicom']} from {SITE_DIR}")
syntax = {"label": "first file per series (sorted file names, header only)",
          "sample": first_file_syntaxes(comp / "train_series", sample),
          "visible_test": first_file_syntaxes(comp / "test_series", test_series)}
stats = {"sample": series_stats(sample), "visible_test": series_stats(test_series)}

# ---- rates per chunk, projection from the steady chunk ----
chunks = []
for r, role in zip(report["chunks"], ("warm-up", "steady"), strict=True):
    chunks.append({"chunk": r["chunk"], "role": role, "studies": r["studies"], "series": r["series"],
                   "cache_s": r["cache_seconds"], "cache_s_per_series": round(r["cache_seconds"] / r["series"], 4),
                   "cache_bytes": r["cache_bytes"], "member_s": {m: r["member_seconds"][m] for m in MEMBER_MODULES},
                   "member_s_per_study": {m: round(r["member_seconds"][m] / r["studies"], 4) for m in MEMBER_MODULES},
                   "members_wall_s": r["members_wall_seconds"],
                   "members_wall_s_per_study": round(r["members_wall_seconds"] / r["studies"], 4),
                   "member_gpus": r["member_gpus"],
                   "member_gpu_peak_reserved_gb": r["member_gpu_peak_reserved_gb"], "fs_free_gb": r["fs_free_gb"],
                   "predict": {m: {**{k: predict[r["chunk"], m][k] for k in PREDICT_KEYS}, "outside_predict_s":
                                   round(r["member_seconds"][m] - predict[r["chunk"], m]["total_s"], 1)}
                               for m in MEMBER_MODULES}})
warm, steady = chunks
bound = {m: "loader" if steady["predict"][m]["data_wait_s"] > steady["predict"][m]["compute_s"] else "gpu"
         for m in MEMBER_MODULES}
spr = steady["cache_s"] / steady["series"]
sps = {m: steady["member_s"][m] / steady["studies"] for m in MEMBER_MODULES}
swall = steady["members_wall_s"] / steady["studies"]
at_steady = [c["series"] * spr + c["studies"] * swall for c in chunks]
fixed_s = own_wall_s - sum(at_steady)  # everything that does not scale with series / studies
warmup = {"cache": warm["cache_s"] - warm["series"] * spr, "members": warm["members_wall_s"] - warm["studies"] * swall,
          **{m: warm["member_s"][m] - warm["studies"] * sps[m] for m in MEMBER_MODULES}}
part_warmup = {k: warmup[k] for k in ("members", *MEMBER_MODULES)}
fixed = {"fixed_s": round(fixed_s, 1), "install_wheels_s": round(wheels_s, 1), "env_s": report["seconds"]["env"],
         "chunk1_warmup_s": {k: round(v, 1) for k, v in warmup.items()},
         "rest_s": round(fixed_s - wheels_s - report["seconds"]["env"] - warmup["cache"] - warmup["members"], 1),
         "formula": "own_wall_s - sum over chunks (series x steady cache_s_per_series + studies x the steady "
                    "members_wall_s_per_study); chunk1_warmup = chunk 1's actual - chunk 1 at the steady rates, per "
                    "part (cache and members are in fixed_s, each member's only feeds alone_own_total_s); rest = "
                    "child start, parity, member check, csv, rmtree, disk reports"}
r_test = stats["visible_test"]["series_per_study"]["mean"]
proj_test = project(spr, sps, swall, fixed_s, part_warmup, r_test)
d7_member = {m: proj_test["chunk_member_s"][m] <= D7_CHUNK_MEMBER_FRAC * BUDGETS[f"member:{m}"] for m in MEMBER_MODULES}
summary = {
    "basis": f"EXTRAPOLATED from 2 chunks of train studies (N_SAMPLE {N_SAMPLE}, seed {SAMPLE_SEED}, "
             f"{CHUNK_STUDIES} per chunk): steady rates = chunk 2's cache s/series and per-member s/study (every "
             "checkpoint load of the member and the cache pool start per chunk included), fixed = the rest of the "
             "measured own wall. Variant H: 10 checkpoints per member; exp-0004's = its 5 seed-0 files twice (TIMING "
             "ONLY: compute stand-ins for the untrained seed 1, same cost), exp-0005's = seed 0 + seed 1",
    "members": list(MEMBER_MODULES), "n_ckpts": {m: len(v) for m, v in ckpts.items()}, "n_ckpt_files": n_files,
    "weights_datasets": {m: sorted({ds for ds, _ in CKPTS[m]}) for m in MEMBER_MODULES}, "member_gpus": MEMBER_GPUS,
    "mode": "in turn" if MEMBER_GPUS is None else "parallel", "device": DEVICE,
    "workers": WORKERS, "chunks": chunks,
    "own": {"own_wall_s": round(own_wall_s, 1), "child_s": round(child_s, 1), "install_wheels_s": round(wheels_s, 1),
            "child_total_s": report["total_seconds"], "steps": report["seconds"], "own_budget_s": OWN_BUDGET_S,
            "cache_peak_bytes": report["cache_peak_bytes"], "series_errors": report["cache"]["series_errors"],
            "error_counts": report["cache"]["error_counts"], "unusable": len(report["unusable"]),
            "parity": report["parity"], "member_check": report["member_check"]},
    "resources": {"every_s": SAMPLE_EVERY_S, "samples": peaks["samples"], "error": peaks["error"],
                  "ram_total_gb": round(ram_total, 2), "ram_used_gb_before": round(ram_before, 2),
                  "ram_used_gb_peak": round(peaks["ram_used_gb"], 2), "gpu_mem_used_mib_before": gpu_before,
                  "gpu_mem_used_mib_peak": peaks["gpu_mem_used_mib"],
                  "maxrss_gb_largest_descendant": round(maxrss_kb * 1024 / 1e9, 2),
                  "maxrss_gb_before_child": round(maxrss_before_kb * 1024 / 1e9, 3)},
    "disk": disk, "transfer_syntax": syntax, "series_stats": stats,
    "projection": {"basis": "EXTRAPOLATED from 2 chunks of train studies", "n_hidden": N_HIDDEN,
                   "chunk_studies": CHUNK_STUDIES, "steady_cache_s_per_series": round(spr, 4),
                   "steady_member_s_per_study": {m: round(v, 4) for m, v in sps.items()},
                   "steady_members_wall_s_per_study": round(swall, 4), "fixed": fixed,
                   "visible_test_series_per_study": proj_test,
                   "sample_series_per_study": project(spr, sps, swall, fixed_s, part_warmup,
                                                      stats["sample"]["series_per_study"]["mean"])},
    "d7_two_members": {"basis": f"visible test's {r_test} series/study, EXTRAPOLATED from 2 chunks of train studies",
                       f"own_total_le_{D7_OWN_S}": proj_test["own_total_s"] <= D7_OWN_S,
                       f"chunk_cache_le_{D7_CHUNK_CACHE_S}": proj_test["chunk_cache_s"] <= D7_CHUNK_CACHE_S,
                       f"chunk_member_le_{D7_CHUNK_MEMBER_FRAC}_of_watchdog": d7_member,
                       "all": (proj_test["own_total_s"] <= D7_OWN_S and proj_test["chunk_cache_s"] <= D7_CHUNK_CACHE_S
                               and all(d7_member.values()))},
    "bound": {"basis": "steady chunk's predict_studies split (chunks[1].predict): loader if data_wait_s > compute_s, "
                       "else gpu; outside_predict_s = member step - predict_studies (checkpoint loads, exp-0005's "
                       "ROIs; chunk 1 also the module import and CUDA init) is neither",
              **bound},
}
(ctx.output_dir / "timing.json").write_text(json.dumps(summary, indent=1))
print("TIMING_SUMMARY " + json.dumps(summary), flush=True)
if peaks["error"] is not None:
    raise RuntimeError(f"resource sampler failed (its peaks cover {peaks['samples']} samples): {peaks['error']}")
