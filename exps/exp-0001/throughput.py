"""exp-0001: hidden-runtime budget — throughput of A's stages and of S6fall's component B on train studies (T4 x2).

Each stage runs twice, on its own disjoint random train studies (20 and 100; cold page cache, as on the ~1300-study
hidden test, which does not fit in RAM): Raptor 384 -> Raptor 224 -> R6 -> reader -> S6fall B, A's order. Fit per stage:
fixed seconds + seconds per series, projected to 1300 studies at 5.0 and 5.75 series per study (src/kernel_throughput).
GPU utilization / load average are sampled per stage (can a candidate share a T4 with A?). Measurement only: no
submission; a failed stage is recorded as THROUGHPUT_STAGE_FAILED and the next one runs.

Kernel inputs: competition rsna-knee-abnormality-detection; datasets goodpjw2008/rsna-knee-2-5d-convnext-reader,
heliosli/rsna-knee-r6-frozen-fe055406-w1, nartaa/rsna-knee-publication-swa-weights-20261007,
heliosli/rsna-orthofoundation-fold0-epoch7-20261004, bebra3222/rsna-knee-s6fall-weights; T4 x2; A's docker image.
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
import shutil  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

from utils.ctx import get_ctx  # noqa: E402

ctx = get_ctx()

from src import kernel_throughput as kt  # noqa: E402
from src.log_transport import emit_file  # noqa: E402
from src.public_fork import pipeline as A  # noqa: E402
from src.public_fork import stages  # noqa: E402
from src.public_fork_s6fall import pipeline as S6  # noqa: E402

SEED = 20261018
SIZES = (20, 100)
STAGES = ("raptor384", "raptor224", "r6", "reader", "s6fall")
N_HIDDEN = 1300
SERIES_PER_STUDY = (5.0, 5.75)
S6_BUDGET_S = 2 * 3600

if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
    raise RuntimeError("throughput.py runs only inside a Kaggle kernel (T4 x2; the Raptor scripts read /kaggle/input). "
                       "Launch: kaggle/run_kernel.py --exp exp-0001 --script throughput --accelerator t4 ... (see description.md)")

DATA = ctx.mounts[A.DATA_MOUNT]
READER_DIR = ctx.mounts[A.READER_MOUNT]
BUNDLE_SRC = ctx.mounts[A.BUNDLE_MOUNT]
CODE_SRC = ctx.mounts[A.CODE_MOUNT]
ctx.mounts[A.RAPTOR_W_MOUNT]          # attached for the Raptor scripts (they find it under /kaggle/input themselves)
S6_WEIGHTS = ctx.mounts[S6.WEIGHTS_MOUNT]
WORK = ctx.output_dir
SCRATCH = (Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")) / "throughput"
K1_SCRIPT = S6.materialize_k1(SCRATCH / "k1")
K1_DRIVER = Path(S6.__file__).resolve().parent / "k1_driver.py"

samples = kt.sample_disjoint(DATA / "train_series.csv", {"%s_%d" % (s, n): n for s in STAGES for n in SIZES}, SEED)


def run_stage(stage, root, key):
    """One stage on the fake root -> extra info (the wall time is measured by the caller)."""
    if stage in ("raptor384", "raptor224"):
        kt.run_raptor_on(stage[len("raptor"):], root, SCRATCH / key / "raptor_work", WORK, A.RAPTOR_TIMEOUT_S)
        return {}
    if stage == "r6":
        r6 = stages.run_compact_r6(root, CODE_SRC, BUNDLE_SRC, SCRATCH / key / "probe", WORK, time.time(), A.ARM, A.BUNDLE_ID,
                                   A.DEADLINE_HOURS, A.FLIP_TTA, A.CODE_SHA, A.PROBE_CLASS, None)
        return {k: r6["receipt"].get(k) for k in ("rows", "scored", "fallback_studies", "seconds", "per_study_mean")}
    if stage == "reader":
        stages.run_reader(root, READER_DIR, WORK, A.OWN_GLOB)
        return {}
    if stage == "s6fall":
        d = kt.run_s6fall_b_on(K1_SCRIPT, K1_DRIVER, root, S6_WEIGHTS, CODE_SRC, SCRATCH / key / "kroot", S6_BUDGET_S)
        return {"steps_s": d["seconds"], "parts": d["parts"], "retry": d["retry"], "failed_after_retry": d["failed_after_retry"],
                "per_part_s": [r.get("seconds") for r in d["receipts"]]}
    raise ValueError(stage)


runs = []
for stage in STAGES:
    for n in SIZES:
        key = "%s_%d" % (stage, n)
        cost = kt.fake_competition_root(DATA, samples[key], SCRATCH / key / "comp")
        rec = {"key": key, "stage": stage, "studies": cost["studies"], "series": cost["series"], "files": cost["files"]}
        print("THROUGHPUT_STAGE_START", json.dumps(rec), flush=True)
        t = time.time()
        with kt.UsageSampler() as us:
            try:
                rec["info"] = run_stage(stage, cost["root"], key)
                rec["ok"] = True
            except Exception as e:
                rec["ok"], rec["error"] = False, repr(e)[:2000]
                traceback.print_exc()
                print("THROUGHPUT_STAGE_FAILED", key, repr(e)[:500], flush=True)
        rec["seconds"] = round(time.time() - t, 1)
        rec["usage"] = us.summary()
        runs.append(rec)
        print("THROUGHPUT_STAGE", json.dumps(rec), flush=True)
        shutil.rmtree(SCRATCH / key, ignore_errors=True)

fits = {}
by_key = {r["key"]: r for r in runs}
for stage in STAGES:
    small, large = by_key["%s_%d" % (stage, SIZES[0])], by_key["%s_%d" % (stage, SIZES[1])]
    if small["ok"] and large["ok"]:
        fits[stage] = kt.fit_projection(small, large, N_HIDDEN, SERIES_PER_STUDY)
a_stages = ("raptor384", "raptor224", "r6", "reader")
totals = {}
for name, members in (("A", a_stages), ("A+S6fall", a_stages + ("s6fall",))):
    if all(s in fits for s in members):
        totals[name] = {kind: {k: round(sum(fits[s][kind][k] for s in members), 3) for k in fits[members[0]][kind]}
                        for kind in ("proj_h", "prop_h", "proj_h_files")}
result = {"seed": SEED, "sizes": SIZES, "n_hidden": N_HIDDEN, "series_per_study": SERIES_PER_STUDY, "runs": runs,
          "fits": fits, "totals": totals, "samples": samples}
(WORK / "throughput.json").write_text(json.dumps(result, indent=1))
print("THROUGHPUT_SUMMARY", json.dumps({"fits": fits, "totals": totals}), flush=True)
emit_file(WORK / "throughput.json", "throughput.json", 3000)
