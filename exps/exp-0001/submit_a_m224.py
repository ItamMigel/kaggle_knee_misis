"""exp-0001: component A with Raptor 224 under the anatomical mirror TTA that A already applies to Raptor 384
(src.public_fork.mirror224, lead 18.10). MIRROR224 False = A v7 itself (pipeline.run_public, byte-identical to ffe2).

    uv run --project kaggle python kaggle/run_kernel.py --exp exp-0001 --script submit_a_m224 --accelerator t4 \
      --docker-image gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f \
      --ds goodpjw2008/rsna-knee-2-5d-convnext-reader,heliosli/rsna-knee-r6-frozen-fe055406-w1,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004 \
      [--submit-file submission.csv]

Validation (the 3 visible test studies): mirror224.unit_check() (the 224 mirror is the 384 code path, one window per
plane), then mirror224.diagnose() (Raptor 224 plain / pair / mirror_only on N_TRAIN cold train studies + gold58:
Spearman plain vs mirror, gold58 AUC, runtime), then A. Hidden: A only. No fallback: any failure raises.
Log: M224_UNIT_CHECK, M224_DIAG_RUN, M224_DIAG, A's EXP0001_* lines, A_M224_SUMMARY.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import time  # noqa: E402

T0 = time.time()   # A's deadlines (R6 8.3 h, Part B 8.2 h) count from the kernel start

import json  # noqa: E402
import shutil  # noqa: E402

import pandas as pd  # noqa: E402

from utils.ctx import get_ctx  # noqa: E402

ctx = get_ctx()

from src.log_transport import emit_file  # noqa: E402
from src.public_fork import mirror224, pipeline  # noqa: E402

MIRROR224 = True                # lead 18.10: Raptor 224 under A's 384 mirror TTA
N_TRAIN, DIAG_SEED = 100, 20261018
VISIBLE_STUDIES = ["1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191",
                   "1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776",
                   "1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402"]

if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
    raise RuntimeError("submit_a_m224.py runs only inside a Kaggle kernel; launch it with the kaggle skill, command in the docstring.")
for key in (pipeline.DATA_MOUNT, pipeline.READER_MOUNT, pipeline.BUNDLE_MOUNT, pipeline.RAPTOR_W_MOUNT, pipeline.CODE_MOUNT):
    ctx.mounts[key]             # every input attached, else KeyError here
WORK = ctx.output_dir
test_ids = pd.read_csv(ctx.mounts[pipeline.DATA_MOUNT] / "test.csv", dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
mode = "validation" if sorted(test_ids) == sorted(VISIBLE_STUDIES) else "hidden"
print("A_M224_MODE", json.dumps({"mode": mode, "n_studies": len(test_ids), "mirror224": MIRROR224}), flush=True)

diag = None
if mode == "validation":
    mirror224.unit_check()
    scratch = (Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")) / "m224_diag"
    diag = mirror224.diagnose(ctx, N_TRAIN, DIAG_SEED, scratch, pipeline.RAPTOR_TIMEOUT_S)
    shutil.rmtree(scratch)

t_a = time.time()
final_sub, _, _, a_timings, part_b = mirror224.run_public_m224(ctx, T0, MIRROR224)
summary = {"mode": mode, "mirror224": MIRROR224, "rows": len(final_sub), "part_b": part_b,
           "timings": {"a": a_timings, "a_wall": time.time() - t_a, "total": time.time() - T0},
           "submission_head": final_sub.head(3).to_dict(orient="records")}
print("A_M224_SUMMARY " + json.dumps(summary, default=str), flush=True)
if mode == "validation":
    for name in ("submission.csv", "m224_diag.json", *("m224_raw_%s.csv" % tag for tag, _ in mirror224.DIAG_RUNS)):
        emit_file(WORK / name, name, 3000)
