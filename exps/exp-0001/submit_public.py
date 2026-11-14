"""exp-0001: public notebook A v7 (heliosli, LB 0.958) as one kernel script, members saved separately.

Raptor 384 -> Raptor 224 -> R6 compact -> stack rank blend -> goodpjw 2.5D reader (Part B) -> submission.csv.
Order and constants are A's and live in src/public_fork/pipeline.py (run_public, this script's former body);
the stages in src/public_fork/stages.py, A's embedded sources in src/public_fork/vendored/. Kaggle-only: CUDA,
/kaggle/working, and the vendored Raptor scripts look up their weights and the test data under /kaggle/input
themselves (they assert the checkpoint sha256).

Kernel inputs: competition rsna-knee-abnormality-detection; datasets goodpjw2008/rsna-knee-2-5d-convnext-reader,
heliosli/rsna-knee-r6-frozen-fe055406-w1, nartaa/rsna-knee-publication-swa-weights-20261007,
heliosli/rsna-orthofoundation-fold0-epoch7-20261004; T4 x2; A's docker image (see description.md).
The R6 bundle comes from the -w1 copy: A mounts heliosli/rsna-knee-r6-frozen-fe055406, whose v2 (14.10)
holds R7 under the same file names; -w1 is byte-identical to its v1 (manifest sha256 6508ce69...).
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

T0 = time.time()   # A: NB_T0 (start of cell 3); the R6 deadline (8.3 h) and the Part B cut (8.2 h) count from it

from utils.ctx import get_ctx  # noqa: E402

ctx = get_ctx()

from src.public_fork import pipeline  # noqa: E402

pipeline.run_public(ctx, T0)
