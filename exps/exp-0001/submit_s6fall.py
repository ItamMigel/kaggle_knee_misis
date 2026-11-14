"""exp-0001: public kernel nartaa/rsna-knee-s6fall-solo v2 (LB 0.950) as one kernel script — a candidate member next to A.

S6fall = Heliosli's S6 ConvNeXt retrained by Nartaa (no OAI); the kernel runs unchanged (src/public_fork_s6fall,
run_s6fall: Nartaa's 0.949 Raptor for rows B cannot read, rank(B) as the submission) from our immutable weights copy.
Fingerprint: on the 3 public test studies its submission.csv sha256 must equal the scored v2 commit's (00031c53...);
check with exps/exp-0001/check_s6fall_fingerprint.py.

Kernel inputs: competition rsna-knee-abnormality-detection; datasets bebra3222/rsna-knee-s6fall-weights,
nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-orthofoundation-fold0-epoch7-20261004; T4 x2;
S6fall's docker image (the same digest as A's).
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

T0 = time.time()

from utils.ctx import get_ctx  # noqa: E402

ctx = get_ctx()

from src.public_fork_s6fall import pipeline  # noqa: E402

pipeline.run_s6fall(ctx, T0)
