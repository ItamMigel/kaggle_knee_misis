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
