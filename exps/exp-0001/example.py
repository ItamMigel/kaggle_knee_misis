"""Пример скрипта эксперимента. Новый эксп = копия папки exps/_template/.

Скриптов в папке экспа может быть сколько угодно и называться они могут как
угодно: `prepare.py`, `fit_lgbm.py`, `blend_submit.py`. Никаких обязательных
стадий нет.

Запуск — одной командой, без сборок и инжекторов (какой именно — в секции фактов
CLAUDE.md: бывает системный питон, конда, `uv run`, `docker run ...`):

    python exps/<exp>/example.py

Тот же самый файл едет на кагл телом кернела и работает там без единой правки:
всю разницу сред разруливает get_ctx().
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import random  # noqa: E402

from utils.ctx import get_ctx  # noqa: E402

# Сид — константа скрипта, ctx про него ничего не знает. Сей явно то, что реально
# используешь, здесь же: `np.random.seed(SEED)`, `torch.manual_seed(SEED)`.
SEED = 42
random.seed(SEED)

ctx = get_ctx()

# Данные. Ключ — кагловый слаг; локально ровно то же имя лежит в кэше данных.
#   train = pd.read_csv(ctx.mounts["<competition-slug>"] / "train.csv")
#
# Артефакт чужого запуска. Ключ — его id; локально ищется в exps/*/runs/,
# на кагле — среди приаттаченных кернелов (--kernel <id> при запуске).
#   model = torch.load(ctx.mounts["<run_id>"] / "model.pt")
#
# Чужой датасет — ключ "<owner>/<slug>" (на кагле: --ds <owner>/<slug>).

# Всё, что нужно сохранить, — в ctx.output_dir. Тяжёлое (веса) класть туда же:
# в гит оно не поедет и с кагла по умолчанию не качается.
(ctx.output_dir / "hello.txt").write_text(
    "замени этот скрипт своим\n", encoding="utf-8"
)

# Запись о запуске (каким кодом, что прочитано, что произведено) появится в
# ctx.output_dir сама — её пишет хук на завершение процесса. Даже если упадём.
print(f"[example] готово | {ctx.output_dir}")
