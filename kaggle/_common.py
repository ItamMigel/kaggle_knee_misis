"""Общее для скриптов kaggle/. Хостовая сторона, на кагл не едет.

Запускать всё отсюда только так:

    uv run --project kaggle python kaggle/<script>.py ...

Корень репы берётся от расположения этого файла, а не от cwd: скрипты живут
внутри репы, и гадать по текущей директории незачем.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SCRIPTS = (
    REPO_ROOT / "vendor" / "nvidia-kaggle" / "skills" / "nvidia-kaggle-skill" / "scripts"
)

sys.path.insert(0, str(REPO_ROOT))  # utils/ctx.py — общий хеш, id и раскладка реестра

from utils import ctx  # noqa: E402


def git(*args: str) -> str:
    """git в корне репы. Реализация одна — в utils/ctx.py."""
    return ctx.git(REPO_ROOT, *args)


def require_env() -> None:
    """Проверить, что мы в тулинг-окружении, и подсказать команду, если нет.

    Без этого забытый `--project kaggle` даёт невнятный ImportError глубоко
    внутри чужого скрипта.
    """
    try:
        import kagglesdk  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Нет kagglesdk — скрипт запущен не в тулинг-окружении.\n"
            f"Запускай так: uv run --project kaggle python kaggle/"
            f"{Path(sys.argv[0]).name} ..."
        ) from None


def vendor_script(name: str) -> Path:
    """Путь к скрипту апстрима. vendor/ — зеркало, мы там ничего не правим."""
    path = VENDOR_SCRIPTS / name
    if not path.exists():
        raise SystemExit(
            f"Нет {path.relative_to(REPO_ROOT)} — апстрим переименовал скрипт?\n"
            "Посмотри vendor/nvidia-kaggle/skills/nvidia-kaggle-skill/scripts/."
        )
    return path


def _read(path: Path, hint: str) -> dict:
    if not path.exists():
        raise SystemExit(f"Нет {path.name} — {hint}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict:
    """Конфиг соревнования — из ДВУХ файлов, потому что оси разные.

    competition.json — трекается в гите: слаг соревнования одинаков для всей
        команды, и вбивать его каждому руками значит однажды разойтись и молча
        примонтировать не те данные.
    comp_config.json — в .gitignore: sources_dataset у каждого свой, иначе двое
        затирали бы друг другу сурсы.

    Про worktree: неотслеживаемые файлы git между воркtree НЕ переносит, поэтому
    у каждого воркtree свой comp_config.json автоматически — то есть свой
    sources_dataset. Это то, что нужно для параллельных экспов, и это свойство
    надо сохранять, а не «чинить».

    kaggle_username отдельным полем не держим: владелец уже зашит в
    sources_dataset, а правда всё равно в токене (её сверяет push_sources).
    """
    cfg = _read(REPO_ROOT / "competition.json", "он трекается в гите, должен быть в репе.")
    cfg |= _read(
        REPO_ROOT / "comp_config.json",
        "скопируй comp_config.example.json (файл персональный, в .gitignore).\n"
        "В новом worktree его надо создать заново и указать ОТДЕЛЬНЫЙ "
        "sources_dataset, иначе воркtree затрут друг другу сурсы.",
    )
    for key in ("competition", "sources_dataset"):
        value = cfg.get(key, "")
        # кириллица/пробелы = незаполненный плейсхолдер из example
        if not value or any(ord(ch) > 127 or ch.isspace() for ch in value):
            raise SystemExit(f"Поле '{key}' не заполнено: '{value}'.")
    owner, sep, ds_slug = cfg["sources_dataset"].partition("/")
    if not sep or not owner or not ds_slug:
        raise SystemExit("sources_dataset должен быть 'owner/slug'.")
    # Слаг обязан быть каноничным ЗАРАНЕЕ: upload_dataset прогоняет title через
    # slugify и молча перепишет 'my_src' в 'my-src' — датасет создастся под
    # другим ref, чем ссылается весь конвейер.
    if not re.fullmatch(r"[a-z0-9-]{6,50}", ds_slug):
        raise SystemExit(
            f"Слаг sources_dataset '{ds_slug}' не kaggle-каноничен: нужны только "
            "[a-z0-9-], длина 6-50."
        )
    cfg["owner"] = owner
    return cfg


def kaggle_cli() -> str:
    path = shutil.which("kaggle")
    if not path:
        raise SystemExit("kaggle CLI не найден — запусти через uv run --project kaggle.")
    return path


def stream(cmd: list[str], cwd: Path | None = None) -> int:
    """Запустить и лить вывод в терминал по мере поступления."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=cwd or REPO_ROOT,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    return proc.returncode
