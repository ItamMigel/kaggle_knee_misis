#!/usr/bin/env python3
"""pull_data.py — скачать данные в локальный кэш с кагловой раскладкой.

    uv run --project kaggle python kaggle/pull_data.py            # своё соревнование
    uv run --project kaggle python kaggle/pull_data.py --comp other-competition
    uv run --project kaggle python kaggle/pull_data.py --ds owner/slug

Смысл ровно один: разложить скачанное так же, как это выглядит на кагле, чтобы
`ctx.mounts[...]` резолвился одним и тем же кодом в обеих средах.

    кагл:     /kaggle/input/competitions/titanic/
    локально: <kaggle_cache>/competitions/titanic/

    кагл:     /kaggle/input/datasets/owner/slug/
    локально: <kaggle_cache>/datasets/owner/slug/

Корень кэша — поле kaggle_cache в local_paths.json.

Уже скачанное не перекачивается: если папка непустая, скрипт говорит об этом и
ничего не делает. Нужно перекачать — удали папку руками.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

c.require_env()

from utils.ctx import local_paths  # noqa: E402


def cache_root() -> Path:
    root = Path(local_paths(c.REPO_ROOT)["kaggle_cache"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def ready(target: Path) -> bool:
    if target.is_dir() and any(target.iterdir()):
        print(f"[pull_data] уже на месте: {target} (удали папку, чтобы перекачать)")
        return True
    target.mkdir(parents=True, exist_ok=True)
    return False


def unzip_all(target: Path) -> None:
    """competitions download отдаёт архив и распаковывать не умеет (в отличие от
    datasets download --unzip) — делаем сами."""
    for archive in sorted(target.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        archive.unlink()


def run(cmd: list[str], what: str) -> None:
    if c.stream(cmd) != 0:
        raise SystemExit(f"[pull_data] не удалось скачать {what} (вывод выше).")


def main() -> None:
    parser = argparse.ArgumentParser(description="скачать данные в локальный кэш")
    parser.add_argument("--comp", default=None,
                        help="слаг соревнования (по умолчанию — своё, из competition.json)")
    parser.add_argument("--ds", default=None, help="датасет owner/slug")
    args = parser.parse_args()

    if args.comp and args.ds:
        raise SystemExit("--comp и --ds сразу не бывает: за один раз один источник.")

    root = cache_root()

    if args.ds:
        if "/" not in args.ds:
            raise SystemExit("--ds должен быть 'owner/slug'.")
        target = root / "datasets" / args.ds
        if ready(target):
            return
        run([c.kaggle_cli(), "datasets", "download", args.ds, "-p", str(target), "--unzip"],
            args.ds)
    else:
        comp = args.comp or c.load_config()["competition"]
        target = root / "competitions" / comp
        if ready(target):
            return
        run([c.kaggle_cli(), "competitions", "download", comp, "-p", str(target)], comp)
        unzip_all(target)

    files = sorted(p for p in target.rglob("*") if p.is_file())
    print(f"[pull_data] OK | {target} | файлов: {len(files)}")
    for p in files[:10]:
        print(f"  {p.relative_to(target)}")
    if len(files) > 10:
        print(f"  ... ещё {len(files) - 10}")


if __name__ == "__main__":
    main()
