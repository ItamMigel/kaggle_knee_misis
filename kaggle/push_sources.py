#!/usr/bin/env python3
"""push_sources.py — залить payload (src/ + utils/) в sources-датасет.

    uv run --project kaggle python kaggle/push_sources.py

Нужен только когда менялся payload. Поток:
  1. payload обязан быть закоммичен — git_sha в стампе должен описывать ровно
     то, что уезжает. Остальное дерево (эксп-скрипты, заметки) может быть
     грязным сколько угодно, оно в датасет не едет.
  2. staging ← копия PAYLOAD_DIRS. Состав задаёт единственная константа
     utils/ctx.PAYLOAD_DIRS — та же, по которой хеш пересчитывается внутри
     кернела. Симметрия hash(репа) == hash(staging) ассертится тут же.
  3. _stamp.json = {src_hash, git_sha, pushed_at}. Его читают двое: пре-флайт
     в run_kernel (не устарел ли датасет) и ctx внутри кернела (целостность).
  4. upload_dataset.py апстрима: create при первом пуше, version дальше.

Пушить можно только в СВОЙ датасет: у каждого члена команды он свой, конфиг
персональный.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

c.require_env()

from utils.ctx import PAYLOAD_DIRS, STAMP_NAME, compute_src_hash  # noqa: E402

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".git")


MAX_LIST_PAGES = 50


def listed_as_mine(ds_ref: str, status_output: str) -> bool:
    """Есть ли ds_ref среди СВОИХ датасетов: все страницы `datasets list --mine`.

    Вывод «нет» делается только по полному списку: страницы идут до явного
    «No datasets found»; упавший вызов, незнакомый вывод или слишком много
    страниц → крэш с обоими выводами (status и list).
    """
    slug = ds_ref.partition("/")[2]
    refs: set[str] = set()
    for page in range(1, MAX_LIST_PAGES + 1):
        res = subprocess.run(
            [c.kaggle_cli(), "datasets", "list", "--mine", "--search", slug, "--csv",
             "-p", str(page)],
            capture_output=True, text=True,
        )
        out = res.stdout.strip()
        if res.returncode == 0 and out == "No datasets found":
            return ds_ref in refs
        lines = out.splitlines()
        rows = list(csv.reader(lines[1:])) if lines and lines[0].startswith("ref,") else []
        if res.returncode != 0 or not rows:
            raise SystemExit(
                f"Не смог проверить {ds_ref} по списку своих датасетов (страница {page}).\n"
                f"datasets status:\n{status_output}\n"
                f"datasets list STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        refs |= {row[0] for row in rows}
    raise SystemExit(
        f"Список своих датасетов по '{slug}' длиннее {MAX_LIST_PAGES} страниц — "
        f"по неполному списку отсутствие {ds_ref} не выводим.\ndatasets status:\n{status_output}"
    )


def dataset_exists(ds_ref: str) -> bool:
    """Развилка create/version. Неоднозначность ветку НЕ выбирает:
    rc==0 → есть; явный 404 → нет; 403 → решает полный список своих датасетов;
    всё прочее (сеть/авторизация) → крэш."""
    res = subprocess.run(
        [c.kaggle_cli(), "datasets", "status", ds_ref], capture_output=True, text=True
    )
    if res.returncode == 0:
        return True
    output = (res.stdout + res.stderr).lower()
    if "404" in output or "not found" in output:
        return False
    # 2026-10-17: кагл отвечает 403, а не 404, на ОТСУТСТВУЮЩИЙ датасет в своём
    # пространстве имён (существующий свой по-прежнему даёт rc 0). «Нет» от
    # «нельзя» отличаем по полному списку своих датасетов.
    if "403" in output:
        status_output = f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        if not listed_as_mine(ds_ref, status_output):
            return False
        raise SystemExit(
            f"{ds_ref} есть в списке своих датасетов, но status отвечает 403 — "
            f"разберись с токеном/правами:\n{status_output}"
        )
    raise SystemExit(
        f"Не смог определить, существует ли {ds_ref} (это не 404 — сеть? токен?):\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )


def main() -> None:
    cfg = c.load_config()
    ds_ref = cfg["sources_dataset"]
    ds_slug = ds_ref.partition("/")[2]

    dirty = c.git("status", "--porcelain", "--", *PAYLOAD_DIRS).strip()
    if dirty:
        raise SystemExit(
            "Payload не закоммичен — git_sha в стампе описывал бы не то, что уедет:\n"
            f"{dirty}\nЗакоммить и повтори."
        )
    git_sha = c.git("rev-parse", "HEAD").strip()

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "staging"
        staging.mkdir()
        copied = []
        for name in PAYLOAD_DIRS:
            source = c.REPO_ROOT / name
            if source.is_dir():
                shutil.copytree(source, staging / name, ignore=_IGNORE)
                copied.append(name)
        if not copied:
            raise SystemExit(f"В репе нет ни одной payload-папки из {PAYLOAD_DIRS}.")

        src_hash = compute_src_hash(staging)
        local_hash = compute_src_hash(c.REPO_ROOT)
        if src_hash != local_hash:
            # Копия обязана хешироваться как оригинал, иначе кернел свалится на
            # сверке, а причина будет неочевидна.
            raise SystemExit(
                "Staging разошёлся с репой (баг копирования?):\n"
                f"  staging: {src_hash}\n  репа:    {local_hash}"
            )

        (staging / STAMP_NAME).write_text(
            json.dumps(
                {
                    "src_hash": src_hash,
                    "git_sha": git_sha,
                    "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        notes = f"src {src_hash[7:19]} @ git {git_sha[:10]}"
        cmd = [
            sys.executable, str(c.vendor_script("upload_dataset.py")),
            str(staging), "--title", ds_slug,
        ]
        if dataset_exists(ds_ref):
            cmd += ["--version-notes", notes]
            print(f"[push_sources] новая версия {ds_ref}: {notes}")
        else:
            print(f"[push_sources] первый пуш — создаю {ds_ref}")

        print(f"[push_sources] payload: {', '.join(copied)} | {src_hash}")
        if c.stream(cmd) != 0:
            raise SystemExit("[push_sources] upload_dataset.py упал (вывод выше).")

    print(f"[push_sources] OK | {ds_ref} | {src_hash}")


if __name__ == "__main__":
    main()
