#!/usr/bin/env python3
"""run_kernel.py — запустить скрипт эксперимента кернелом на кагле.

    uv run --project kaggle python kaggle/run_kernel.py \\
        --exp exp-0001 --script fit --accelerator t4
    ... --kernel <run_id>,owner/slug        # свои запуски и/или чужие кернелы
    ... --ds owner/slug                     # датасеты
    ... --comp other-competition            # данные другого соревнования
    ... --submit-file submission.csv        # после кернела сабмитнуть файл
    ... --docker-image gcr.io/...@sha256:…  # образ кернела (дефолт — свежий образ кагла)
    ... --internet                          # включить интернет (дефолт — выключен)
    ... --dry-run                           # показать metadata и выйти

Скрипт уезжает телом кернела КАК ЕСТЬ: кагл сохранит на странице кернела
буквально тот код, который выполнился. Внутри кернела run_id не нужен и не
известен — скрипт пишет в /kaggle/working, а id живёт в слаге и в launch.json.

Две фазы: plan() — всё локальное, сети не касается (её и показывает --dry-run);
launch() — сетевые проверки, реестр и пуш.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

c.require_env()

from utils.ctx import (  # noqa: E402
    LAUNCH_NAME,
    PAYLOAD_DIRS,
    RECORD_NAME,
    STAMP_NAME,
    compute_src_hash,
    new_run_id,
    read_stamp,
    run_dir,
)

# P100 (sm_60) может не поддерживаться торчем текущего кагл-образа
# («no kernel image is available»), поэтому дефолт GPU = t4. НЕ полагаться на
# дефолт API — там P100.
GPU_SHAPES = {"t4": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100"}

MAX_SLUG_LEN = 50


def split(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_kernels(values: list[str]) -> list[str]:
    """--kernel принимает два вида значения, и различить их можно однозначно:

      'owner/slug'  — чужой (или любой готовый) кернел, идёт как есть;
      '<run_id>'    — наш запуск: в id слеша не бывает, поэтому лезем в его
                      запись за полем kernel.
    """
    refs = []
    for value in values:
        if "/" in value:
            refs.append(value)
            continue
        matches = list(c.REPO_ROOT.glob(f"exps/*/runs/{value}"))
        if not matches:
            raise SystemExit(
                f"'{value}' не похож на 'owner/slug' и не найден в exps/*/runs/ — "
                "опечатка в id?"
            )
        record = matches[0] / RECORD_NAME
        if not record.exists():
            raise SystemExit(
                f"У запуска {value} нет {RECORD_NAME} — он ещё не зафетчен или "
                "упал до записи."
            )
        kernel = json.loads(record.read_text(encoding="utf-8")).get("kernel")
        if not kernel:
            raise SystemExit(
                f"Запуск {value} был локальным — его артефактов на кагле нет.\n"
                "Залей их датасетом и подключи через --ds."
            )
        refs.append(kernel)
    return refs


def plan(args: argparse.Namespace, cfg: dict) -> dict:
    """Локальная часть: проверки, минт id, аттачи, kernel-metadata."""
    # Имя экспа уезжает в слаг КАК ЕСТЬ — кагл принимает только [a-z0-9-].
    # Молча санитизировать нельзя: слаг разъедется с тем, по которому ctx потом
    # ищет приаттаченный запуск. Имя скрипта в слаг не идёт — там можно всё.
    if not re.fullmatch(r"[a-z0-9-]+", args.exp):
        raise SystemExit(
            f"Имя экспа '{args.exp}' не kaggle-safe: только [a-z0-9-] (конвенция: exp-0001)."
        )
    # Сабмит идёт через competition_submit_code, а code competition принимает
    # только кернелы с выключенным интернетом — иначе прогон потрачен зря.
    if args.internet and args.submit_file:
        raise SystemExit(
            "--internet несовместим с --submit-file: code competition не примет "
            "кернел с включённым интернетом."
        )

    script_file = c.REPO_ROOT / "exps" / args.exp / f"{args.script}.py"
    if not script_file.exists():
        raise SystemExit(f"Нет скрипта: {script_file.relative_to(c.REPO_ROOT)}")

    rel_script = str(script_file.relative_to(c.REPO_ROOT))
    dirty = c.git("status", "--porcelain", "--", *PAYLOAD_DIRS, rel_script).strip()
    if dirty:
        raise SystemExit(
            "Незакоммичено то, что уедет на кагл (payload и/или сам скрипт) — "
            f"запись о запуске врала бы про git_sha:\n{dirty}\nЗакоммить и повтори."
        )

    run_id = new_run_id()
    slug = f"{args.exp}-{run_id}"
    if len(slug) > MAX_SLUG_LEN:
        raise SystemExit(f"Слаг '{slug}' длиннее {MAX_SLUG_LEN} — укороти имя экспа.")

    # Соревнование из конфига и sources-датасет подставляются всегда; ручки
    # только ДОБАВЛЯЮТ. dict.fromkeys — дедуп с сохранением порядка.
    metadata = {
        "id": f"{cfg['owner']}/{slug}",
        "title": slug,
        "code_file": script_file.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": args.accelerator != "none",
        "enable_internet": args.internet,
        "dataset_sources": list(dict.fromkeys([cfg["sources_dataset"], *split(args.ds)])),
        "competition_sources": list(dict.fromkeys([cfg["competition"], *split(args.comp)])),
        "kernel_sources": list(dict.fromkeys(resolve_kernels(split(args.kernel)))),
    }
    if args.accelerator != "none":
        metadata["machine_shape"] = GPU_SHAPES[args.accelerator]
    # Без поля кагл берёт свежий образ. Закрепить образ форкнутого ноутбука —
    # единственный способ получить его окружение (питон, torch, timm).
    if args.docker_image:
        metadata["docker_image"] = args.docker_image

    return {
        "run_id": run_id,
        "slug": slug,
        "script_file": script_file,
        "metadata": metadata,
        "git_sha": c.git("rev-parse", "HEAD").strip(),
        "sources_hash": compute_src_hash(c.REPO_ROOT),
    }


def preflight_sources(cfg: dict, local_hash: str) -> None:
    """Стамп последней версии sources-датасета против локального payload'а.

    Это единственная проверка, которая реально ловит «правил src, забыл
    push_sources». Сверка внутри кернела на это не способна: в устаревшем
    датасете лежит его же стамп, и хеш с ним сойдётся.
    """
    with tempfile.TemporaryDirectory() as td:
        res = subprocess.run(
            [c.kaggle_cli(), "datasets", "download", cfg["sources_dataset"],
             "-f", STAMP_NAME, "-p", td, "--force"],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise SystemExit(
                f"Не удалось скачать {STAMP_NAME} из {cfg['sources_dataset']} — "
                "датасета ещё нет? Запусти kaggle/push_sources.py.\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        zipped = Path(td) / f"{STAMP_NAME}.zip"
        # CLI отдаёт мелкие файлы как есть (проверено), но ветка оставлена, чтобы
        # при смене поведения диагноз не врал.
        if not (Path(td) / STAMP_NAME).exists() and zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extract(STAMP_NAME, td)
        try:
            attached = read_stamp(td)["src_hash"]
        except FileNotFoundError:
            raise SystemExit(
                f"В sources-датасете нет {STAMP_NAME} — он залит не push_sources'ом."
            ) from None
    if attached != local_hash:
        raise SystemExit(
            "Sources-датасет отстал от локального payload'а:\n"
            f"  локально:   {local_hash}\n"
            f"  в датасете: {attached}\n"
            "Запусти kaggle/push_sources.py и повтори."
        )


def launch(args: argparse.Namespace, cfg: dict, p: dict) -> int:
    """Сетевая часть: проверки, запись о запуске, пуш кернела."""
    preflight_sources(cfg, p["sources_hash"])

    # launch.json — половина записи, которую знает только хост: эксп, скрипт,
    # ref кернела и АКТУАЛЬНАЯ git_sha (в стампе лежит sha момента пуша сурсов,
    # а скрипт мог закоммититься позже). Фетч сольёт её с записью кернела.
    (run_dir(c.REPO_ROOT, args.exp, p["run_id"]) / LAUNCH_NAME).write_text(
        json.dumps(
            {
                "run_id": p["run_id"],
                "exp": args.exp,
                "script": p["script_file"].name,
                "kernel": p["metadata"]["id"],
                "git_sha": p["git_sha"],
                "sources_hash": p["sources_hash"],
                "attached": {
                    "competitions": p["metadata"]["competition_sources"],
                    "datasets": p["metadata"]["dataset_sources"],
                    "kernels": p["metadata"]["kernel_sources"],
                },
                "launched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        shutil.copy2(p["script_file"], staging / p["script_file"].name)
        (staging / "kernel-metadata.json").write_text(
            json.dumps(p["metadata"], indent=2) + "\n", encoding="utf-8"
        )
        cmd = [sys.executable, str(c.vendor_script("submit_kernel.py")), str(staging)]
        if args.submit_file:
            # --message = run_id: по нему потом джойнятся LB-скоры.
            cmd += ["--file", args.submit_file, "--message", p["run_id"]]

        print(f"[run_kernel] {p['metadata']['id']} | accelerator={args.accelerator}")
        if p["metadata"]["kernel_sources"]:
            print(f"[run_kernel] аттачу выходы: {', '.join(p['metadata']['kernel_sources'])}")
        # Апстрим ищет корень проекта по pyproject.toml; в корне репы его нет, и
        # без подсказки лог сабмитов уехал бы в vendor/nvidia-kaggle/data/.
        os.environ["PROJECT_ROOT"] = str(c.REPO_ROOT)
        return c.stream(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="запустить скрипт экспа кернелом")
    parser.add_argument("--exp", required=True, help="имя экспа: exp-0001")
    parser.add_argument("--script", required=True, help="имя скрипта без .py: fit")
    parser.add_argument("--comp", default="",
                        help="доп. соревнования через запятую (своё берётся из конфига)")
    parser.add_argument("--ds", default="",
                        help="датасеты owner/slug через запятую (sources берётся из конфига)")
    parser.add_argument("--kernel", default="",
                        help="через запятую: run_id своего запуска или owner/slug чужого кернела")
    parser.add_argument(
        "--accelerator", choices=["none", "t4", "p100"], default="none",
        help="t4 — дефолт для GPU; p100 может не работать с torch",
    )
    parser.add_argument("--submit-file", default=None,
                        help="имя файла сабмишна в /kaggle/working")
    parser.add_argument("--docker-image", default="",
                        help="образ кернела, напр. образ форкнутого ноутбука; дефолт — свежий")
    parser.add_argument("--internet", action="store_true",
                        help="включить интернет в кернеле (несовместимо с --submit-file)")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать kernel-metadata и выйти")
    args = parser.parse_args()

    cfg = c.load_config()
    p = plan(args, cfg)

    if args.dry_run:
        print(json.dumps(p["metadata"], indent=2, ensure_ascii=False))
        print(f"[run_kernel] --dry-run; run_id был бы {p['run_id']}")
        return

    code = launch(args, cfg, p)
    where = f"https://www.kaggle.com/code/{p['metadata']['id']}"
    if code == 0:
        print(f"[run_kernel] OK | {p['run_id']} | {where}")
        print("[run_kernel] дальше: uv run --project kaggle python kaggle/fetch.py "
              f"--kernel {p['metadata']['id']}")
    else:
        print(f"[run_kernel] FAILED (exit {code}) | {where}")
        print("[run_kernel] причина — в логе кернела: kaggle/fetch.py --kernel "
              f"{p['metadata']['id']} --log-only")
    sys.exit(code)


if __name__ == "__main__":
    main()
