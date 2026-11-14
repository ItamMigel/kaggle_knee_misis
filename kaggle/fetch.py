#!/usr/bin/env python3
"""fetch.py — забрать выход кернела в реестр и достроить запись о запуске.

    uv run --project kaggle python kaggle/fetch.py --kernel <owner>/<slug>
    ... --all        # вместе с тяжёлыми артефактами (нужно, чтобы гонять локально)
    ... --log-only   # только лог, ничего не трогая (разбор упавшего кернела)
    ... --from-log [--skip-file NAME]  # выход не качать: запись и файлы — из лога

Куда класть — не гадаем и слаг не разбираем: ищем `launch.json`, который положил
run_kernel, по совпадению поля `kernel`. Оттуда же берём то, чего кернел про себя
знать не мог.

Запись собирается из двух половин:
  кернел  → env, status, время, sources_hash, used, produced
  хост    → run_id, exp, script, kernel, attached и АКТУАЛЬНАЯ code.git_sha
            (в стампе лежит sha момента пуша сурсов — она описывает payload, но
            не скрипт, который мог закоммититься позже; стампная уезжает в
            отдельное поле sources_git_sha)

Заодно тут ловится дыра, которую не ловит ничто другое: если кернелу приехала
устаревшая версия sources-датасета, его sources_hash не совпадёт с тем, что был
локально в момент запуска. До запуска это не поймать (в устаревшем датасете
лежит его же стамп, и они сходятся между собой), а постфактум — можно.

--from-log — для кернелов, чей выход не скачать (www.kaggleusercontent.com
недоступен): record.json берётся из блока @@RECORD, который печатает хук
utils/ctx.py, файлы — из блоков src/log_transport.emit_file. Всё сверяется
(sha256 записи, crc/sha256 файлов, хеши файлов против produced записи); чего-то
не хватает — падаем. Никакого перехода между скачиванием и логом ни в одну
сторону: какой режим выбран, тот и работает.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

c.require_env()

from src.log_transport import LogTransportError, decode_log  # noqa: E402
from utils.ctx import (  # noqa: E402
    LAUNCH_NAME,
    RECORD_LOG_BEGIN,
    RECORD_LOG_END,
    RECORD_LOG_VERSION,
    RECORD_NAME,
)

# Тяжёлое не качаем по умолчанию: оно и так лежит в выходе кернела, а таскать
# веса в репу незачем. Список должен совпадать с .gitignore — правишь здесь,
# правь и там.
HEAVY_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".h5",
    ".pkl", ".npy", ".npz", ".parquet", ".feather",
}

# Кагл докидывает своё в выход кернела — это не наши артефакты.
KAGGLE_SERVICE = ("__script__", "__results__", "__output__", "custom.css")


def find_launch(kernel_ref: str) -> tuple[Path, dict]:
    """Папка запуска и его launch.json по ref кернела."""
    for path in sorted(c.REPO_ROOT.glob(f"exps/*/runs/*/{LAUNCH_NAME}")):
        launch = json.loads(path.read_text(encoding="utf-8"))
        if launch.get("kernel") == kernel_ref:
            return path.parent, launch
    raise SystemExit(
        f"Не нашёл запуск с kernel={kernel_ref} среди exps/*/runs/*/{LAUNCH_NAME}.\n"
        "Кернел запускали не через kaggle/run_kernel.py? Тогда неизвестно, в какую "
        "папку реестра его класть."
    )


def list_output(kernel_ref: str) -> tuple[list, str]:
    """Пофайловый listing выхода кернела + лог. Пагинация учтена.

    Не `kaggle kernels output` — тот тянет весь выход целиком, включая веса.
    """
    from kagglesdk import KaggleClient
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )

    owner, _, slug = kernel_ref.partition("/")
    if not slug:
        raise SystemExit(f"--kernel должен быть 'owner/slug', а не '{kernel_ref}'.")
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("KAGGLE_API_TOKEN не выставлен в окружении.")

    files, log, page = [], "", None
    with KaggleClient(api_token=token) as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = owner, slug
            if page:
                req.page_token = page
            out = client.kernels.kernels_api_client.list_kernel_session_output(req)
            files.extend(out.files or [])
            log = out.log or log
            page = out.next_page_token
            if not page:
                break
    return files, log


def write_log(raw: str, dest: Path) -> None:
    """Лог кернела приходит json-массивом {stream_name, time, data}."""
    try:
        text = "".join(f"[{e['stream_name']}] {e['data']}" for e in json.loads(raw))
    except (json.JSONDecodeError, KeyError, TypeError):
        print("[fetch] лог не в ожидаемом json-формате — пишу как есть")
        text = raw
    (dest / "log.txt").write_text(text, encoding="utf-8")


def kernel_stdout(raw: str) -> str:
    """stdout кернела, как он печатался: data всех записей stream_name=stdout подряд.

    Без префиксов «[stdout] » и без stderr-строк между нашими: строка, которую
    кагл разрезал на две записи, склеивается обратно сама.
    """
    try:
        return "".join(e["data"] for e in json.loads(raw) if e["stream_name"] == "stdout")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise SystemExit(
            f"[fetch] --from-log: лог не json-массив {{stream_name, time, data}} ({e!r}) — "
            "восстанавливать запись не из чего."
        ) from None


def record_from_log(stdout: str) -> bytes:
    """Байты record.json из блока @@RECORD в stdout; версия и sha256 проверены."""
    lines = stdout.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if RECORD_LOG_BEGIN in line]
    if len(hits) != 1:
        raise SystemExit(
            f"[fetch] --from-log: в логе {len(hits)} блоков {RECORD_LOG_BEGIN}, нужен ровно один. "
            "Ноль — кернел убит до хука записи или бежал на ctx без него; запись из лога "
            "не восстановить."
        )
    begin = hits[0]
    header = lines[begin][lines[begin].index(RECORD_LOG_BEGIN) + len(RECORD_LOG_BEGIN):].strip()
    m = re.fullmatch(r"v=(\S+) sha256=([0-9a-f]{64})", header)
    if m is None:
        raise SystemExit(f"[fetch] --from-log: битый заголовок {RECORD_LOG_BEGIN}: {header[:200]!r}")
    if m.group(1) != RECORD_LOG_VERSION:
        raise SystemExit(
            f"[fetch] --from-log: блок записи формата v={m.group(1)}, этот фетч знает "
            f"v={RECORD_LOG_VERSION}. Разбирай кодом той версии, что его напечатала."
        )
    end = next((i for i in range(begin + 1, len(lines))
                if lines[i].rstrip("\r\n") == RECORD_LOG_END), None)
    if end is None:
        raise SystemExit(f"[fetch] --from-log: блок записи без {RECORD_LOG_END} — лог обрезан.")
    data = "".join(lines[begin + 1:end]).encode("utf-8")
    if hashlib.sha256(data).hexdigest() != m.group(2):
        raise SystemExit(
            "[fetch] --from-log: sha256 блока записи не сходится с заголовком — в лог "
            "вклинился чужой вывод или он побит."
        )
    if not isinstance(json.loads(data), dict):
        raise SystemExit("[fetch] --from-log: блок записи — не json-объект.")
    return data


def restore_from_log(raw: str, dest: Path, skip: list[str]) -> list[str]:
    """--from-log: record.json и файлы emit_file из лога в папку запуска; всё сверено.

    skip — имена, которые не проверяются и не пишутся (например, нарочно
    обрезанный probe); остальное обязано прийти целым и совпасть с produced.
    """
    stdout = kernel_stdout(raw)
    data = record_from_log(stdout)
    try:
        files = decode_log(stdout, tuple(skip))
    except LogTransportError as e:
        raise SystemExit(f"[fetch] --from-log: {e}") from None
    clash = sorted({RECORD_NAME, LAUNCH_NAME, "log.txt"} & set(files))
    if clash:
        raise SystemExit(f"[fetch] --from-log: в логе файлы со служебными именами: {clash}")
    produced = {p["name"]: p["hash"] for p in json.loads(data)["produced"]}
    for name, content in files.items():
        if name not in produced:
            raise SystemExit(
                f"[fetch] --from-log: {name} есть в логе, но не в produced записи — "
                "сверить его не с чем."
            )
        if f"sha256:{hashlib.sha256(content).hexdigest()}" != produced[name]:
            raise SystemExit(
                f"[fetch] --from-log: {name} из лога не совпадает с хешем в produced записи."
            )
    (dest / RECORD_NAME).write_bytes(data)
    for name, content in files.items():
        (dest / name).write_bytes(content)
    rest = sorted(set(produced) - set(files))
    if rest:
        print(f"[fetch] в логе нет (остались только в выходе кернела): {', '.join(rest)}")
    return [RECORD_NAME, *sorted(files)]


def merge(launch: dict, kernel_record: dict | None) -> dict:
    """Две половины в одну запись."""
    if kernel_record is None:
        # Записи в выходе нет — процесс убили снаружи (таймаут, OOM), хук не
        # успел отработать. Факт запуска не теряем, но produced не выдумываем.
        record = {"env": "kaggle", "status": "killed", "used": [], "produced": []}
    else:
        record = dict(kernel_record)

    record["run_id"] = launch["run_id"]
    record["exp"] = launch["exp"]
    record["script"] = launch["script"]
    record["kernel"] = launch["kernel"]
    record["attached"] = launch["attached"]

    # sha из стампа описывает payload на момент пуша сурсов; актуальная,
    # покрывающая и скрипт, — у хоста.
    stamp_sha = (record.get("code") or {}).get("git_sha")
    if stamp_sha:
        record["sources_git_sha"] = stamp_sha
    record["code"] = {"git_sha": launch["git_sha"], "dirty": False}

    actual, expected = record.get("sources_hash"), launch["sources_hash"]
    if actual and actual != expected:
        record["sources_hash_expected"] = expected
        print(
            "\n[fetch] ВНИМАНИЕ: кернел бежал не на том payload'е, что был локально.\n"
            f"  локально при запуске: {expected}\n"
            f"  реально в кернеле:    {actual}\n"
            "  Кагл подсунул устаревшую версию sources-датасета. Выводы по этому "
            "запуску недостоверны — перезапусти.\n"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="забрать выход кернела в реестр")
    parser.add_argument("--kernel", required=True, help="owner/slug (печатает run_kernel)")
    parser.add_argument("--all", action="store_true",
                        help="качать и тяжёлые артефакты (веса и т.п.)")
    parser.add_argument("--log-only", action="store_true",
                        help="только лог, запись не трогать")
    parser.add_argument("--from-log", action="store_true",
                        help="выход не качать: record.json и файлы emit_file — из лога")
    parser.add_argument("--skip-file", action="append", default=[], metavar="NAME",
                        help="только с --from-log: файл из лога не проверять и не писать")
    args = parser.parse_args()
    if args.from_log and (args.all or args.log_only):
        parser.error("--from-log несовместим с --all и --log-only")
    if args.skip_file and not args.from_log:
        parser.error("--skip-file работает только с --from-log")

    dest, launch = find_launch(args.kernel)
    files, log = list_output(args.kernel)
    write_log(log, dest)

    if args.log_only:
        print(f"[fetch] лог: {(dest / 'log.txt').relative_to(c.REPO_ROOT)}")
        return

    downloaded, skipped = [], []
    if args.from_log:
        downloaded = restore_from_log(log, dest, args.skip_file)
    else:
        for f in files:
            name = Path(f.file_name)
            if name.name.startswith(KAGGLE_SERVICE):
                continue
            if name.suffix in HEAVY_SUFFIXES and not args.all:
                skipped.append(f.file_name)
                continue
            target = dest / f.file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f.url) as resp, target.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            downloaded.append(f.file_name)

    record_path = dest / RECORD_NAME
    kernel_record = (
        json.loads(record_path.read_text(encoding="utf-8"))
        if record_path.exists() else None
    )
    if kernel_record is None:
        print("[fetch] в выходе нет record.json — кернел убили снаружи (таймаут?). "
              "Регистрирую запуск со status=killed, смотри log.txt.")

    record_path.write_text(
        json.dumps(merge(launch, kernel_record), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # launch.json остаётся: это единственный способ найти запуск по ref кернела,
    # а фетч повторяют — сначала лёгкое, потом --all, когда веса понадобились
    # локально. Поля с записью пересекаются, зато фетч идемпотентен.

    print(f"[fetch] OK | {args.kernel} → {dest.relative_to(c.REPO_ROOT)}/")
    print(f"[fetch] забрал: {', '.join(downloaded + ['log.txt'])}")
    if skipped:
        print(f"[fetch] не качал (тяжёлое, живёт в выходе кернела): {', '.join(skipped)}")
        print("[fetch] нужно локально — повтори с --all")


if __name__ == "__main__":
    main()
