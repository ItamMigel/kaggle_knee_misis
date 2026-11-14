"""utils/ctx.py — контекст запуска. ЕДИНСТВЕННОЕ место, где различаются среды.

Скрипт эксперимента видит ровно три поля:

    ctx.sources_dir   корень кода (репа локально / маунт датасета на кагле)
    ctx.mounts        всё доступное: соревнования, датасеты, чужие запуски
    ctx.output_dir    куда писать всё

Чего в ctx НЕТ и почему:
  run_id / exp / script — скрипту не нужны (он пишет в output_dir), а главное:
      локально id известен в момент запуска, внутри кернела нет. Положи их в ctx —
      и контракт станет разным в двух средах.
  seed — константа в самом скрипте.
  env  — если он есть, кто-нибудь напишет `if ctx.env == "kaggle"`, и мы получим
      ровно то разное поведение в двух средах, от которого уходим.
  git_sha / dirty — нужны записи, не скрипту.

Только stdlib: модуль едет в sources-датасете и импортится ВНУТРИ кернела.

Запись пишет хук на завершение процесса (см. _install_recorder), поэтому не бывает
запуска без записи: упал скрипт — запись всё равно есть, со status="failed".
Оговорка: если процесс убит снаружи (таймаут кагла, SIGKILL), хук не отработает —
тогда фетч заводит папку с логом и записью status="killed".

Импортируется бутстрапом в начале каждого скрипта:

    import sys, glob
    from pathlib import Path
    _p  = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
    _p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
    sys.path[:0] = _p[:1]
    from utils.ctx import get_ctx
"""

from __future__ import annotations

import atexit
import hashlib
import json
import secrets
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

RECORD_NAME = "record.json"
LAUNCH_NAME = "launch.json"  # пишет хост при кагл-запуске; фетч сольёт с record
DIFF_NAME = "diff.patch"
TRACEBACK_NAME = "traceback.txt"
# Копия записи в stdout (выход кернела бывает не скачать, лог — можно; читает
# kaggle/fetch.py --from-log). Меняешь формат блока — подними версию.
RECORD_LOG_BEGIN = "@@RECORD BEGIN"
RECORD_LOG_END = "@@RECORD END"
RECORD_LOG_VERSION = "1"
LOCAL_PATHS_NAME = "local_paths.json"

# Что НЕ попадает в produced. Список короткий и намеренно лежит на виду.
#   record.json — физически не может: он пишется ПОСЛЕ хеширования и описывал бы
#                 сам себя.
#   мусор кагла — он докидывает своё в output кернела, это не наши артефакты.
# Всё остальное, что скрипт положил в output_dir, включая diff.patch и
# traceback.txt, в produced попадает.
_NOT_ARTIFACTS = {RECORD_NAME}
_KAGGLE_SERVICE = ("__script__", "__results__", "__output__", "custom.css")

_CHUNK = 1 << 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Читаемый, сортируемый, уникальный id: YYYYmmddtHHMMSS-<ms>-<4hex>.

    Только lowercase и дефисы → id валиден в слаге kaggle-кернела КАК ЕСТЬ.
    По нему же ctx находит приаттаченный запуск (поиск подстроки в имени папки).
    """
    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%dt%H%M%S}-{now.microsecond // 1000:03d}-{secrets.token_hex(2)}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------- payload

# Что уезжает в sources-датасет и что хешируется. Единственный источник правды
# о составе payload'а: и хеш считается по нему, и kaggle/push_sources.py собирает
# staging копированием ровно этих папок.
PAYLOAD_DIRS = ("src", "utils")

STAMP_NAME = "_stamp.json"

# Не участвует в хеше: сам стамп (кладётся уже после подсчёта), кэши питона,
# мусор macOS, git-служебное.
_EXCLUDED_DIRS = {"__pycache__", ".git"}
_EXCLUDED_FILES = {STAMP_NAME, ".DS_Store"}
_EXCLUDED_SUFFIXES = {".pyc"}


def payload_files(root: Path | str) -> list[Path]:
    """Все файлы PAYLOAD_DIRS под root, кроме исключений.

    Папка из PAYLOAD_DIRS может отсутствовать — тогда она просто не вносит
    вклада; git и staging пустых папок тоже не переносят, так что обе стороны
    консистентны.
    """
    root = Path(root)
    files = []
    for dir_name in PAYLOAD_DIRS:
        base = root / dir_name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _EXCLUDED_DIRS for part in path.parts):
                continue
            if path.name in _EXCLUDED_FILES or path.suffix in _EXCLUDED_SUFFIXES:
                continue
            files.append(path)
    return files


def compute_src_hash(root: Path | str) -> str:
    """Каноничный дерево-хеш содержимого PAYLOAD_DIRS под root → "sha256:<hex>".

    Файлы сортируются по относительному пути; в хеш для каждого подаётся
    rel_path + разделитель + содержимое. Значит переименование меняет хеш так же,
    как правка байтов, а порядок обхода ФС не влияет.

    root — локально корень репы, на кагле корень приаттаченного sources-датасета.
    """
    root = Path(root)
    h = hashlib.sha256()
    for path in sorted(payload_files(root), key=lambda p: str(p.relative_to(root))):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def read_stamp(directory: Path | str) -> dict:
    """Прочитать _stamp.json из папки датасета. Нет файла → FileNotFoundError.

    Стамп кладёт kaggle/push_sources.py: src_hash (целостность) и git_sha момента
    пуша (единственный провенанс кода на кагле — git-репы там нет).
    """
    text = (Path(directory) / STAMP_NAME).read_text(encoding="utf-8")
    return json.loads(text)


# ---------------------------------------------------------------- mounts


class Mounts:
    """Словарь «ключ → папка»: резолвит лениво и помнит, о чём его спрашивали.

    Ленивость важна: «доступны все запуски репы» не стоит ничего, потому что
    список целиком никогда не строится заранее.

    touched отдаётся в запись как поле used — БЕЗ фильтрации: там будет видно и
    соревнование, и датасеты, и запуски, то есть буквально всё, что читал скрипт.
    """

    def __init__(self, scan: Callable[[], list[tuple[str, Path, bool]]], empty_hint: str):
        self._scan = scan
        self._empty_hint = empty_hint
        self.touched: set[str] = set()

    def _describe(self) -> str:
        lines = [f"  {k}{'  (запуск)' if is_run else ''}" for k, _, is_run in self._scan()]
        return "\n".join(lines) or self._empty_hint

    def __getitem__(self, key: str) -> Path:
        # Запуски матчатся ПО ПОДСТРОКЕ: слаг кернела собираем мы сами, но ctx о
        # его формате ничего не знает — значит формат можно менять свободно.
        found = [p for k, p, is_run in self._scan() if k == key or (is_run and key in k)]
        if len(found) != 1:
            # RuntimeError, а не KeyError: у KeyError текст печатается через repr,
            # и многострочная подсказка схлопывается в нечитаемую строку.
            if not found:
                raise RuntimeError(
                    f"Нет источника с ключом '{key}'.\nДоступно:\n{self._describe()}\n"
                    "На кагле источник надо приаттачить при запуске: --kernel "
                    "<run_id|owner/slug>, --ds <owner/slug>, --comp <slug>.\n"
                    "Локально данные кладёт kaggle/pull_data.py."
                )
            listed = "\n".join(f"  {p}" for p in found)
            raise RuntimeError(f"Ключ '{key}' подходит нескольким источникам:\n{listed}")
        self.touched.add(key)
        return found[0]

    def __repr__(self) -> str:
        return f"Mounts(\n{self._describe()}\n)"


def _mounts(data_root: Path, runs: Callable[[], list[Path]], empty_hint: str) -> Mounts:
    """Единая раскладка источников; отличается только корень.

        <корень>/competitions/<slug>/      → ключ "<slug>"
        <корень>/datasets/<owner>/<slug>/  → ключ "<owner>/<slug>"
        папки запусков                     → ключ = id запуска

    На кагле корень — /kaggle/input (раскладка проверена пробами, см.
    kaggle_facts.md), локально — кэш из local_paths.json, который наполняет
    pull_data ровно по этой же схеме. Запуски в кэш не попадают: локально они
    лежат в репе (exps/*/runs/), на кагле — среди приаттаченных кернелов.
    """

    def scan() -> list[tuple[str, Path, bool]]:
        out = [(p.name, p, False) for p in sorted((data_root / "competitions").glob("*"))
               if p.is_dir()]
        out += [(f"{p.parent.name}/{p.name}", p, False)
                for p in sorted((data_root / "datasets").glob("*/*")) if p.is_dir()]
        out += [(p.name, p, True) for p in runs() if p.is_dir()]
        return out

    return Mounts(scan, empty_hint)


# ---------------------------------------------------------------- запись


def _artifacts(output_dir: Path) -> list[dict]:
    """Всё, что появилось в output_dir, с хешем и размером.

    Хешируется содержимое файла как есть (без пути в хеше): те же байты под
    другим именем — тот же хеш.
    """
    out = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir)
        if rel.name in _NOT_ARTIFACTS or rel.name.startswith(_KAGGLE_SERVICE):
            continue
        if "__pycache__" in rel.parts:
            continue
        out.append(
            {"name": str(rel), "hash": _sha256(path), "bytes": path.stat().st_size}
        )
    return out


def _install_recorder(record: dict, output_dir: Path, mounts: Mounts) -> None:
    """Повесить запись на завершение процесса.

    Падение ловится через sys.excepthook (он срабатывает раньше atexit), поэтому
    упавший запуск оставляет запись со status="failed" и трейсбеком рядом.
    Записанные байты record.json хук печатает и в stdout блоком @@RECORD (с их
    sha256) — по нему фетч восстанавливает запись из лога кернела.
    Оговорка: SystemExit мимо excepthook не проходит, а убитый снаружи процесс
    не доходит и до atexit — такой запуск опознаётся по отсутствию записи.
    """
    failure: dict[str, str] = {}
    previous = sys.excepthook

    def hook(exc_type, exc, tb):  # noqa: ANN001
        failure["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = hook

    def finish() -> None:
        record["finished"] = _now()
        record["status"] = "failed" if failure else "ok"
        if failure:
            (output_dir / TRACEBACK_NAME).write_text(
                failure["traceback"], encoding="utf-8"
            )
        record["used"] = sorted(mounts.touched)
        record["produced"] = _artifacts(output_dir)
        data = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        (output_dir / RECORD_NAME).write_bytes(data)
        _print_record(data)

    atexit.register(finish)


def _print_record(data: bytes) -> None:
    """Байты record.json в stdout между маркерами @@RECORD BEGIN v=.. sha256=.. / END.

    JSON с indent=2 и экранированными переводами строк: каждая его строка
    начинается с «{», «}» или пробела, маркеры с ним не спутать.
    """
    sys.stdout.flush()
    print(f"{RECORD_LOG_BEGIN} v={RECORD_LOG_VERSION} "
          f"sha256={hashlib.sha256(data).hexdigest()}", flush=True)
    sys.stdout.write(data.decode("utf-8"))
    print(RECORD_LOG_END, flush=True)


# ---------------------------------------------------------------- ctx


@dataclass(frozen=True)
class Ctx:
    sources_dir: Path
    mounts: Mounts
    output_dir: Path


def local_paths(repo: Path) -> dict:
    """Машинный конфиг. Публичная: её же читает kaggle/pull_data.py, чтобы знать,
    куда раскладывать скачанное."""
    path = repo / LOCAL_PATHS_NAME
    if not path.exists():
        raise SystemExit(
            f"Нет {path}. Скопируй local_paths.example.json в {LOCAL_PATHS_NAME} "
            "и заполни пути этой машины."
        )
    paths = json.loads(path.read_text(encoding="utf-8"))
    if "kaggle_cache" not in paths:
        raise SystemExit(
            f"В {LOCAL_PATHS_NAME} нет поля 'kaggle_cache' — это путь к локальному "
            "кэшу данных с кагловой раскладкой (competitions/<slug>/, "
            "datasets/<owner>/<slug>/). Смотри local_paths.example.json."
        )
    return paths


def _script_identity(repo: Path) -> tuple[str, str]:
    """(exp, script) из пути запускаемого файла: exps/<exp>/<script>.py."""
    main = sys.modules.get("__main__")
    file = getattr(main, "__file__", None)
    if file is None:
        raise SystemExit(
            "ctx не понял, какой файл запущен (нет __main__.__file__). "
            "Запускай скрипт файлом: python exps/<exp>/<script>.py"
        )
    path = Path(file).resolve()
    try:
        rel = path.relative_to(repo)
    except ValueError:
        raise SystemExit(f"Скрипт {path} лежит вне репы {repo}.") from None
    if len(rel.parts) < 3 or rel.parts[0] != "exps":
        raise SystemExit(f"Скрипт должен лежать в exps/<exp>/..., а он в {rel}.")
    # Эксп — второй сегмент, скрипт — весь остаток пути: подпапки внутри экспа
    # разрешены (exps/exp-0001/steps/fit.py → exp-0001, steps/fit.py).
    return rel.parts[1], str(Path(*rel.parts[2:]))


def git(repo: Path, *args: str) -> str:
    """git в заданной репе. Публичная: ею же пользуются скрипты kaggle/, чтобы
    реализация была одна."""
    out = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout


def run_dir(repo: Path, exp: str, run_id: str) -> Path:
    """Папка запуска. Раскладку реестра знает только этот модуль — и хост, и
    кернел ходят сюда, а не собирают путь руками."""
    d = repo / "exps" / exp / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kaggle_ctx(sources_dir: Path) -> Ctx:
    """Кагл: id не минтим (его знает только хост), сурсы сверяем со стампом."""
    actual = compute_src_hash(sources_dir)
    meta = read_stamp(sources_dir)
    for field in ("src_hash", "git_sha"):
        if field not in meta:
            raise RuntimeError(
                f"В {STAMP_NAME} sources-датасета нет поля '{field}' — он залит "
                "старой версией push_sources. Перезалей: kaggle/push_sources.py"
            )
    if actual != meta["src_hash"]:
        raise RuntimeError(
            "Содержимое sources-датасета не совпадает с его же стампом — "
            "битая или недокачанная заливка.\n"
            f"  в стампе: {meta['src_hash']}\n"
            f"  на месте: {actual}"
        )

    # sources-датасет из mounts НЕ исключается: всё примонтированное видно по
    # ключу, без спецслучаев. ctx.mounts["<owner>/<sources-slug>"] отдаст ровно
    # ctx.sources_dir — это одно и то же место, просто два способа его назвать.
    mounts = _mounts(
        KAGGLE_INPUT,
        lambda: sorted((KAGGLE_INPUT / "notebooks").glob("*/*")),
        empty_hint="  (ничего не приаттачено)",
    )
    record = {
        "run_id": None,  # проставит фетч: только он знает слаг кернела
        "exp": None,
        "script": None,
        "env": "kaggle",
        "started": _now(),
        "code": {"git_sha": meta["git_sha"], "dirty": False},
        "sources_hash": actual,
    }
    _install_recorder(record, KAGGLE_WORKING, mounts)
    return Ctx(sources_dir=sources_dir, mounts=mounts, output_dir=KAGGLE_WORKING)


def _local_ctx(repo: Path) -> Ctx:
    """Локально: git под рукой, поэтому id минтим сами, а провенанс снимаем как есть.

    Коммит НЕ требуется. Грязное дерево — не запрет, а факт: пишем dirty=true и
    кладём рядом diff.patch, и такой запуск восстанавливается так же, как чистый.
    """
    exp, script = _script_identity(repo)
    cache = Path(local_paths(repo)["kaggle_cache"]).expanduser().resolve()

    run_id = new_run_id()
    output_dir = run_dir(repo, exp, run_id)

    # Грязным считается то, что влияет на воспроизводимость: payload и сам скрипт.
    watched = [*PAYLOAD_DIRS, f"exps/{exp}/{script}"]
    dirty = bool(git(repo, "status", "--porcelain", "--", *watched).strip())
    if dirty:
        # Untracked-файлы сюда не попадут (git diff их не знает) — но флаг dirty
        # их видит, так что молчаливой дыры нет.
        (output_dir / DIFF_NAME).write_text(
            git(repo, "diff", "--", *watched), encoding="utf-8"
        )

    mounts = _mounts(
        cache,
        lambda: sorted(repo.glob("exps/*/runs/*")),
        empty_hint=f"  (пусто; кэш данных: {cache})",
    )
    record = {
        "run_id": run_id,
        "exp": exp,
        "script": script,
        "env": "local",
        "started": _now(),
        "code": {"git_sha": git(repo, "rev-parse", "HEAD").strip(), "dirty": dirty},
        # Локально сверять не с чем: рабочее дерево и есть истина. Пишем для истории.
        "sources_hash": compute_src_hash(repo),
    }
    _install_recorder(record, output_dir, mounts)
    return Ctx(sources_dir=repo, mounts=mounts, output_dir=output_dir)


def get_ctx() -> Ctx:
    """Собрать контекст запуска. Зовётся первой строкой скрипта после бутстрапа.

    sources_dir не ищется: это папка на два уровня выше utils/ctx.py, то есть
    откуда импортировался сам модуль. Локально — корень репы, на кагле — маунт
    нашего датасета (чужой датасет спутать нельзя, в нём нет utils/ctx.py).
    """
    sources_dir = Path(__file__).resolve().parent.parent
    if KAGGLE_INPUT.is_dir():
        return _kaggle_ctx(sources_dir)
    return _local_ctx(sources_dir)
