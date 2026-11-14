#!/usr/bin/env python3
"""fetch_writeup_by_slug.py — standalone-врайтап по URL → markdown.

    uv run --project kaggle python kaggle/fetch_writeup_by_slug.py \\
        https://www.kaggle.com/writeups/<user>/<slug> [output.md]

Дополняет апстримный fetch_writeup.py, который умеет ТОЛЬКО competition-scoped
врайтапы (`.../competitions/<comp>/writeups/<slug>`, резолвит через лидерборд).
Этот берёт standalone-врайтап (`kaggle.com/writeups/<user>/<slug>`) напрямую.

Механика (разведка network-инспекцией SPA): страница — пустой шелл, данные
грузит внутренний web-сервис `discussions.WriteUpsService/GetWriteUpBySlug`
с payload {userName, slug} → {title, topicId, message.rawMarkdown}. Комментарии
не тянем, как и апстрим.

Кандидат на PR в апстрим: дыру он затыкает общую, не нашу.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

c.require_env()

sys.path.insert(0, str(c.VENDOR_SCRIPTS))  # runtime.py апстрима
from runtime import kaggle_web_service  # noqa: E402

# competition-scoped URL сюда НЕ подходит — для него апстримный fetch_writeup.py
URL_RE = re.compile(r"kaggle\.com/writeups/(?P<user>[^/]+)/(?P<slug>[^/?#]+)")


def fetch_writeup_by_slug(url: str) -> str:
    match = URL_RE.search(url)
    if not match:
        raise SystemExit(
            "URL должен быть standalone-врайтапом вида "
            "https://www.kaggle.com/writeups/<user>/<slug>. Для competition-врайтапа "
            "(/competitions/<comp>/writeups/<slug>) — апстримный fetch_writeup.py."
        )
    data = kaggle_web_service().post(
        "discussions.WriteUpsService/GetWriteUpBySlug",
        {"userName": match.group("user"), "slug": match.group("slug")},
    )
    message = data.get("message") or {}
    markdown = (message.get("rawMarkdown") or message.get("content") or "").strip()
    if not markdown:
        raise SystemExit(f"Врайтап '{match.group('slug')}' вернул пустой контент.")

    title = data.get("title") or ""
    if title and not markdown.lstrip().startswith("#"):
        markdown = f"# {title}\n\n{markdown}"
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="standalone Kaggle writeup → markdown")
    parser.add_argument("url", help="https://www.kaggle.com/writeups/<user>/<slug>")
    parser.add_argument("output_path", nargs="?", help="куда сохранить md (иначе stdout)")
    args = parser.parse_args()

    if not re.match(r"^https?://", args.url):
        parser.error("url должен быть абсолютным http(s) URL врайтапа")

    content = fetch_writeup_by_slug(args.url)

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        print(f"[writeup] сохранено: {out} ({len(content)} симв)", file=sys.stderr)
    else:
        sys.stdout.buffer.write(content.encode("utf-8"))


if __name__ == "__main__":
    main()
