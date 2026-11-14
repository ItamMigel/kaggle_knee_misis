"""Files through the kernel log: kernel output files cannot be downloaded, the log can.

Kernel side `emit_file` prints a gzip+base64 file as marker lines; host side
`decode_log` reassembles it from `log.txt` (built by kaggle/fetch.py as
"[<stream>] <data>" per log entry) and verifies everything. Stdlib only: the
module runs inside kernels.

Wire format v=1, one line each, ASCII, every line < MAX_LINE_CHARS:

    @@LOGFILE BEGIN v=1 <name> bytes=<raw bytes> sha256=<hex of raw> chunks=<k>
    @@LOGFILE CHUNK <name> <i>/<k> crc32=<8 hex of the chunk text> <base64>
    @@LOGFILE END <name>

Our marker lines never contain "[", so a "[<stream>] " that fetch.py glued into
the middle of one (Kaggle split the line into two log entries) is removed
exactly, and the chunk crc32 + whole-file sha256 prove the result. Anything else
off — unknown version, missing/duplicate/corrupt chunk, block without END —
raises; a partial file is never returned. A new codec must bump FORMAT_VERSION.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import re
import sys
import zlib
from pathlib import Path

FORMAT_VERSION = "1"
SUPPORTED_VERSIONS = ("1",)
MARKER = "@@LOGFILE "
# A write below PIPE_BUF (4096 bytes on Linux) is atomic, so lines of processes
# that share stdout (vLLM workers) cannot interleave inside ours.
MAX_LINE_CHARS = 4000

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
_STREAM_PREFIX_RE = re.compile(r"\[[A-Za-z_]+\] ")
_BEGIN_RE = re.compile(r"BEGIN (\S+) (\S+) bytes=(\d+) sha256=([0-9a-f]{64}) chunks=(\d+)")
_CHUNK_RE = re.compile(r"CHUNK (\S+) (\d+)/(\d+) crc32=([0-9a-f]{8}) ([A-Za-z0-9+/=]+)")
_END_RE = re.compile(r"END (\S+)")


class LogTransportError(RuntimeError):
    """Log does not contain the emitted files intact; the message names files and chunks."""


def _check_name(name: str) -> None:
    """Refuse names that would break the space-separated markers or escape out_dir on decode."""
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"emit name {name!r} must match {_NAME_RE.pattern}: it travels as one "
            "space-separated token and becomes a file name on decode"
        )


def emit_file(path: Path, name: str, chunk_chars: int) -> None:
    """Print a file to stdout as gzip+base64 marker lines (BEGIN v=1 / CHUNK i/k / END)."""
    _check_name(name)
    raw = Path(path).read_bytes()
    text = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    chunks = [text[i: i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    k = len(chunks)
    longest = len(f"{MARKER}CHUNK {name} {k}/{k} crc32=00000000 ") + chunk_chars
    if chunk_chars < 1 or longest >= MAX_LINE_CHARS:
        raise ValueError(
            f"chunk_chars={chunk_chars} gives chunk lines of {longest} chars for {name!r}; "
            f"they must stay below {MAX_LINE_CHARS} to be written atomically"
        )
    lines = [f"{MARKER}BEGIN v={FORMAT_VERSION} {name} bytes={len(raw)} "
             f"sha256={hashlib.sha256(raw).hexdigest()} chunks={k}"]
    lines += [f"{MARKER}CHUNK {name} {i}/{k} crc32={zlib.crc32(c.encode('ascii')):08x} {c}"
              for i, c in enumerate(chunks, start=1)]
    lines.append(f"{MARKER}END {name}")
    sys.stdout.flush()  # nothing of ours pending in the buffer → one write per line
    for line in lines:
        print(line, flush=True)


def _marker_name(body: str) -> str | None:
    """File name a marker line refers to, read before full validation (None if not even that)."""
    parts = body.split(" ")
    idx = 2 if parts[0] == "BEGIN" else 1
    return parts[idx] if len(parts) > idx else None


def _parse(log_text: str, ignore: tuple[str, ...]) -> dict[str, dict]:
    """Marker lines → per-file blocks; structural defects raise, crc failures are recorded."""
    blocks: dict[str, dict] = {}
    for lineno, line in enumerate(log_text.splitlines(), start=1):
        pos = line.find(MARKER)
        if pos < 0:
            continue
        body = _STREAM_PREFIX_RE.sub("", line[pos + len(MARKER):].rstrip("\r"))
        if _marker_name(body) in ignore:
            continue
        where = f"log line {lineno}"
        if body.startswith("BEGIN "):
            m = _BEGIN_RE.fullmatch(body)
            if m is None or not m.group(1).startswith("v="):
                raise LogTransportError(
                    f"{where}: malformed BEGIN marker (expected 'BEGIN v=<version> <name> "
                    f"bytes=.. sha256=.. chunks=..'): {body[:200]!r}")
            version, name = m.group(1)[2:], m.group(2)
            if version not in SUPPORTED_VERSIONS:
                raise LogTransportError(
                    f"{where}: {name!r} uses log format v={version}, this decoder knows "
                    f"{SUPPORTED_VERSIONS}; decode it with the code version that emitted it"
                )
            if name in blocks:
                raise LogTransportError(f"{where}: {name!r} is emitted twice in this log")
            blocks[name] = {"bytes": int(m.group(3)), "sha256": m.group(4), "k": int(m.group(5)),
                            "chunks": {}, "bad_crc": [], "ended": False}
        elif body.startswith("CHUNK "):
            m = _CHUNK_RE.fullmatch(body)
            if m is None:
                raise LogTransportError(f"{where}: malformed CHUNK marker: {body[:200]!r}")
            name, i, k, crc, data = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
            block = blocks.get(name)
            if block is None or block["ended"]:
                raise LogTransportError(f"{where}: chunk {i}/{k} of {name!r} outside its BEGIN..END")
            if k != block["k"] or not 1 <= i <= k:
                raise LogTransportError(
                    f"{where}: chunk {i}/{k} of {name!r} does not fit BEGIN chunks={block['k']}")
            if i in block["chunks"]:
                raise LogTransportError(f"{where}: chunk {i}/{k} of {name!r} appears twice")
            block["chunks"][i] = data
            if f"{zlib.crc32(data.encode('ascii')):08x}" != crc:
                block["bad_crc"].append(i)
        elif body.startswith("END "):
            m = _END_RE.fullmatch(body)
            if m is None:
                raise LogTransportError(f"{where}: malformed END marker: {body[:200]!r}")
            block = blocks.get(m.group(1))
            if block is None or block["ended"]:
                raise LogTransportError(f"{where}: END of {m.group(1)!r} without its BEGIN")
            block["ended"] = True
        else:
            raise LogTransportError(f"{where}: unknown {MARKER.strip()} marker: {body[:200]!r}")
    for block in blocks.values():
        block["missing"] = [i for i in range(1, block["k"] + 1) if i not in block["chunks"]]
    return blocks


def log_inventory(log_text: str) -> dict[str, dict]:
    """Per emitted file: bytes, chunks expected/received, missing and crc-failed indices, END seen."""
    return {name: {"bytes": b["bytes"], "chunks": b["k"], "received": len(b["chunks"]),
                   "missing": b["missing"], "bad_crc": b["bad_crc"], "ended": b["ended"]}
            for name, b in _parse(log_text, ()).items()}


def decode_log(log_text: str, ignore: tuple[str, ...] = ()) -> dict[str, bytes]:
    """Reassemble every emitted file from a kernel log; any defect raises LogTransportError.

    ignore: names whose blocks are skipped unchecked and unreturned (e.g. a
    deliberately oversized probe); every other file must be intact.
    """
    problems: list[str] = []
    files: dict[str, bytes] = {}
    for name, block in _parse(log_text, tuple(ignore)).items():
        k = block["k"]
        if block["bad_crc"]:
            problems.append(f"{name}: chunks {_ranges(block['bad_crc'])} of {k} fail crc32")
        if block["missing"]:
            problems.append(f"{name}: chunks {_ranges(block['missing'])} of {k} missing")
        if not block["ended"]:
            problems.append(f"{name}: no END marker — block truncated")
        if block["bad_crc"] or block["missing"] or not block["ended"]:
            continue
        raw = gzip.decompress(base64.b64decode(
            "".join(block["chunks"][i] for i in range(1, k + 1)), validate=True))
        if len(raw) != block["bytes"] or hashlib.sha256(raw).hexdigest() != block["sha256"]:
            problems.append(f"{name}: reassembled file fails bytes/sha256 check")
            continue
        files[name] = raw
    if problems:
        raise LogTransportError(
            f"{len(problems)} problem(s) in the log, no file returned:\n  " + "\n  ".join(problems))
    return files


def _ranges(indices: list[int]) -> str:
    """[1, 2, 3, 7] → '1-3,7' (for error messages)."""
    spans, start, prev = [], indices[0], indices[0]
    for cur in indices[1:] + [None]:
        if cur is None or cur != prev + 1:
            spans.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = cur
        prev = cur
    return ",".join(spans)


def write_decoded(log_path: Path, out_dir: Path, ignore: tuple[str, ...] = ()) -> list[Path]:
    """Decode every file emitted into a kernel log.txt and write them into out_dir (no overwrite)."""
    files = decode_log(Path(log_path).read_text(encoding="utf-8"), ignore)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, raw in files.items():
        with (out_dir / name).open("xb") as f:
            f.write(raw)
        written.append(out_dir / name)
    return written
