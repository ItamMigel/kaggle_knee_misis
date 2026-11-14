"""src/zip_range.py — the competition's download-all zip read over HTTP Range.

The archive is never downloaded as a whole: Kaggle redirects download-all to a
signed GCS URL that serves byte ranges, the ZIP64 central directory gives every
member's local-header offset, and members are raw-inflated in memory.

    reader = ArchiveReader("rsna-knee-abnormality-detection")
    index = read_central_directory(reader)          # one row per member
    rows = index[index.study == study]              # contiguous in archive order
    a, b = int(rows.lho.min()), int(rows.next_lho.max())
    for name, raw in iter_members(reader.read_range(a, b), a, rows):
        ...

stdlib HTTP only (urllib): the `knee` env has no requests. The signed URL is a
credential: it never goes into messages, logs or files.
"""

from __future__ import annotations

import http.client
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from typing import Iterator

import pandas as pd

DOWNLOAD_ALL_API = "https://www.kaggle.com/api/v1/competitions/data/download-all/{slug}"

# Retry knobs of ArchiveReader. They change how hard we try, never what is read.
MAX_ATTEMPTS = 6
BACKOFF_S = 2.0  # sleep before attempt k+1 = BACKOFF_S * 2**(k-1)
TIMEOUT_S = 120.0  # per socket operation, not per whole read

EOCD_SEARCH_BYTES = 1 << 16  # EOCD + ZIP64 locator sit in the archive's last 64 KB
DEFLATE = 8
INDEX_COLUMNS = ["name", "split", "study", "series", "sop", "lho", "csize", "usize",
                 "method", "crc32", "next_lho"]

_SIG_LOCAL = b"PK\x03\x04"
_SIG_CENTRAL = b"PK\x01\x02"
_SIG_EOCD = b"PK\x05\x06"
_SIG_ZIP64_LOCATOR = b"PK\x06\x07"
_SIG_ZIP64_EOCD = b"PK\x06\x06"
_U32_MAX = 0xFFFFFFFF


class ArchiveError(RuntimeError):
    """The archive did not read as expected: retries exhausted, bad header, size or CRC mismatch."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: the 302 Location is the answer we want."""

    def redirect_request(self, *args, **kwargs):  # noqa: ANN002, ANN003
        """Refuse; urllib then raises HTTPError carrying the 302 and its Location."""
        return None


def signed_archive_url(slug: str) -> str:
    """Signed GCS URL of the competition's download-all zip (the 302 Location of the Kaggle API)."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is not set. ~/.bashrc exports it from ~/.kaggle/access_token "
            "before the interactive guard; run from a login shell or export it yourself."
        )
    request = urllib.request.Request(
        DOWNLOAD_ALL_API.format(slug=slug), headers={"Authorization": f"Bearer {token}"}
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=TIMEOUT_S) as resp:
            status, body = resp.status, resp.read(2000)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            location = e.headers.get("Location")
            if not location:
                raise RuntimeError(f"download-all/{slug}: {e.code} without a Location header") \
                    from None
            return location
        status, body = e.code, e.read(2000)
    raise RuntimeError(
        f"download-all/{slug}: expected a redirect to the signed archive URL, got HTTP "
        f"{status}: {body!r}. Rules not accepted or a wrong token give 401/403 here."
    )


class ArchiveReader:
    """Byte ranges of the download-all zip; refreshes the signed URL on 400/403, retries the rest."""

    def __init__(self, slug: str, max_attempts: int = MAX_ATTEMPTS, backoff_s: float = BACKOFF_S,
                 timeout_s: float = TIMEOUT_S):
        self.slug = slug
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self._url = signed_archive_url(slug)

    def __repr__(self) -> str:
        return f"ArchiveReader({self.slug!r})"  # never shows the signed URL

    def read_range(self, start: int, end: int) -> bytes:
        """Archive bytes [start, end); raises ArchiveError after max_attempts failed tries."""
        if not 0 <= start < end:
            raise ValueError(f"bad range [{start}, {end})")
        data, _ = self._get(start, end)
        return data

    def total_size(self) -> int:
        """Size of the whole archive in bytes, from the Content-Range of a 1-byte read."""
        _, content_range = self._get(0, 1)
        total = content_range.rpartition("/")[2]
        if not total.isdigit():
            raise ArchiveError(f"Content-Range without a total size: {content_range!r}")
        return int(total)

    def _get(self, start: int, end: int) -> tuple[bytes, str]:
        """One Range GET with retries → (body, Content-Range header)."""
        what = f"bytes {start}-{end - 1} of the {self.slug} archive"
        last = None
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                time.sleep(self.backoff_s * 2 ** (attempt - 2))
            try:
                return self._get_once(start, end)
            except urllib.error.HTTPError as e:
                if e.code in (400, 403):
                    # An expired or revoked signature: GCS answers 400 ExpiredToken or 403.
                    last = f"HTTP {e.code}"
                    print(f"[zip_range] {what}: {last}, refreshing the signed URL "
                          f"(attempt {attempt}/{self.max_attempts})", file=sys.stderr)
                    try:
                        self._url = signed_archive_url(self.slug)
                    except (OSError, http.client.HTTPException) as refresh_error:
                        last = f"HTTP {e.code}, then the URL refresh failed: {refresh_error}"
                    continue
                if e.code == 429 or e.code >= 500:
                    last = f"HTTP {e.code}"
                else:
                    raise ArchiveError(f"{what}: HTTP {e.code}: {e.read(2000)!r}") from None
            except _ShortRead as e:
                last = str(e)
            except (OSError, http.client.HTTPException) as e:
                # URLError, timeouts, resets, IncompleteRead: the connection, not the data.
                last = f"{type(e).__name__}: {e}"
            print(f"[zip_range] {what}: {last} (attempt {attempt}/{self.max_attempts})",
                  file=sys.stderr)
        raise ArchiveError(f"{what}: failed after {self.max_attempts} attempts, last: {last}")

    def _get_once(self, start: int, end: int) -> tuple[bytes, str]:
        request = urllib.request.Request(self._url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
            content_range = resp.headers.get("Content-Range", "")
            # 200 means the server ignored Range and is about to send the whole 265 GB.
            if resp.status != 206 or not content_range.startswith(f"bytes {start}-{end - 1}/"):
                raise ArchiveError(
                    f"bytes {start}-{end - 1}: expected 206 with that Content-Range, got "
                    f"{resp.status} {content_range!r}"
                )
            data = resp.read()
        if len(data) != end - start:
            raise _ShortRead(f"got {len(data)} of {end - start} bytes")
        return data, content_range


class _ShortRead(Exception):
    """The body ended before the requested range did (retriable)."""


def _zip64_central_directory(tail: bytes, tail_off: int) -> tuple[int, int, int]:
    """(n_entries, cd_size, cd_offset) from the archive tail via EOCD → ZIP64 locator → ZIP64 EOCD."""
    eocd = tail.rfind(_SIG_EOCD)
    if eocd < 0:
        raise ArchiveError("no end-of-central-directory record in the archive tail")
    loc = eocd - 20
    if loc < 0 or tail[loc:loc + 4] != _SIG_ZIP64_LOCATOR:
        raise ArchiveError("no ZIP64 end-of-central-directory locator before the EOCD record")
    z64_abs, = struct.unpack("<Q", tail[loc + 8:loc + 16])
    z64 = z64_abs - tail_off
    if not 0 <= z64 <= len(tail) - 56 or tail[z64:z64 + 4] != _SIG_ZIP64_EOCD:
        raise ArchiveError(f"ZIP64 EOCD record not found at offset {z64_abs}")
    n_entries, cd_size, cd_off = struct.unpack("<QQQ", tail[z64 + 32:z64 + 56])
    return n_entries, cd_size, cd_off


def _layout(name: str) -> tuple[str, str | None, str | None, str | None]:
    """(split, study, series, sop) for `<split>/<study>/<series>/<sop>.dcm`; other names keep only split."""
    parts = name.split("/")
    if len(parts) == 4 and all(parts) and parts[3].endswith(".dcm") and len(parts[3]) > 4:
        return parts[0], parts[1], parts[2], parts[3][:-4]
    return parts[0], None, None, None


def read_central_directory(reader: ArchiveReader) -> pd.DataFrame:
    """One row per archive member, in archive (lho) order, with study/series/sop parsed from the path."""
    size = reader.total_size()
    tail_off = max(0, size - EOCD_SEARCH_BYTES)
    n_entries, cd_size, cd_off = _zip64_central_directory(reader.read_range(tail_off, size),
                                                          tail_off)
    cd = reader.read_range(cd_off, cd_off + cd_size)

    rows = []
    p = 0
    while p < len(cd):
        if cd[p:p + 4] != _SIG_CENTRAL:
            raise ArchiveError(f"central directory entry {len(rows)}: bad signature at {p}")
        flags, method = struct.unpack("<HH", cd[p + 8:p + 12])
        crc, csize, usize = struct.unpack("<III", cd[p + 16:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode("utf-8" if flags & 0x800 else "cp437")
        if flags & 0x1:
            raise ArchiveError(f"{name}: encrypted member")
        if _U32_MAX in (usize, csize, lho):
            # ZIP64 extra field (id 1) holds exactly the saturated ones, in this order.
            extra = cd[p + 46 + nl:p + 46 + nl + el]
            wide = None
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    wide = list(struct.unpack(f"<{hlen // 8}Q", extra[q + 4:q + 4 + hlen // 8 * 8]))
                    break
                q += 4 + hlen
            if wide is None:
                raise ArchiveError(f"{name}: saturated sizes but no ZIP64 extra field")
            try:
                if usize == _U32_MAX:
                    usize = wide.pop(0)
                if csize == _U32_MAX:
                    csize = wide.pop(0)
                if lho == _U32_MAX:
                    lho = wide.pop(0)
            except IndexError:
                raise ArchiveError(f"{name}: ZIP64 extra field too short") from None
        rows.append((name, *_layout(name), lho, csize, usize, method, crc))
        p += 46 + nl + el + cl
    if len(rows) != n_entries:
        raise ArchiveError(f"central directory: parsed {len(rows)} entries, EOCD says {n_entries}")

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS[:-1])
    df = df.sort_values("lho", kind="stable").reset_index(drop=True)
    if df.lho.duplicated().any():
        raise ArchiveError("central directory: two members share a local-header offset")
    df["next_lho"] = df.lho.shift(-1).fillna(cd_off).astype("int64")
    if (df.next_lho - df.lho < 30 + df.csize).any():
        raise ArchiveError("central directory: a member overlaps the next one")
    return df[INDEX_COLUMNS]


def member_range(rows: pd.DataFrame) -> tuple[int, int]:
    """Archive byte range [start, end) of index rows that are contiguous in archive order."""
    lho = rows.lho.to_numpy()
    nxt = rows.next_lho.to_numpy()
    if len(rows) == 0:
        raise ValueError("no members")
    if (lho[1:] != nxt[:-1]).any():
        # One Range read would also fetch the members in between: never silently.
        raise ArchiveError("members are not contiguous in archive order (or not sorted by lho)")
    return int(lho[0]), int(nxt[-1])


def iter_members(buf: bytes, base_off: int, rows: pd.DataFrame) -> Iterator[tuple[str, bytes]]:
    """Yield (name, inflated bytes) for each row whose member lies in buf (archive offset base_off)."""
    view = memoryview(buf)
    for r in rows[["name", "lho", "csize", "usize", "method", "crc32"]].itertuples(index=False):
        o = r.lho - base_off
        if o < 0 or o + 30 > len(buf) or view[o:o + 4] != _SIG_LOCAL:
            raise ArchiveError(f"{r.name}: no local header at archive offset {r.lho}")
        if r.method != DEFLATE:
            raise ArchiveError(f"{r.name}: compression method {r.method}, only deflate is read")
        nl, el = struct.unpack("<HH", view[o + 26:o + 30])
        if bytes(view[o + 30:o + 30 + nl]).decode("utf-8", "replace") != r.name:
            raise ArchiveError(f"{r.name}: local header names another member")
        d = o + 30 + nl + el
        if d + r.csize > len(buf):
            raise ArchiveError(f"{r.name}: member data runs past the read range")
        inflater = zlib.decompressobj(-15)
        try:
            raw = inflater.decompress(view[d:d + r.csize]) + inflater.flush()
        except zlib.error as e:
            raise ArchiveError(f"{r.name}: corrupt deflate stream: {e}") from None
        if not inflater.eof:
            raise ArchiveError(f"{r.name}: deflate stream is truncated")
        if len(raw) != r.usize:
            raise ArchiveError(f"{r.name}: inflated {len(raw)} bytes, expected {r.usize}")
        if zlib.crc32(raw) != r.crc32:
            raise ArchiveError(f"{r.name}: CRC32 mismatch")
        yield r.name, raw
