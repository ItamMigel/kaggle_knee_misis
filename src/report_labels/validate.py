"""Validation of raw extractor output: JSON + EXTRACTION_SCHEMA + consistency rules + quote check.

Issues are strings `"<scope>|<code>"`: scope is a target name (the error invalidates only that
cell) or `*` (record-level: an error invalidates all 12 cells). Codes in WARNING_CODES never invalidate.
Invalid cells go to the review queue and become NaN targets downstream — never zeros.

rubric-v2: a finding the report does not mention comes as the SILENT form
`{"state": "not_addressed"}` and is expanded to rubric.SILENT_RECORD, so the long table keeps all fields.
Quote tiers, first match wins: exact substring → ok; equal after deleting all whitespace → warning
quote_ws_normalised; equal after also folding case, diacritics, Turkish dotless/dotted i and
đ ł ø æ œ ß → warning quote_folded; folded match after repairs (one edge ellipsis stripped,
backslashes removed and ∆→Δ on both sides, the quote also tried mojibake-repaired) → warning
quote_repaired; an interior ellipsis splitting the repaired quote into segments that occur in
the report in order, each within SPLICE_MAX_GAP folded chars of the previous → warning
quote_spliced; otherwise error quote_not_in_report. The long table records the tier per cell
(`quote_tier`). A raw control char inside a JSON string
(xgrammar length-limited strings allow a copied TAB) is accepted with the `*` warning json_control_char.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.report_labels.rubric import (EXTRACTION_SCHEMA, PROMPT_VERSION, SILENT_RECORD, SILENT_STATE,
                                      TARGET_FIELDS, TARGETS)

UID = "StudyInstanceUID"
PASSES = ("base", "reasoning")
WARNING_CODES = frozenset({"quote_ws_normalised", "quote_folded", "quote_repaired", "quote_spliced",
                           "json_control_char"})
if tuple(SILENT_RECORD) != tuple(TARGET_FIELDS):  # the long table reads TARGET_FIELDS from every cell
    raise ImportError(f"rubric SILENT_RECORD fields {tuple(SILENT_RECORD)} differ from TARGET_FIELDS {TARGET_FIELDS}")
LINE_FIELDS = (UID, "raw", "finish_reason", "model", "revision", "prompt_version", "pass")
_EXCERPT_CHARS = 400


def split_issue(issue: str) -> tuple[str, str]:
    """`"<scope>|<code>"` -> (scope, code); scope is a target name or `*`."""
    scope, code = issue.rsplit("|", 1)
    return scope, code


def is_error(issue: str) -> bool:
    """True if the issue invalidates its scope (everything except WARNING_CODES)."""
    return split_issue(issue)[1] not in WARNING_CODES


class _DuplicateKey(ValueError):
    """A JSON object repeats a key (json.loads would silently keep the last one)."""


class _NonFinite(ValueError):
    """A NaN/Infinity literal, which strict JSON does not allow."""


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    """json object_pairs_hook that raises on duplicate keys."""
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise _DuplicateKey(f"duplicate keys {dup}")
    return dict(pairs)


def _reject_constant(name: str) -> None:
    """json parse_constant hook that rejects NaN/Infinity."""
    raise _NonFinite(f"non-finite number {name}")


_JSON_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}
_SCHEMA_KEYWORDS = {"type", "properties", "required", "additionalProperties", "enum", "minLength", "maxLength",
                    "anyOf"}


def _schema_errors(value, schema: dict, path: tuple) -> list[tuple[tuple, str]]:
    """(path, code) for every violation of the JSON-Schema subset EXTRACTION_SCHEMA uses."""
    unknown = set(schema) - _SCHEMA_KEYWORDS
    if unknown:
        raise NotImplementedError(f"validator does not implement schema keywords {sorted(unknown)} "
                                  f"(at {path}); extend _schema_errors")
    field = path[-1] if path else "root"
    if "anyOf" in schema:
        # No branch matches → report the closest branch (fewest violations; first on a tie), so a
        # FULL record with a bad enum says schema_enum.<field>, not "matches neither form".
        branches = [_schema_errors(value, sub, path) for sub in schema["anyOf"]]
        return [] if not all(branches) else min(branches, key=len)
    if "type" in schema and not _JSON_TYPES[schema["type"]](value):
        return [(path, f"schema_type.{field}")]
    out = []
    if "enum" in schema and value not in schema["enum"]:
        out.append((path, f"schema_enum.{field}"))
    if "minLength" in schema and len(value) < schema["minLength"]:
        out.append((path, f"schema_min_length.{field}"))
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        out.append((path, f"schema_max_length.{field}"))
    if "properties" in schema:
        props = schema["properties"]
        for key in schema.get("required", []):
            if key not in value:
                out.append((path + (key,), f"schema_missing.{key}"))
        if schema.get("additionalProperties") is False:
            out += [(path, f"schema_unexpected_key.{field}") for key in value if key not in props]
        for key, sub in props.items():
            if key in value:
                out += _schema_errors(value[key], sub, path + (key,))
    return out


def _no_ws(s: str) -> str:
    """Delete every whitespace character (incl. newlines, NBSP): a dropped report line break still matches."""
    return "".join(s.split())


# Letters NFKD does not decompose, as they are usually ASCII-fied; applied after casefold
# (İ casefolds to i + combining dot, stripped below; ẞ/ß casefold to ss).
_FOLD_LETTERS = str.maketrans({"ı": "i", "đ": "d", "ł": "l", "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss"})


def _fold(s: str) -> str:
    """_no_ws + casefold, ı đ ł ø æ œ ß → i d l o ae oe ss, NFKD with combining marks stripped (ASCII-fied text matches accented)."""
    s = s.casefold().translate(_FOLD_LETTERS)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return _no_ws(s)


SPLICE_MAX_GAP = 200  # folded report chars allowed between consecutive segments of a spliced quote
_ELLIPSIS = r"(?:\.\.\.|\u2026)"
_LEADING_ELLIPSIS = re.compile(rf"^\s*{_ELLIPSIS}\s*")
_TRAILING_ELLIPSIS = re.compile(rf"\s*{_ELLIPSIS}\s*$")
_INTERIOR_ELLIPSIS = re.compile(rf"\s*(?:{_ELLIPSIS}\s*)+")  # consecutive ellipses are one cut
# Both sides: RTF escapes in Dutch reports (\'e0, \~) — the schema forbids backslashes in quotes —
# and ∆ (U+2206 increment) written for Δ (U+0394).
_REPAIR_BOTH = str.maketrans({"\\": None, "\u2206": "\u0394"})


def _unmojibake(s: str) -> str | None:
    """UTF-8 text that was decoded as cp1252 (é written as Ã©), decoded back; None unless the round trip succeeds and changes s.

    Characters cp1252 cannot encode fall back to latin-1 (C1 controls for its 5 undefined bytes).
    """
    raw = bytearray()
    for c in s:
        try:
            raw += c.encode("cp1252")
        except UnicodeEncodeError:
            try:
                raw += c.encode("latin-1")
            except UnicodeEncodeError:  # not single-byte text: not mojibake
                return None
    try:
        fixed = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return fixed if fixed != s else None


def _repaired_quotes(quote: str) -> list[str]:
    """The quote with one leading/trailing ellipsis stripped, and that also mojibake-repaired when it round-trips."""
    q = _TRAILING_ELLIPSIS.sub("", _LEADING_ELLIPSIS.sub("", quote, count=1), count=1)
    fixed = _unmojibake(q)
    return [q] if fixed is None else [q, fixed]


def _in_order(segments: list[str], text: str, max_gap: int) -> bool:
    """True if the segments occur in text in this order, each starting 0..max_gap chars after the previous one's end."""
    ends = set()
    i = text.find(segments[0])
    while i != -1:
        ends.add(i + len(segments[0]))
        i = text.find(segments[0], i + 1)
    for seg in segments[1:]:  # every reachable end, not just the leftmost one: exact for any layout
        nxt = set()
        for e in ends:
            i = text.find(seg, e)
            while i != -1 and i - e <= max_gap:
                nxt.add(i + len(seg))
                i = text.find(seg, i + 1)
        ends = nxt
    return bool(ends)


def _repair_tier(quote: str, report: str) -> str | None:
    """'quote_repaired' / 'quote_spliced' if the quote matches the report after the repairs, else None."""
    text = _fold(report.translate(_REPAIR_BOTH))
    candidates = _repaired_quotes(quote)
    for q in candidates:
        f = _fold(q.translate(_REPAIR_BOTH))
        if f and f in text:
            return "quote_repaired"
    for q in candidates:
        segments = [_fold(p.translate(_REPAIR_BOTH)) for p in _INTERIOR_ELLIPSIS.split(q)]
        if len(segments) >= 2 and all(segments) and _in_order(segments, text, SPLICE_MAX_GAP):
            return "quote_spliced"
    return None


def _target_issues(t: str, rec: dict, report: str) -> list[str]:
    """Consistency rules and the quote check for one schema-valid target record."""
    codes = []
    quote, state, meets = rec["quote"], rec["state"], rec["meets_rubric"]
    if state == "not_addressed" and quote != "":
        codes.append("consistency_quote_on_not_addressed")
    if state != "not_addressed" and quote == "":
        codes.append("consistency_empty_quote")
    if meets == "yes" and state != "present":
        codes.append("consistency_yes_without_present")
    if state in ("absent", "not_addressed") and meets != "no":
        codes.append("consistency_negative_not_no")
    if quote != "":
        if not quote.strip():
            codes.append("quote_blank")
        elif quote in report:
            pass
        elif _no_ws(quote) in _no_ws(report):
            codes.append("quote_ws_normalised")
        elif _fold(quote) and _fold(quote) in _fold(report):
            codes.append("quote_folded")
        else:
            codes.append(_repair_tier(quote, report) or "quote_not_in_report")
    return [f"{t}|{c}" for c in codes]


def parse_record(raw: str, report: str) -> tuple[dict | None, list[str]]:
    """Parse + validate one raw extractor output against its report; returns (object with SILENT cells expanded, or None if unparseable; issues)."""
    if not isinstance(raw, str) or not isinstance(report, str):
        raise TypeError(f"raw and report must be str, got {type(raw).__name__}, {type(report).__name__}")
    issues = []
    try:
        try:
            obj = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
        except json.JSONDecodeError:
            # xgrammar's length-limited strings allow raw control chars (a TAB copied from the
            # report): accept them as literal characters, flagged with a record-level warning.
            obj = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant, strict=False)
            issues.append("*|json_control_char")
    except _DuplicateKey:
        return None, ["*|json_duplicate_key"]
    except _NonFinite:
        return None, ["*|json_nonfinite"]
    except json.JSONDecodeError:  # truncated or garbled output
        return None, ["*|json_decode"]
    if not isinstance(obj, dict):
        return None, ["*|schema_type.root"]

    bad_targets = set()
    for path, code in _schema_errors(obj, EXTRACTION_SCHEMA, ()):
        if len(path) >= 2 and path[0] == "targets" and path[1] in TARGETS:
            scope = path[1]
            if len(path) == 2 and code.startswith("schema_missing."):
                code = "missing_target"
            bad_targets.add(scope)
        else:
            scope = "*"
        issues.append(f"{scope}|{code}")
    if any(split_issue(i)[0] == "*" and is_error(i) for i in issues):
        return obj, issues
    for t in TARGETS:
        if t not in bad_targets:
            if obj["targets"][t] == {"state": SILENT_STATE}:
                obj["targets"][t] = dict(SILENT_RECORD)
            issues += _target_issues(t, obj["targets"][t], report)
    return obj, issues


def _excerpt(raw: str, obj: dict | None, scope: str) -> str:
    """Review-queue excerpt: the target's JSON for a target issue, else head + tail of the raw output."""
    if scope != "*" and obj is not None and isinstance(obj.get("targets"), dict) and scope in obj["targets"]:
        return json.dumps(obj["targets"][scope], ensure_ascii=False)[:_EXCERPT_CHARS]
    if len(raw) <= _EXCERPT_CHARS:
        return raw
    return raw[: _EXCERPT_CHARS - 100] + " … " + raw[-100:]


def _cell_value(v):
    """Scalar as is; anything else (a wrongly typed list/dict) as JSON text for the long table."""
    return v if v is None or isinstance(v, (str, int, float)) else json.dumps(v, ensure_ascii=False)


QUOTE_TIERS = ("silent", "exact", "ws_normalised", "folded", "repaired", "spliced", "not_in_report", "blank",
               "unchecked")


def _quote_tier(quote, target_codes: list[str], star_codes: list[str]) -> str:
    """Which QUOTE_TIERS a cell's quote fell into; `unchecked` when JSON/schema errors stopped the check."""
    if any(c.startswith(("json_", "schema_")) and c not in WARNING_CODES for c in star_codes) or \
            any(c.startswith("schema_") or c == "missing_target" for c in target_codes):
        return "unchecked"
    for code, tier in (("quote_not_in_report", "not_in_report"), ("quote_blank", "blank"),
                       ("quote_spliced", "spliced"), ("quote_repaired", "repaired"),
                       ("quote_folded", "folded"), ("quote_ws_normalised", "ws_normalised")):
        if code in target_codes:
            return tier
    return "silent" if quote == "" else "exact"


def _line_issues(rec: dict, report: str) -> tuple[dict | None, list[str], dict[str, list[str]]]:
    """parse_record + finish_reason check for one JSONL line -> (object, issues, error codes per target incl. `*`)."""
    obj, issues = parse_record(rec["raw"], report)
    if rec["finish_reason"] != "stop":
        issues.insert(0, f"*|finish_reason_{rec['finish_reason']}")
    star = [c for s, c in map(split_issue, issues) if s == "*" and c not in WARNING_CODES]
    cell_errors = {t: star + [c for s, c in map(split_issue, issues) if s == t and c not in WARNING_CODES]
                   for t in TARGETS}
    return obj, issues, cell_errors


def check_batch(lines: list[str], reports: dict[str, str]) -> dict:
    """Fail-fast summary of one batch of extractor JSONL lines: invalid records, cell validity, top codes (no pandas).

    A record is invalid when it has any `*` error (bad JSON, top-level schema, truncation,
    finish_reason != stop); a cell is valid when it has neither a `*` nor its own error.
    """
    if not lines:
        raise ValueError("check_batch got an empty batch")
    n_record_invalid = n_cells_valid = 0
    errors, warnings, finish = Counter(), Counter(), Counter()
    for line in lines:
        rec = json.loads(line)
        if rec[UID] not in reports:
            raise ValueError(f"check_batch: {rec[UID]} is not in reports")
        _, issues, cell_errors = _line_issues(rec, reports[rec[UID]])
        finish[rec["finish_reason"]] += 1
        n_record_invalid += any(s == "*" and c not in WARNING_CODES for s, c in map(split_issue, issues))
        n_cells_valid += sum(not e for e in cell_errors.values())
        for s, c in map(split_issue, issues):
            (warnings if c in WARNING_CODES else errors)[c] += 1
    return {
        "n": len(lines),
        "n_record_invalid": n_record_invalid,
        "cell_valid_share": n_cells_valid / (len(lines) * len(TARGETS)),
        "top_errors": errors.most_common(10),
        "top_warnings": warnings.most_common(5),
        "finish_reasons": dict(finish),
    }


def validate_run(raw_jsonl: Path | Sequence[Path], reports: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate extractor JSONL file(s) -> (long table study×pass×target with `valid`/`selected`, review queue).

    Several files are read as one stream (e.g. base shards + a reasoning re-run). Per
    (study, target) the `selected` row is the valid `reasoning` cell if there is one, else the
    `base` cell (valid or not). The review queue lists the issues of selected cells plus the
    record-level (`*`) issues of every record that has a selected cell.
    """
    paths = [Path(raw_jsonl)] if isinstance(raw_jsonl, (str, Path)) else [Path(p) for p in raw_jsonl]
    if not paths:
        raise ValueError("validate_run needs at least one raw.jsonl")
    if reports[UID].duplicated().any():
        raise ValueError("reports has duplicate StudyInstanceUID")
    report_of = dict(zip(reports[UID], reports["Report"]))

    rows, issue_rows, seen = [], [], set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                where = f"{path}:{lineno}"
                missing = [k for k in LINE_FIELDS if k not in rec]
                if missing:
                    raise ValueError(f"{where}: line lacks fields {missing}")
                uid, p = rec[UID], rec["pass"]
                if p not in PASSES:
                    raise ValueError(f"{where}: pass {p!r} not in {PASSES}")
                if rec["prompt_version"] != PROMPT_VERSION:
                    raise ValueError(f"{where}: extracted with prompt_version {rec['prompt_version']!r}, but "
                                     f"rubric.py is {PROMPT_VERSION!r}; validate with the matching commit")
                if (uid, p) in seen:
                    raise ValueError(f"{where}: duplicate record for ({uid}, {p})")
                seen.add((uid, p))
                if uid not in report_of:
                    raise ValueError(f"{where}: {uid} is not in reports")

                obj, issues, cell_errors = _line_issues(rec, report_of[uid])
                star_codes = [c for s, c in map(split_issue, issues) if s == "*"]
                targets = obj.get("targets") if obj is not None else None
                for t in TARGETS:
                    mine = [split_issue(i)[1] for i in issues if split_issue(i)[0] == t]
                    cell = targets.get(t) if isinstance(targets, dict) else None
                    cell = cell if isinstance(cell, dict) else {}
                    rows.append({
                        UID: uid, "pass": p, "target": t,
                        "model": rec["model"], "revision": rec["revision"],
                        "prompt_version": rec["prompt_version"], "finish_reason": rec["finish_reason"],
                        "report_side": _cell_value(obj.get("report_side")) if obj is not None else None,
                        **{k: _cell_value(cell.get(k)) for k in TARGET_FIELDS},
                        "valid": not cell_errors[t],
                        "errors": ";".join(cell_errors[t]),  # record-level (`*`) codes first
                        "warnings": ";".join(c for c in mine if c in WARNING_CODES),
                        "quote_tier": _quote_tier(cell.get("quote"), mine, star_codes),
                    })
                for i in issues:
                    scope, code = split_issue(i)
                    issue_rows.append({UID: uid, "pass": p, "target": scope,
                                       "level": "error" if is_error(i) else "warning",
                                       "error": code, "excerpt": _excerpt(rec["raw"], obj, scope)})
    if not rows:
        raise ValueError(f"no records in {paths}")

    long = pd.DataFrame(rows)
    rank = np.where((long["pass"] == "reasoning") & long["valid"], 0, np.where(long["pass"] == "base", 1, 2))
    chosen = pd.Series(rank, index=long.index).groupby([long[UID], long["target"]]).idxmin()
    long["selected"] = long.index.isin(chosen.to_numpy())

    review_cols = [UID, "pass", "target", "level", "error", "excerpt"]
    issues_df = pd.DataFrame(issue_rows, columns=review_cols)
    sel = long.loc[long["selected"], [UID, "pass", "target"]]
    keep_cell = issues_df.merge(sel, on=[UID, "pass", "target"], how="inner")
    keep_star = issues_df[issues_df["target"] == "*"].merge(
        sel[[UID, "pass"]].drop_duplicates(), on=[UID, "pass"], how="inner")
    review = pd.concat([keep_star, keep_cell], ignore_index=True)[review_cols]
    return long, review


def validity_summary(records_long: pd.DataFrame) -> dict:
    """Valid shares per pass (records and cells, per target) and for the selected cells, plus finish_reason counts."""
    def block(df: pd.DataFrame) -> dict:
        per_record = df.groupby([UID, "pass"])["valid"].all()
        return {
            "n_studies": int(df[UID].nunique()),
            "n_records": int(len(per_record)),
            "record_valid_share": float(per_record.mean()),
            "cell_valid_share": float(df["valid"].mean()),
            "cell_valid_share_by_target": {t: float(df.loc[df["target"] == t, "valid"].mean()) for t in TARGETS},
        }

    first = records_long.drop_duplicates([UID, "pass"])
    return {
        "by_pass": {p: block(records_long[records_long["pass"] == p])
                    for p in PASSES if (records_long["pass"] == p).any()},
        "selected": block(records_long[records_long["selected"]]),
        "finish_reason": {f"{p}/{r}": int(n) for (p, r), n in first.groupby(["pass", "finish_reason"]).size().items()},
    }


def quote_tier_table(records_long: pd.DataFrame) -> pd.DataFrame:
    """Selected cells per target (+ TOTAL) by quote tier (QUOTE_TIERS: silent / exact / ws_normalised / folded / repaired / spliced / not_in_report / blank / unchecked)."""
    sel = records_long[records_long["selected"]]
    unknown = set(sel["quote_tier"]) - set(QUOTE_TIERS)
    if unknown:
        raise ValueError(f"unknown quote tiers {sorted(unknown)}")
    t = pd.crosstab(sel["target"], sel["quote_tier"]).reindex(index=list(TARGETS), columns=list(QUOTE_TIERS),
                                                               fill_value=0)
    t.loc["TOTAL"] = t.sum()
    return t.rename_axis(index="target", columns=None).reset_index()


def validity_by_group(records_long: pd.DataFrame, groups: pd.Series, name: str) -> pd.DataFrame:
    """Selected cells by a study-level grouping (uid -> label, e.g. language): validity, silent and quote-tier shares (+ ALL)."""
    sel = records_long[records_long["selected"]]
    label = sel[UID].map(groups)
    if label.isna().any():
        raise ValueError(f"{int(label.isna().sum())} selected cells have no {name}, e.g. {sel.loc[label.isna(), UID].iloc[0]}")
    sel = sel.assign(**{name: label})

    def row(df: pd.DataFrame) -> dict:
        return {"n_studies": int(df[UID].nunique()),
                "record_valid_share": float(df.groupby(UID)["valid"].all().mean()),
                "cell_valid_share": float(df["valid"].mean()),
                **{f"{t}_share": float((df["quote_tier"] == t).mean())
                   for t in ("silent", "ws_normalised", "folded", "repaired", "spliced", "not_in_report")}}

    rows = [{name: g, **row(df)} for g, df in sel.groupby(name)]
    out = pd.DataFrame(rows).sort_values("n_studies", ascending=False, kind="stable")
    return pd.concat([out, pd.DataFrame([{name: "ALL", **row(sel)}])], ignore_index=True)
