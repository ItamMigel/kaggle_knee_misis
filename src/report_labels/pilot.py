"""Which studies an extraction run labels: smoke, pilot (languages/bilateral/disagreements), shards.

Deterministic across machines: order comes from sha256(seed:uid), not from an
RNG, so the kernel and a local run select identically. gold58 (train rows with
labels) never enters smoke or pilot; shards cover all of train.csv, gold included.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

from src.report_labels.rubric import TARGETS

# The 9 languages of the corpus, by frequency (research/report_labels.md, languages).
LANGUAGES = ("English", "Spanish", "Turkish", "Croatian", "Greek", "German", "Bulgarian",
             "Dutch", "French")
BILATERAL_MARK = "BILATERAL NOTE"
# Findings where S4 and P disagree for a reason worth auditing (report_labels.md failure modes).
DISAGREEMENT_TARGETS = ("Synovitis", "Medial OA", "Lateral OA", "PF OA", "Contusion", "Fracture",
                        "ACL", "MCL")

# --- lexical language detector: ported as-is from scratch/scout-labels/languages.py ---
_WORDS = {
    'English': 'the and with without there is are knee findings normal tear intact anterior posterior joint effusion no evidence medial lateral meniscus technique',
    'Spanish': 'rodilla hallazgos menisco cruzados ligamentos derrame sinovitis impresion articular rotura cartilago conservado conservados condilo senal sin lesiones alteraciones',
    'French': 'genou constatations trouvailles epanchement ligaments ligament menisque menisques osseux osseuse kyste pas aucune aucun sans avec une des les cartilage',
    'German': 'knie kniegelenk unauffallig unauffallige unauffalliges knorpel knorpelbesatz kreuzband innenmeniskus aussenmeniskus keine kein der die das mit ist sowie gelenkerguss intakte',
    'Dutch': 'knie klinische inlichtingen bevindingen besluit vraagstelling geen kraakbeen ter van het een met kruisband scheur voorste achterste mediale laterale meniscus vocht',
    'Turkish': 'diz eklemi meniskus normaldir izlendi izleniyor izlenmistir izlenmedi sivi bag capraz ve arka on medyal medial lateral yirtik sinyal kollateral eklem',
    'Croatian': 'koljena koljeno medijalnog lateralnog meniska menisk uredan uredne uredna urednog nema zglobne hrskavice krizni prednji straznji te uz na je se u i izljeva edem',
}
_LEX = {k: set(v.split()) for k, v in _WORDS.items()}
_ANCHORS = {
    'English': r'\b(the|findings|technique|knee|there|without)\b',
    'Spanish': r'\b(rodilla|hallazgos|resultados|impresion|rotura|condropatia)\b',
    'French': r'\b(constatations|trouvailles|epanchement|croise|genou|menisque|menisques|osseuse)\b',
    'German': r'\b(kniegelenk|knorpel|knorpelbesatz|kreuzbander|kreuzband|kompartiment|keine|kein|intakter)\b',
    'Dutch': r'\b(bevindingen|klinische|inlichtingen|kruisband|kraakbeen|voorste|besluit|geen)\b',
    'Turkish': r'\b(diz|meniskus|eklemi|eklem|izleniyor|izlenmistir|bulgular|capraz|yirtik|sivi)\b',
    'Croatian': r'\b(koljena|koljeno|hrskavice|krizni|uredan|uredna|zglobne|izljeva|izljev|kostani)\b',
}


def _norm(s: str) -> str:
    """Lowercase, dotless ı → i, accents stripped (NFKD) — detector input."""
    return ''.join(c for c in unicodedata.normalize('NFKD', s.lower().replace('ı', 'i'))
                   if not unicodedata.combining(c))


def detect_language(report: str) -> str:
    """Lexical language of a report: one of LANGUAGES or 'Unresolved' (Greek/Cyrillic by script)."""
    s = str(report)
    greek = len(re.findall(r'[Ͱ-Ͽἀ-῿]', s))
    cyr = len(re.findall(r'[Ѐ-ӿ]', s))
    if greek > 20:
        return 'Greek'
    if cyr > 20:
        return 'Bulgarian'
    ns = _norm(s)
    toks = re.findall(r'\b[a-z]+\b', ns)
    freq = {k: sum(t in v for t in toks) + 10 * len(re.findall(_ANCHORS[k], ns)) for k, v in _LEX.items()}
    order = sorted(freq, key=freq.get, reverse=True)
    return order[0] if freq[order[0]] > 0 else 'Unresolved'
# --- end of port ---


def _rank(uids: pd.Series, seed: int) -> pd.Series:
    """Seeded, platform-independent sort key per uid (hex of sha256(seed:uid))."""
    return uids.map(lambda u: hashlib.sha256(f"{seed}:{u}".encode()).hexdigest())


def _candidates(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Non-gold studies whose report text is not also a gold report, seeded order, with language."""
    missing = {"StudyInstanceUID", "Report", *TARGETS} - set(train.columns)
    if missing:
        raise ValueError(f"train lacks columns {sorted(missing)}")
    if not train["StudyInstanceUID"].is_unique:
        raise ValueError("train has duplicated StudyInstanceUID")
    gold = train[list(TARGETS)].notna().any(axis=1)
    pool = train.loc[~gold & ~train["Report"].isin(set(train.loc[gold, "Report"])),
                     ["StudyInstanceUID", "Report"]].copy()
    pool["language"] = pool["Report"].map(detect_language)
    pool["_key"] = _rank(pool["StudyInstanceUID"], seed)
    return pool.sort_values("_key", kind="mergesort").reset_index(drop=True)


def _take(pool: pd.DataFrame, n: int, reason: str, taken: set[str], texts: set[str]) -> list[dict]:
    """First n pool rows (pool order) not yet taken and with a not-yet-seen report text."""
    rows = []
    for r in pool.itertuples(index=False):
        if len(rows) == n:
            break
        if r.StudyInstanceUID in taken or r.Report in texts:
            continue
        taken.add(r.StudyInstanceUID)
        texts.add(r.Report)
        rows.append({"StudyInstanceUID": r.StudyInstanceUID, "reason": reason, "language": r.language})
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} candidates for {reason!r}, quota {n}")
    return rows


def select_smoke(train: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n non-gold studies, one per language in LANGUAGES order (n ≤ 9): the smoke run."""
    if not 1 <= n <= len(LANGUAGES):
        raise ValueError(f"smoke n must be 1..{len(LANGUAGES)}, got {n}")
    pool, taken, texts, rows = _candidates(train, seed), set(), set(), []
    for lang in LANGUAGES[:n]:
        rows += _take(pool[pool["language"] == lang], 1, f"smoke:{lang}", taken, texts)
    return pd.DataFrame(rows)


def select_pilot(train: pd.DataFrame, s4: pd.DataFrame, p: pd.DataFrame, n_per_language: int,
                 n_bilateral: int, n_disagreement: int, seed: int) -> pd.DataFrame:
    """Pilot selection (no gold58): language quotas, BILATERAL NOTE reports, top S4-vs-P disagreements.

    Taken in that order without repeats (a study or report text counts once):
    n_per_language per each of LANGUAGES, n_bilateral reports with BILATERAL_MARK,
    then n_disagreement per DISAGREEMENT_TARGETS by |S4 − P| (ties in seeded order;
    studies missing from S4 or P are not ranked). Returns StudyInstanceUID, reason, language.
    """
    pool, taken, texts, rows = _candidates(train, seed), set(), set(), []
    for lang in LANGUAGES:
        rows += _take(pool[pool["language"] == lang], n_per_language, f"language:{lang}", taken, texts)
    rows += _take(pool[pool["Report"].str.contains(BILATERAL_MARK, regex=False)], n_bilateral,
                  "bilateral", taken, texts)
    for name, ref in (("s4", s4), ("p", p)):
        if not ref["StudyInstanceUID"].is_unique:
            raise ValueError(f"{name} has duplicated StudyInstanceUID")
        lacking = set(DISAGREEMENT_TARGETS) - set(ref.columns)
        if lacking:
            raise ValueError(f"{name} lacks target columns {sorted(lacking)}")
    both = (pool.merge(s4[["StudyInstanceUID", *DISAGREEMENT_TARGETS]], on="StudyInstanceUID")
                .merge(p[["StudyInstanceUID", *DISAGREEMENT_TARGETS]], on="StudyInstanceUID",
                       suffixes=("__s4", "__p")))
    for t in DISAGREEMENT_TARGETS:
        d = both.assign(_gap=(both[f"{t}__s4"] - both[f"{t}__p"]).abs()).dropna(subset=["_gap"])
        d = d.sort_values(["_gap", "_key"], ascending=[False, True], kind="mergesort")
        rows += _take(d, n_disagreement, f"disagreement:{t}", taken, texts)
    return pd.DataFrame(rows)


def select_shard(train: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """Shard k of n over ALL train studies (gold included), by sha256(uid) mod n."""
    if not 0 <= k < n:
        raise ValueError(f"shard k must be in 0..{n - 1}, got {k}")
    if not train["StudyInstanceUID"].is_unique:
        raise ValueError("train has duplicated StudyInstanceUID")
    part = train["StudyInstanceUID"].map(lambda u: int(hashlib.sha256(u.encode()).hexdigest(), 16) % n)
    uids = train.loc[part == k, "StudyInstanceUID"].sort_values(kind="mergesort")
    return pd.DataFrame({"StudyInstanceUID": uids.to_numpy(), "reason": f"shard:{k}/{n}"})
