"""Local reference for the Kaggle preprocessing parity check: fingerprints of the series kaggle_parity.py re-preprocesses.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/parity_ref.py

Series: every series of the visible test studies (load_series_dir on the local test_series copy),
and from the full train cache up to N_PER_TS series per non-LE transfer syntax (different
studies), N_LE random Explicit-LE series, every series of the two reported-corrupt studies and up
to N_ERR failed series (their SeriesError code is the expected result). Cache == load_series_dir
locally is exp-0002's parity result. Writes parity_ref.json (paste its "series" into kaggle_parity.py).
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import json  # noqa: E402
from collections import Counter  # noqa: E402

import pandas as pd  # noqa: E402

from src.kernel_env import env_report, volume_fingerprint  # noqa: E402
from src.knee_volume import load_series_dir  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SEED = 0
CACHE = "exp-0002/knee-vol04-pilot"  # LE only; the full cache adds the compressed syntaxes
LE = "1.2.840.10008.1.2.1"
N_PER_TS, N_LE, N_ERR = 3, 12, 3
CORRUPT_STUDIES = ["1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035",
                   "1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680"]

ctx = get_ctx()
comp = ctx.mounts["rsna-knee-abnormality-detection"]
cache = KneeVolCache(ctx.mounts[CACHE])
s = cache.series.copy()
s["ts"] = [",".join(t) if t is not None else None for t in s.transfer_syntaxes]
ok, err = s[s.error_code.isna()], s[s.error_code.notna()]

pick = []
for ts, g in ok.groupby("ts"):
    if ts != LE:
        g = g.drop_duplicates("study")
        pick += g.sample(min(N_PER_TS, len(g)), random_state=SEED).series.tolist()
pick += ok[ok.ts == LE].sample(N_LE, random_state=SEED).series.tolist()
pick += ok[ok.study.isin(CORRUPT_STUDIES)].series.tolist()
pick = list(dict.fromkeys(pick))

rows = []
for sr in pick:
    vol, pos, meta = cache.load(sr, mmap=False)
    rows.append({"split": "train", "study": meta["study"], "series": sr, "ts": ",".join(meta["transfer_syntaxes"]),
                 "fp": volume_fingerprint(vol, pos, meta)})
for r in err.drop_duplicates("error_code").head(N_ERR).itertuples():
    rows.append({"split": "train", "study": r.study, "series": r.series, "ts": None, "fp": {"error": r.error_code}})
test = pd.read_csv(comp / "test_series.csv")
for st, sr in zip(test.StudyInstanceUID, test.SeriesInstanceUID):
    vol, pos, meta = load_series_dir(comp / "test_series" / st / sr)
    rows.append({"split": "test", "study": st, "series": sr, "ts": ",".join(meta["transfer_syntaxes"]),
                 "fp": volume_fingerprint(vol, pos, meta)})

out = {"cache": CACHE, "env": env_report(), "series": rows,
       "cache_ts_counts": Counter(s.ts.dropna()).most_common(), "cache_error_counts": err.error_code.value_counts().to_dict()}
(ctx.output_dir / "parity_ref.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"n": len(rows), "by_ts": Counter(r["ts"] for r in rows), "cache_ts_counts": out["cache_ts_counts"],
                  "cache_error_counts": out["cache_error_counts"]}, default=str))
print(f"[parity_ref] {ctx.output_dir}")
