"""Pilot review (contract D): PNG montages of a cache slug and pilot_report.json.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0002/montage.py

Set BUILD_RUN to the build_cache.py run that produced the pilot first. Writes to ctx.output_dir:
  montages/<plane>_<i>_<series>.png  every k-th slice of N_PER_PLANE random series per plane
                                     (fg bbox in yellow, edge letters from the stored geometry);
  montages/reported_corrupt_<sop>.png  each SOP reported as corrupt, highlighted in red among its
                                     neighbouring slices;
  pilot_report.json                  transfer syntaxes, plane/laterality agreement, bilateral
                                     suspects, dropped rows, error codes, per-study bytes and seconds
                                     of BUILD_RUN, and the full-pass size/time extrapolation.
"""

# --- bootstrap: find utils/. Constant text, the same in every script ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- end of bootstrap ---

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.knee_volume import PLANES  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from src.vol_report import pilot_report, render_tiles, series_title, write_png  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SLUG = "knee-vol04"
INDEX_RUN = "20261017t111437-152-050c"  # index_archive.py
BUILD_RUN = "20261017t114458-985-14c1"  # build_cache.py run id of the pilot (its run_log.parquet / throughput.json)
SEED = 0
N_PER_PLANE = 3
MAX_TILES = 24  # every k-th slice with k = ceil(Z / MAX_TILES)
NEIGHBOURS = 2  # slices on each side of a reported-corrupt SOP
KNOWN_REPORTED_CORRUPT = [  # research/discussions.md, «Data and inference traps»
    "1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320",
    "1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626",
]
CORRUPT_NOTE = (
    "Reported on the forum as half-size pixel payloads (409600/819200 and 173056/346112 bytes). In the "
    "current archive both members are complete (usize 820652 / 347616, PixelData = Rows*Cols*2) and decode, "
    "so contract B's policy keeps them; no special case by decision of orch-0002. Both are edge slices of "
    "their stacks (InstanceNumber 36/36 and 1/38): judge them visually in the PNGs."
)

if not BUILD_RUN:
    raise SystemExit("montage.py: set BUILD_RUN to the build_cache.py run id that built the pilot")

ctx = get_ctx()
cache = KneeVolCache(ctx.mounts[f"exp-0002/{SLUG}"])
out = ctx.output_dir / "montages"
out.mkdir()
rows = {r["series"]: r for r in cache.series.to_dict("records")}
ok = cache.series[cache.series.error_code.isna()]
rng = np.random.default_rng(SEED)

montages = []
for plane in PLANES:
    cand = sorted(ok.loc[ok.plane_computed == plane, "series"])
    for i, series in enumerate(sorted(rng.choice(cand, min(N_PER_PLANE, len(cand)), replace=False))):
        vol, pos, meta = cache.load(series)
        ks = list(range(0, meta["Z"], max(1, math.ceil(meta["Z"] / MAX_TILES))))
        name = f"{plane}_{i}_{series}.png"
        write_png(out / name, render_tiles(vol, pos, meta, ks, series_title(meta, rows[series])))
        montages.append({"png": name, "plane": plane, "study": meta["study"], "series": series, "ks": ks})

corrupt_pngs = {}
for sop in KNOWN_REPORTED_CORRUPT:
    hit = cache.slices[cache.slices.sop == sop]
    if hit.empty:
        continue  # dropped or not in this cache: pilot_report says which
    series, k = hit.series.iloc[0], int(hit.k.iloc[0])
    vol, pos, meta = cache.load(series)
    lo = max(0, min(k - NEIGHBOURS, meta["Z"] - 2 * NEIGHBOURS - 1))
    ks = list(range(lo, min(meta["Z"], lo + 2 * NEIGHBOURS + 1)))
    name = f"reported_corrupt_{sop}.png"
    title = [f"reported-corrupt SOP ..{sop[-12:]} = k={k} (red)"] + series_title(meta, rows[series])
    write_png(out / name, render_tiles(vol, pos, meta, ks, title, highlight={k}))
    corrupt_pngs[sop] = name

run_dir = ctx.mounts[BUILD_RUN]
index = pd.read_parquet(ctx.mounts[INDEX_RUN] / "archive_index.parquet",
                        columns=["split", "study", "series", "lho", "next_lho"])
report = pilot_report(cache, pd.read_parquet(run_dir / "run_log.parquet"),
                      json.loads((run_dir / "throughput.json").read_text(encoding="utf-8")),
                      index, KNOWN_REPORTED_CORRUPT, corrupt_pngs, CORRUPT_NOTE)
report.update(slug=SLUG, build_run=BUILD_RUN, index_run=INDEX_RUN, montages=montages)
(ctx.output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

ex = report["extrapolation"]
print(f"[montage] {len(montages)} montages + {len(corrupt_pngs)} reported-corrupt PNGs | "
      f"series ok {report['n_series_ok']}/{report['n_series']}, dropped {report['n_dropped']} | "
      f"full pass ≈ {ex['cache_gb_by_compressed']:.1f} GB, ≈ {ex['wall_h_by_compressed']:.2f} h wall | "
      f"{ctx.output_dir}")
