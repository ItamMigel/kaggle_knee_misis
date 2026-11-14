"""C5 ROI parity: member_rois (the inference path) on a random sample of train studies vs the C3 run ROI_RUN.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/roi_parity.py

member_rois runs on the full training cache in one process (as in the kernel) over N_STUDIES random
studies plus the studies of ROI_RUN's degenerate series. Every ROI field of every series and every
study side must equal knee_rois.parquet; the side-source counts must equal the parquet's. A C3-
degenerate series has no parquet row: member_rois gives it the image centre, reported separately
(and no other series may get "image_centre"). Writes parity.json, then raises on any mismatch.
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.joint_reader import CROP_MM_BY_PLANE, LOCATE, MM_PER_PX, OUT_PX, member_rois  # noqa: E402
from src.joint_view import ROI_FIELDS, load_roi_run  # noqa: E402
from src.knee_volume import canon_px_to_lps  # noqa: E402
from src.vol_cache import KneeVolCache  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

SEED = 0
N_STUDIES = 300
CACHE = "exp-0002/knee-vol04"
ROI_RUN = "20261017t124739-029-511f"  # the C3 run train.py reads
N_EXAMPLES = 20  # mismatches listed in parity.json


def main():
    ctx = get_ctx()
    cv2.setNumThreads(1)
    roi_dir = ctx.mounts[ROI_RUN]
    ref, ref_sides, degenerate = load_roi_run(roi_dir, cache_key=CACHE, crop_mm_by_plane=CROP_MM_BY_PLANE,
                                              mm_per_px=MM_PER_PX, out_px=OUT_PX, locate=LOCATE)
    table = pd.read_parquet(roi_dir / "knee_rois.parquet")
    deg_rows = pd.read_csv(roi_dir / "degenerate.csv")
    cache = KneeVolCache(ctx.mounts[CACHE])

    all_studies = sorted(set(cache.series.study))
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(all_studies, size=N_STUDIES, replace=False).tolist())
    forced = sorted(set(deg_rows.study.astype(str)) - set(sample))
    studies = sample + forced
    rois, sides, counts = member_rois(studies, cache, canon_px_to_lps)

    in_studies = cache.series.study.isin(studies)
    want_series = set(cache.series.series[in_studies & cache.series.error_code.isna()])
    deg_here = degenerate & want_series
    problems = {"missing_series": sorted(want_series - set(rois)), "extra_series": sorted(set(rois) - want_series),
                "not_in_c3": sorted(want_series - set(ref) - deg_here), "n_rois_differ": [], "fields": Counter(),
                "sides": [], "image_centre_outside_degenerate": [], "degenerate_not_image_centre": [],
                "side_sources": {}}
    examples, n_rois = [], 0
    for sr in sorted(want_series & set(rois) & set(ref)):
        got, exp = rois[sr], ref[sr]
        if len(got) != len(exp):
            problems["n_rois_differ"].append(sr)
            continue
        for i, (g, e) in enumerate(zip(got, exp)):
            n_rois += 1
            for f in ROI_FIELDS:
                if g[f] != e[f]:
                    problems["fields"][f] += 1
                    if len(examples) < N_EXAMPLES:
                        examples.append({"series": sr, "roi_idx": i, "field": f, "member": g[f], "c3": e[f]})
    problems["image_centre_outside_degenerate"] = sorted(
        sr for sr, lst in rois.items() if sr not in deg_here and any(r["method"] == "image_centre" for r in lst))
    problems["sides"] = [{"study": st, "member": sides.get(st, "<none>"), "c3": ref_sides[st]}
                         for st in studies if st in ref_sides and (st not in sides or sides[st] != ref_sides[st])]
    # side source per study: knee_rois.parquet, or degenerate.csv for a study whose usable series are all degenerate
    src = pd.concat([table[["study", "side_source"]], deg_rows[["study", "side_source"]]]).drop_duplicates("study")
    src = src.set_index("study").side_source
    want_sources = dict(Counter(src[st] for st in studies if st in src.index))
    got_sources = {k: v for k, v in counts["studies_by_side_source"].items() if v}
    if got_sources != want_sources:
        problems["side_sources"] = {"member": got_sources, "c3": want_sources}

    deg_report = []
    for row in deg_rows[deg_rows.series.isin(deg_here)].itertuples(index=False):
        c3_side = None if pd.isna(row.side) else row.side
        deg_report.append({"study": row.study, "series": row.series, "plane": row.plane,
                           "c3_degenerate_row": {k: (None if pd.isna(v) else v) for k, v in row._asdict().items()},
                           "member_rois": rois[row.series], "member_side": sides[row.study], "c3_side": c3_side})
        if [r["method"] for r in rois[row.series]] != ["image_centre"] or sides[row.study] != c3_side:
            problems["degenerate_not_image_centre"].append(row.series)

    failed = sorted(k for k, v in problems.items() if v)
    res = {"roi_run": ROI_RUN, "cache": CACHE, "seed": SEED, "n_sample": len(sample), "forced_degenerate_studies": forced,
           "n_studies": len(studies), "n_series_compared": len(want_series & set(ref)), "n_rois_compared": n_rois,
           "n_degenerate_series": len(deg_here), "degenerate": deg_report,
           "counts": counts, "seconds_per_series": round(counts["seconds"] / len(rois), 4),
           "problems": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in problems.items()},
           "examples": examples, "pass": not failed}
    (ctx.output_dir / "parity.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("n_studies", "n_series_compared", "n_rois_compared", "n_degenerate_series",
                                          "seconds_per_series", "pass")}), flush=True)
    print(json.dumps({"counts": counts, "degenerate": deg_report}, default=str), flush=True)
    if failed:
        raise AssertionError(f"member_rois differs from {ROI_RUN}: {failed}; see {ctx.output_dir / 'parity.json'}")
    print(f"[parity] member_rois == {ROI_RUN} on {len(studies)} studies | {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
