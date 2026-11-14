"""Validate extractor output and map it to the soft-label table (local, CPU).

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0003/build_labels.py

Reads the raw.part-NNNN.jsonl of every extract.py kernel run in EXTRACT_RUNS (fetched with
`kaggle/fetch.py --kernel <ref> --from-log` into exps/exp-0003/runs/<run_id>/), checked by
collect_parts. Writes records_long.csv (the selected cells of validate_run's long table),
labels_long.csv, labels_wide.csv, review_queue.csv, validity.json, policy.json, quote_tiers.csv,
tokens.json, validity_by_language.csv, compare_s4.csv (extracted unlabeled studies) and, with
RENDER_REVIEW, pilot_review.md.
Never reads gold label values: train.csv labels are used only as the NaN mask that tells
gold studies from unlabeled ones.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import os  # noqa: E402

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json  # noqa: E402

import pandas as pd  # noqa: E402

from src.report_labels.compare import compare_with_reference, render_review, split_gold  # noqa: E402
from src.report_labels.mapping import POLICY_V1, map_to_targets, to_wide  # noqa: E402
from src.report_labels.pilot import detect_language  # noqa: E402
from src.report_labels.rubric import TARGETS  # noqa: E402
from src.report_labels.runs import collect_parts, token_summary  # noqa: E402
from src.report_labels.validate import (quote_tier_table, validate_run, validity_by_group,  # noqa: E402
                                        validity_summary)
from utils.ctx import get_ctx  # noqa: E402

EXTRACT_RUNS: list[str] = ["20261017t152927-764-4766"]  # run ids of extract.py kernels, fetched with --from-log
ALLOW_UNFINISHED = False  # True: also use the parts of a kernel that stopped early (no stats.json)
POLICY = POLICY_V1
RENDER_REVIEW = True  # pilot: per-study markdown review; set False for the full 4,407-study run
COMP = "rsna-knee-abnormality-detection"
S4 = ("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv")
UID = "StudyInstanceUID"

if not EXTRACT_RUNS:
    raise SystemExit("EXTRACT_RUNS is empty: put the run ids of the extract.py kernels there")

ctx = get_ctx()
out = ctx.output_dir

raw_files = []
for run in EXTRACT_RUNS:
    parts = collect_parts(ctx.mounts[run], allow_unfinished=ALLOW_UNFINISHED)
    raw_files += parts
    print(f"[build_labels] {run}: {len(parts)} raw parts")

train = pd.read_csv(ctx.mounts[COMP] / "train.csv")
_, unlabeled = split_gold(train)
reports = train[[UID, "Report"]]
del train  # gold label values are not used past the NaN mask above

records_long, review = validate_run(raw_files, reports)
labels_long = map_to_targets(records_long, POLICY)
labels_wide = to_wide(labels_long)
validity = validity_summary(records_long)

records_long[records_long["selected"]].to_csv(out / "records_long.csv", index=False)
labels_long.to_csv(out / "labels_long.csv", index=False)
labels_wide.to_csv(out / "labels_wide.csv", index=False)
review.to_csv(out / "review_queue.csv", index=False)
(out / "validity.json").write_text(json.dumps(validity, indent=2), encoding="utf-8")
(out / "policy.json").write_text(json.dumps(POLICY, indent=2, ensure_ascii=False), encoding="utf-8")
tiers = quote_tier_table(records_long)
tiers.to_csv(out / "quote_tiers.csv", index=False)
tokens = token_summary(raw_files)
(out / "tokens.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
extracted = pd.Index(labels_wide[UID])
language = reports.set_index(UID).loc[extracted, "Report"].map(detect_language)
by_language = validity_by_group(records_long, language, "language")
by_language.to_csv(out / "validity_by_language.csv", index=False)

s4 = pd.read_csv(ctx.mounts[S4[0]] / S4[1])[[UID, *TARGETS]]
cmp_uids = extracted.intersection(unlabeled)
compare = compare_with_reference(labels_wide, s4, cmp_uids, require_complete_ref=True)
compare.to_csv(out / "compare_s4.csv", index=False)

if RENDER_REVIEW:
    render_review(records_long, labels_long, reports, s4, sorted(labels_wide[UID]), out / "pilot_review.md")

sel = validity["selected"]
print(f"[build_labels] studies {sel['n_studies']} | selected cells valid {sel['cell_valid_share']:.4f} | "
      f"records fully valid {sel['record_valid_share']:.4f} | review rows {len(review)} "
      f"(errors {int((review['level'] == 'error').sum())})")
for p, b in validity["by_pass"].items():
    print(f"[build_labels] pass {p}: records {b['n_records']}, record valid share {b['record_valid_share']:.4f}")
for k, t in tokens.items():
    print(f"[build_labels] tokens {k}: output {t['output_tokens']} | prompt {t['prompt_tokens']} | "
          f"finish {t['finish_reasons']}")
print("[build_labels] quote tiers (selected cells):")
print(tiers.to_string(index=False))
print(by_language.to_string(index=False, float_format="%.3f"))
print(f"[build_labels] vs S4 on {len(cmp_uids)} unlabeled studies:")
print(compare[["target", "pos_rate_ours", "pos_rate_ref", "silent_rate", "invalid_rate", "agreement", "kappa",
               "spearman"]].to_string(index=False, float_format="%.3f"))
print(f"[build_labels] done | {out}")
