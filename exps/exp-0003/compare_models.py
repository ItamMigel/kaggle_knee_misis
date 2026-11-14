"""Two extractions (build_labels.py runs) side by side: cell validity and state agreement, soft-label agreement (local, CPU).

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0003/compare_models.py

Reads records_long.csv and labels_wide.csv of the build_labels.py runs RUN_A and RUN_B. Writes
model_agreement.csv (compare_models on the studies both extracted) and model_compare_wide.csv
(compare_with_reference, ours = A, ref = B, on the common unlabeled studies). Never reads gold
label values: train.csv labels are used only as the NaN mask that tells gold studies apart.
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

import pandas as pd  # noqa: E402

from src.report_labels.compare import compare_models, compare_with_reference, split_gold  # noqa: E402
from src.report_labels.rubric import TARGETS  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

RUN_A = "20261017t191223-797-3f77"  # 14B pilot+gold58; build_labels.py run id of model A
RUN_B = "20261017t154557-714-758b"  # 32B pilot; build_labels.py run id of model B (the reference side of model_compare_wide.csv)
COMP = "rsna-knee-abnormality-detection"
UID = "StudyInstanceUID"

if not RUN_A or not RUN_B:
    raise SystemExit("set RUN_A and RUN_B to build_labels.py run ids")

ctx = get_ctx()
out = ctx.output_dir

train = pd.read_csv(ctx.mounts[COMP] / "train.csv")
_, unlabeled = split_gold(train)
del train  # gold label values are not used past the NaN mask above

records = {r: pd.read_csv(ctx.mounts[r] / "records_long.csv") for r in (RUN_A, RUN_B)}
wide = {r: pd.read_csv(ctx.mounts[r] / "labels_wide.csv") for r in (RUN_A, RUN_B)}
for r in (RUN_A, RUN_B):
    print(f"[compare_models] {r}: {records[r][UID].nunique()} studies, models {sorted(records[r]['model'].unique())}")

agreement = compare_models(records[RUN_A], records[RUN_B])
agreement.to_csv(out / "model_agreement.csv", index=False)
common = pd.Index(wide[RUN_A][UID]).intersection(pd.Index(wide[RUN_B][UID])).intersection(unlabeled)
if len(common) == 0:
    raise SystemExit("the two runs have no unlabeled study in common")
compare = compare_with_reference(wide[RUN_A], wide[RUN_B], common, require_complete_ref=False)
compare.to_csv(out / "model_compare_wide.csv", index=False)

n_studies = int(agreement["n_common"].iloc[-1]) // len(TARGETS)
print(f"[compare_models] cells on {n_studies} common studies (A = {RUN_A}, B = {RUN_B}):")
print(agreement.to_string(index=False, float_format="%.3f"))
print(f"[compare_models] soft labels, A vs B, on {len(common)} common unlabeled studies:")
print(compare[["target", "n_compared", "pos_rate_ours", "pos_rate_ref", "agreement", "kappa",
               "spearman"]].to_string(index=False, float_format="%.3f"))
print(f"[compare_models] done | {out}")
