"""exp-0001: submit_a_s6fall.py (kernel e775, public 0.958) with W_S6FALL 0.50 instead of 0.10 (lead 21.10; validated
alongside w0.40 and submitted only if w0.40 passes 6000 on the static-team detector), nothing else changed: S6fall's
component B, then public notebook A v7, then the per-target rank blend (1 - W) * pct(A) + W * pct(S6fall) -> submission.csv.

    uv run --project kaggle python kaggle/run_kernel.py --exp exp-0001 --script submit_a_s6fall_w50 --accelerator t4 \
      --docker-image gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f \
      --ds goodpjw2008/rsna-knee-2-5d-convnext-reader,heliosli/rsna-knee-r6-frozen-fe055406-w1,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004,bebra3222/rsna-knee-s6fall-weights \
      [--submit-file submission.csv]

B first (lead 18.10): K1 installs nothing, while A's R6 pip-installs timm 1.0.25 / pydicom 3.0.1 over the image's 1.0.29 /
3.0.2, so B-then-A runs both pipelines in their scored environments. B runs in a child process
(src.public_fork_s6fall.run_s6fall_b); A (src.public_fork.pipeline.run_public, untouched) starts only after both GPUs are
released. A's internal deadlines (R6 8.3 h, Part B 8.2 h) count from T0, B included: the margins are logged per stage, and
a study R6 did not reach before its deadline or a Part B skipped for time raises. The blend is src.blend_kernel.rank_blend:
a study S6fall could not read takes A's rank. No fallback anywhere (HIDDEN_FALLBACK False semantics): any failure raises.
Validation (the 3 visible studies): rank(B) must reproduce the scored S6fall v2 submission.csv sha256.
Log: S6FALL_B_SUMMARY, GPU_RELEASED, A's EXP0001_* lines, A_DEADLINE_MARGIN, A_S6FALL_SUMMARY.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import time  # noqa: E402

T0 = time.time()   # A's deadlines (R6 8.3 h, Part B 8.2 h) count from the kernel start, S6fall's B included

import json  # noqa: E402
import os  # noqa: E402

import pandas as pd  # noqa: E402

from utils.ctx import get_ctx  # noqa: E402

ctx = get_ctx()

from src.blend_kernel import _check_submission, rank_blend  # noqa: E402
from src.log_transport import emit_file  # noqa: E402
from src.public_fork import pipeline as A  # noqa: E402
from src.public_fork_s6fall import pipeline as S6  # noqa: E402

W_S6FALL = 0.50                 # lead 21.10 (one step above w0.40): S6fall's rank weight, A gets 1 - it
MAX_UNREADABLE_FRAC = 0.01      # more S6fall-unreadable studies than this raises
B_BUDGET_S = 2.5 * 3600         # run_b's deadline: 2.2x the 1.16 h projection at 5.75 series/study (run 20261018t105255-901-9816);
                                # even at the cap A (4.9 h there) ends by 7.4 h, before its 8.2 / 8.3 h deadlines
GPU_FREE_MIB, GPU_WAIT_S = 500, 120
VISIBLE_STUDIES = ["1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191",
                   "1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776",
                   "1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402"]
K1_SOLO_SHA256 = "00031c5337550d13acaf2e8dbf8743e260169d93da64296d479f2c0fb0c0d6b0"   # scored S6fall v2 commit; run 20261018t115028-846-7c6e
A_STAGES = ("raptor384", "raptor224", "r6", "blend_stack", "part_b")                   # run_public's order and timings keys
R6_DEADLINE_S, PART_B_CUTOFF_S = A.DEADLINE_HOURS * 3600, A.PART_B_CUTOFF_HOURS * 3600

if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
    raise RuntimeError("submit_a_s6fall_w50.py runs only inside a Kaggle kernel; launch it with the kaggle skill, command in the docstring.")
for key in (A.DATA_MOUNT, A.READER_MOUNT, A.BUNDLE_MOUNT, A.RAPTOR_W_MOUNT, A.CODE_MOUNT, S6.WEIGHTS_MOUNT):
    ctx.mounts[key]             # every input attached, else KeyError here, before B's hour
WORK = ctx.output_dir
test_ids = pd.read_csv(ctx.mounts[A.DATA_MOUNT] / "test.csv", dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
mode = "validation" if sorted(test_ids) == sorted(VISIBLE_STUDIES) else "hidden"
print("A_S6FALL_MODE", json.dumps({"mode": mode, "n_studies": len(test_ids), "w_s6fall": W_S6FALL}), flush=True)


def margin(stage, end_s):
    """A_DEADLINE_MARGIN line: hours left to R6's deadline and Part B's cutoff when `stage` ended, end_s after T0."""
    rec = {"stage": stage, "end_h": round(end_s / 3600, 3), "r6_deadline_margin_h": round((R6_DEADLINE_S - end_s) / 3600, 3),
           "part_b_cutoff_margin_h": round((PART_B_CUTOFF_S - end_s) / 3600, 3)}
    print("A_DEADLINE_MARGIN " + json.dumps(rec), flush=True)
    return rec


# ---- S6fall B (child process), then both GPUs released ----
probs, unreadable, b_timings, b_receipt = S6.run_s6fall_b(ctx, T0, B_BUDGET_S, MAX_UNREADABLE_FRAC)
if mode == "validation" and b_receipt["k1_solo_sha256"] != K1_SOLO_SHA256:
    raise RuntimeError("S6fall parity: rank(B) gives submission sha256 %s, the scored v2 commit %s"
                       % (b_receipt["k1_solo_sha256"], K1_SOLO_SHA256))
gpu_used = S6.wait_gpus_released(GPU_FREE_MIB, GPU_WAIT_S)

# ---- A v7, untouched; its time cuts raise ----
a_start = time.time() - T0
margins = [margin("s6fall_b", a_start)]
final_a, _, _, a_timings, part_b = A.run_public(ctx, T0)
end = a_start
for stage in A_STAGES:
    end += a_timings[stage]
    margins.append(margin(stage, end))
r6_receipt = json.loads((WORK / "submission_receipt.json").read_text())
cut = sorted(u for w in r6_receipt["workers"] for u, why in (w.get("fallback") or {}).items() if "deadline" in str(why).lower())
timed_out = [w.get("worker", i) for i, w in enumerate(r6_receipt["workers"]) if w.get("timed_out")]
if cut or timed_out:
    raise RuntimeError("A's R6 hit its %.1f h deadline: workers %s timed out, %d studies not reached, e.g. %s"
                       % (A.DEADLINE_HOURS, timed_out, len(cut), cut[:3]))
if part_b["fired"] and part_b["reason"].startswith("time budget"):
    raise RuntimeError("A's Part B (reader) was skipped for time: %s" % part_b["reason"])

# ---- blend -> submission.csv (atomically, over A's) ----
blend, receipt = rank_blend(final_a, {"s6fall": probs}, {"s6fall": W_S6FALL}, {"s6fall": unreadable}, S6.LABELS)
_check_submission(blend, final_a)
sub_path = WORK / "submission.csv"
tmp = sub_path.with_name(sub_path.name + ".a_s6fall.tmp")
blend.to_csv(tmp, index=False)
os.replace(tmp, sub_path)

total_s = time.time() - T0
summary = {"mode": mode, "rows": len(blend), "weights": receipt["weights"], "unreadable": len(unreadable),
           "k1_solo_sha256": b_receipt["k1_solo_sha256"], "spearman_vs_public": receipt["spearman_vs_public"],
           "gpu_used_mib_before_a": gpu_used, "part_b": part_b, "margins": margins,
           "timings": {"s6fall_b": b_timings, "a": a_timings, "total": total_s},
           "submission_head": blend.head(3).to_dict(orient="records")}
(WORK / "a_s6fall_receipt.json").write_text(json.dumps(summary | {"blend": receipt, "s6fall_b": b_receipt}, indent=1, default=str))
print("A_S6FALL_SUMMARY " + json.dumps(summary, default=str), flush=True)
if mode == "validation":
    for path, name in ((sub_path, "submission.csv"), (WORK / "s6fall" / "s6fall_prob.csv", "s6fall_prob.csv"),
                       (WORK / "a_s6fall_receipt.json", "a_s6fall_receipt.json")):
        emit_file(path, name, 3000)
