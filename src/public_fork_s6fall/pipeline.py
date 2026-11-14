"""Public kernel nartaa/rsna-knee-s6fall-solo v2 (LB 0.950) as one function, same contract as public_fork.run_public.

The kernel ("K1") runs unchanged as its own process: component A = Nartaa's 0.949 Raptor (used only for a study B cannot
read), component B = S6fall (Heliosli's S6 ConvNeXt retrained by Nartaa), final = rank(B) (W_A 0, W_B 1). K1 checks the
model, anatomy.pt, every code file and A's source against the scored sha256 itself and hard-fails on any component error.
Here on top: the vendored text's sha256 and the sha256 of every file of our immutable weights copy, before K1 starts.
Its inputs and their keys are module constants on purpose: they are the identity of the fork (as in public_fork.pipeline).
Kaggle-only: K1 reads /kaggle/input and writes /kaggle/working itself.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_fork import stages

VENDORED = Path(__file__).resolve().parent / "vendored"
K1_FILE = "rsna-knee-s6fall-solo.py"
K1_TIMEOUT_S = 8 * 3600 + 20 * 60      # K1's own hang guard fires at 8 h + 10 min backstop; this only catches a wedged child
B_DRIVER = Path(__file__).resolve().parent / "k1_driver.py"
B_DRIVER_SLACK_S = 900                  # on top of run_b's budget: the driver's env / inputs / probe steps and K1's retry bookkeeping
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# ---- inputs (ctx.mounts keys) ----
DATA_MOUNT = "rsna-knee-abnormality-detection"
# our immutable copy of the inference files of nartaa/rsna-knee-s6fall-weights v5 (that dataset was re-versioned v1->v5;
# the checkpoint and code_manifest.json are those of the scored v2 run: MANIFEST b66c574b..., model 0c7297d5...).
# Same slug as Nartaa's: K1's find_mount looks for datasets/*/rsna-knee-s6fall-weights.
WEIGHTS_MOUNT = "bebra3222/rsna-knee-s6fall-weights"
WEIGHTS_SHA = {
    "E_prior_s42_fall.pt": "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12",
    "E_prior_s42_fall.DONE.json": "acc99ed61393404feaa4f84a2e20ce27ad9f55a257b8e0d0760e83e105a20c6a",
    "E_prior_s42_fall.run_config.json": "36e49cf0c8db97e7f995482933911f018c32fca8c5933b4d64399c128c19c886",
    "checkpoint-manifest.json": "ed3c6d8d2e69ae08c4a97eab3c860a9ec20c5c69484e35de6a3119d396d731cd",
    "code_manifest.json": "b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22",
    "s6_extras.bin": "ddc428d9f2cda0e1378b6731ca7efbdd0397978204763f2a2f81169979e50c8e",
    "README.md": "04de798cfe142110322212be0af141044003039e0a8c3493a258e701ce0f6a75",
}
RAPTOR_W_MOUNT = "nartaa/rsna-knee-publication-swa-weights-20261007"     # component A (fallback rows only)
CODE_MOUNT = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"      # anatomy.pt + Heliosli's code (B)


def materialize_k1(dest_dir):
    """Write the vendored K1 source to dest_dir/rsna-knee-s6fall-solo.py, sha256-checked against vendored/sha256.json."""
    expected = json.loads((VENDORED / "sha256.json").read_text())[K1_FILE]
    dest = Path(dest_dir) / K1_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((VENDORED / (K1_FILE + ".txt")).read_bytes())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert digest == expected, ("vendored K1 differs from sha256.json", digest, expected)
    return dest


def check_weights(weights_dir):
    """Every file of the weights copy has its pinned sha256 and nothing else is there (a different copy fails loudly)."""
    weights_dir = Path(weights_dir)
    found = sorted(p.name for p in weights_dir.iterdir() if p.is_file())
    assert found == sorted(WEIGHTS_SHA), ("weights copy holds other files than pinned", found, sorted(WEIGHTS_SHA))
    got = stages.sha256_files([weights_dir / n for n in WEIGHTS_SHA])
    bad = {n: got[str(weights_dir / n)] for n in WEIGHTS_SHA if got[str(weights_dir / n)] != WEIGHTS_SHA[n]}
    assert not bad, ("weights copy differs from the pinned sha256", bad)


def run_s6fall(ctx, t0):
    """S6fall solo kernel in one call -> (final_sub, members, masks, timings, fallback), as public_fork.run_public.

    t0: the caller's script start (timings["total"] counts from it). members: "raptor0949" (component A's file, used only
    where B failed), "s6fall" (B's probabilities), "final"; masks: the studies B failed on (they took rank(A)). fallback:
    {"fired", "reason"} for B-failed studies, in the shape of run_public's part_b. Writes members/ like run_public.
    """
    if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
        raise RuntimeError("run_s6fall runs only inside a Kaggle kernel (K1 reads /kaggle/input and writes /kaggle/working). "
                           "Launch it with the kaggle skill, attaching %s, %s, %s and the competition."
                           % (WEIGHTS_MOUNT, RAPTOR_W_MOUNT, CODE_MOUNT))
    ctx.mounts[DATA_MOUNT], ctx.mounts[RAPTOR_W_MOUNT], ctx.mounts[CODE_MOUNT]      # all attached, else KeyError here
    check_weights(ctx.mounts[WEIGHTS_MOUNT])
    WORK = ctx.output_dir
    k1 = materialize_k1(Path("/tmp/k1run"))
    t = time.time()
    rc = subprocess.run([sys.executable, "-u", str(k1)], cwd="/tmp", timeout=K1_TIMEOUT_S).returncode
    if rc != 0:
        raise RuntimeError("S6fall kernel (K1) exited with %d; reason in %s" % (rc, WORK / "k1" / "k1_error.json"))
    k1_seconds = time.time() - t
    receipt = json.loads((WORK / "k1" / "k1_receipt.json").read_text())
    final_sub = pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str})
    ids = final_sub["StudyInstanceUID"].tolist()
    a = pd.read_csv(WORK / "k1" / "A" / "submission_A.csv", dtype={"StudyInstanceUID": str})
    b = pd.read_csv(WORK / "k1" / "B" / "B_probs.csv", dtype={"StudyInstanceUID": str})
    cols = ["StudyInstanceUID"] + LABELS

    def aligned(df):
        assert sorted(df["StudyInstanceUID"]) == sorted(ids), "member rows differ from submission.csv"
        return df.set_index("StudyInstanceUID").loc[ids, LABELS].reset_index()[cols]

    b_failed = receipt["blend"]["b_failed_rank_A_only"]
    mask_b = np.array([u in b_failed for u in ids])
    members = {"raptor0949": aligned(a), "s6fall": aligned(b), "final": aligned(final_sub)}
    masks = {"raptor0949": np.zeros(len(ids), dtype=bool), "s6fall": mask_b, "final": np.zeros(len(ids), dtype=bool)}
    stages.save_members(WORK, members, masks, receipt["blend"]["weights"])
    timings = {k: v.get("seconds") for k, v in receipt["stages"].items()}
    timings.update(k1=k1_seconds, total=time.time() - t0)
    fallback = {"fired": bool(b_failed), "reason": ("%d studies B failed on took rank(A)" % len(b_failed)) if b_failed else ""}
    if fallback["fired"]:
        print("!!! S6FALL_FALLBACK %s !!!" % fallback["reason"], flush=True)
    (WORK / "timings.json").write_text(json.dumps(timings, indent=1))
    print("S6FALL_SUMMARY", json.dumps({"timings": timings, "fallback": fallback, "rows": len(final_sub),
                                        "submission_sha256": receipt["blend"]["submission_sha256"]}), flush=True)
    return final_sub, members, masks, timings, fallback


def run_s6fall_b(ctx, t0, budget_s, max_unreadable_frac=0.01):
    """S6fall's component B alone on the competition test split -> (probs, unreadable, timings, receipt).

    K1's own functions (prepare_inputs -> probe_b_stack -> run_b) in a child process (k1_driver.py), so nothing K1 puts on
    sys.path reaches the caller; K1's component A, its blend and its submission.csv are not run. budget_s: run_b's deadline
    (K1 raises on a deadline hit and on more failed studies than its cap). probs: DataFrame indexed by StudyInstanceUID in
    test.csv's order, columns LABELS, values in [0, 1]; an unreadable study's row holds the column mean of the scored rows
    (K1's fill, never used: callers pass `unreadable` to the blend). unreadable: studies B could not score after K1's retry;
    raises if more than max_unreadable_frac of the studies. receipt["k1_solo_sha256"]: sha256 of the submission.csv K1 v2
    writes from this B (rank(B), W_A 0), None if a study is unreadable. Writes s6fall/s6fall_prob.csv and s6fall/receipt.json.
    """
    if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
        raise RuntimeError("run_s6fall_b runs only inside a Kaggle kernel (K1 reads /kaggle/input and writes /kaggle/working). "
                           "Launch it with the kaggle skill, attaching %s, %s and the competition." % (WEIGHTS_MOUNT, CODE_MOUNT))
    data = ctx.mounts[DATA_MOUNT]
    ctx.mounts[CODE_MOUNT]                                  # attached, else KeyError here
    check_weights(ctx.mounts[WEIGHTS_MOUNT])
    tmp = (Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")) / "k1b"
    if tmp.exists():
        raise RuntimeError("%s exists before S6fall B: a stale K1 scratch dir would mix runs" % tmp)
    k1 = materialize_k1(tmp / "src")
    out_json = tmp / "driver_out.json"
    env = dict(os.environ, K1_KAGGLE_ROOT="/kaggle", K1_TMP=str(tmp / "k1"))
    t = time.time()
    rc = subprocess.run([sys.executable, "-u", str(B_DRIVER), str(k1), "%.0f" % budget_s, str(out_json)], cwd=str(tmp),
                        env=env, timeout=budget_s + B_DRIVER_SLACK_S).returncode
    if rc != 0:
        raise RuntimeError("S6fall B (k1_driver) exited with %d; K1's reason is in the log above" % rc)
    wall = time.time() - t
    out = json.loads(out_json.read_text())

    ids = pd.read_csv(data / "test.csv", dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
    b = pd.read_csv(out["probs_csv"], dtype={"StudyInstanceUID": str})
    if sorted(b["StudyInstanceUID"]) != sorted(ids) or len(set(ids)) != len(ids):
        raise RuntimeError("S6fall B rows (%d) differ from test.csv's %d studies" % (len(b), len(ids)))
    probs = b.set_index("StudyInstanceUID").loc[ids, LABELS].astype(np.float64)
    unreadable = set(out["failed_after_retry"])
    nan_rows = set(probs.index[probs.isna().any(axis=1)])
    if nan_rows != unreadable:
        raise RuntimeError("S6fall B: NaN rows %s != failed studies %s" % (sorted(nan_rows)[:5], sorted(unreadable)[:5]))
    if len(unreadable) > max_unreadable_frac * len(ids):
        raise RuntimeError("S6fall B could not read %d of %d studies (> %.1f%%): %s" % (
            len(unreadable), len(ids), 100 * max_unreadable_frac, dict(list(out["failed_after_retry"].items())[:10])))
    ok = ~probs.index.isin(unreadable)
    v = probs.to_numpy()[ok]
    if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
        raise RuntimeError("S6fall B probabilities non-finite or outside [0, 1] on scored rows")
    if unreadable:
        probs = probs.fillna(probs[ok].mean())

    d = ctx.output_dir / "s6fall"
    d.mkdir(exist_ok=False)
    probs.to_csv(d / "s6fall_prob.csv", float_format="%.17g")
    timings = {"driver_steps": out["seconds"], "wall": wall, "since_t0": time.time() - t0}
    receipt = {k: out[k] for k in ("studies", "parts", "retry", "fail_cap", "probs_sha256", "k1_solo_sha256")}
    receipt.update(unreadable=sorted(unreadable), failed_reasons=out["failed_after_retry"], budget_s=budget_s, timings=timings)
    (d / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    shutil.rmtree(tmp)                                      # K1's scratch only; its outputs are under /kaggle/working/k1
    print("S6FALL_B_SUMMARY", json.dumps({k: receipt[k] for k in ("studies", "parts", "retry", "fail_cap", "k1_solo_sha256")}
                                         | {"unreadable": len(unreadable), "timings": timings}), flush=True)
    return probs, unreadable, timings, receipt


def wait_gpus_released(max_used_mib, timeout_s):
    """Print nvidia-smi, then wait until every GPU uses at most max_used_mib MiB; raises after timeout_s. Returns {index: MiB}."""
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout, flush=True)
    t = time.time()
    while True:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()
        used = {int(i): int(m) for i, m in (r.split(",") for r in rows)}
        if max(used.values()) <= max_used_mib:
            print("GPU_RELEASED", json.dumps({"used_mib": used, "waited_s": round(time.time() - t, 1)}), flush=True)
            return used
        if time.time() - t > timeout_s:
            raise RuntimeError("GPU memory not released after %ds: %s MiB used (limit %d)" % (timeout_s, used, max_used_mib))
        time.sleep(2)
