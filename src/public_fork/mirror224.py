"""Raptor 224 with the anatomical mirror TTA of A's Raptor 384 script (exp-0001, lead 18.10), as a switch around A's path.

mirror224_text(variant) builds the 224 script from the two vendored texts. Copied from the 384 text verbatim:
- SLOT_SIZES .. infer_pair_probs (_slot_index, _mirror_tri); one edit, infer_pair_probs' shape assert
  (94,3,320,320) -> (94,3,224,224), the 224 arm's windows;
- gpu_windows' signature with mirror=False and its two mirror lines right after the uint8 triplet gather (the same
  gather line in both scripts), so the mirror runs before crop / resize / normalisation as in 384;
- the consumer's TTA lines, which replace the 224 consumer's xla / else block.
Nothing else of the 224 script changes. "pair" = A's 384 TTA (probability mean of the plain and the mirrored pass);
"mirror_only" (diagnostic) scores the mirrored pass alone. Each variant's text is pinned by sha256.

installed(variant) makes stages.materialize write that text for raptor224_run.py while it is active, so A's own code
(pipeline.run_public -> stages.run_raptor, kernel_throughput.run_raptor_on) runs the mirror script and is otherwise
untouched. run_public_m224(ctx, t0, False) is pipeline.run_public itself (A's scored path, byte-identical to ffe2).
"""

import ast
import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.public_fork import pipeline, stages

SCRIPT = "raptor224_run.py"
BLOCK_START = "SLOT_SIZES = [26, 22, 18, 12, 18]\n"                       # 384: the mirror code, up to its eval_windows
BLOCK_END_384 = "def eval_windows(vol, mask, k, res, norm=NORM, mirror=False):\n"
EVAL_224 = "def eval_windows(vol, mask, k, res, norm=NORM):\n"            # 224: the block goes right before it
PAIR_ASSERT_384 = "    assert plain.shape==mirrored.shape==(94,3,320,320)\n"
PAIR_ASSERT_224 = "    assert plain.shape==mirrored.shape==(94,3,224,224)\n"
GPU_SIG_224 = "def gpu_windows(vol, mask, k, res, dev, norm=NORM):\n"
GPU_SIG_384 = "def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):\n"
GATHER = ("    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)"
          "   # (k,3,IMG,IMG) uint8\n")
MIRROR_LINES = "    if mirror:\n        tri = _mirror_tri(tri, cs, D)\n"
CONSUMER_224 = ("                    if xla is not None:\n"
                "                        with torch.no_grad():\n"
                "                            o = torch.sigmoid(model(xw.unsqueeze(0).to(d)).float()); xla.mark_step()\n"
                "                            arm_probs[a][i] = o[0].cpu().numpy()\n"
                "                    else:\n"
                "                        arm_probs[a][i] = infer_probs(model, xw, d)\n")
CONSUMER_384 = ("                    assert xla is None, \"Anatomical L/R TTA requires CUDA/fp16\"\n"
                "                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)\n"
                "                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)\n"
                "                    del xm\n")
PAIR_LINE = "                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)\n"
MIRROR_ONLY_LINE = "                    arm_probs[a][i] = infer_probs(model, xm, d)   # [mirror_only diagnostic]\n"
TEXT_SHA = {   # sha256 of mirror224_text(variant), pinned from the vendored texts (vendored/sha256.json)
    "pair": "efe551916eddd895f506ae43de19ea607027d76dddd7684dd05e3be0a8a09c61",
    "mirror_only": "46bdd6dc60d3ab079d82bb14499601f04f4021c226393963cc69bfda72514fa5",
}
MIRROR_DEFS = ("SLOT_SIZES", "SAGITTAL_SLOTS", "_slot_index", "_mirror_tri")
DIAG_RUNS = (("plain_cold", None), ("pair", "pair"), ("mirror_only", "mirror_only"), ("plain_warm", None))


def _vendored(name):
    """Text of a vendored file, sha256-checked against vendored/sha256.json."""
    raw = (stages.VENDORED / (name + ".txt")).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == stages._vendored_sha()[name], ("vendored file differs from sha256.json", name, digest)
    return raw.decode("utf-8")


def _once(text, part, what):
    if text.count(part) != 1:
        raise AssertionError("%s: expected exactly one occurrence, found %d" % (what, text.count(part)))


def mirror224_text(variant):
    """The 224 script with the 384 script's mirror TTA (see the module docstring); variant "pair" or "mirror_only"."""
    if variant not in TEXT_SHA:
        raise ValueError("variant must be one of %s, got %r" % (sorted(TEXT_SHA), variant))
    t224, t384 = _vendored("raptor224_run.py"), _vendored("raptor384_run.py")
    for part, what in ((BLOCK_START, "384 block start"), (BLOCK_END_384, "384 eval_windows"), (PAIR_ASSERT_384, "384 pair assert"),
                       (GPU_SIG_384, "384 gpu_windows"), (GATHER + MIRROR_LINES, "384 gather + mirror lines"),
                       (CONSUMER_384, "384 consumer")):
        _once(t384, part, what)
    for part, what in ((EVAL_224, "224 eval_windows"), (GPU_SIG_224, "224 gpu_windows"), (GATHER, "224 gather"),
                       (CONSUMER_224, "224 consumer"), (BLOCK_START, "224 has no mirror block")):
        if what == "224 has no mirror block":
            assert part not in t224, what
        else:
            _once(t224, part, what)
    i = t384.index(BLOCK_START)
    block = t384[i:t384.index(BLOCK_END_384, i)]
    _once(block, PAIR_ASSERT_384, "pair assert inside the block")
    block = block.replace(PAIR_ASSERT_384, PAIR_ASSERT_224)
    consumer = CONSUMER_384 if variant == "pair" else CONSUMER_384.replace(PAIR_LINE, MIRROR_ONLY_LINE)
    out = t224.replace(EVAL_224, block + EVAL_224)
    out = out.replace(GPU_SIG_224, GPU_SIG_384)
    out = out.replace(GATHER, GATHER + MIRROR_LINES)
    return out.replace(CONSUMER_224, consumer)


def materialize_mirror224(dest_dir, variant):
    """Write mirror224_text(variant) to dest_dir/raptor224_run.py, checked against its pinned sha256 -> path."""
    dest = Path(dest_dir) / SCRIPT
    dest.write_text(mirror224_text(variant), encoding="utf-8")
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert digest == TEXT_SHA[variant], ("mirror224 %s text sha256 differs from the pinned one" % variant, digest)
    return dest


@contextlib.contextmanager
def installed(variant):
    """While active, stages.materialize writes the mirror224 `variant` text for raptor224_run.py (any other file as before)."""
    orig = stages.materialize

    def materialize(name, dest_dir):
        return materialize_mirror224(dest_dir, variant) if name == SCRIPT else orig(name, dest_dir)

    stages.materialize = materialize
    try:
        yield
    finally:
        stages.materialize = orig


def run_public_m224(ctx, t0, mirror224):
    """pipeline.run_public with Raptor 224 under the 384 mirror TTA (mirror224 True); False = pipeline.run_public itself."""
    if not mirror224:
        return pipeline.run_public(ctx, t0)
    with installed("pair"):
        return pipeline.run_public(ctx, t0)


def _defs(text, names):
    """Namespace of the top-level assignments / functions `names` of a script text, executed with np and torch only."""
    import torch

    tree = ast.parse(text)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in keep}
    assert found == set(names), ("definitions missing", sorted(set(names) - found))
    ns = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<mirror_defs>", "exec"), ns)
    return ns


def unit_check():
    """Proof that the 224 mirror is the 384 code path -> dict; raises on any failure.

    Static: removing the inserted 384 text from mirror224_text("pair") gives back the vendored 224 text exactly, every
    inserted piece is a verbatim substring of the 384 text (the pair assert's one edit aside), and the plain 224 bytes
    stages.materialize writes outside installed() are the vendored ones. One window: _mirror_tri of the 224 text and of the
    384 text give identical tensors on a sagittal and on a coronal window, equal to the channel reversal / width flip.
    """
    import torch

    t224, t384 = _vendored("raptor224_run.py"), _vendored("raptor384_run.py")
    pair = mirror224_text("pair")
    i = t384.index(BLOCK_START)
    block = t384[i:t384.index(BLOCK_END_384, i)].replace(PAIR_ASSERT_384, PAIR_ASSERT_224)
    back = pair.replace(block, "").replace(GPU_SIG_384, GPU_SIG_224).replace(GATHER + MIRROR_LINES, GATHER)
    back = back.replace(CONSUMER_384, CONSUMER_224)
    assert back == t224, "mirror224 text minus the inserted 384 pieces != the vendored 224 text"
    for piece in (block.replace(PAIR_ASSERT_224, PAIR_ASSERT_384), GPU_SIG_384, GATHER + MIRROR_LINES, CONSUMER_384):
        assert piece in t384, "an inserted piece is not verbatim 384 text"
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / ("m224_check_%d" % os.getpid())
    tmp.mkdir(parents=True, exist_ok=True)
    plain_sha = hashlib.sha256(stages.materialize(SCRIPT, tmp).read_bytes()).hexdigest()
    assert plain_sha == stages._vendored_sha()[SCRIPT], "plain 224 bytes outside installed() are not the vendored ones"
    with installed("pair"):
        got = hashlib.sha256(stages.materialize(SCRIPT, tmp).read_bytes()).hexdigest()
    assert got == TEXT_SHA["pair"], "installed('pair') did not write the pinned mirror text"
    assert stages.materialize.__module__ == stages.__name__, "installed() did not restore stages.materialize"
    shutil.rmtree(tmp)

    n224, n384 = _defs(pair, MIRROR_DEFS), _defs(t384, MIRROR_DEFS)
    D = sum(n384["SLOT_SIZES"])
    rng = np.random.default_rng(0)
    vol = torch.from_numpy(rng.integers(0, 256, size=(D, 16, 16), dtype=np.uint8))
    windows = {}
    for plane, c in (("sagittal", 10), ("coronal", 60)):        # slot 0 (sagittal, slices 0-25) / slot 2 (coronal, 48-65)
        tri = vol[[c - 1, c, c + 1]][None]                        # (1, 3, H, W) uint8: one window
        a, b = n224["_mirror_tri"](tri, [c], D), n384["_mirror_tri"](tri, [c], D)
        assert torch.equal(a, b), "%s window: the 224 and 384 _mirror_tri differ" % plane
        want = tri.flip(1) if plane == "sagittal" else tri.flip(-1)
        assert torch.equal(a, want), "%s window: mirror is not the %s" % (plane, "channel reversal" if plane == "sagittal" else "width flip")
        windows[plane] = {"centre": c, "equal_224_384": True, "op": "channel reversal" if plane == "sagittal" else "width flip"}
    out = {"pair_sha256": TEXT_SHA["pair"], "plain_sha256": plain_sha, "roundtrip_to_vendored_224": True,
           "pieces_verbatim_384": True, "windows": windows}
    print("M224_UNIT_CHECK " + json.dumps(out), flush=True)
    return out


def diagnose(ctx, n_train, seed, scratch, timeout_s):
    """Validation diagnostics: Raptor 224 plain / pair / mirror_only on n_train cold non-gold train studies + the gold58.

    One fake test root (kernel_throughput.fake_competition_root) and, in this order, DIAG_RUNS: plain (cold page cache),
    pair, mirror_only, plain again (warm), each through kernel_throughput.run_raptor_on (the vendored script, or the
    mirror224 text under installed()). Each run's raptor224_raw.csv goes to <output_dir>/m224_raw_<run>.csv and the
    scripts' raptor224_{sub,raw}.csv are removed from /kaggle/working, so A's own run later cannot read them. Reports per
    target the Spearman of plain vs mirror_only (train studies), the gold58 AUC of plain / pair / mirror_only (official
    labels of train.csv), plain cold-vs-warm determinism and the pair - plain runtime (warm runs). Writes m224_diag.json
    (top level of output_dir: log_transport names have no folders).
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from src.kernel_throughput import fake_competition_root, run_raptor_on
    from src.knee_labels import LABELS

    data = ctx.mounts[pipeline.DATA_MOUNT]
    out = ctx.output_dir
    tr = pd.read_csv(data / "train.csv", dtype={"StudyInstanceUID": str})
    gold = tr[tr[LABELS].notna().all(axis=1)]
    if len(gold) != 58:
        raise RuntimeError("expected the 58 gold studies in train.csv, found %d" % len(gold))
    pool = sorted(set(tr["StudyInstanceUID"]) - set(gold["StudyInstanceUID"]))
    train_ids = [pool[i] for i in sorted(np.random.default_rng(seed).choice(len(pool), n_train, replace=False))]
    ids = train_ids + gold["StudyInstanceUID"].tolist()
    root = fake_competition_root(data, ids, Path(scratch) / "root")
    raw, seconds = {}, {}
    for tag, variant in DIAG_RUNS:
        ctxm = installed(variant) if variant else contextlib.nullcontext()
        with ctxm:
            seconds[tag] = run_raptor_on("224", root["root"], Path(scratch) / ("work_" + tag), ctx.output_dir, timeout_s)
        shutil.move(str(ctx.output_dir / "raptor224_raw.csv"), str(out / ("m224_raw_%s.csv" % tag)))
        (ctx.output_dir / "raptor224_sub.csv").unlink()
        df = pd.read_csv(out / ("m224_raw_%s.csv" % tag), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        if sorted(df.index) != sorted(ids) or df[LABELS].isna().to_numpy().any():
            raise RuntimeError("Raptor 224 %s: rows differ from the %d diagnostic studies or hold NaN" % (tag, len(ids)))
        failed = int((df[LABELS] == 0.5).all(axis=1).sum())
        if failed:
            raise RuntimeError("Raptor 224 %s left %d studies at 0.5 (its per-study FALLBACK)" % (tag, failed))
        raw[tag] = df.loc[ids, LABELS]
        print("M224_DIAG_RUN " + json.dumps({"run": tag, "seconds": round(seconds[tag], 1), "studies": len(ids)}), flush=True)

    p, m, q = raw["plain_cold"].loc[train_ids], raw["mirror_only"].loc[train_ids], raw["pair"].loc[train_ids]
    spear = {c: round(float(spearmanr(p[c], m[c])[0]), 4) for c in LABELS}
    pct = lambda d: d.rank(pct=True)  # noqa: E731
    rank_shift = {c: round(float((pct(q[c]) - pct(p[c])).abs().mean()), 4) for c in LABELS}
    g = gold.set_index("StudyInstanceUID")
    auc = {}
    for tag in ("plain_cold", "pair", "mirror_only"):
        per = {c: float(roc_auc_score(g[c].astype(int), raw[tag].loc[g.index, c])) for c in LABELS}
        auc[tag] = {"macro": round(float(np.mean(list(per.values()))), 4), "per_target": {c: round(v, 4) for c, v in per.items()}}
    det = float((raw["plain_cold"] - raw["plain_warm"]).abs().to_numpy().max())
    series = root["series"]
    extra_s_per_series = (seconds["pair"] - seconds["plain_warm"]) / series
    diag = {"n_train": n_train, "n_gold": len(gold), "seed": seed, "series": series, "files": root["files"],
            "seconds": {k: round(v, 1) for k, v in seconds.items()},
            "pair_minus_plain_warm_s_per_series": round(extra_s_per_series, 4),
            "pair_extra_h_1300": {"%.2f" % s: round(extra_s_per_series * 1300 * s / 3600, 3) for s in (5.0, 5.75)},
            "plain_cold_vs_warm_max_abs_diff": det, "spearman_plain_vs_mirror_train": spear,
            "spearman_mean": round(float(np.mean(list(spear.values()))), 4), "mean_abs_pct_shift_pair_vs_plain_train": rank_shift,
            "gold58_auc": auc, "gold58_note": "Raptor weights README: gold58 was not trained on but was reused for checkpoint "
                                              "selection, so it is not independent"}
    (out / "m224_diag.json").write_text(json.dumps(diag, indent=1))
    print("M224_DIAG " + json.dumps({k: diag[k] for k in ("seconds", "pair_extra_h_1300", "plain_cold_vs_warm_max_abs_diff",
                                                         "spearman_mean", "gold58_auc")}), flush=True)
    return diag
