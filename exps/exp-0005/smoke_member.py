"""Synthetic CPU smoke of C5 (src/joint_reader.py): member_rois vs the C3 pass, predict_member end-to-end.

On smoke_joint_view's fake cache: locate_all.run (C3) -> member_rois must give the same ROIs and
sides for every located series and the image centre for the C3-degenerate ones. Then predict_member
on CPU (bf16 autocast: MEMBER_AMP patched for the smoke) with the GPU smoke's fold0.pt over a fake
cache of 2 studies (L tag; R geom with bilateral coronal/axial and a reversed sagittal), compared
with predict_studies over the C3 run's ROIs: identical probabilities. First of all, infer_check.py's
path on CPU: the competition's 3 example test studies from raw DICOM (build_dir_cache) ->
predict_member -> infer_check.format_checks. It runs first because build_dir_cache's fork-pool
initializer (cv2.setNumThreads) hangs once this process has run threaded cv2 work (member_rois, C3):
the blend kernel must build its cache before any member runs. exp-0004's src/infer25d.py is not on
this branch yet: it is loaded from INFER25D_REV's blob (git, read-only).

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0005/smoke_member.py
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import locate_all  # noqa: E402
import smoke_joint_view as sj  # noqa: E402
import src.joint_reader as jr  # noqa: E402
from src.joint_view import JointCache, load_roi_run  # noqa: E402
from src.knee_labels import LABELS  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

INFER25D_REV = "0fe73fb"  # exp-0004: predict_studies accepts torch.float32
SMOKE_RUN = "20261017t125202-486-6244"  # GPU smoke of train.py: fold0.pt
CPU_THREADS = 8  # shared machine
PREDICT_STUDIES = ["S1", "S2"]
COMP = "rsna-knee-abnormality-detection"
CACHE_WORKERS = 4


def load_infer25d(repo):
    """sys.modules['src.infer25d'] from the INFER25D_REV blob; raises once the module is on this branch."""
    if (repo / "src" / "infer25d.py").exists():
        raise RuntimeError("src/infer25d.py is on this branch now: import it instead of the blob")
    code = subprocess.run(["git", "-C", str(repo), "show", f"{INFER25D_REV}:src/infer25d.py"], check=True,
                          capture_output=True, text=True).stdout
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("src.infer25d", loader=None))
    mod.__file__ = f"git:{INFER25D_REV}:src/infer25d.py"
    exec(compile(code, mod.__file__, "exec"), mod.__dict__)
    sys.modules["src.infer25d"] = mod
    return mod


def comp_series_of(cache):
    """The competition *_series.csv a cache was built from (cache_study_table checks it matches)."""
    cs = cache.series
    return pd.DataFrame({"StudyInstanceUID": cs.study, "SeriesInstanceUID": cs.series,
                         "Fat_Suppression": cs.Fat_Suppression, "Anatomical_Plane": cs.Anatomical_Plane})


def load_c3(c3):
    return load_roi_run(c3, cache_key=locate_all.CACHE_KEY, crop_mm_by_plane=jr.CROP_MM_BY_PLANE,
                        mm_per_px=jr.MM_PER_PX, out_px=jr.OUT_PX, locate=jr.LOCATE)


def check_member_rois(out):
    """member_rois on the whole fake cache == the C3 run's ROIs and sides; degenerate -> image centre."""
    root, c3 = out / "fake_cache", out / "c3"
    sj.build_fake_cache(root)
    c3.mkdir()
    locate_all.run(root, c3, 2, sj.FakeCache, sj.canon_stub)
    ref, ref_sides, degenerate = load_c3(c3)
    st = json.loads((c3 / "stats.json").read_text())
    cache = sj.FakeCache(root)
    studies = sorted(set(cache.series.study))
    rois, sides, counts = jr.member_rois(studies, cache, sj.canon_stub)
    assert set(rois) == set(ref) | degenerate and "S1-failed" not in rois, sorted(rois)
    assert all(rois[s] == ref[s] for s in ref), [s for s in ref if rois[s] != ref[s]]
    assert degenerate == {"S5-cor", "S5-sag"} and all([r["method"] for r in rois[s]] == ["image_centre"]
                                                      for s in degenerate), {s: rois[s] for s in degenerate}
    for s in degenerate:  # image centre of the stored volume, no foreground bbox
        r = rois[s][0]
        assert (r["cy"], r["cx"]) == ((r["vol_h"] - 1) // 2, (r["vol_w"] - 1) // 2) and r["r0"] is None, r
    assert {k: v for k, v in sides.items() if k in ref_sides} == ref_sides and set(sides) == set(studies), sides
    assert counts["studies_by_side"] == st["studies_by_side"], counts
    assert counts["studies_by_side_source"] == st["studies_by_side_source"], counts
    want = {p: {**m, "degenerate": 0, "image_centre": m["degenerate"]} for p, m in st["series_by_plane_method"].items()}
    assert counts["series_by_plane_method"] == want, (counts["series_by_plane_method"], want)
    assert (counts["n_studies"], counts["n_series"], counts["studies_without_usable_series"]) == (6, 13, 0), counts
    sj.expect_raise(ValueError, jr.member_rois, ["S1", "S1"], cache, sj.canon_stub)
    sj.expect_raise(KeyError, jr.member_rois, ["S1", "nope"], cache, sj.canon_stub)
    print(f"[smoke] member_rois == C3 on the fake cache; counts {json.dumps(counts)}", flush=True)
    return root, c3, counts


def check_predict_member(out, root, c3, ckpt, infer):
    """predict_member on CPU over a 2-study fake cache == predict_studies over the C3 ROIs; contract checks."""
    full = sj.FakeCache(root)
    sub_root = out / "fake_cache_2"
    sj.build_fake_cache(sub_root)
    tab = pd.read_parquet(sub_root / "series.parquet")
    tab[tab.study.isin(PREDICT_STUDIES)].to_parquet(sub_root / "series.parquet", index=False)
    cache = sj.FakeCache(sub_root)
    comp_series = comp_series_of(cache)
    jr.MEMBER_AMP = torch.bfloat16  # T4 runs fp16; the CPU smoke runs bf16 (for the rest of this process)
    got = jr.predict_member(PREDICT_STUDIES, cache, comp_series, [ckpt], "cpu")
    # every study of the cache with a usable series must be predicted (JointCache needs its side)
    sj.expect_raise(KeyError, jr.predict_member, PREDICT_STUDIES, full, comp_series_of(full), [ckpt], "cpu")
    k_eval, jr.MEMBER_K_EVAL = jr.MEMBER_K_EVAL, 8  # checkpoints scored with another k_eval -> refused
    sj.expect_raise(ValueError, jr.predict_member, PREDICT_STUDIES, cache, comp_series, [ckpt], "cpu")
    jr.MEMBER_K_EVAL = k_eval
    assert list(got.index) == PREDICT_STUDIES and list(got.columns) == LABELS, got
    p = got.to_numpy()
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), got

    # the training eval path over the C3 run's ROIs (S1/S2 have no degenerate series): same crops, same numbers
    ref, ref_sides, _ = load_c3(c3)
    keep = set(cache.series.series)
    rois = {s: r for s, r in ref.items() if s in keep}
    sides = {s: v for s, v in ref_sides.items() if s in PREDICT_STUDIES}
    crop_fn = jr.make_joint_crop_fn(rois, sides, jr.OUT_PX, jr.CROP_MM_BY_PLANE, jr.MM_PER_PX, jitter_px=0)
    models = infer.load_models([ckpt], "cpu")
    want = infer.predict_studies(PREDICT_STUDIES, JointCache(cache, sides, set()), comp_series, models, crop_fn,
                                 jr.MEMBER_K_EVAL, jr.MEMBER_BS, jr.MEMBER_WORKERS, "cpu", torch.bfloat16)
    d = float(np.abs(got.to_numpy() - want.to_numpy()).max())
    assert d == 0.0, d
    print(f"[smoke] predict_member on CPU (bf16) == predict_studies over the C3 ROIs; p range "
          f"[{p.min():.3f}, {p.max():.3f}]", flush=True)
    return {"studies": PREDICT_STUDIES, "max_abs_dp_vs_c3_path": d, "p_min": float(p.min()), "p_max": float(p.max()),
            "sides": sides}


def check_test_examples(ctx, ckpt):
    """infer_check.py's path on CPU: example test studies, raw DICOM -> KneeVolCache -> predict_member -> format."""
    import infer_check  # imports src.infer25d: after load_infer25d

    comp = ctx.mounts[COMP]
    studies = pd.read_csv(comp / "test.csv").StudyInstanceUID.tolist()
    comp_series = pd.read_csv(comp / "test_series.csv")
    cache = infer_check.build_dir_cache(comp / "test_series", comp / "test_series.csv", ctx.output_dir / "test_cache",
                                        CACHE_WORKERS)
    pred = jr.predict_member(studies, cache, comp_series, [ckpt], "cpu")  # MEMBER_AMP: bf16 (set above)
    checks = infer_check.format_checks(pred, studies, pd.read_csv(comp / "sample_submission.csv", nrows=0).columns)
    assert all(checks.values()), checks
    print(f"[smoke] example test studies: {checks}\n{pred.round(3).to_string()}", flush=True)
    return {"checks": checks, "n_series": len(comp_series), "pred": pred.round(4).to_dict("index")}


def main():
    ctx = get_ctx()
    torch.set_num_threads(CPU_THREADS)
    repo = Path(__file__).resolve().parents[2]
    infer = load_infer25d(repo)
    if torch.float32 not in infer.AMP_DTYPES:
        raise AssertionError(f"{INFER25D_REV}: predict_studies does not accept torch.float32")
    ckpt = ctx.mounts[SMOKE_RUN] / "fold0.pt"
    test_examples = check_test_examples(ctx, ckpt)  # before any cv2 work in this process, see the docstring
    root, c3, counts = check_member_rois(ctx.output_dir)
    res = {"infer25d": infer.__file__, "member_rois_counts": counts, "test_examples": test_examples,
           "predict_member": check_predict_member(ctx.output_dir, root, c3, ckpt, infer)}
    (ctx.output_dir / "smoke_member.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[smoke] C5 checks passed | {ctx.output_dir}", flush=True)


if __name__ == "__main__":
    main()
