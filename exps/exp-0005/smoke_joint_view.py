"""Synthetic smoke of src/joint_view.py (C1a, C1b, C2) and locate_all.run (C3); needs no real cache.

Tiny volumes in exp-0002's cache frame, a stub canon_px_to_lps with exp-0002's mapping and
meta keys (origin_lps + r*sy*row_dir + c*sx*col_dir + (pos_mm[k] - pos_mm[0])*normal_dir)
and a stub KneeVolCache. Checks ROIs, bilateral side picking, the per-plane crop spec (sagittal
375 px resampled to 300), crop shape/padding/flip, sagittal reversal (JointCache), degenerate modes, the ROI table round trip, then runs the whole C3 pass
on 2 workers into ctx.output_dir/c3 and checks JointCache over that run. CPU only, seconds:

    ~/conda/envs/knee/bin/python exps/exp-0005/smoke_joint_view.py
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.joint_view import (  # noqa: E402
    DEGENERATE_ERROR_CODE, JointCache, crop_padded, locate_knee, make_joint_crop_fn, resize_slices, roi_rows, rois_from_table,
    series_info, study_side)
from utils.ctx import get_ctx  # noqa: E402

SEED = 0
MM = 0.4
OUT_PX = 300
CROP_MM = {"Sagittal": 150.0, "Coronal": 120.0, "Axial": 120.0}
SRC_PX = {"Sagittal": 375, "Coronal": 300, "Axial": 300}
LOCATE = dict(mm_per_px=MM, bilateral_min_mm=50.0, degenerate_min_mm=30.0, open_px=5, min_area_frac=0.01)
# canonical LPS directions (row, col, normal) of exp-0002's cache frame
FRAME = {
    "Sagittal": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "Coronal": ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
    "Axial": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
}


def canon_stub(meta, pos_mm, k, r, c):
    """exp-0002's canonical pixel -> LPS mapping (integer k), for synthetic metas."""
    sy, sx = meta["spacing_rc"]
    return (np.asarray(meta["origin_lps"], float) + r * sy * np.asarray(meta["row_dir"], float)
            + c * sx * np.asarray(meta["col_dir"], float)
            + (float(pos_mm[k]) - float(pos_mm[0])) * np.asarray(meta["normal_dir"], float))


class FakeCache:
    """Stub of exp-0002's KneeVolCache(root): .series table, .load(series) -> (vol, pos_mm, meta), .meta, .pos_mm."""

    def __init__(self, root):
        self.root = Path(root)
        self.series = pd.read_parquet(self.root / "series.parquet")

    def load(self, series, mmap=True):
        row = self.series[self.series["series"] == series]
        if len(row) != 1:
            raise KeyError(series)
        meta = row.iloc[0].to_dict()
        vol = np.load(self.root / "vol" / meta["study"] / f"{series}.npy", mmap_mode="r" if mmap else None)
        return vol, np.load(self.root / "pos" / f"{series}.npy"), meta

    def meta(self, series):
        row = self.series[self.series["series"] == series]
        if len(row) != 1 or row["error_code"].notna().any():
            raise KeyError(f"no usable series {series}")
        return row.iloc[0].to_dict()

    def pos_mm(self, series):
        return np.load(self.root / "pos" / f"{series}.npy")


def blob_volume(h, w, blobs, z=8, seed=0):
    """uint8 [z,h,w]: ellipses (cy, cx, ry, rx) at ~150 over a dark noisy background."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.uint8)
    for cy, cx, ry, rx in blobs:
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 150, -1)
    vol = np.repeat(img[None], z, 0).astype(np.int16)
    vol += rng.integers(0, 20, vol.shape, dtype=np.int16)
    vol[:, img == 0] //= 4
    return vol.clip(0, 255).astype(np.uint8)


def make_meta(study, series, plane, w, lat_tag=None, lat_geom=None, bil=False):
    row_dir, col_dir, normal = FRAME[plane]
    origin = (-(w / 2) * MM, 0.0, 0.0) if plane != "Sagittal" else (0.0, 0.0, 0.0)
    return dict(study=study, series=series, plane_computed=plane, lat_tag=lat_tag, lat_geom=lat_geom,
                bilateral_suspect=bil, origin_lps=list(origin), row_dir=list(row_dir), col_dir=list(col_dir),
                normal_dir=list(normal), spacing_rc=[MM, MM])


def expect_raise(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


UNI = [(220, 180, 150, 120)]  # one knee, 96 mm wide
BIL = [(200, 250, 150, 130), (200, 650, 140, 110)]  # R knee (larger, x = -80 mm), L knee (x = +80 mm)
NARROW = [(200, 200, 150, 25)]  # 20 mm wide
WIDE_SAG = [(220, 250, 150, 200)]  # 160 mm AP: wider than the 150 mm sagittal crop


def check_units(out):
    pos = np.arange(8, dtype=np.float32) * 3.0 + 10.0
    loc = dict(crop_mm_by_plane=CROP_MM, canon_px_to_lps=canon_stub, **LOCATE)

    # C1a study_side
    m = lambda **kw: make_meta("S", kw.pop("se", "x"), "Coronal", 400, **kw)  # noqa: E731
    assert study_side([m(lat_tag="L"), m(se="y")]) == ("L", "tag")
    assert study_side([m(lat_tag="L"), m(se="y", lat_tag="R")]) == (None, "tag_conflict")
    assert study_side([m(lat_tag=float("nan"), lat_geom="R"), m(se="y", lat_tag=pd.NA)]) == ("R", "geom")
    assert study_side([m(lat_geom="L"), m(se="y", lat_geom="R")]) == (None, "geom_conflict")
    assert study_side([m(), m(se="y")]) == (None, "unknown")
    assert study_side([m(lat_tag="B"), m(se="y", lat_geom="L")]) == ("L", "geom")
    assert study_side([m(lat_tag="U"), m(se="y", lat_tag="R")]) == ("R", "tag")
    assert study_side([m(lat_tag="RIGHT"), m(se="y", lat_tag="R")]) == ("R", "tag")
    assert study_side([m(lat_tag="LEFT"), m(se="y", lat_geom="L")]) == ("L", "tag")
    assert study_side([m(lat_tag="LEFT"), m(se="y", lat_geom="R")]) == (None, "tag_geom_conflict")
    assert study_side([m(lat_tag="L", lat_geom="R")]) == (None, "tag_geom_conflict")
    assert series_info(m(lat_tag="RIGHT")).lat_tag == "R" and series_info(m(lat_tag="RIGHT")).lat_tag_raw == "RIGHT"
    assert series_info(m(lat_tag="B")).lat_tag_raw == "B" and series_info(m(lat_tag="B")).lat_tag is None
    assert series_info(m(bil=None)).bilateral_suspect is None
    expect_raise(ValueError, study_side, [m(lat_tag="X")])
    expect_raise(ValueError, study_side, [m(lat_geom="B")])
    expect_raise(KeyError, study_side, [{"study": "S"}])

    # C1b locate_knee: unilateral coronal -> bbox centre
    vol = blob_volume(400, 400, UNI)
    rois = locate_knee(vol, pos, make_meta("S", "c", "Coronal", 400), "L", **loc)
    assert len(rois) == 1 and rois[0]["method"] == "bbox" and rois[0]["n_components"] == 1, rois
    assert abs(rois[0]["cy"] - 220) <= 1 and abs(rois[0]["cx"] - 180) <= 1 and rois[0]["src_px"] == 300, rois

    # bilateral coronal and axial: side picking by LPS x
    for plane in ("Coronal", "Axial"):
        vol = blob_volume(400, 900, BIL)
        meta = make_meta("S", "b", plane, 900)
        r_l = locate_knee(vol, pos, meta, "L", **loc)
        r_r = locate_knee(vol, pos, meta, "R", **loc)
        r_n = locate_knee(vol, pos, meta, None, **loc)
        assert [r["method"] for r in r_l] == ["bilateral_side"] and abs(r_l[0]["cx"] - 650) <= 1, r_l
        assert [r["method"] for r in r_r] == ["bilateral_side"] and abs(r_r[0]["cx"] - 250) <= 1, r_r
        assert [r["method"] for r in r_n] == ["bilateral_both"] * 2, r_n
        assert abs(r_n[0]["cx"] - 250) <= 1 and abs(r_n[1]["cx"] - 650) <= 1, r_n  # larger first
        assert r_n[0]["area_px"] > r_n[1]["area_px"]

    # sagittal with two disjoint wide blobs: no bilateral logic, union bbox
    vol = blob_volume(400, 900, BIL)
    r_s = locate_knee(vol, pos, make_meta("S", "s", "Sagittal", 900), "R", **loc)
    assert [r["method"] for r in r_s] == ["bbox"] and r_s[0]["n_components"] == 2, r_s
    assert r_s[0]["c0"] <= 120 and r_s[0]["c1"] >= 760 and r_s[0]["src_px"] == 375, r_s

    # degenerate: empty and narrow, all three modes
    for vol in (np.zeros((8, 400, 400), np.uint8), blob_volume(400, 400, NARROW)):
        meta = make_meta("S", "d", "Coronal", 400)
        msg = expect_raise(ValueError, locate_knee, vol, pos, meta, "L", **loc)
        assert "degenerate" in msg, msg
        rep = locate_knee(vol, pos, meta, "L", on_degenerate="report", **loc)
        assert len(rep) == 1 and rep[0]["method"] == "degenerate" and rep[0]["cy"] is None, rep
        ic = locate_knee(vol, pos, meta, "L", on_degenerate="image_centre", **loc)
        assert ic[0]["method"] == "image_centre" and (ic[0]["cy"], ic[0]["cx"]) == (199, 199), ic
    expect_raise(ValueError, locate_knee, vol, pos, meta, "L", on_degenerate="centre", **loc)

    # C2 crop_fn: shape/dtype, padding at FOV edges, column flip only for R coronal/axial
    rng = np.random.default_rng(SEED)
    sl = rng.integers(1, 256, (5, 400, 900), dtype=np.uint8)
    edge = dict(cy=50, cx=880, src_px=300, method="bbox", n_components=1, side=None, area_px=1,
                vol_h=400, vol_w=900, r0=0, r1=100, c0=860, c1=900)
    for plane in ("Sagittal", "Coronal", "Axial"):
        crops = {}
        for side in ("L", "R", None):
            fn = make_joint_crop_fn({"e": [dict(edge, side=side, src_px=SRC_PX[plane])]}, {"S": side},
                                    OUT_PX, CROP_MM, MM, 20)
            crops[side] = fn(sl, make_meta("S", "e", plane, 900), False, None)
            assert crops[side].dtype == np.uint8 and crops[side].shape == (5, OUT_PX, OUT_PX)
        src = crop_padded(sl, 50, 880, SRC_PX[plane])
        want = src if plane != "Sagittal" else np.stack(
            [cv2.resize(x, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA) for x in src])
        assert (crops["L"] == want).all(), plane  # sagittal: 375 px (150 mm) area-resampled to 300
        assert (crops["L"] == crops[None]).all()
        flipped = (crops["R"] == crops["L"][:, :, ::-1]).all()
        assert flipped == (plane != "Sagittal"), plane
        if plane == "Sagittal":
            assert (crops["R"] == crops["L"]).all()
    c = crops["L"]  # rows [-100, 200), cols [730, 1030) of a 400 x 900 image
    assert (c[:, :100] == 0).all() and (c[:, :, 170:] == 0).all()
    assert (c[:, 100:, :170] == sl[:, :200, 730:]).all()
    assert c[:, 100:, :170].min() > 0

    # bilateral_both: eval -> rois[0]; train -> both, jitter within +-20 px
    both = [dict(edge, cy=200, cx=250, method="bilateral_both"), dict(edge, cy=200, cx=650, method="bilateral_both")]
    fn = make_joint_crop_fn({"b": both}, {"S": None}, OUT_PX, CROP_MM, MM, 20)
    meta = make_meta("S", "b", "Coronal", 900)
    assert (fn(sl, meta, False, None) == crop_padded(sl, 200, 250, OUT_PX)).all()
    seen = set()
    for _ in range(40):
        got = fn(sl[:1], meta, True, rng)
        # no padding within +-20 px of these centres: crop == sl[rows 50+dy.., cols cx-150+dx..]
        hit = [(cx, dy, dx) for cx in (250, 650) for dy in range(-20, 21) for dx in range(-20, 21)
               if (got[0, :8, :8] == sl[0, 50 + dy:58 + dy, cx - 150 + dx:cx - 142 + dx]).all()]
        assert len(hit) == 1, hit
        cx, dy, dx = hit[0]
        assert (got == crop_padded(sl[:1], 200 + dy, cx + dx, OUT_PX)).all()
        seen.add(cx)
    assert seen == {250, 650}, seen
    expect_raise(KeyError, fn, sl, make_meta("S", "zzz", "Coronal", 900), False, None)
    expect_raise(ValueError, fn, sl[:, :300], meta, False, None)
    expect_raise(ValueError, make_joint_crop_fn, {"d": [dict(edge, method="degenerate")]}, {}, OUT_PX, CROP_MM, MM, 0)
    expect_raise(ValueError, make_joint_crop_fn, {"e": [dict(edge, src_px=256)]}, {}, OUT_PX, CROP_MM, MM, 0)
    expect_raise(ValueError, make_joint_crop_fn, {"e": [edge]}, {}, OUT_PX, {"Sagittal": 150.0}, MM, 0)
    # a 300 px ROI on a sagittal series (spec 375) raises at crop time
    fn = make_joint_crop_fn({"e": [edge]}, {"S": "L"}, OUT_PX, CROP_MM, MM, 0)
    msg = expect_raise(ValueError, fn, sl, make_meta("S", "e", "Sagittal", 900), False, None)
    assert "src_px" in msg, msg
    # sagittal train jitter is in source pixels: equals the resampled 375 crop at the shifted centre
    sag = dict(edge, cy=200, cx=450, src_px=375)
    fn = make_joint_crop_fn({"e": [sag]}, {"S": "R"}, OUT_PX, CROP_MM, MM, 20)
    jr = np.random.default_rng(1)
    dy, dx = np.random.default_rng(1).integers(-20, 21, size=2)
    got = fn(sl, make_meta("S", "e", "Sagittal", 900), True, jr)
    assert (got == resize_slices(crop_padded(sl, 200 + int(dy), 450 + int(dx), 375), OUT_PX)).all()

    # ROI table round trip (parquet)
    rows = roi_rows(series_info(meta), "unknown", both) + roi_rows(
        series_info(make_meta("T", "e", "Axial", 900)), "tag", [dict(edge, side="R")])
    path = out / "roundtrip.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    rois_rt, sides_rt = rois_from_table(pd.read_parquet(path))
    assert rois_rt == {"b": both, "e": [dict(edge, side="R")]}, rois_rt
    assert sides_rt == {"S": None, "T": "R"}, sides_rt
    print("[smoke] unit checks passed", flush=True)


def build_fake_cache(root):
    """6 studies / 13 series covering every method, side source and disagreement case."""
    spec = [  # study, series, plane, blobs, width, lat_tag, lat_geom, bilateral_suspect
        ("S1", "S1-cor", "Coronal", UNI, 400, "L", None, True),
        ("S1", "S1-sag", "Sagittal", UNI, 400, None, None, False),
        ("S1", "S1-ax", "Axial", UNI, 400, None, "L", False),
        ("S2", "S2-cor", "Coronal", BIL, 900, None, "R", True),
        ("S2", "S2-sag", "Sagittal", UNI, 400, None, "R", False),
        ("S2", "S2-ax", "Axial", BIL, 900, None, "R", False),
        ("S3", "S3-cor", "Coronal", BIL, 900, "B", None, True),
        ("S3", "S3-sag", "Sagittal", WIDE_SAG, 500, None, None, False),
        ("S4", "S4-cor", "Coronal", UNI, 400, "L", None, None),
        ("S4", "S4-ax", "Axial", UNI, 400, "R", None, False),
        ("S5", "S5-cor", "Coronal", [], 400, None, "L", None),
        ("S5", "S5-sag", "Sagittal", NARROW, 400, None, "R", False),
        ("S6", "S6-cor", "Coronal", UNI, 400, "L", "R", False),
    ]
    metas = []
    for j, (study, series, plane, blobs, w, tag, geom, bil) in enumerate(spec):
        vol = blob_volume(400, w, blobs, seed=j) if blobs else np.zeros((8, 400, w), np.uint8)
        (root / "vol" / study).mkdir(parents=True, exist_ok=True)
        (root / "pos").mkdir(exist_ok=True)
        np.save(root / "vol" / study / f"{series}.npy", vol)
        np.save(root / "pos" / f"{series}.npy", np.arange(len(vol), dtype=np.float32) * 3.0)
        metas.append({**make_meta(study, series, plane, w, tag, geom, bil), "Z": len(vol),
                      "error_code": None, "error_detail": None, "Anatomical_Plane": plane, "Fat_Suppression": 1})
    # exp-0002 keeps a failed series as an error row with no meta: C3 must skip and count it
    metas.append({"study": "S1", "series": "S1-failed", "error_code": "flat_intensity", "error_detail": "x",
                  "Anatomical_Plane": "Coronal", "Fat_Suppression": 1})
    pd.DataFrame(metas).to_parquet(root / "series.parquet", index=False)


def check_c3(out):
    import locate_all

    root, c3 = out / "fake_cache", out / "c3"
    build_fake_cache(root)
    c3.mkdir()
    locate_all.run(root, c3, 2, FakeCache, canon_stub)
    st = json.loads((c3 / "stats.json").read_text())
    assert st["n_series_failed_in_cache"] == 1 and st["failed_in_cache_by_code"] == {"flat_intensity": 1}, st
    assert "crop_px" not in st and (st["crop_mm_by_plane"], st["mm_per_px"], st["out_px"]) == (CROP_MM, MM, OUT_PX), st
    assert st["wider_than_crop"] == {"Sagittal": {"n": 1, "n_located": 3, "series": ["S3-sag"]},
                                     "Coronal": {"n": 0, "n_located": 5, "series": []},
                                     "Axial": {"n": 0, "n_located": 3, "series": []}}, st["wider_than_crop"]
    nz = {p: {k: v for k, v in c.items() if v} for p, c in st["series_by_plane_method"].items()}
    assert nz == {"Sagittal": {"bbox": 3, "degenerate": 1},
                  "Coronal": {"bbox": 3, "bilateral_side": 1, "bilateral_both": 1, "degenerate": 1},
                  "Axial": {"bbox": 2, "bilateral_side": 1}}, nz
    # S6: tag L but geom R -> unknown side, no mirroring (lead, 17.10)
    assert st["studies_by_side"] == {"L_kept": 1, "R_mirrored": 1, "unknown": 4}, st["studies_by_side"]
    assert st["studies_by_side_source"] == {"tag": 1, "geom": 1, "tag_conflict": 1, "tag_geom_conflict": 1,
                                            "geom_conflict": 1, "unknown": 1}, st["studies_by_side_source"]
    lat = st["lat_tag_vs_geom"]
    assert lat["per_series"]["n_disagree"] == 1 and lat["per_series"]["study_ids"] == ["S6"], lat
    assert lat["per_study"]["study_ids"] == ["S6"] and lat["per_study"]["n_both"] == 2, lat
    bil = st["bilateral_vs_exp0002_suspect"]
    assert {k: bil["Coronal"][k] for k in ("both", "ours_only", "theirs_only", "neither", "theirs_missing")} == \
        {"both": 2, "ours_only": 0, "theirs_only": 1, "neither": 1, "theirs_missing": 1}, bil
    assert st["lat_tag_without_side"] == {"n_series": 1, "by_value": {"B": 1}, "study_ids": ["S3"]}, st
    assert bil["Axial"]["ours_only"] == 1 and bil["Axial"]["neither"] == 2 and bil["n_degenerate_excluded"] == 1, bil
    canon = st["series_by_plane_canonicalisation"]
    assert canon["Sagittal"] == {"kept_L": 1, "mirrored_R": 1, "unknown_side": 2}, canon

    table = pd.read_parquet(c3 / "knee_rois.parquet")
    all_series = set(pd.read_parquet(root / "series.parquet")["series"])
    assert len(table) == 12 and set(table["series"]) == all_series - {"S5-cor", "S5-sag", "S1-failed"}, table
    assert (table["src_px"] == table["plane"].map(SRC_PX)).all(), table[["plane", "src_px"]]
    rois, sides = rois_from_table(table)
    assert abs(rois["S2-cor"][0]["cx"] - 250) <= 1 and sides["S2"] == "R"
    assert [r["method"] for r in rois["S3-cor"]] == ["bilateral_both"] * 2 and sides["S3"] is None
    assert len(pd.read_csv(c3 / "degenerate.csv")) == 2

    # C2 JointCache over the fake cache and this run: degenerate series -> error rows, sagittal R reversed
    cache = FakeCache(root)
    deg = set(pd.read_csv(c3 / "degenerate.csv")["series"])
    jc = JointCache(cache, sides, deg)
    js = jc.series
    assert list(js["series"]) == list(cache.series["series"]) and cache.series["error_code"].notna().sum() == 1
    assert set(js.series[js.error_code == DEGENERATE_ERROR_CODE]) == {"S5-cor", "S5-sag"}, js[["series", "error_code"]]
    assert set(js.series[js.error_code == "flat_intensity"]) == {"S1-failed"}
    expect_raise(ValueError, jc.load, "S5-cor")
    v, p, _ = jc.load("S2-sag")  # study S2: side R (geom) -> reversed, pos_mm negated and still increasing
    v0, p0, _ = cache.load("S2-sag")
    assert (np.asarray(v) == np.asarray(v0)[::-1]).all() and np.allclose(p, -p0[::-1]) and (np.diff(p) > 0).all()
    assert not (np.asarray(v) == np.asarray(v0)).all()
    for s in ("S2-cor", "S2-ax", "S1-sag", "S3-sag"):  # R coronal/axial (flipped in the crop), L / unknown sagittal
        v, p, _ = jc.load(s)
        v0, p0, _ = cache.load(s)
        assert (np.asarray(v) == np.asarray(v0)).all() and (p == p0).all(), s
    assert (jc.pos_mm("S2-sag") == cache.pos_mm("S2-sag")).all()  # .pos_mm keeps the original order
    assert jc.meta("S2-sag").keys() == cache.meta("S2-sag").keys() and jc.meta("S2-sag")["series"] == "S2-sag"
    assert not any(hasattr(jc, a) for a in ("slices", "dropped", "root")), "JointCache must not pass through"
    expect_raise(KeyError, JointCache, cache, sides, deg | {"nope"})
    expect_raise(ValueError, JointCache, cache, sides, deg | {"S1-failed"})
    expect_raise(KeyError, JointCache, cache, {k: v for k, v in sides.items() if k != "S1"}, deg)
    import src.reader25d as r25

    hi, r25.DEPTH_HI = r25.DEPTH_HI, 0.9  # an asymmetric depth range must break JointCache loudly
    expect_raise(ValueError, JointCache, cache, sides, deg)
    r25.DEPTH_HI = hi
    pages = st["montages"]
    assert all((c3 / p).is_file() for p in pages), pages
    idx = pd.read_csv(c3 / "montages" / "index.csv")
    assert set(idx["group"]) == {"random", "bilateral", "degenerate", "wider_than_crop", "exp0002_suspect_only"} \
        and len(idx) == 13 + 3 + 2 + 1 + 1, idx  # S1-cor: suspect for exp-0002, bbox for us
    print(f"[smoke] C3 pass checks passed; montages: {pages}", flush=True)


def main():
    ctx = get_ctx()
    check_units(ctx.output_dir)
    check_c3(ctx.output_dir)
    print(f"[smoke] all checks passed | {ctx.output_dir}")


if __name__ == "__main__":
    main()
