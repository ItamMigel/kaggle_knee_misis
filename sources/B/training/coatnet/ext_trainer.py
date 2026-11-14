#!/usr/bin/env python3
"""ext_trainer.py - run a train_knee trainer UNCHANGED with an external corpus part (OAI) appended to the TRAINING set only.

Production version of research/rnd5/external-data/ext_trainer.py (2026-09-29, oai/trainer-integration). The trainer file is
loaded by path and only four module globals are patched; nothing in it is edited.

  python ext_trainer.py --trainer <train_knee*.py> --out <dir> [--ext_dir <dir> --ext_weight 1.0 ...] <trainer args ...>

FLAGS (all OFF by default; with no --ext_dir the trainer runs bit-identically, only its output dir moves to --out):
  --trainer PATH      trainer to wrap (default research/bo/pb12/train_knee_r2.py). Keep it on the command line: the laptop
                      queues wait on `pgrep -f train_knee`, and the path is what makes this process match.
  --out DIR           where raptor_ft_/raptor_gold_/raptor_oof_* go (the trainer's HERE). Default runs/<tag> next to this file.
  --ext_dir DIR       external part: ext_ids.npy (N,) str | ext_vols.npy (N,96,384,384) u8 | ext_masks.npy (N,96) u8 |
                      ext_labels.parquet (StudyInstanceUID + finding columns, 0..1 or NaN) | ext_lat.parquet (StudyInstanceUID,
                      is_left; REQUIRED with --lat_norm). Build geometry exactly like cache/build_corpus96.py.
  --ext_weight W      per-source loss weight: every labelled external cell is weighted W (RSNA cells 1). Default 1.0.
  --ext_findings L    comma list of findings the external labels may supervise; every other external cell is forced to NaN
                      (masked). Default "PF OA,Lateral OA,Synovitis" (the pre-registered MOAKS cuts). Effusion, Contusion,
                      Fracture and MCL can never be enabled (pre-registered NaN).
  --ext_max N         use only the first N labelled external studies (ext_ids.npy order); 0 = all.
  --ext_allow_empty   accept a part that trains 0 external rows or leaves an allow-listed finding without a single labelled
                      external cell (tests only; without it such a run is refused: e.g. slug-named label columns 'PF_OA' are
                      ignored as unknown and would silently turn the arm into a copy of the control).
  --ext_pin_pw auto|off  auto (default): when external rows are trained and the command has no --pw_labels, pos_weight is
                      computed from the RSNA label file over the RSNA fold-train studies only (via the trainer's own --pw_labels),
                      so adding OAI rows does not move the RSNA cells' loss weighting. It uses the trainer's --pw_labels rows
                      (fold-train BEFORE --train_subset); without a subset that is the control's set exactly, and the float64
                      vs float32 mean differs from the control's pos_weight by <= 1 float32 ulp (tests/IDENTITY.txt, C_D).
  --threads N         torch.set_num_threads(N) before training (tests; production leaves it unset).

GUARANTEES (checked, the run refuses otherwise):
  - external ids: unique, disjoint from the corpus ids, from train.csv (gold), and absent from --fold_file (so the fold hook
    `fmap.get(u, -1) != fold` keeps every one of them in the fold-train set: never OOF, never gold);
  - after training, every raptor_oof_<tag>*.npz and raptor_gold_<tag>*.npz in --out is re-opened and must hold no external id;
  - NaN cells: the trainer gives them weight 0 (its CELL path); non-allow-listed findings are NaN before it sees them;
  - --limit / --smoke are refused with --ext_dir (both truncate train_ids and would silently drop the external rows);
    --train_subset is extended with the external ids (the subset is an RSNA-side selection);
  - with --lat_norm every trained external id needs a laterality row (the trainer's default is 'right knee', silently);
  - ext_lat.parquet (whenever present): unique ids, is_left bool (or 0/1, never NaN/strings: astype(bool) maps NaN and
    'False' to True), and it must agree with the side in ids of the form OAI_<7 digits>_<R|L>_...;
  - --roi is not supported with --ext_dir (roi_boxes.npz has no external rows; the trainer itself stops on the missing ids).
A manifest (args before/after rewriting, md5s, counts, guard results, and the part's file fingerprint ext.files, which the
member chain's G0 compares against the part it is about to train) is written to <out>/ext_manifest_<tag>.json.
"""
import os, re, sys, json, time, hashlib, argparse, importlib.util

sys.dont_write_bytecode = True          # never drop __pycache__ next to the wrapped trainer
import numpy as np

R = "/home/danial/rsna-knee"
HERE_SELF = os.path.dirname(os.path.abspath(__file__))
DEF_TRAINER = f"{R}/research/bo/pb12/train_knee_r2.py"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
DEFAULT_FINDINGS = ("PF OA", "Lateral OA", "Synovitis")
NEVER = ("Effusion", "Contusion", "Fracture", "MCL")    # pre-registered NaN for OAI (2026-09-29 plan)
D_SLICES, IMG = 96, 384


def _md5(p, cap=None):
    h = hashlib.md5()
    with open(p, "rb") as f:
        n = 0
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b); n += len(b)
            if cap and n >= cap: break
    return h.hexdigest()


class _Chain:
    """Row-indexable view over several (N_i, D, H, W) arrays; StudyWindows only uses .shape, len() and [int]."""
    def __init__(self, *parts):
        self.parts = parts
        self.offs = np.cumsum([0] + [p.shape[0] for p in parts])
        self.shape = (int(self.offs[-1]),) + tuple(parts[0].shape[1:])
        for p in parts[1:]:
            if tuple(p.shape[1:]) != self.shape[1:]:
                raise SystemExit(f"[ext] external part shape {p.shape} != corpus {self.shape}")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        row = int(row)
        if row < 0 or row >= self.shape[0]:
            raise IndexError(row)
        k = int(np.searchsorted(self.offs, row, side="right") - 1)
        return self.parts[k][row - self.offs[k]]


# ---------------------------------------------------------------- argv helpers (trainer args are read, never re-parsed by it)
def _get(argv, flag):
    """Value of --flag in argv (last occurrence; '--flag v' or '--flag=v'), or None."""
    v = None
    for i, t in enumerate(argv):
        if t == flag and i + 1 < len(argv): v = argv[i + 1]
        elif t.startswith(flag + "="): v = t.split("=", 1)[1]
    return v


def _has(argv, flag):
    return any(t == flag or t.startswith(flag + "=") for t in argv)


def _set(argv, flag, value):
    """Replace every occurrence of --flag's value; append '--flag value' if absent."""
    out, i, hit = [], 0, False
    while i < len(argv):
        t = argv[i]
        if t == flag and i + 1 < len(argv):
            out += [flag, str(value)]; i += 2; hit = True; continue
        if t.startswith(flag + "="):
            out += [f"{flag}={value}"]; i += 1; hit = True; continue
        out.append(t); i += 1
    if not hit: out += [flag, str(value)]
    return out


EXT_FILES = ("ext_ids.npy", "ext_masks.npy", "ext_labels.parquet", "ext_lat.parquet", "ext_vols.npy")
_SIDE_RE = r"^OAI_\d{7}_([RL])_"


def ext_fingerprint(X):
    """{file: {bytes, md5}} of the part (ext_vols.npy: size + md5 of its first 64 MiB) - proves a later run trains the SAME part."""
    out = {}
    for f in EXT_FILES:
        p = os.path.join(X, f)
        if os.path.exists(p):
            out[f] = {"bytes": int(os.path.getsize(p)), "md5": _md5(p, cap=(64 << 20) if f == "ext_vols.npy" else None)}
    return out


def read_ext_lat(p):
    """ext_lat.parquet, validated: unique ids, is_left strictly boolean, and consistent with the side in OAI_<ID>_<R|L>_ ids."""
    import pandas as pd
    lx = pd.read_parquet(p)
    if "StudyInstanceUID" not in lx.columns or "is_left" not in lx.columns:
        raise SystemExit(f"[ext] ext_lat.parquet needs StudyInstanceUID + is_left, has {list(lx.columns)}")
    lx = lx[["StudyInstanceUID", "is_left"]].copy(); lx["StudyInstanceUID"] = lx["StudyInstanceUID"].astype(str)
    if lx["StudyInstanceUID"].duplicated().any(): raise SystemExit("[ext] duplicate StudyInstanceUID in ext_lat.parquet")
    v = lx["is_left"]
    if v.dtype != bool:
        n = pd.to_numeric(v, errors="coerce")
        if n.isna().any() or not n.isin([0, 1]).all():
            raise SystemExit(f"[ext] ext_lat.parquet is_left must be bool (or 0/1), got dtype {v.dtype} with values "
                             f"{sorted(map(str, v.unique()))[:5]} (astype(bool) would turn NaN and 'False' into LEFT)")
        lx["is_left"] = n.astype(int).astype(bool)
    side = lx["StudyInstanceUID"].str.extract(_SIDE_RE)[0]
    bad = lx[side.notna() & ((side == "L") != lx["is_left"])]
    if len(bad):
        raise SystemExit(f"[ext] {len(bad)} ext_lat rows contradict the side in their id, e.g. {bad['StudyInstanceUID'].iloc[0]} "
                         f"is_left={bool(bad['is_left'].iloc[0])}")
    return lx


def _read_uids(p):
    import pandas as pd
    if p.endswith(".json"):
        j = json.load(open(p)); return [str(u) for u in (j["ids"] if isinstance(j, dict) else j)]
    if p.endswith(".parquet"): return pd.read_parquet(p)["StudyInstanceUID"].astype(str).tolist()
    if p.endswith(".csv"): return pd.read_csv(p)["StudyInstanceUID"].astype(str).tolist()
    return [s.strip() for s in open(p) if s.strip()]


# ---------------------------------------------------------------- external part: load + validate + prepare trainer inputs
def prepare_ext(e, targv, T, out, rsna_ids):
    """Validate the external part and rewrite the trainer argv. Returns (argv, info, ext_ids_all, trained_ext_ids)."""
    import pandas as pd
    X = e.ext_dir; info = {"ext_dir": os.path.abspath(X)}
    if _has(targv, "--limit") and int(_get(targv, "--limit") or 0) > 0:
        raise SystemExit("[ext] --limit truncates train_ids (RSNA first) and would drop every external row; use --train_subset")
    if _has(targv, "--smoke"):
        raise SystemExit("[ext] --smoke sets --limit 40 internally; refused with --ext_dir")
    allow = [s.strip() for s in e.ext_findings.split(",") if s.strip()]
    bad = [f for f in allow if f not in LAB]
    if bad: raise SystemExit(f"[ext] --ext_findings: unknown finding(s) {bad}; names: {LAB}")
    nev = [f for f in allow if f in NEVER]
    if nev: raise SystemExit(f"[ext] --ext_findings: {nev} are pre-registered NaN for the external part")
    if not (e.ext_weight >= 0 and np.isfinite(e.ext_weight)): raise SystemExit("[ext] --ext_weight must be finite and >= 0")

    ids = np.load(f"{X}/ext_ids.npy", allow_pickle=True).astype(str)
    em = np.load(f"{X}/ext_masks.npy")
    ev = np.load(f"{X}/ext_vols.npy", mmap_mode="r")
    if len(set(ids)) != len(ids): raise SystemExit("[ext] duplicate external ids")
    if ev.dtype != np.uint8 or em.dtype != np.uint8: raise SystemExit(f"[ext] dtypes vols {ev.dtype} masks {em.dtype} (need uint8)")
    if ev.shape != (len(ids), D_SLICES, IMG, IMG) or em.shape != (len(ids), D_SLICES):
        raise SystemExit(f"[ext] shapes ids {ids.shape} vols {ev.shape} masks {em.shape}; need (N,{D_SLICES},{IMG},{IMG}) / (N,{D_SLICES})")
    rng = np.random.default_rng(0)                                  # spot check: mask == 'slice is non-zero' (build_corpus96 rule)
    for r in sorted(rng.choice(len(ids), min(32, len(ids)), replace=False).tolist()):   # <= 32 rows x 14 MB
        nz = (np.asarray(ev[r]).reshape(D_SLICES, -1).max(1) > 0).astype(np.uint8)
        if not np.array_equal(nz, em[r]): raise SystemExit(f"[ext] ext_masks row {r} ({ids[r]}) disagrees with its non-zero slices")
    if (em.sum(1) < 3).any(): raise SystemExit(f"[ext] {int((em.sum(1) < 3).sum())} external studies have < 3 valid slices")

    coll = set(ids) & set(rsna_ids)
    if coll: raise SystemExit(f"[ext] {len(coll)} external ids collide with corpus ids, e.g. {sorted(coll)[:2]}")
    tcsv = os.path.join(T.RSNA, "train.csv")
    tr_ids = set(pd.read_csv(tcsv, usecols=["StudyInstanceUID"])["StudyInstanceUID"].astype(str))
    if set(ids) & tr_ids: raise SystemExit("[ext] external ids appear in train.csv (gold/RSNA)")
    ff = _get(targv, "--fold_file")
    if ff:
        fm = json.load(open(ff))["folds"]
        inf = [u for u in ids if u in fm]
        if inf: raise SystemExit(f"[ext] {len(inf)} external ids are in the fold file {ff} (could land in OOF): {inf[:2]}")
    latx = f"{X}/ext_lat.parquet"
    LX = read_ext_lat(latx) if os.path.exists(latx) else None          # validated whenever present (member too: G1 requires it)
    info["files"] = ext_fingerprint(X)

    # ---- labels: RSNA file untouched + external rows, non-allow-listed cells forced NaN
    labp = _get(targv, "--labels")
    if not labp: raise SystemExit("[ext] --labels is required with --ext_dir")
    rs = pd.read_parquet(labp); rs["StudyInstanceUID"] = rs["StudyInstanceUID"].astype(str)
    for nm, p_ in (("--labels", labp), ("--pw_labels", _get(targv, "--pw_labels"))):   # a pre-merged RSNA+OAI file would supervise
        if p_ and set(_read_uids(p_)) & set(ids):                                         # every OAI cell and move pos_weight
            raise SystemExit(f"[ext] {nm} {p_} already holds external ids (a pre-merged file such as *_plus_moaks.parquet?): "
                             f"pass the RSNA-only file; ext_trainer appends ext_labels.parquet itself, allow-list applied")
    xl = pd.read_parquet(f"{X}/ext_labels.parquet"); xl["StudyInstanceUID"] = xl["StudyInstanceUID"].astype(str)
    if xl["StudyInstanceUID"].duplicated().any(): raise SystemExit("[ext] duplicate StudyInstanceUID in ext_labels.parquet")
    unknown = set(xl["StudyInstanceUID"]) - set(ids)
    if unknown: raise SystemExit(f"[ext] {len(unknown)} ext_labels rows have no image in ext_ids.npy, e.g. {sorted(unknown)[:2]}")
    extra_cols = [c for c in xl.columns if c != "StudyInstanceUID" and c not in LAB]
    for c in LAB:
        if c not in xl.columns: xl[c] = np.nan
    vals = xl[LAB].astype("float64")
    fin = vals.values[np.isfinite(vals.values)]
    if fin.size and (fin.min() < 0 or fin.max() > 1): raise SystemExit("[ext] external label values outside [0,1]")
    forced = {c: int(vals[c].notna().sum()) for c in LAB if c not in allow and vals[c].notna().any()}
    for c in LAB:
        if c not in allow: vals[c] = np.nan
    xl = pd.concat([xl[["StudyInstanceUID"]], vals], axis=1)
    order = {u: i for i, u in enumerate(ids)}
    xl = xl.assign(_o=xl["StudyInstanceUID"].map(order)).sort_values("_o").drop(columns="_o")
    n_all_nan = int(xl[LAB].isna().all(axis=1).sum())
    xl = xl[~xl[LAB].isna().all(axis=1)]
    if e.ext_max > 0: xl = xl.iloc[:e.ext_max]
    keep = xl["StudyInstanceUID"].tolist()
    empty = [c for c in allow if not xl[c].notna().any()]
    if (not keep or empty) and not e.ext_allow_empty:
        raise SystemExit(f"[ext] {'0 external rows would be trained' if not keep else f'allow-listed {empty} have no labelled external cell'}"
                         f" (label columns ignored as unknown: {extra_cols}; names must be exactly {LAB}); --ext_allow_empty overrides")
    info.update(n_ext_ids=int(len(ids)), n_ext_labelled_rows=int(len(keep)), n_dropped_all_nan=n_all_nan,
                ext_findings=allow, forced_nan_cells=forced, ignored_columns=extra_cols, ext_weight=e.ext_weight,
                per_finding={c: {"n": int(xl[c].notna().sum()), "mean": (round(float(xl[c].mean()), 4) if xl[c].notna().any() else None)}
                             for c in allow})
    miss_cols = [c for c in LAB if c not in rs.columns]
    if miss_cols: raise SystemExit(f"[ext] RSNA label file lacks {miss_cols}")
    merged = pd.concat([rs[["StudyInstanceUID"] + LAB], xl[["StudyInstanceUID"] + LAB]], ignore_index=True)
    os.makedirs(out, exist_ok=True)
    tag = _get(targv, "--tag") or "dev"
    mp = os.path.join(out, f"ext_labels_merged_{tag}.parquet"); merged.to_parquet(mp, index=False)
    argv = _set(list(targv), "--labels", mp)
    info["labels_rsna"] = os.path.abspath(labp); info["labels_merged"] = mp

    # ---- laterality: merged copy of the RSNA flags + external rows
    latp = _get(targv, "--lat_file")
    if _has(targv, "--lat_norm"):
        if LX is None: raise SystemExit("[ext] --lat_norm needs ext_lat.parquet in --ext_dir")
        gap = set(keep) - set(LX["StudyInstanceUID"])
        if gap: raise SystemExit(f"[ext] {len(gap)} trained external ids have no laterality row (would default to right knee)")
    if latp and LX is not None:
        lr = pd.read_parquet(latp); lr["StudyInstanceUID"] = lr["StudyInstanceUID"].astype(str)
        lx = LX
        if set(lx["StudyInstanceUID"]) & set(lr["StudyInstanceUID"]): raise SystemExit("[ext] ext_lat ids collide with RSNA laterality ids")
        lm = pd.concat([lr[["StudyInstanceUID", "is_left"]], lx[["StudyInstanceUID", "is_left"]].astype({"is_left": bool})], ignore_index=True)
        lp = os.path.join(out, f"ext_lat_merged_{tag}.parquet"); lm.to_parquet(lp, index=False)
        argv = _set(argv, "--lat_file", lp)
        info["lat_merged"] = lp
        info["ext_left_knees"] = int(lx[lx["StudyInstanceUID"].isin(keep)]["is_left"].astype(bool).sum())

    if keep:
        # ---- train subset: the RSNA selection plus every external id
        sp = _get(targv, "--train_subset")
        if sp:
            su = _read_uids(sp) + keep
            np_ = os.path.join(out, f"ext_subset_{tag}.json"); json.dump(su, open(np_, "w"))
            argv = _set(argv, "--train_subset", np_); info["train_subset_merged"] = np_
        # ---- per-source loss weight through the trainer's own --study_weights (multiplies all 12 cells of a study)
        swp = _get(targv, "--study_weights")
        if e.ext_weight != 1.0 or swp:
            sw = (pd.read_parquet(swp) if (swp and swp.endswith(".parquet")) else pd.read_csv(swp)) if swp else \
                pd.DataFrame({"StudyInstanceUID": [], "weight": []})
            sw["StudyInstanceUID"] = sw["StudyInstanceUID"].astype(str)
            base = dict(zip(sw["StudyInstanceUID"], sw["weight"].astype(float)))
            for u in keep: base[u] = base.get(u, 1.0) * e.ext_weight
            wp = os.path.join(out, f"ext_study_weights_{tag}.parquet")
            pd.DataFrame({"StudyInstanceUID": list(base), "weight": list(base.values())}).to_parquet(wp, index=False)
            argv = _set(argv, "--study_weights", wp); info["study_weights_merged"] = wp
        # ---- pos_weight pinned to the RSNA rows
        if e.ext_pin_pw == "auto" and not _has(targv, "--pw_labels"):
            argv = _set(argv, "--pw_labels", os.path.abspath(labp)); info["pw_pinned_to"] = os.path.abspath(labp)
    return argv, info, ids, keep


def post_guard(out, tag, ext_ids):
    """Every OOF / gold prediction file of this tag must be free of external ids."""
    ext = set(map(str, ext_ids)); res = {}
    for f in sorted(os.listdir(out)):
        if (f.startswith(f"raptor_oof_{tag}_") or f.startswith(f"raptor_gold_{tag}")) and f.endswith(".npz"):
            z = np.load(os.path.join(out, f), allow_pickle=True)
            ids = set(map(str, z["ids"])); res[f] = {"n": len(ids), "ext": len(ids & ext)}
    bad = {k: v for k, v in res.items() if v["ext"]}
    return res, bad


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--trainer", default=DEF_TRAINER)
    ap.add_argument("--out", default=os.environ.get("RAPTOR_OUT"))
    ap.add_argument("--ext_dir", default=None)          # CLI only: a stale RAPTOR_EXT in a shell must never switch it on
    ap.add_argument("--ext_weight", type=float, default=1.0)
    ap.add_argument("--ext_findings", default=",".join(DEFAULT_FINDINGS))
    ap.add_argument("--ext_max", type=int, default=0)
    ap.add_argument("--ext_pin_pw", choices=["auto", "off"], default="auto")
    ap.add_argument("--ext_allow_empty", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    e, targv = ap.parse_known_args(argv)
    tag = _get(targv, "--tag") or "dev"
    out = os.path.abspath(e.out or os.path.join(HERE_SELF, "runs", tag)); os.makedirs(out, exist_ok=True)
    os.environ.setdefault("RAPTOR_INPUT", f"{R}/public/kinput96")    # the trainer binds _find's default root at import
    if e.threads > 0:
        import torch; torch.set_num_threads(e.threads)
    trainer = os.path.abspath(e.trainer)
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(trainer))[0], trainer)
    T = importlib.util.module_from_spec(spec); sys.modules[spec.name] = T; spec.loader.exec_module(T)
    if list(T.LAB) != LAB: raise SystemExit(f"[ext] trainer LAB order differs: {T.LAB}")
    T.HERE = out                                                      # outputs only; the corpus is found under RAPTOR_INPUT
    if os.path.exists(os.path.join(out, "all_vols.npy")) or os.path.exists(os.path.join(out, "all_ids.npy")):
        raise SystemExit("[ext] --out holds all_vols/all_ids.npy: the trainer would read it as the corpus")
    man = {"when": time.strftime("%F %T"), "trainer": trainer, "trainer_md5": _md5(trainer), "out": out, "tag": tag,
           "argv_in": argv, "raptor_input": os.environ["RAPTOR_INPUT"], "crop_frac": os.environ.get("RAPTOR_CROP_FRAC", "1.0")}
    ext_ids = []
    if e.ext_dir:
        _orig_ids, _orig_corpus = T._open_ids, T._open_corpus
        X = e.ext_dir; cache = {}

        def _open_ids():
            ids = _orig_ids()
            return np.concatenate([ids, np.load(f"{X}/ext_ids.npy", allow_pickle=True).astype(str)])

        def _open_corpus():
            if "c" not in cache:
                vols, masks = _orig_corpus()
                cache["c"] = (_Chain(vols, np.load(f"{X}/ext_vols.npy", mmap_mode="r")),
                              np.concatenate([masks, np.load(f"{X}/ext_masks.npy")], 0))
            return cache["c"]
        targv, info, ext_ids, keep = prepare_ext(e, targv, T, out, _orig_ids())
        T._open_ids = _open_ids
        T._open_corpus = _open_corpus                                  # patched only after validation passed
        man["ext"] = info
        print(f"[ext] part {info['ext_dir']}: {info['n_ext_ids']} ids, {info['n_ext_labelled_rows']} trained "
              f"(all-NaN dropped {info['n_dropped_all_nan']}) | findings {info['ext_findings']} | weight {e.ext_weight} | "
              f"forced-NaN cells {info['forced_nan_cells']} | pos_weight pin {info.get('pw_pinned_to', 'no')} | "
              f"left knees {info.get('ext_left_knees', 'n/a')}", flush=True)
        for c, v in info["per_finding"].items(): print(f"[ext]   {c:12s} n={v['n']} mean={v['mean']}", flush=True)
    else:
        print(f"[ext] OFF (no --ext_dir): trainer {os.path.basename(trainer)} runs unchanged | outputs -> {out}", flush=True)
    man["argv_trainer"] = targv
    json.dump(man, open(os.path.join(out, f"ext_manifest_{tag}.json"), "w"), indent=1)
    sys.argv = [trainer] + targv
    T.main()
    if e.ext_dir:
        res, bad = post_guard(out, tag, ext_ids)
        man["post_guard"] = {"files": res, "violations": bad}
        json.dump(man, open(os.path.join(out, f"ext_manifest_{tag}.json"), "w"), indent=1)
        if bad: raise SystemExit(f"[ext] GUARD FAILED: external ids inside held-out/gold predictions: {bad}")
        print(f"[ext] post-run guard OK: {len(res)} OOF/gold prediction files, no external id in any", flush=True)


if __name__ == "__main__":
    main()
