"""Public notebook A v7 end to end as one function (the body of exps/exp-0001/submit_public.py; exp-0004 calls it).

A's constants (cells 3/5, 6, 8, 9) and the keys of the five inputs are module constants here on purpose: they are
the identity of the fork, like vendored/sha256.json, so run_public(ctx, t0) gives A v7 exactly (lead-approved
exception to "nothing hardcoded in src/", exp-0001 D5). Kaggle-only: CUDA, /kaggle/working, and the vendored
Raptor scripts look up their weights and the test data under /kaggle/input themselves.
"""

import glob
import importlib.metadata
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

# ---- A's constants (cells 3/5, 6, 8, 9) ----
RAPTOR_TIMEOUT_S = 2 * 3600                       # cells 3 / 5
RAPTOR_WORK = Path("/tmp")                        # cells 2 / 4 wrote the scripts to /tmp
RAPTOR384_CKPT = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"        # cell 2 _weight_name
RAPTOR224_CKPT = "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"    # cell 4 _weight_name
ARM = "E_lrprior"                                 # cell 6
BUNDLE_ID = "fe055406fad6a12f"
DEADLINE_HOURS = 8.3
FLIP_TTA = True
PROBE_CLASS = None
CODE_SHA = {
    "native_cache.py": "c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9",
    "native_data.py": "f59afe205c8e193369fa3ccd22edcce8df3b293205fc48ddb1dd45169ee43b36",
    "input_provenance.py": "b3a22500c7f976f6ee040239ada5cacecf2326861cb49af1c7d8cfdeeb55fcda",
    "intensity_mapping.py": "19be2d8cbd6f8971c1288d7512a8cca41d143470c2ba6e9b9096fde035fa9d41",
    "coronal_views.py": "841f216d2682ae2c0f8afd82fc902b9fa74924d7145b952069dea8b0bbd03c40",
    "exact_duplicates.py": "51b38db21ce9c5fe2b6278a835983414756b61ac6af81aba380b44b41b075897",
    "dicom_geometry.py": "12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1",
    "common.py": "8283fa1f8e7ef8cfacbba52df1715636be3c9ad688e39912814e7d5bcf57105f",
}
W_RAPTOR, W_OURS = 0.5, 0.5                       # cell 8
A224 = 0.35
W_RAPTOR_C = {"Medial OA": 0.5, "Lateral OA": 0.15, "PF OA": 0.65, "Synovitis": 0.85, "Baker's": 0.30}
PART_B_CUTOFF_HOURS = 8.2                         # cell 9
OWN_W = 0.15
OWN_GLOB = "cnxt_v0_fold[012].pt"
OWN_W_PER_TARGET = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02,
                    "MCL": 0.02, "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008,
                    "Effusion": 0.004, "Synovitis": 0.003}
# ---- the five inputs (ctx.mounts keys) ----
DATA_MOUNT = "rsna-knee-abnormality-detection"
READER_MOUNT = "goodpjw2008/rsna-knee-2-5d-convnext-reader"
# the -w1 copy: A mounts heliosli/rsna-knee-r6-frozen-fe055406, whose v2 (14.10) holds R7 under the same file
# names; -w1 is byte-identical to its v1 (manifest sha256 6508ce69...)
BUNDLE_MOUNT = "heliosli/rsna-knee-r6-frozen-fe055406-w1"
RAPTOR_W_MOUNT = "nartaa/rsna-knee-publication-swa-weights-20261007"
CODE_MOUNT = "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"


def run_public(ctx, t0):
    """A v7 public pipeline in one call: Raptor384 → Raptor224 → R6 → stack blend → reader (Part B) → submission.csv + members.

    t0: the caller's script start (A: NB_T0, start of cell 3); the R6 deadline (8.3 h) and the Part B cut (8.2 h)
    count from it. Returns (final_sub, members, masks, timings, part_b): the final submission.csv, the dicts given
    to save_members, timings.json and fallbacks.json["part_b"].
    """
    if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
        raise RuntimeError(
            "submit_public.py runs only inside a Kaggle kernel (CUDA, /kaggle/working; the vendored Raptor "
            "scripts read /kaggle/input directly). Launch it with the kaggle skill: "
            "kaggle/run_kernel.py --exp exp-0001 --script submit_public ... with the four datasets of A attached."
        )

    # All mounts resolved up front: a missing attachment fails here, before any GPU time is spent
    # (A finds them lazily per stage; its Part B would fall back on a missing reader dataset).
    DATA = ctx.mounts[DATA_MOUNT]
    READER_DIR = ctx.mounts[READER_MOUNT]
    BUNDLE_SRC = ctx.mounts[BUNDLE_MOUNT]
    RAPTOR_W = ctx.mounts[RAPTOR_W_MOUNT]
    CODE_SRC = ctx.mounts[CODE_MOUNT]
    WORK = ctx.output_dir
    READER_CKPTS = sorted(c for g in OWN_GLOB.split() for c in glob.glob(str(READER_DIR) + "/" + g))   # as cell 9's _o_ck

    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()
    print("EXP0001_ENV", json.dumps({"python": sys.version, "torch": importlib.metadata.version("torch"),
                                     "docker_image": os.environ.get("KAGGLE_DOCKER_IMAGE"), "gpus": gpus}), flush=True)

    timings = {}
    r384 = stages.run_raptor("384", RAPTOR_WORK, WORK, RAPTOR_TIMEOUT_S)
    timings["raptor384"] = r384["seconds"]
    r224 = stages.run_raptor("224", RAPTOR_WORK, WORK, RAPTOR_TIMEOUT_S)
    timings["raptor224"] = r224["seconds"]

    TMP = (Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")) / "probe"   # cell 6
    r6 = stages.run_compact_r6(DATA, CODE_SRC, BUNDLE_SRC, TMP, WORK, t0, ARM, BUNDLE_ID, DEADLINE_HOURS, FLIP_TTA,
                               CODE_SHA, PROBE_CLASS, None)
    timings["r6"] = r6["seconds"]

    t = time.time()
    test_ids, cols, labels, inputs = stages.read_stack_inputs(DATA, WORK)
    stack, blend, fail_masks = stages.blend_stack(test_ids, labels, cols, inputs, r6["pred_dir"], W_RAPTOR, W_OURS,
                                                  W_RAPTOR_C, A224, r384["seconds"], r224["seconds"], t0)
    stack.to_csv(WORK / "submission.csv", index=False)
    (WORK / "blend_receipt.json").write_text(json.dumps(blend, indent=1))
    timings["blend_stack"] = time.time() - t

    # ---- Part B (cell 9): A's semantics, plus a loud line and fallbacks.json (approved exception for this fork) ----
    t = time.time()
    pub, bak = WORK / "submission.csv", WORK / "_public_stack.csv"
    own = None
    if time.time() - t0 > PART_B_CUTOFF_HOURS * 3600:
        print("Part B skipped: time budget used, the Part A3 submission is kept", flush=True)
        part_b = {"fired": True, "reason": "time budget: %.0f s since start > %.1f h" % (time.time() - t0, PART_B_CUTOFF_HOURS)}
    else:
        shutil.copy(pub, bak)
        try:
            t_reader = time.time()
            own_csv = stages.run_reader(DATA, READER_DIR, WORK, OWN_GLOB)
            stack_csv = pd.read_csv(bak, dtype={"StudyInstanceUID": str})
            own = pd.read_csv(own_csv, dtype={"StudyInstanceUID": str})
            final = stages.blend_reader(stack_csv, own, labels, OWN_W, OWN_W_PER_TARGET, len(READER_CKPTS), t_reader)
            final.to_csv(pub, index=False)
            part_b = {"fired": False, "reason": ""}
        except Exception as e:
            shutil.copy(bak, pub)
            print("own leg FAILED, public stack submission kept:", repr(e))
            part_b = {"fired": True, "reason": repr(e)}
    if part_b["fired"]:
        print("!!! EXP0001_FALLBACK part_b %s — submission = stack without the reader !!!" % part_b["reason"], flush=True)
    (WORK / "fallbacks.json").write_text(json.dumps({"part_b": part_b}, indent=1))
    timings["part_b"] = time.time() - t

    # ---- members, timings, input hashes, summary (after submission.csv is final) ----
    t = time.time()
    final_sub = pd.read_csv(pub, dtype={"StudyInstanceUID": str})

    def aligned(df):
        return df.set_index("StudyInstanceUID").loc[test_ids, labels].reset_index()

    members = {"raptor384": aligned(inputs["raptor_raw"]), "raptor224": aligned(inputs["raptor224_raw"]),
               "r6": aligned(inputs["ours"])}
    masks = {"raptor384": fail_masks["raptor384"], "raptor224": fail_masks["raptor224"], "r6": fail_masks["r6"]}
    if not part_b["fired"]:
        members["reader"] = aligned(own)
        masks["reader"] = np.zeros(len(test_ids), dtype=bool)
    members["stack"], masks["stack"] = stack, np.zeros(len(test_ids), dtype=bool)
    members["final"], masks["final"] = final_sub, np.zeros(len(test_ids), dtype=bool)
    stages.save_members(WORK, members, masks, {"raptor": W_RAPTOR, "ours": W_OURS, "raptor_per_class": W_RAPTOR_C,
                                               "a224": A224, "reader": OWN_W, "reader_per_target": OWN_W_PER_TARGET,
                                               "part_b_fired": part_b["fired"]})
    timings["save_members"] = time.time() - t
    timings["total"] = time.time() - t0
    (WORK / "timings.json").write_text(json.dumps(timings, indent=1))

    inputs_sha = {
        "reader_checkpoints": stages.sha256_files(READER_CKPTS),
        "r6_bundle": stages.sha256_files(sorted(p for p in BUNDLE_SRC.rglob("*") if p.is_file())),
        "raptor_checkpoints": stages.sha256_files([RAPTOR_W / RAPTOR384_CKPT, RAPTOR_W / RAPTOR224_CKPT]),
    }
    (WORK / "inputs_sha256.json").write_text(json.dumps(inputs_sha, indent=1))

    print("EXP0001_SUMMARY", json.dumps({"timings": timings, "fallbacks": {"part_b": part_b}, "rows": len(final_sub),
                                         "submission_head": final_sub.head(10).to_dict(orient="records")}), flush=True)
    return final_sub, members, masks, timings, part_b
