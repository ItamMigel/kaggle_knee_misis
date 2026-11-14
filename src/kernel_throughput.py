"""Throughput of public inference stages on train studies dressed up as the test split (exp-0001, hidden-runtime budget).

The hidden test cannot be timed, so each stage runs on its own random train studies: a fake competition root whose
test.csv / test_series.csv / test_series/ hold them (every DICOM a symlink into train_series/). Two sample sizes per
stage give fixed cost + seconds per series; the projection is to N studies at a given series per study.
Stdlib + pandas only (runs inside kernels).
"""

import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

from src.public_fork import stages

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RAPTOR_ROOT_LINE = "    ROOT = find_test_root()\n"


def sample_disjoint(series_csv, sizes, seed):
    """{name: n} -> {name: [study ids]}: disjoint uniform samples of train studies (sorted ids shuffled with `seed`)."""
    ids = sorted(pd.read_csv(series_csv, dtype=str, usecols=["StudyInstanceUID"])["StudyInstanceUID"].unique())
    random.Random(seed).shuffle(ids)
    need = sum(sizes.values())
    assert need <= len(ids), ("samples need more studies than train has", need, len(ids))
    out, k = {}, 0
    for name, n in sizes.items():
        out[name] = ids[k: k + n]
        k += n
    return out


def fake_competition_root(data_dir, study_ids, dest):
    """Competition layout with `study_ids` as the test split -> {"root", "studies", "series", "files"}.

    test.csv / test_series.csv (rows of train_series.csv) / sample_submission.csv (0.5) and test_series/<study>/<series>/
    as real directories holding one symlink per train DICOM, so os.walk and glob see the same tree as on the real test.
    """
    data_dir, dest = Path(data_dir), Path(dest)
    ser = pd.read_csv(data_dir / "train_series.csv", dtype=str)
    ser = ser[ser["StudyInstanceUID"].isin(set(study_ids))]
    assert set(ser["StudyInstanceUID"]) == set(study_ids), "studies without rows in train_series.csv"
    (dest / "test_series").mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"StudyInstanceUID": study_ids}).to_csv(dest / "test.csv", index=False)
    ser.to_csv(dest / "test_series.csv", index=False)
    sub = pd.DataFrame({"StudyInstanceUID": study_ids})
    for c in LABELS:
        sub[c] = 0.5
    sub.to_csv(dest / "sample_submission.csv", index=False)
    n_files = 0
    for study, series in zip(ser["StudyInstanceUID"], ser["SeriesInstanceUID"]):
        src = data_dir / "train_series" / study / series
        assert src.is_dir(), "no DICOM directory %s" % src
        dst = dest / "test_series" / study / series
        dst.mkdir(parents=True)
        for e in os.scandir(src):
            os.symlink(e.path, dst / e.name)
            n_files += 1
    return {"root": dest, "studies": len(study_ids), "series": len(ser), "files": n_files}


def run_raptor_on(variant, test_root, work_dir, out_dir, timeout_s):
    """Vendored Raptor 384/224 (sha-checked) on `test_root`: its find_test_root() call is replaced by that path -> seconds.

    The only edit to the script text; it must match exactly once, else AssertionError.
    """
    script, stem, _, _ = stages._RAPTOR[variant]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = stages.materialize(script, work_dir)
    text = path.read_text()
    assert text.count(RAPTOR_ROOT_LINE) == 1, ("Raptor %s: the find_test_root() call line is not unique" % variant)
    path.write_text(text.replace(RAPTOR_ROOT_LINE, "    ROOT = os.environ['TP_TEST_ROOT']   # [throughput]\n"))
    t = time.time()
    rc = subprocess.run([sys.executable, "-u", str(path)], env=dict(os.environ, TP_TEST_ROOT=str(test_root)),
                        timeout=timeout_s).returncode
    if rc != 0:
        raise RuntimeError("Raptor %s exited with %d" % (variant, rc))
    for f in (stem + "_sub.csv", stem + "_raw.csv"):
        assert (Path(out_dir) / f).is_file(), f + " missing"
    return time.time() - t


def run_s6fall_b_on(k1_script, driver, test_root, weights_dir, helios_dir, kroot, budget_s):
    """S6fall's component B alone (k1_driver.py) on `test_root` -> the driver's json (seconds per step, receipts).

    kroot/input mirrors Kaggle's mount layout with symlinks, so K1's find_mount resolves the fake competition root.
    """
    kroot = Path(kroot)
    for link, target in (("input/competitions/rsna-knee-abnormality-detection", test_root),
                         ("input/datasets/k1/rsna-knee-s6fall-weights", weights_dir),
                         ("input/datasets/k1/" + Path(helios_dir).name, helios_dir)):
        (kroot / link).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, kroot / link)
    (kroot / "working").mkdir(parents=True)
    out_json = kroot / "driver.json"
    env = dict(os.environ, K1_KAGGLE_ROOT=str(kroot), K1_TMP=str(kroot / "tmp"))
    rc = subprocess.run([sys.executable, "-u", str(driver), str(k1_script), str(budget_s), str(out_json)], env=env,
                        timeout=budget_s + 1800).returncode
    if rc != 0:
        raise RuntimeError("S6fall component B (k1_driver) exited with %d" % rc)
    return json.loads(out_json.read_text())


class UsageSampler:
    """Context manager: nvidia-smi utilization / memory per GPU and 1-min load average every `every_s` seconds."""

    def __init__(self, every_s=5.0):
        self.every_s, self.rows, self._stop = every_s, [], threading.Event()

    def _loop(self):
        while not self._stop.wait(self.every_s):
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20).stdout
                gpus = [[float(x) for x in line.split(",")] for line in q.strip().splitlines()]
                self.rows.append({"t": time.time(), "load1": os.getloadavg()[0], "gpus": gpus})
            except Exception as e:      # a missed sample is recorded, not hidden
                self.rows.append({"t": time.time(), "error": repr(e)[:200]})

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=30)
        return False

    def summary(self):
        """{"samples", "errors", "load1_mean", "gpu": {idx: {"util_mean", "util_busy_frac" (>10 %), "mem_max_mib"}}}."""
        ok = [r for r in self.rows if "gpus" in r]
        out = {"samples": len(ok), "errors": len(self.rows) - len(ok),
               "load1_mean": round(sum(r["load1"] for r in ok) / len(ok), 2) if ok else None, "gpu": {}}
        for r in ok:
            for idx, util, mem in r["gpus"]:
                g = out["gpu"].setdefault(str(int(idx)), {"util": [], "mem": []})
                g["util"].append(util)
                g["mem"].append(mem)
        for idx, g in out["gpu"].items():
            out["gpu"][idx] = {"util_mean": round(sum(g["util"]) / len(g["util"]), 1),
                               "util_busy_frac": round(sum(u > 10 for u in g["util"]) / len(g["util"]), 3),
                               "mem_max_mib": max(g["mem"])}
        return out


def fit_projection(small, large, n_studies, series_per_study):
    """Two runs {"seconds", "series", "files"} of one stage -> fixed s, s/series, s/file and projected hours.

    proj_h[spp]: fixed + s_per_series * spp * n_studies (the fit); prop_h[spp]: the large run scaled proportionally,
    fixed cost included in the rate (an upper bound when fixed > 0). files: the same fit on DICOM files, projected with
    the large sample's files per series.
    """
    s_per_series = (large["seconds"] - small["seconds"]) / (large["series"] - small["series"])
    fixed = small["seconds"] - s_per_series * small["series"]
    s_per_file = (large["seconds"] - small["seconds"]) / (large["files"] - small["files"])
    fixed_f = small["seconds"] - s_per_file * small["files"]
    files_per_series = large["files"] / large["series"]
    out = {"fixed_s": round(fixed, 1), "s_per_series": round(s_per_series, 3), "fixed_s_files": round(fixed_f, 1),
           "s_per_file": round(s_per_file, 4), "files_per_series_large": round(files_per_series, 2),
           "proj_h": {}, "prop_h": {}, "proj_h_files": {}}
    for spp in series_per_study:
        k = "%.2f" % spp
        out["proj_h"][k] = round((fixed + s_per_series * spp * n_studies) / 3600, 3)
        out["prop_h"][k] = round(large["seconds"] / large["series"] * spp * n_studies / 3600, 3)
        out["proj_h_files"][k] = round((fixed_f + s_per_file * files_per_series * spp * n_studies) / 3600, 3)
    return out
