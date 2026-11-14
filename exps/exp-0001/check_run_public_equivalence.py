"""CPU check of D5: submit_public.py's body moved to src/public_fork/pipeline.run_public changes nothing.

    nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0001/check_run_public_equivalence.py [--keep NEW_DIR]

(--keep: build the harness in NEW_DIR, which must not exist yet, and leave it there: per scenario and side the
working dir, stdout.txt / stderr.txt and calls.json.)

OLD = `git show caff3c8:exps/exp-0001/submit_public.py` with caff3c8's src/ and utils/ (the script of the scored
kernel ffe2); NEW = the working-tree submit_public.py, src/ and utils/. Each side runs in its own fresh interpreter
(this file with --child) under the same harness:
- fake Kaggle without touching the code under test: a user+mount namespace (`unshare -Urm`) chroots into a tmpfs
  root that bind-mounts every top-level directory of / and adds /kaggle/input (five small fake inputs named like
  the real ones, the code tree at datasets/bebra3222/rsna-knee-sources-exp-0001), /kaggle/working and /kaggle/temp;
  so the guard really sees /kaggle/input and output_dir == /kaggle/working;
- the child patches, before it runs the script with runpy: utils.ctx.get_ctx (utils.ctx's own Mounts over
  /kaggle/input, output_dir /kaggle/working, no recorder), time.time (a counter, +1 s per call) and
  stages.run_raptor / run_compact_r6 / run_reader (they copy the fixtures of check_blend_equivalence.synth where the
  real stages write, return the same dict shapes); everything else in stages runs for real, and every stages.* call
  is logged (name, args and result digests); PATH starts with a stub nvidia-smi; KAGGLE_DOCKER_IMAGE and
  PYTHONHASHSEED are fixed.
Scenarios: a) 3 studies, normal; b) run_reader raises -> Part B fallback; c) R6 takes 8.25 h on the clock -> Part B
time cut; d) no fake environment -> both raise the guard's RuntimeError.
Per scenario OLD == NEW: exit code, stdout bytes, stderr bytes, the output tree (names + bytes), the stage-call log;
in d) the tracebacks differ by construction (OLD raises at module level, NEW inside run_public), so there the
exception type + message and the last stderr line are compared instead of the full stderr. For NEW the returned
(final_sub, members, masks, timings, part_b) must equal submission.csv, members/, timings.json, fallbacks.json.
Then check_fingerprint.py on a)'s two stdouts and a re-run of check_blend_equivalence.py.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import contextlib  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

# numpy's OpenBLAS starts a thread per core on import (128 here, a shared machine): one thread in every process
os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

REPO = Path(_p[0])
HERE = Path(__file__).resolve()
OLD_REV = "caff3c8"
SCRIPT = "exps/exp-0001/submit_public.py"
CODE_MOUNT = "datasets/bebra3222/rsna-knee-sources-exp-0001"   # where the sources dataset sat in kernel ffe2
CLOCK_START = 1_800_000_000.0
DOCKER_IMAGE = "gcr.io/kaggle-gpu-images/python@sha256:harness"
GPUS = "0, Tesla T4, 15360 MiB\n1, Tesla T4, 15360 MiB\n"
# fake inputs: mount key -> file names (the real names; contents are a placeholder line)
FAKE_INPUTS = {
    "datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader": ["cnxt_v0_fold0.pt", "cnxt_v0_fold1.pt", "cnxt_v0_fold2.pt"],
    "datasets/heliosli/rsna-knee-r6-frozen-fe055406-w1": ["manifest.json"] + ["E_lrprior_s0_f%d.pt" % f for f in range(4)],
    "datasets/nartaa/rsna-knee-publication-swa-weights-20261007": ["raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
                                                                   "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"],
    "datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004": ["receipt.json"],
}
# what may differ between the OLD and NEW code the pipeline can reach: utils/ and src/public_fork/ (public_fork
# imports nothing else from src/, so modules other experiments add to src/ are out of scope)
REACHABLE = ("utils/", "src/public_fork/")
ALLOWED_PAYLOAD_DIFF = {"src/public_fork/pipeline.py", "src/public_fork/__init__.py"}

# user+mount namespace: tmpfs root with every top-level dir of / bound in, plus /kaggle; then chroot and run the child.
# Mounts stay private to the namespace (unshare's default propagation), nothing on the host changes.
NS_SETUP = r"""set -e
R=$1 INPUT=$2 CODE=$3 OUT=$4 TEMP=$5
shift 5
mount -t tmpfs tmpfs "$R"
for e in /*; do
    n=${e#/}
    if [ -L "$e" ]; then ln -s "$(readlink "$e")" "$R/$n"
    elif [ -d "$e" ]; then mkdir "$R/$n"; mount --rbind "$e" "$R/$n"
    fi
done
mkdir -p "$R/kaggle/input" "$R/kaggle/working" "$R/kaggle/temp"
mount --rbind "$INPUT" "$R/kaggle/input"
mount --bind "$CODE" "$R/kaggle/input/CODE_MOUNT"
mount --bind "$OUT" "$R/kaggle/working"
mount --bind "$TEMP" "$R/kaggle/temp"
exec chroot "$R" /bin/sh -c 'cd /kaggle/working && exec "$@"' sh "$@"
""".replace("CODE_MOUNT", CODE_MOUNT)


# ---------------------------------------------------------------- child (one side, one scenario)

class Clock:
    """time.time replacement: +1 s per call; a stub can jump it forward."""

    def __init__(self, start):
        self.now = start

    def __call__(self):
        self.now += 1.0
        return self.now

    def advance(self, seconds):
        self.now += seconds


def child(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    code_root = cfg["code_root"]
    sys.path[:] = [code_root] + [p for p in sys.path if p not in (str(REPO), str(HERE.parent))]
    import numpy as np
    import pandas as pd

    import utils.ctx as uctx
    from src.public_fork import stages
    assert Path(stages.__file__).resolve().is_relative_to(Path(code_root).resolve()), stages.__file__
    pipeline = None
    if (Path(code_root) / "src/public_fork/pipeline.py").is_file():
        from src.public_fork import pipeline
    norm_map = cfg["normalize"]

    def norm_str(s):
        for real, tag in norm_map.items():
            s = s.replace(real, tag)
        return s

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def digest(x):
        if isinstance(x, pd.DataFrame):
            return {"DataFrame": list(x.shape), "csv_sha256": sha(x.to_csv().encode("utf-8"))}
        if isinstance(x, np.ndarray):
            return {"ndarray": str(x.dtype), "shape": list(x.shape), "sha256": sha(np.ascontiguousarray(x).tobytes())}
        if isinstance(x, Path):
            return {"path": norm_str(str(x))}
        if isinstance(x, str):
            return norm_str(x)
        if x is None or isinstance(x, (bool, int, float)):
            return x
        if isinstance(x, dict):
            return {str(k): digest(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [digest(v) for v in x]
        raise TypeError("no digest for %r" % type(x))

    clock = Clock(cfg["clock_start"])
    fx = Path(cfg["fixtures"])

    def run_raptor(variant, work_dir, out_dir, timeout_s):
        _, stem, _, marker = stages._RAPTOR[variant]
        t = time.time()
        for suffix in ("_sub.csv", "_raw.csv"):
            shutil.copyfile(fx / "work" / (stem + suffix), Path(out_dir) / (stem + suffix))
        seconds = time.time() - t
        print("%s_STAGE_DONE seconds %.0f" % (marker, seconds), flush=True)
        return {"seconds": seconds, "rc": 0}

    def run_compact_r6(data_dir, code_src, bundle_src, tmp_dir, out_dir, t0, arm, bundle_id, deadline_hours,
                       flip_tta, code_sha, probe_class, w_src):
        t_stage = time.time()
        clock.advance(cfg["r6_extra_seconds"])
        pred = Path(tmp_dir) / "prediction"
        for w in sorted((fx / "pred").glob("worker*")):
            (pred / w.name).mkdir(parents=True)
            shutil.copyfile(w / "predictions.csv", pred / w.name / "predictions.csv")
        shutil.copyfile(fx / "work" / "ours_sub.csv", Path(out_dir) / "ours_sub.csv")
        rows = len(pd.read_csv(fx / "work" / "ours_sub.csv"))
        receipt = {"status": "ok", "arm": arm, "rows": rows, "scored": rows, "fallback_studies": 0,
                   "bundle_id": bundle_id, "stub": True}
        (Path(out_dir) / "submission_receipt.json").write_text(json.dumps(receipt, indent=1))
        print("CONTROLLER_DONE", json.dumps({"rows": rows, "stub": True}), flush=True)
        print("OURS_STAGE_COMPLETE", json.dumps(receipt), flush=True)
        return {"receipt": receipt, "pred_dir": pred, "seconds": time.time() - t_stage}

    def run_reader(data_dir, reader_dir, out_dir, ckpt_glob):
        if cfg["reader_raises"]:
            raise RuntimeError("stub run_reader: simulated infer.py failure")
        own_csv = Path(out_dir) / "_own.csv"
        shutil.copyfile(fx / "work" / "_own.csv", own_csv)
        return own_csv

    calls = []
    stubs = {"run_raptor": run_raptor, "run_compact_r6": run_compact_r6, "run_reader": run_reader}

    def logged(name, fn):
        def wrapper(*args, **kwargs):
            entry = {"call": name, "stub": name in stubs, "args": digest(args), "kwargs": digest(kwargs)}
            calls.append(entry)
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                entry["raised"] = repr(e)
                raise
            entry["returned"] = digest(out)
            return out
        return wrapper

    for name, fn in list(vars(stages).items()):
        if inspect.isfunction(fn) and fn.__module__ == stages.__name__:
            setattr(stages, name, logged(name, stubs.get(name, fn)))
    assert set(stubs) <= set(vars(stages)), "a stubbed stage is missing in stages.py"

    def get_ctx():
        mounts = uctx._mounts(Path(cfg["input_root"]), lambda: [], empty_hint="  (harness: nothing mounted)")
        return uctx.Ctx(sources_dir=Path(code_root), mounts=mounts, output_dir=Path(cfg["output_dir"]))

    uctx.get_ctx = get_ctx
    returned = []
    if pipeline is not None:
        real_run_public = pipeline.run_public

        def run_public(ctx, t0):
            out = real_run_public(ctx, t0)
            returned.append(out)
            return out

        pipeline.run_public = run_public

    script = str(Path(code_root) / SCRIPT)
    sys.argv = [script]
    exc = None
    time.time = clock
    try:
        runpy.run_path(script, run_name="__main__")
    except BaseException as e:
        exc = {"type": type(e).__name__, "message": str(e)}
        raise
    finally:
        Path(cfg["calls_out"]).write_text(json.dumps(calls, indent=1))
        Path(cfg["exc_out"]).write_text(json.dumps(exc))
        if returned:
            with open(cfg["returned_out"], "wb") as f:
                pickle.dump(returned[0], f)


# ---------------------------------------------------------------- parent

def tree(d):
    """Every file and dir under d -> {relative path: bytes (file) or "<dir>"}."""
    return {str(p.relative_to(d)): (p.read_bytes() if p.is_file() else "<dir>") for p in sorted(Path(d).rglob("*"))}


def show_diff(what, a, b):
    if isinstance(a, dict) and isinstance(b, dict) and any(isinstance(v, bytes) for v in [*a.values(), *b.values()]):
        for k in sorted(set(a) | set(b)):   # a file tree
            if a.get(k) != b.get(k):
                print("  %s: %s differs (only OLD: %s, only NEW: %s)" % (what, k, k not in b, k not in a))
                if isinstance(a.get(k), bytes) and isinstance(b.get(k), bytes):
                    show_diff(what + "/" + k, a[k], b[k])
        return

    def text(x):
        if isinstance(x, bytes):
            return x.decode("utf-8", "replace")
        return x if isinstance(x, str) else json.dumps(x, indent=1, sort_keys=True, default=str)

    a_l, b_l = text(a).splitlines(), text(b).splitlines()
    for line in list(difflib.unified_diff(a_l, b_l, "OLD " + what, "NEW " + what, lineterm="", n=1))[:40]:
        print("   ", line)


def same(what, a, b):
    if a != b:
        show_diff(what, a, b)
    check(a == b, "OLD == NEW: " + what)


def build_code_trees(h):
    """OLD: caff3c8's src/, utils/ and submit_public.py; NEW: the same paths from the working tree."""
    old, new = h / "code_old", h / "code_new"
    old.mkdir()
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", OLD_REV, "src", "utils", SCRIPT],
                         capture_output=True, check=True).stdout
    tar_path = h / "old.tar"
    tar_path.write_bytes(tar)
    with tarfile.open(tar_path) as t:
        t.extractall(old, filter="data")
    skip = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("src", "utils"):
        shutil.copytree(REPO / d, new / d, ignore=skip)
    (new / SCRIPT).parent.mkdir(parents=True)
    shutil.copyfile(REPO / SCRIPT, new / SCRIPT)
    old_script = subprocess.run(["git", "-C", str(REPO), "show", "%s:%s" % (OLD_REV, SCRIPT)],
                                capture_output=True, check=True).stdout
    check((old / SCRIPT).read_bytes() == old_script, "OLD submit_public.py == git show %s:%s" % (OLD_REV, SCRIPT))
    t_old = {k: v for k, v in tree(old).items() if not k.startswith("exps")}
    t_new = {k: v for k, v in tree(new).items() if not k.startswith("exps")}
    differ = {k for k in set(t_old) | set(t_new) if t_old.get(k) != t_new.get(k) and k.startswith(REACHABLE)}
    check(differ <= ALLOWED_PAYLOAD_DIFF and "src/public_fork/pipeline.py" in differ,
          "payload OLD vs NEW differs only in %s (stages.py, vendored/, utils/ identical)" % sorted(differ))
    return old, new


def build_inputs(h, labels):
    """Fixtures (check_blend_equivalence.synth, 3 studies) and the fake /kaggle/input tree."""
    import numpy as np

    from check_blend_equivalence import SEED, synth
    fx = h / "fixtures"
    synth(fx, 3, [], [], [], {}, labels, np.random.default_rng(SEED))
    inp = h / "input"
    comp = inp / "competitions/rsna-knee-abnormality-detection"
    comp.mkdir(parents=True)
    for f in ("test.csv", "sample_submission.csv"):
        shutil.copyfile(fx / "data" / f, comp / f)
    for key, names in FAKE_INPUTS.items():
        (inp / key).mkdir(parents=True)
        for n in names:
            (inp / key / n).write_text("harness placeholder for %s/%s\n" % (key, n))
    (inp / CODE_MOUNT).mkdir(parents=True)
    bin_dir = h / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nprintf '%s'\n" % GPUS.replace("\n", "\\n"))
    (bin_dir / "nvidia-smi").chmod(0o755)
    return fx, inp, bin_dir


def run_side(h, scenario, side, code_root, fx, inp, bin_dir, fake_kaggle, reader_raises, r6_extra_seconds):
    d = h / ("%s_%s" % (scenario, side))
    out, temp, newroot = d / "working", d / "temp", d / "root"
    for p in (out, temp, newroot):
        p.mkdir(parents=True)
    cfg = {"code_root": "/kaggle/input/" + CODE_MOUNT if fake_kaggle else str(code_root),
           "input_root": "/kaggle/input" if fake_kaggle else str(inp),
           "output_dir": "/kaggle/working" if fake_kaggle else str(out),
           "fixtures": str(fx), "clock_start": CLOCK_START, "reader_raises": reader_raises,
           "r6_extra_seconds": r6_extra_seconds,
           "normalize": {str(code_root): "<CODE>", str(out): "<OUT>", str(inp): "<INPUT>", str(h): "<HARNESS>"},
           "calls_out": str(d / "calls.json"), "exc_out": str(d / "exc.json"), "returned_out": str(d / "returned.pkl")}
    (d / "cfg.json").write_text(json.dumps(cfg, indent=1))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"PATH": str(bin_dir) + os.pathsep + env["PATH"], "KAGGLE_DOCKER_IMAGE": DOCKER_IMAGE,
                "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    cmd = [sys.executable, str(HERE), "--child", str(d / "cfg.json")]
    if fake_kaggle:
        cmd = ["unshare", "-Urm", "sh", "-c", NS_SETUP, "sh", str(newroot), str(inp), str(code_root), str(out),
               str(temp)] + cmd
    p = subprocess.run(cmd, capture_output=True, env=env)
    (d / "stdout.txt").write_bytes(p.stdout)
    (d / "stderr.txt").write_bytes(p.stderr)
    res = {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "tree": tree(out),
           "calls": json.loads((d / "calls.json").read_text()) if (d / "calls.json").is_file() else None,
           "exc": json.loads((d / "exc.json").read_text()) if (d / "exc.json").is_file() else None,
           "returned": None, "out": out}
    if (d / "returned.pkl").is_file():
        with open(d / "returned.pkl", "rb") as f:
            res["returned"] = pickle.load(f)
    return res


def check_returned(res, labels):
    """NEW's returned tuple == what it wrote."""
    import numpy as np
    import pandas as pd

    final_sub, members, masks, timings, part_b = res["returned"]
    out = res["out"]
    sub_file = pd.read_csv(out / "submission.csv", dtype={"StudyInstanceUID": str})
    pd.testing.assert_frame_equal(final_sub, sub_file, check_exact=True)
    check(final_sub.to_csv(index=False).encode("utf-8") == (out / "submission.csv").read_bytes(),
          "returned final_sub == submission.csv (values and bytes)")
    check(timings == json.loads((out / "timings.json").read_text()), "returned timings == timings.json")
    check(part_b == json.loads((out / "fallbacks.json").read_text())["part_b"], "returned part_b == fallbacks.json")
    man = json.loads((out / "members/manifest.json").read_text())
    check(list(members) == list(masks) == list(man["members"]), "returned members/masks names == manifest %s" % list(members))
    for name, df in members.items():
        prob = out / "members" / (name + "_prob.csv")
        check(df[["StudyInstanceUID"] + labels].to_csv(index=False).encode("utf-8") == prob.read_bytes(),
              "returned members[%r] == %s" % (name, prob.name))
        ids = df["StudyInstanceUID"].astype(str).tolist()
        mask = masks[name]
        check(isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (len(ids),)
              and int(mask.sum()) == man["members"][name]["failed"]
              and [u for u, m in zip(ids, mask) if m] == man["members"][name]["failed_ids"],
              "returned masks[%r] == manifest failed / failed_ids" % name)


def compare(name, old, new, labels, full_stderr=True):
    print("== scenario %s" % name)
    same("exit code", old["rc"], new["rc"])
    same("stdout", old["stdout"], new["stdout"])
    if full_stderr:
        same("stderr", old["stderr"], new["stderr"])
    same("output tree (%d entries)" % len(old["tree"]), old["tree"], new["tree"])
    same("stage-call log (%d calls)" % len(old["calls"]), old["calls"], new["calls"])
    if old["rc"] == 0:
        check(old["returned"] is None and new["returned"] is not None, "only NEW returned a tuple (run_public)")
        check_returned(new, labels)


def main(keep):
    from src.knee_labels import LABELS
    labels = list(LABELS)
    check(not Path("/kaggle").exists(), "host has no /kaggle (scenario d needs the real local environment)")
    check(shutil.which("unshare") is not None, "unshare is available")
    harness = tempfile.TemporaryDirectory(prefix="exp0001_d5_") if keep is None else contextlib.nullcontext(keep)
    with harness as td:
        h = Path(td)
        if keep is not None:
            h.mkdir()   # a new directory, left in place for inspection
        print("== code trees")
        old_code, new_code = build_code_trees(h)
        fx, inp, bin_dir = build_inputs(h, labels)
        runs = {}
        for name, fake, raises, extra in (("a_normal", True, False, 0.0), ("b_reader_raises", True, True, 0.0),
                                          ("c_time_cut", True, False, 8.25 * 3600), ("d_guard", False, False, 0.0)):
            runs[name] = [run_side(h, name, side, code, fx, inp, bin_dir, fake, raises, extra)
                          for side, code in (("old", old_code), ("new", new_code))]

        old, new = runs["a_normal"]
        compare("a_normal", old, new, labels)
        part_b = new["returned"][4]
        check(old["rc"] == 0 and part_b == {"fired": False, "reason": ""} and "reader" in new["returned"][1]
              and b"own leg: blended 3 checkpoints at weight 0.15" in old["stdout"],
              "a: Part B ran (reader member saved, own leg line printed)")
        check(len([c for c in old["calls"] if c["stub"]]) == 4 and {c["call"] for c in old["calls"]} >= {
            "read_stack_inputs", "blend_stack", "blend_reader", "save_members", "sha256_files"},
            "a: 4 stub calls; read_stack_inputs, blend_stack, blend_reader, save_members, sha256_files ran for real")

        old, new = runs["b_reader_raises"]
        compare("b_reader_raises", old, new, labels)
        part_b = new["returned"][4]
        check(part_b["fired"] and "stub run_reader" in part_b["reason"] and b"own leg FAILED" in old["stdout"]
              and b"!!! EXP0001_FALLBACK part_b RuntimeError(" in old["stdout"] and "reader" not in new["returned"][1]
              and old["tree"]["submission.csv"] == old["tree"]["_public_stack.csv"],
              "b: fallback fired, submission.csv == _public_stack.csv, no reader member")

        old, new = runs["c_time_cut"]
        compare("c_time_cut", old, new, labels)
        part_b = new["returned"][4]
        check(part_b["fired"] and part_b["reason"].startswith("time budget:") and b"Part B skipped" in old["stdout"]
              and "_public_stack.csv" not in old["tree"] and not any(c["call"] == "run_reader" for c in old["calls"]),
              "c: Part B skipped by the 8.2 h cut (run_reader never called), reason %r" % part_b["reason"])

        old, new = runs["d_guard"]
        compare("d_guard", old, new, labels, full_stderr=False)
        last = [r["stderr"].decode().strip().splitlines()[-1] for r in (old, new)]
        same("last stderr line", last[0], last[1])
        same("exception", old["exc"], new["exc"])
        check(old["rc"] != 0 and old["exc"]["type"] == "RuntimeError"
              and old["exc"]["message"].startswith("submit_public.py runs only inside a Kaggle kernel")
              and old["tree"] == {} and old["calls"] == [],
              "d: guard raised RuntimeError before any work (empty output dir, no stage calls)")

        print("== check_fingerprint.py on scenario a (OLD stdout as ref, NEW stdout as new)")
        old, new = runs["a_normal"]
        (h / "a_old_stdout.txt").write_bytes(old["stdout"])
        (h / "a_new_stdout.txt").write_bytes(new["stdout"])
        fp = subprocess.run([sys.executable, str(REPO / "exps/exp-0001/check_fingerprint.py"),
                             str(h / "a_old_stdout.txt"), str(h / "a_new_stdout.txt")],
                            capture_output=True, text=True, check=True).stdout
        print(fp.rstrip())
        check("DIFF" not in fp and "missing" not in fp and "False" not in fp, "check_fingerprint: OLD == NEW")

    print("== check_blend_equivalence.py")
    be = subprocess.run([sys.executable, str(REPO / "exps/exp-0001/check_blend_equivalence.py")],
                        capture_output=True, text=True)
    print("\n".join(be.stdout.rstrip().splitlines()[-4:]))
    check(be.returncode == 0 and be.stdout.rstrip().endswith("ALL CHECKS PASSED"),
          "check_blend_equivalence.py passed (rc %d)%s" % (be.returncode, "" if be.returncode == 0 else ": " + be.stderr[-500:]))
    print("ALL CHECKS PASSED")


def check(cond, what):
    if not cond:
        raise SystemExit("FAIL: " + what)
    print("ok  ", what)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2])
    elif sys.argv[1:2] == ["--keep"]:
        main(Path(sys.argv[2]).resolve())
    else:
        assert len(sys.argv) == 1, "usage: check_run_public_equivalence.py [--keep NEW_DIR]"
        main(None)
