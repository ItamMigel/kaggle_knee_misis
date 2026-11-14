"""CPU check of the exp-0001 port against notebook A v7 itself (no GPU, no ctx, no run folder).

    ~/conda/envs/knee/bin/python exps/exp-0001/check_blend_equivalence.py

1. vendored/sha256.json == the notebook's SOURCE_SHA / _PB_SHA; every vendored file hashes to it; the Raptor
   scripts equal the %%writefile bodies of cells 2 / 4; materialize() reproduces each file.
2. The constants of src/public_fork/pipeline.py (submit_public.py's body since D5) equal the notebook's (cells 6, 8, 9).
3. Equivalence: A's cell 8 and A's Part B "end"-mode blend (text extracted from the .ipynb and executed) vs
   stages.read_stack_inputs + blend_stack and stages.blend_reader on the same synthetic files: submission
   values (np.array_equal), CSV bytes, blend receipt and the printed marker lines must be identical.
   Scenarios: 3 studies without failures; 70 studies with a Raptor-384-only, a 224-only and a both-failed row,
   R6-unscored rows (one overlapping each Raptor failure kind), malformed worker rows and tied values
   (70, not ~50: A stops above 3 % failures per Raptor model, and two failures each need n >= 67);
   3 Raptor-384 failures of 70 -> both must raise the same error.
4. save_members on the 70-study scenario: file set, shapes, ranks in (0, 1], manifest fail counts.
Exits non-zero on the first mismatch.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import ast  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.public_fork import stages  # noqa: E402

REPO = Path(_p[0])
NOTEBOOK = (REPO / "data/notebooks/rsna-knee-abnormality-detection/heliosli_this-is-best-public-dataset-new-version-0-9586"
            / "this-is-best-public-dataset-new-version-0-9586.ipynb")
PIPELINE = REPO / "src/public_fork/pipeline.py"
SEED = 0
FROZEN_TIME = 1_000_000.0   # time.time() in both executions, so the receipts' seconds fields are comparable
MARKERS = ("BLEND_FALLBACK", "NOTEBOOK_COMPLETE", "own leg:")
A = {}   # A's blend weights, read from the notebook in main() (section 2 checks pipeline.py has the same)


def cell(nb, i):
    s = nb["cells"][i]["source"]
    return "".join(s) if isinstance(s, list) else s


def literals(src, names):
    """Top-level `NAME = <literal>` (also `A, B = x, y`) of a source text -> {name: value}."""
    found = {}
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id in names:
            found[tgt.id] = ast.literal_eval(node.value)
        elif isinstance(tgt, ast.Tuple) and all(isinstance(e, ast.Name) for e in tgt.elts) \
                and any(e.id in names for e in tgt.elts):
            vals = ast.literal_eval(node.value)
            for e, v in zip(tgt.elts, vals):
                if e.id in names:
                    found[e.id] = v
    missing = set(names) - set(found)
    assert not missing, ("not found", missing)
    return found


def indented_literals(src, names):
    """`NAME = <literal>` lines inside an indented block (cell 9's runner) -> {name: value}."""
    found = {}
    for line in src.splitlines():
        s = line.strip()
        for n in names:
            if s.startswith(n + " = "):
                found[n] = ast.literal_eval(s[len(n) + 3:].split("  #")[0].strip())
    missing = set(names) - set(found)
    assert not missing, ("not found", missing)
    return found


def check(cond, what):
    if not cond:
        raise SystemExit("FAIL: " + what)
    print("ok  ", what)


# ---------------------------------------------------------------- 1. vendored files

def check_vendored(nb, tmp):
    sha = json.loads((stages.VENDORED / "sha256.json").read_text(encoding="utf-8"))
    c6 = literals(cell(nb, 6), ["SOURCE_SHA"])["SOURCE_SHA"]
    pb = None
    for node in ast.parse(cell(nb, 9)).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "_PB_SHA":
            pb = ast.literal_eval(node.value)
    assert pb is not None, "_PB_SHA not found in cell 9"
    check({k[len("compact/"):]: v for k, v in sha.items() if k.startswith("compact/")} == c6,
          "sha256.json compact/* == cell 6 SOURCE_SHA (%d files)" % len(c6))
    check({k[len("reader/"):]: v for k, v in sha.items() if k.startswith("reader/")} == pb,
          "sha256.json reader/* == cell 9 _PB_SHA (%d files)" % len(pb))
    on_disk = sorted(str(p.relative_to(stages.VENDORED))[: -len(".txt")] for p in stages.VENDORED.rglob("*.txt"))
    check(on_disk == sorted(sha), "vendored/*.txt file set == sha256.json keys (%d)" % len(sha))
    for k, v in sha.items():
        assert hashlib.sha256((stages.VENDORED / (k + ".txt")).read_bytes()).hexdigest() == v, k
    check(True, "every vendored file hashes to sha256.json")
    for i, k in ((2, "raptor384_run.py"), (4, "raptor224_run.py")):
        first, body = cell(nb, i).split("\n", 1)
        assert first.startswith("%%writefile /tmp/raptor"), first
        check((stages.VENDORED / (k + ".txt")).read_bytes() == body.encode("utf-8"),
              "%s == cell %d %%%%writefile body" % (k, i))
    d = tmp / "materialized"
    d.mkdir()
    for k in sha:
        p = stages.materialize(k, d)
        assert p == d / Path(k).name and hashlib.sha256(p.read_bytes()).hexdigest() == sha[k], k
    check(True, "materialize() writes every vendored file byte-exact")


# ---------------------------------------------------------------- 2. constants

def check_constants(nb):
    mine = literals(PIPELINE.read_text(encoding="utf-8"),
                    ["ARM", "BUNDLE_ID", "DEADLINE_HOURS", "FLIP_TTA", "PROBE_CLASS", "CODE_SHA", "W_RAPTOR", "W_OURS",
                     "A224", "W_RAPTOR_C", "OWN_W", "OWN_GLOB", "OWN_W_PER_TARGET", "PART_B_CUTOFF_HOURS",
                     "RAPTOR384_CKPT", "RAPTOR224_CKPT"])
    a = literals(cell(nb, 6), ["ARM", "BUNDLE_ID", "DEADLINE_HOURS", "FLIP_TTA", "PROBE_CLASS", "CODE_SHA"])
    a.update(literals(cell(nb, 8), ["W_RAPTOR", "W_OURS", "A224", "W_RAPTOR_C"]))
    a.update(indented_literals(cell(nb, 9), ["OWN_W", "OWN_GLOB", "OWN_W_PER_TARGET", "OWN_MODE", "OWN_P33"]))
    check(a.pop("OWN_MODE") == "end" and a.pop("OWN_P33") == 0.0, 'A: OWN_MODE == "end", OWN_P33 == 0.0 (dead branches)')
    check("> 8.2 * 3600" in cell(nb, 9), "A: Part B cut at 8.2 h")
    a["PART_B_CUTOFF_HOURS"] = 8.2
    for i, k in ((2, "RAPTOR384_CKPT"), (4, "RAPTOR224_CKPT")):
        a[k] = indented_literals(cell(nb, i), ["_weight_name"])["_weight_name"]
    for k in sorted(a):
        assert mine[k] == a[k], (k, mine[k], a[k])
    check(True, "pipeline.py constants == notebook (%d)" % len(a))


# ---------------------------------------------------------------- 3. equivalence

def part_b_block(nb):
    """A's Part B code from reading the stack to writing the final submission (the try body after infer.py)."""
    lines = cell(nb, 9).split("\n")
    start = [i for i, s in enumerate(lines) if s.strip().startswith("_o_stack = _o_pd.read_csv(_o_bak")]
    end = [i for i, s in enumerate(lines) if s == "    except Exception as _o_e:"]
    assert len(start) == 1 and len(end) == 1 and start[0] < end[0], (start, end)
    return textwrap.dedent("\n".join(lines[start[0]:end[0]])) + "\n"


def synth(root, n, r384_fail_rows, r224_fail_rows, r6_unscored, bad_worker_rows, labels, rng):
    """Write test.csv / sample_submission.csv (DATA), the five member CSVs (WORK), worker predictions (pred)."""
    data, work, pred = root / "data", root / "work", root / "pred"
    for d in (data, work, pred / "worker0", pred / "worker1"):
        d.mkdir(parents=True)
    ids = ["1.2.826.0.1.3680043.8.498.%d" % (10 ** 20 + i * 7919) for i in range(n)]
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(data / "test.csv", index=False)
    pd.DataFrame([[ids[0]] + [0.5] * len(labels)], columns=["StudyInstanceUID"] + labels).to_csv(
        data / "sample_submission.csv", index=False)
    order = list(rng.permutation(n))   # members come in their own row order: A reindexes them to test.csv

    def frame(values):
        df = pd.DataFrame(values, columns=labels)
        df.insert(0, "StudyInstanceUID", ids)
        return df.iloc[order].reset_index(drop=True)

    def raptor(stem, fail_rows):
        raw = rng.random((n, len(labels))).astype(np.float32)
        raw[list(fail_rows)] = 0.5                       # a failed Raptor study stays 0.5 in every column
        sub = pd.DataFrame(raw).rank(method="average", pct=True).to_numpy()   # its rank-mean submission
        frame(raw).to_csv(work / (stem + "_raw.csv"), index=False)
        frame(sub).to_csv(work / (stem + "_sub.csv"), index=False)

    raptor("raptor", r384_fail_rows)
    raptor("raptor224", r224_fail_rows)
    ours = np.round(rng.random((n, len(labels))), 2)   # rounded: ties for the average rank
    scored = [i for i in range(n) if i not in r6_unscored]
    ours[list(r6_unscored)] = ours[scored].mean(axis=0)   # the controller fills an unscored study with the column mean
    frame(ours).to_csv(work / "ours_sub.csv", index=False)
    for w in (0, 1):
        with open(pred / ("worker%d" % w) / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["StudyInstanceUID"] + labels)
            for i in scored[w::2]:
                wr.writerow([ids[i]] + ["%.9g" % x for x in ours[i]])
            for i, kind in bad_worker_rows.items():
                if i % 2 == w:   # a row the worker wrote but did not score: empty field / out of [0, 1]
                    wr.writerow([ids[i]] + (["", *["0.3"] * (len(labels) - 1)] if kind == "empty"
                                            else ["1.5", *["0.3"] * (len(labels) - 1)]))
    own = rng.random((n, len(labels)))
    pd.DataFrame(own, columns=labels).assign(StudyInstanceUID=ids)[["StudyInstanceUID"] + labels] \
        .iloc[rng.permutation(n)].to_csv(work / "_own.csv", index=False)
    return data, work, pred


def run_a(nb, data, work, pred, part_b):
    """Execute A's cell 8 (and the Part B block) as text with the globals the notebook had."""
    out = io.StringIO()
    g8 = {"DATA": data, "WORK": work, "pred": pred, "RAPTOR_SECONDS": 50.4, "RAPTOR224_SECONDS": 22.2,
          "NB_T0": FROZEN_TIME - 443.0, "json": json, "time": time, "display": lambda x: None}
    with contextlib.redirect_stdout(out):
        exec(compile(cell(nb, 8), "A-cell-8", "exec"), g8)
    res = {"sub": g8["sub"], "blend": g8["blend"], "sub_csv": (work / "submission.csv").read_bytes(),
           "receipt": (work / "blend_receipt.json").read_text()}
    if part_b:
        shutil.copy(work / "submission.csv", work / "_public_stack.csv")
        cfg = indented_literals(cell(nb, 9), ["OWN_W", "OWN_GLOB", "OWN_MODE", "OWN_W_PER_TARGET", "OWN_P33"])
        g9 = dict(cfg, _o_pd=pd, _o_time=time, _o_t0=FROZEN_TIME - 24.0, _o_ck=["a.pt", "b.pt", "c.pt"],
                  _o_pub=str(work / "submission.csv"), _o_bak=str(work / "_public_stack.csv"))
        code = part_b_block(nb).replace('"/kaggle/working/', '"%s/' % work)
        assert "/kaggle/" not in code, "unpatched Kaggle path in the Part B block"
        with contextlib.redirect_stdout(out):
            exec(compile(code, "A-cell-9-partB", "exec"), g9)
        res["final"] = g9["_o_final"]
        res["final_csv"] = (work / "submission.csv").read_bytes()
    res["markers"] = [s for s in out.getvalue().splitlines() if s.startswith(MARKERS)]
    return res


def run_port(data, work, pred, part_b):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        test_ids, cols, labels, inputs = stages.read_stack_inputs(data, work)
        sub, blend, masks = stages.blend_stack(test_ids, labels, cols, inputs, pred, A["W_RAPTOR"], A["W_OURS"],
                                               A["W_RAPTOR_C"], A["A224"], 50.4, 22.2, FROZEN_TIME - 443.0)
    res = {"sub": sub, "blend": blend, "masks": masks, "sub_csv": sub.to_csv(index=False).encode("utf-8"),
           "receipt": json.dumps(blend, indent=1), "inputs": inputs, "test_ids": test_ids, "labels": labels}
    if part_b:
        stack_csv = pd.read_csv(io.BytesIO(res["sub_csv"]), dtype={"StudyInstanceUID": str})
        own = pd.read_csv(work / "_own.csv", dtype={"StudyInstanceUID": str})
        with contextlib.redirect_stdout(out):
            final = stages.blend_reader(stack_csv, own, labels, A["OWN_W"], A["OWN_W_PER_TARGET"], 3, FROZEN_TIME - 24.0)
        res["final"] = final
        res["final_csv"] = final.to_csv(index=False).encode("utf-8")
        res["own"] = own
    res["markers"] = [s for s in out.getvalue().splitlines() if s.startswith(MARKERS)]
    return res


def same_frame(a, b):
    return (list(a.columns) == list(b.columns) and a["StudyInstanceUID"].tolist() == b["StudyInstanceUID"].tolist()
            and np.array_equal(a.drop(columns="StudyInstanceUID").to_numpy(), b.drop(columns="StudyInstanceUID").to_numpy()))


def scenario(nb, tmp, name, n, r384, r224, unscored, bad, labels, seed):
    root = tmp / name
    rng = np.random.default_rng(seed)
    data, work, pred = synth(root, n, r384, r224, unscored, bad, labels, rng)
    with mock.patch.object(time, "time", return_value=FROZEN_TIME):
        a = run_a(nb, data, work, pred, part_b=True)
        p = run_port(data, work, pred, part_b=True)
    check(same_frame(a["sub"], p["sub"]), "%s: stack submission values identical (np.array_equal)" % name)
    check(a["sub_csv"] == p["sub_csv"], "%s: stack submission.csv bytes identical" % name)
    check(a["blend"] == p["blend"] and a["receipt"] == p["receipt"], "%s: blend receipt identical" % name)
    check(same_frame(a["final"], p["final"]), "%s: final (reader) submission values identical (np.array_equal)" % name)
    check(a["final_csv"] == p["final_csv"], "%s: final submission.csv bytes identical" % name)
    check(a["markers"] == p["markers"] and len(a["markers"]) >= 3,
          "%s: printed markers identical (%d lines)" % (name, len(a["markers"])))
    b = p["blend"]
    print("     receipt: rows %d raptor384_failed %d raptor224_failed %d raptor_failed %d ours_failed %d both_failed %d"
          % (b["rows"], b["raptor384_failed"], b["raptor224_failed"], b["raptor_failed"], b["ours_failed"], b["both_failed"]))
    return root, p


def check_raises(nb, tmp, labels):
    root = tmp / "too_many_384_failures"
    data, work, pred = synth(root, 70, [3, 9, 27], [], [], {}, labels, np.random.default_rng(SEED + 7))
    errs = []
    for fn in (lambda: run_a(nb, data, work, pred, part_b=False), lambda: run_port(data, work, pred, part_b=False)):
        try:
            with mock.patch.object(time, "time", return_value=FROZEN_TIME):
                fn()
            errs.append(None)
        except RuntimeError as e:
            errs.append(str(e))
    check(errs[0] is not None and errs[0] == errs[1], "3 Raptor-384 failures of 70: both raise %r" % errs[0])


# ---------------------------------------------------------------- 4. save_members

def check_save_members(tmp, p):
    test_ids, labels, inputs, masks = p["test_ids"], p["labels"], p["inputs"], p["masks"]

    def aligned(df):
        return df.set_index("StudyInstanceUID").loc[test_ids, labels].reset_index()

    zeros = np.zeros(len(test_ids), dtype=bool)
    members = {"raptor384": aligned(inputs["raptor_raw"]), "raptor224": aligned(inputs["raptor224_raw"]),
               "r6": aligned(inputs["ours"]), "reader": aligned(p["own"]), "stack": p["sub"], "final": p["final"]}
    fm = {"raptor384": masks["raptor384"], "raptor224": masks["raptor224"], "r6": masks["r6"],
          "reader": zeros, "stack": zeros, "final": zeros}
    out = tmp / "members_out"
    stages.save_members(out, members, fm, {"a224": A["A224"], "reader_per_target": A["OWN_W_PER_TARGET"]})
    d = out / "members"
    expected = {f"{m}_{k}.csv" for m in members for k in ("prob", "rank")} | {"manifest.json"}
    check({p_.name for p_ in d.iterdir()} == expected, "save_members: file set (%d files)" % len(expected))
    for m in members:
        prob = pd.read_csv(d / f"{m}_prob.csv", dtype={"StudyInstanceUID": str})
        rank = pd.read_csv(d / f"{m}_rank.csv", dtype={"StudyInstanceUID": str})
        assert prob.shape == rank.shape == (len(test_ids), 1 + len(labels)), (m, prob.shape, rank.shape)
        assert prob["StudyInstanceUID"].tolist() == rank["StudyInstanceUID"].tolist() == test_ids, m
        r = rank[labels].to_numpy()
        assert np.isfinite(r).all() and (r > 0).all() and (r <= 1).all(), m
    check(True, "save_members: shapes (n, 13), row order = test.csv, ranks in (0, 1]")
    man = json.loads((d / "manifest.json").read_text())
    check(man["rows"] == len(test_ids) and all(man["members"][m]["failed"] == int(fm[m].sum()) for m in members)
          and man["weights"]["a224"] == A["A224"],
          "save_members: manifest rows / fail counts %s" % {m: man["members"][m]["failed"] for m in members})


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    A.update(literals(cell(nb, 8), ["W_RAPTOR", "W_OURS", "A224", "W_RAPTOR_C"]))
    A.update(indented_literals(cell(nb, 9), ["OWN_W", "OWN_W_PER_TARGET"]))
    labels = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
              "Synovitis", "Baker's", "Contusion", "Fracture"]
    with tempfile.TemporaryDirectory(prefix="exp0001_check_") as td:
        tmp = Path(td)
        print("== 1. vendored files")
        check_vendored(nb, tmp)
        print("== 2. constants")
        check_constants(nb)
        print("== 3. equivalence with A's cell 8 / Part B")
        scenario(nb, tmp, "visible_3", 3, [], [], [], {}, labels, SEED)
        # rows: 5 = 384-only fail, 11 = 224-only fail, 17 = both; R6 unscored 5, 17, 30, 44 (+ malformed 52, 61)
        _, p70 = scenario(nb, tmp, "failures_70", 70, [5, 17], [11, 17], [5, 17, 30, 44, 52, 61],
                          {52: "empty", 61: "out_of_range"}, labels, SEED + 1)
        check(p70["blend"]["raptor384_failed"] == 2 and p70["blend"]["raptor224_failed"] == 2
              and p70["blend"]["raptor_failed"] == 1 and p70["blend"]["ours_failed"] == 6 and p70["blend"]["both_failed"] == 1,
              "failures_70: every failure kind was exercised")
        check_raises(nb, tmp, labels)
        print("== 4. save_members")
        check_save_members(tmp, p70)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
