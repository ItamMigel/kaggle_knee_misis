"""exp-0001: compare a kernel log with the A v7 reference log (markers, tables, own-leg lines; timings ignored).

    ~/conda/envs/knee/bin/python exps/exp-0001/check_fingerprint.py exps/exp-0001/reference/a_v7_kernel_log.txt exps/exp-0001/runs/<run_id>/log.txt
"""
import json, re, sys
ref = open(sys.argv[1]).read().splitlines()
new = [re.sub(r"^\[(stdout|stderr)\] ?", "", l) for l in open(sys.argv[2]).read().splitlines()]
def marker(lines, name):
    out = [json.loads(l[len(name) + 1:]) for l in lines if l.startswith(name + " ")]
    return out[0] if out else None
TIME = re.compile(r"second|time|_s$|decode_s|render_s|gpu_s")
def cmp(a, b, path=""):
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if TIME.search(k):
                continue
            if k not in a or k not in b:
                bad.append(f"{path}.{k}: only in {'ref' if k in a else 'new'}")
            else:
                bad += cmp(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): bad.append(f"{path}: len {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)): bad += cmp(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > 1e-6: bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")
    return bad
for name in ["CONTROLLER_DONE", "OURS_STAGE_COMPLETE", "NOTEBOOK_COMPLETE"]:
    a, b = marker(ref, name), marker(new, name)
    d = cmp(a, b) if a is not None and b is not None else [f"missing ref={a is None} new={b is None}"]

    print(name, "IDENTICAL (non-timing fields, tol 1e-6)" if not d else "DIFF:\n  " + "\n  ".join(d))
# tables
def tables(lines):
    rows = [re.sub(r"\s+", " ", l.strip()) for l in lines if re.match(r"^\s*1\.2\.826\.\S+(\s+[-0-9.e]+){12}\s*$", l)]
    return rows
tr, tn = tables(ref), tables(new)
print("table rows ref", len(tr), "new", len(tn))
uniq_r = list(dict.fromkeys(tr)); uniq_n = list(dict.fromkeys(tn))
print("unique table rows equal:", uniq_r == uniq_n, len(uniq_r), len(uniq_n))
if uniq_r != uniq_n:
    for x in uniq_r:
        if x not in uniq_n: print(" only ref:", x)
    for x in uniq_n:
        if x not in uniq_r: print(" only new:", x)
# own leg lines without timings
o = lambda ls: [re.sub(r"in \d+s", "in Ns", l) for l in ls if l.startswith("own leg")]
print("own leg lines equal:", o(ref) == o(new))
if o(ref) != o(new):
    print(" ref:", o(ref)); print(" new:", o(new))
