"""B5's reference: every member's probabilities on the visible test studies, computed locally as the kernel's member does.

    CUDA_VISIBLE_DEVICES=0 nice -n 19 ~/conda/envs/knee/bin/python exps/exp-0004/member_ref.py

Builds the visible-test cache from the competition's test_series (build_dir_cache, into ctx.output_dir/cache)
and runs each member's predict_member on DEVICE over all its studies in own_part_main's order (csv_studies):
the dtype is predict_member's own (fp16 on cuda), the same function the kernel calls. Writes member_ref.json
({member: {study: [12 probs in LABELS order]}}: the kernel spec's member_expected) and member_ref_ckpts.json
(source, path and sha256 of every checkpoint, in load order), and prints both.
CONFIG picks the members' checkpoints from CONFIGS: "H" = variant H (each member = its 2-seed average, the 5 folds
of two weights datasets, = submit_blend.py's MEMBER_CKPTS), "seed0" = the two-member probe's 5-checkpoint members
(its MEMBER_EXPECTED, run 20261018t034818-649-feac, came from these bytes in this order, then read from exp-0004's
training-run dirs), "smoke" = exp-0004's in22k smoke checkpoint. A member loads its sources in the listed order,
each source's files in the listed order (= submit_blend.py's: seed 0 folds, then seed 1 folds). Every source of
every member is checked before the cache is built: a None source raises (a dataset not published yet), a
weights dataset's files must match its own SHA256SUMS.txt (exactly those files). Pull every weights dataset of
CONFIG before: uv run --project kaggle python kaggle/pull_data.py --ds <slug>.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import importlib  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

from src.blend_kernel import _file_sha256, csv_studies  # noqa: E402
from src.infer25d import build_dir_cache  # noqa: E402
from src.knee_labels import LABELS  # noqa: E402
from utils.ctx import get_ctx  # noqa: E402

FOLDS = [f"fold{k}.pt" for k in range(5)]
# weights datasets (fold0..4.pt + SHA256SUMS.txt each) = submit_blend.py's MEMBER_CKPTS
DS4_S0 = "bebra3222/rsna-knee-exp-0004-w-8f67ee3d"  # copies of training runs 20261017t150919-625-2918 (folds 0-2),
#                                                     20261017t214233-033-af68 (3), 20261018t000939-333-228f (4)
DS4_S1 = "bebra3222/rsna-knee-exp-0004-w-8446e076"  # copies of runs 20261018t023618-204-1085 (folds 0-2), 20261018t045524-429-281f (3-4)
DS5_S0, DS5_S1 = "bebra3222/rsna-knee-exp-0005-w-109a48bc", "bebra3222/rsna-knee-exp-0005-w-e2a42938"
SMOKE_RUN = "20261017t134730-587-6974"  # exp-0004's in22k smoke run: one fold0.pt, no SHA256SUMS.txt
MODULES = {"exp-0004": "src.infer25d", "exp-0005": "src.joint_reader"}
# {config: {member: [(mount key, files, sums)]}}: sums True = a weights dataset, its SHA256SUMS.txt must list exactly
# these files with their sha256; False = a run dir (sha256 recorded only)
CONFIGS = {"H": {"exp-0004": [(DS4_S0, FOLDS, True), (DS4_S1, FOLDS, True)],
                 "exp-0005": [(DS5_S0, FOLDS, True), (DS5_S1, FOLDS, True)]},
           "seed0": {"exp-0004": [(DS4_S0, FOLDS, True)], "exp-0005": [(DS5_S0, FOLDS, True)]},
           "smoke": {"exp-0004": [(SMOKE_RUN, ["fold0.pt"], False)], "exp-0005": [(DS5_S0, FOLDS, True)]}}
CONFIG = "H"
DEVICE = "cuda"
WORKERS = 4  # build_dir_cache processes (the kernel's 4 CPUs)


def read_sums(path):
    """{file name: sha256} of a sha256sum-format SHA256SUMS.txt."""
    return {name: sha for sha, name in (line.split() for line in Path(path).read_text().splitlines() if line.strip())}


def checked_ckpts(member, sources, mounts):
    """[(mount key, path, sha256)] of a member's checkpoints in load order; a dataset off its SHA256SUMS.txt raises."""
    out = []
    for key, files, sums in sources:
        root = mounts[key]
        got = {f: _file_sha256(root / f) for f in files}
        if sums:
            want = read_sums(root / "SHA256SUMS.txt")
            if got != want:
                raise ValueError(f"member {member}: {key}'s checkpoints' sha256 {got} != its SHA256SUMS.txt {want}")
        out += [(key, root / f, got[f]) for f in files]
    return out


if CONFIG not in CONFIGS or set(CONFIGS[CONFIG]) != set(MODULES):
    raise ValueError(f"CONFIG {CONFIG!r} must be one of {list(CONFIGS)}, each naming exactly the members "
                     f"{list(MODULES)}")
for m, sources in CONFIGS[CONFIG].items():
    keys = [key for key, _, _ in sources]
    if None in keys:
        raise ValueError(f"member {m} of CONFIG {CONFIG!r} has a None source {keys}: fill it (DS4_S1 = exp-0004 seed "
                         "1's weights dataset slug, once published)")
    if len(set(keys)) != len(keys):
        raise ValueError(f"member {m} of CONFIG {CONFIG!r} lists a source twice: {keys}")

ctx = get_ctx()
ckpts = {m: checked_ckpts(m, CONFIGS[CONFIG][m], ctx.mounts) for m in MODULES}
print(f"[member_ref] CONFIG {CONFIG}: sha256 ok for {({m: len(v) for m, v in ckpts.items()})} checkpoints", flush=True)
comp = ctx.mounts["rsna-knee-abnormality-detection"]
cache = build_dir_cache(comp / "test_series", comp / "test_series.csv", ctx.output_dir / "cache", WORKERS)
comp_series = pd.read_csv(comp / "test_series.csv")
studies = csv_studies(comp_series)

ref = {}
for m, module in MODULES.items():
    probs = importlib.import_module(module).predict_member(studies, cache, comp_series, [p for _, p, _ in ckpts[m]],
                                                           DEVICE)
    ref[m] = {st: [float(v) for v in probs.loc[st, LABELS]] for st in studies}
shas = {m: [{"source": key, "path": str(p), "sha256": sha} for key, p, sha in v] for m, v in ckpts.items()}

(ctx.output_dir / "member_ref.json").write_text(json.dumps(ref, indent=1))
(ctx.output_dir / "member_ref_ckpts.json").write_text(json.dumps(shas, indent=1))
print(json.dumps(shas))
print(json.dumps(ref))
print(f"[member_ref] CONFIG {CONFIG} -> {ctx.output_dir}")
