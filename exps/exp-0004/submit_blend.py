"""exp-0004 blend kernel: our members (exp-0004, exp-0005) in a child process + component A (run_public) -> per-target rank blend;
OWN_ONLY: our members alone, run_public skipped.

    uv run --project kaggle python kaggle/run_kernel.py --exp exp-0004 --script submit_blend --accelerator t4 \
      --docker-image gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f \
      --ds goodpjw2008/rsna-knee-2-5d-convnext-reader,heliosli/rsna-knee-r6-frozen-fe055406-w1,nartaa/rsna-knee-publication-swa-weights-20261007,heliosli/rsna-orthofoundation-fold0-epoch7-20261004,bebra3222/rsna-knee-wheels-cp313,bebra3222/rsna-knee-exp-0004-w-8f67ee3d,bebra3222/rsna-knee-exp-0004-w-8446e076,bebra3222/rsna-knee-exp-0005-w-109a48bc,bebra3222/rsna-knee-exp-0005-w-e2a42938 \
      [--submit-file submission.csv]

Body: src/blend_kernel.blend_main (B6). Checkpoint sha256 and cwd guards -> mode (validation on the 3 visible
test studies, hidden otherwise) -> our part in a child process (PINS wheels into SITE_DIR; per CHUNK_STUDIES
studies: test cache in CACHE_DIR -> every member of MEMBER_MODULES at once, each in its own child on its own GPU
(MEMBER_GPUS; None = in turn) -> cache deleted, so the disk peak is one chunk's cache; then parity vs
PARITY_EXPECTED and check vs MEMBER_EXPECTED)
-> pipeline.run_public (A v7, untouched) -> rank blend with WEIGHTS -> submission.csv; with OWN_ONLY no run_public:
rank_own of our members with WEIGHTS summing to 1 over test.csv's studies -> submission.csv. Validation: any failure of
ours raises. Hidden: raises too with HIDDEN_FALLBACK False (LB probes); with True (final candidates) it prints
EXP0004_FALLBACK and A's final submission stays. Log: EXP0004_* lines, OWN_* from
the child, A's EXP0001_* lines.
"""

# --- бутстрап: найти utils/. Константный текст, одинаковый во всех скриптах ---
import glob
import sys
from pathlib import Path

_p = [str(p) for p in Path(__file__).resolve().parents if (p / "utils" / "ctx.py").exists()]
_p += [g[: -len("/utils/ctx.py")] for g in glob.glob("/kaggle/input/datasets/*/*/utils/ctx.py")]
sys.path[:0] = _p[:1]
# --- конец бутстрапа ---

import time  # noqa: E402

T0 = time.time()   # D7: A's deadlines (R6 8.3 h, Part B 8.2 h) count from the kernel start, our part included

import json  # noqa: E402

from utils.ctx import get_ctx  # noqa: E402

ctx = get_ctx()

from src.blend_kernel import blend_main  # noqa: E402
from src.public_fork import pipeline  # noqa: E402

# our members, in this order in every chunk (logs, reports); variant H (lead 18.10): each member = its 2-seed
# average, the 5 folds of two weights datasets (fold{k}.pt + SHA256SUMS.txt each)
MEMBER_MODULES = {"exp-0004": "src.infer25d", "exp-0005": "src.joint_reader"}
# {member: [(weights dataset, {"fold{k}.pt": sha256} = its SHA256SUMS.txt)]}; <sha8> = sha256(SHA256SUMS.txt)[:8].
# exp-0004 seed 0: fold0..4.pt of runs 20261017t150919-625-2918 (0-2), 20261017t214233-033-af68 (3),
# 20261018t000939-333-228f (4); exp-0004 seed 1: runs 20261018t023618-204-1085 (folds 0-2; record failed = SIGINT
# after fold 2) + 20261018t045524-429-281f (3-4). exp-0005 seed 0: run 20261017t135225-664-aecb, git 8fb6668e;
# exp-0005 seed 1: run 20261018t003924-017-1d75
MEMBER_CKPTS = {
    "exp-0004": [("bebra3222/rsna-knee-exp-0004-w-8f67ee3d",
                  {"fold0.pt": "c7071ece3734bc4f6c92a9b23b1d88051cc04e8c9ccbccceda0776af2f45d5c7",
                   "fold1.pt": "d87264cb6444f059ea33ac6454e0e9f44fc3231dc11f9aa2ec36771036ab3f06",
                   "fold2.pt": "aa469496d2f941bbae4d0dfb63a6a1295d943527fe2eca21080b3d21cc60a3ec",
                   "fold3.pt": "d5f2323130dfdfcaaae4d0fd8cc72e5cb063cf6f5dc483dc2f27536b29168ece",
                   "fold4.pt": "5a5875680775c44741844d83803506c4d4b49b66512ab2e000bc59fac6beaa4e"}),
                 ("bebra3222/rsna-knee-exp-0004-w-8446e076",
                  {"fold0.pt": "3d0b0d05a18ba9aef86cb1d1846f58de2b3cbfead55da087d1ad57c39fd0c7aa",
                   "fold1.pt": "01d2fc7b335a6024bc5531082d7e2d37b5a7664442ab59d5668c1302cbdefb8f",
                   "fold2.pt": "782ec1a4d979e57adf6963a4607b19d7129668475bb461c9a183e6e8c902417d",
                   "fold3.pt": "4dfb7c4892bd787bc2e9ecae0b13d563e321b0423b00c6c422f9cd0fbf1ab432",
                   "fold4.pt": "1295b81d34a16747aca1d1496ec55f072f72397b215a5e24f0f071d5430ac2d4"})],
    "exp-0005": [("bebra3222/rsna-knee-exp-0005-w-109a48bc",
                  {"fold0.pt": "e0100e381741e7a5457d492957f68d7cbe9b7b6b137a09f33613c954ae39db9a",
                   "fold1.pt": "4e949a1272a98c2ccb8cb46908a2cf55f261a28d873ca6049b94f9fdece549f4",
                   "fold2.pt": "857caf4b91ec8c8b2f3e921a9f3f3fdab9cf4aedef774af74dae430ff7953ad4",
                   "fold3.pt": "9763c9122d1268bdcb9bf3e2c299853de79e663d4439c9c332b69a7bf9ff2f9c",
                   "fold4.pt": "24fa76d42f58807dba5f63f1e8834e4738d76a686836bf93f43d30c6c3e346a2"}),
                 ("bebra3222/rsna-knee-exp-0005-w-e2a42938",
                  {"fold0.pt": "2d1a63a9f48c93f23128a5b57c5aa91b9abe106b836e17c5e7d355bc72de5b6c",
                   "fold1.pt": "15cab94ae909b0bbf809c78098a673b1ebf089f8f40c46563eb94a52d325dcd0",
                   "fold2.pt": "7bb73c97e298eea65dc2b982817eac471275ff59f243e250b61f52da54d007b1",
                   "fold3.pt": "e0e93cd89f7fe2712c531901deee3d91ccb4512d3a200799af9331053a553260",
                   "fold4.pt": "f7e5575a8c12f160391c742655daa046b0ffec73ef7a73c1e9ba47ca69b2f0e6"})],
}
# T4 x2: one GPU per member, both at once (H on one T4 in turn: 7911 s > D7's 7200, both members GPU-bound, timing
# run 20261018t052751-433-0d9c); None = in turn on one GPU (the probes' path)
MEMBER_GPUS = {"exp-0004": 0, "exp-0005": 1}
N_FOLDS = 5
# visible test: the 3 studies of the competition's test.csv
VISIBLE_STUDIES = ["1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191",
                   "1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776",
                   "1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402"]
# volume_fingerprint of the 15 visible test series: parity_ref.py run 20261017t120738-561-0468
PARITY_EXPECTED = json.loads(r"""[{"series":"1.2.826.0.1.3680043.8.498.11580656442259111255675562605155903947","fp":{"shape":[34,375,375],"sha_vol":"c4457295bc2f1860","sha_pos":"67e4e2ab1bb79d11","sum":368441360,"lo":2.9319445800781248,"hi":2090.8626831054685}},{"series":"1.2.826.0.1.3680043.8.498.17811502614030631664517622518906646132","fp":{"shape":[30,375,375],"sha_vol":"7fd5a12a4cd8e25c","sha_pos":"93a2cb77c1a92d6d","sum":345524607,"lo":3.41015625,"hi":1376.033203125}},{"series":"1.2.826.0.1.3680043.8.498.30565395595045942404081022062489758495","fp":{"shape":[30,375,375],"sha_vol":"24b43404ae4ef526","sha_pos":"08f7395a2a3c8bce","sum":375753594,"lo":6.120652856826782,"hi":2226.6173461914054}},{"series":"1.2.826.0.1.3680043.8.498.32856494541816845805947850304653662081","fp":{"shape":[30,375,375],"sha_vol":"8a6cc2e34f6dd0dc","sha_pos":"c01ac7bc9fdbffc7","sum":287809716,"lo":1.0344692993164062,"hi":458.2434936523432}},{"series":"1.2.826.0.1.3680043.8.498.44334485654554877495800965096767429689","fp":{"shape":[160,400,800],"sha_vol":"6b12ba8e8a21658d","sha_pos":"bb9012c3d0d0979c","sum":1386525482,"lo":0.0,"hi":253.765625}},{"series":"1.2.826.0.1.3680043.8.498.10543708582570884867685900269888836671","fp":{"shape":[24,450,450],"sha_vol":"96be98367570256c","sha_pos":"a29d3c2566ca5038","sum":277759769,"lo":-17.5576171875,"hi":5909.636879882809}},{"series":"1.2.826.0.1.3680043.8.498.10818107180155454609278933308651876447","fp":{"shape":[33,375,375],"sha_vol":"df2fe1cfe24e0f29","sha_pos":"e618015f99f75b3f","sum":185178554,"lo":-0.7348284912109375,"hi":3889.8841894531233}},{"series":"1.2.826.0.1.3680043.8.498.34851887739080902195133258975769749352","fp":{"shape":[24,450,450],"sha_vol":"f90a7fbcb0ddd208","sha_pos":"a29d3c2566ca5038","sum":205284587,"lo":-6.3341677856445315,"hi":4368.035205078124}},{"series":"1.2.826.0.1.3680043.8.498.49057071886338907232646925625200929717","fp":{"shape":[24,450,450],"sha_vol":"869dd8d40815d947","sha_pos":"a29d3c2566ca5038","sum":298966235,"lo":-7.626600444316865,"hi":7718.3652783203115}},{"series":"1.2.826.0.1.3680043.8.498.49397095733611495516760005725956804191","fp":{"shape":[30,500,400],"sha_vol":"6711492cb4d55a67","sha_pos":"d2d78c39ac9f84f4","sum":265144398,"lo":0.0,"hi":3983.4005566406213}},{"series":"1.2.826.0.1.3680043.8.498.10927312244766790279817592893017048392","fp":{"shape":[24,425,425],"sha_vol":"8225b92ab15a7083","sha_pos":"8ca850c6aaa26d99","sum":484910035,"lo":0.0,"hi":1861.0731622314443}},{"series":"1.2.826.0.1.3680043.8.498.12207733879434320501816063789123909030","fp":{"shape":[24,425,425],"sha_vol":"592fac6f7d8434a1","sha_pos":"8ca850c6aaa26d99","sum":333085423,"lo":0.0,"hi":752.7015048217768}},{"series":"1.2.826.0.1.3680043.8.498.15858543841290540180647614348005539746","fp":{"shape":[40,425,425],"sha_vol":"9bf42f0f9321a962","sha_pos":"858413852242ebc1","sum":350165666,"lo":0.0,"hi":1141.4560821533196}},{"series":"1.2.826.0.1.3680043.8.498.19236602294493686533219656323303746544","fp":{"shape":[25,450,450],"sha_vol":"3a7ff6c6fb31fad0","sha_pos":"0d2200bb0a7599df","sum":391749146,"lo":0.0,"hi":1213.1723657226562}},{"series":"1.2.826.0.1.3680043.8.498.66069370990398014317043644514003034312","fp":{"shape":[25,450,450],"sha_vol":"355eb5ee637200af","sha_pos":"0d2200bb0a7599df","sum":333545594,"lo":0.0,"hi":678.3313427734365}}]""")
# member_ref.py's member_ref.json, all members from one run (L40, fp16, the MEMBER_CKPTS checkpoints by sha256):
# run 20261018t092503-788-ef6a (variant H: 10 checkpoints per member)
MEMBER_EXPECTED = json.loads(r"""{"exp-0004":{"1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191":[0.04867515712976456,0.1113119125366211,0.9147888422012329,0.12905137240886688,0.8007418513298035,0.26132380962371826,0.4025402069091797,0.5433321595191956,0.282114177942276,0.9530985951423645,0.11700116842985153,0.1227363720536232],"1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776":[0.5746750235557556,0.19956639409065247,0.8451544642448425,0.19212134182453156,0.7631039023399353,0.5771552920341492,0.691726565361023,0.4941844940185547,0.6313955187797546,0.3438168466091156,0.5483414530754089,0.35067272186279297],"1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402":[0.026460746303200722,0.09149554371833801,0.2505183219909668,0.13797567784786224,0.7531742453575134,0.5917847752571106,0.9553059935569763,0.28771209716796875,0.20073798298835754,0.12790098786354065,0.14488087594509125,0.1330348402261734]},"exp-0005":{"1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191":[0.02801366150379181,0.06942478567361832,0.7261844277381897,0.08935003727674484,0.6788792610168457,0.26094189286231995,0.4816475808620453,0.4570194184780121,0.2574915587902069,0.9618465304374695,0.1792157143354416,0.1173609122633934],"1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776":[0.7109455466270447,0.24785716831684113,0.5539120435714722,0.15919774770736694,0.5805604457855225,0.4515198767185211,0.6290048360824585,0.600281298160553,0.6910207867622375,0.3087496757507324,0.6049540042877197,0.39333367347717285],"1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402":[0.02707759104669094,0.03056742623448372,0.2189107984304428,0.09810661524534225,0.6599557995796204,0.4175582528114319,0.8857169151306152,0.23749864101409912,0.1961708813905716,0.12128450721502304,0.09228668361902237,0.10098631680011749]}}""")
# lead 18.10 13:27 (probe 1, A + exp-0004 s0 at 0.2, scored 0.956 < A's 0.958): an own-only LB probe, our real
# level. True: run_public is skipped and submission.csv = our members alone (needs HIDDEN_FALLBACK False); probed as
# kernel run 20261018t103420-654-0b64 (ref 57351808). False (lead 18.10 14:30): A + own H at OWN_TOTAL
OWN_ONLY = False
# D8 (lead 18.10): own split 0.6 : 0.4 from d8_oof run 20261018t025721-274-46f2; OWN_ONLY: the split alone, summing to 1
OWN_SPLIT = {"exp-0004": 0.6, "exp-0005": 0.4}
# blend: the own part's total weight, public gets 1 - it. D8 safe 0.2 (bold 0.3) cost ~0.002 on probe 1; lead 18.10
# 13:50: 0.05, an 'A + a small real improvement' probe read from the rank inside the 0.958 block
OWN_TOTAL = 0.05
WEIGHTS = OWN_SPLIT if OWN_ONLY else {m: round(OWN_TOTAL * w, 6) for m, w in OWN_SPLIT.items()}
# fp16 T4 vs fp16 L40; fp16 vs fp32 moved probs by max 0.013 (exp-0004, D9) / 0.014 (exp-0005 precision_check),
# real bugs by ~0.1+
MEMBER_TOL = 0.02
WHEELS_DS = "bebra3222/rsna-knee-wheels-cp313"
PINS = {"pydicom": "3.0.2", "pylibjpeg": "2.1.0", "pylibjpeg-libjpeg": "2.4.0", "pylibjpeg-openjpeg": "2.6.0",
        "python-gdcm": "3.2.6"}  # as kaggle_parity.py; no opencv: the image's cv2 is parity-verified
SITE_DIR = Path("/kaggle/tmp/exp0004_site")
CACHE_DIR = Path("/kaggle/tmp/exp0004_cache")
DEVICE, WORKERS = "cuda", 4
CHUNK_STUDIES = 150  # ~800 series: cache peak ~6-7 GB at ~8 MB/series, whatever the hidden test size
# D7 for two members (lead 18.10): 2.5 h for our whole part, the binding total; valid only if the two-member
# projection from the first validation kernel is <= 2.0 h
OWN_BUDGET_S = 9000
# watchdogs inside the child: cache and members per chunk, the rest one-off. 150 studies, H timing run
# 20261018t052751-433-0d9c (one T4, members in turn, 10 checkpoints each): cache 165 s, exp-0004 467 s, exp-0005
# 299 s per steady chunk; in parallel each member has its own GPU but they share the 4 CPUs
BUDGETS = {"env": 300, "cache": 900, "parity": 300, "member:exp-0004": 900, "member:exp-0005": 900,
           "member_check": 120}
N_HIDDEN_STUDIES = 1300  # the validation run's projection
# D6 amended (lead 17.10): the hidden rerun's log is invisible. False = any own/blend failure there errors the
# submission, so an LB probe's score is always our blend; True only for final candidates, as a new kernel version
# submitted after its False twin scored.
HIDDEN_FALLBACK = False

for _name, _d in (("MEMBER_CKPTS", MEMBER_CKPTS), ("WEIGHTS", WEIGHTS),
                  ("MEMBER_GPUS", MEMBER_MODULES if MEMBER_GPUS is None else MEMBER_GPUS)):
    if set(_d) != set(MEMBER_MODULES):
        raise ValueError(f"{_name} must name exactly the members {list(MEMBER_MODULES)}, got {sorted(_d)}")
if not set(MEMBER_EXPECTED) <= set(MEMBER_MODULES):
    raise ValueError(f"MEMBER_EXPECTED names {sorted(MEMBER_EXPECTED)}, not all of them members {list(MEMBER_MODULES)}")
for m in MEMBER_MODULES:
    for ds, shas in MEMBER_CKPTS[m]:
        if ds is None or shas is None:
            raise ValueError(f"fill MEMBER_CKPTS[{m!r}]: (weights dataset slug, its SHA256SUMS.txt as {{file: sha256}}) "
                             "for every dataset of the member, once it is published")
        if set(shas) != {f"fold{k}.pt" for k in range(N_FOLDS)}:
            raise ValueError(f"MEMBER_CKPTS[{m!r}] {ds}: must hold fold0..fold{N_FOLDS - 1}.pt, got {sorted(shas)}: "
                             "fill it from the dataset's SHA256SUMS.txt")
    if len({ds for ds, _ in MEMBER_CKPTS[m]}) != len(MEMBER_CKPTS[m]):
        raise ValueError(f"MEMBER_CKPTS[{m!r}] lists a weights dataset twice: {[ds for ds, _ in MEMBER_CKPTS[m]]}")
    if m not in MEMBER_EXPECTED or not MEMBER_EXPECTED[m]:
        raise ValueError(f"MEMBER_EXPECTED has no {m!r}: run exps/exp-0004/member_ref.py with all members on the "
                         "MEMBER_CKPTS checkpoints and paste its whole member_ref.json")
    if WEIGHTS[m] is None:
        raise ValueError(f"fill WEIGHTS[{m!r}]: the lead's D8 weight of member {m}")
if not (Path("/kaggle/input").is_dir() and ctx.output_dir == Path("/kaggle/working")):
    raise RuntimeError("submit_blend.py runs only inside a Kaggle kernel (CUDA, /kaggle/working; run_public's vendored "
                       "scripts read /kaggle/input directly). Launch it with the kaggle skill, command in the docstring.")

comp = ctx.mounts["rsna-knee-abnormality-detection"]
ckpts = {m: {str(ctx.mounts[ds] / name): sha for ds, shas in MEMBER_CKPTS[m] for name, sha in sorted(shas.items())}
         for m in MEMBER_MODULES}  # {member: {path: sha256}}, the member's datasets in order
members = [{"name": m, "module": module, "ckpt_paths": list(ckpts[m])} for m, module in MEMBER_MODULES.items()]
own_spec = {"series_root": str(comp / "test_series"), "series_csv": str(comp / "test_series.csv"),
            "cache_dir": str(CACHE_DIR), "out_dir": str(ctx.output_dir / "own"), "device": DEVICE, "workers": WORKERS,
            "chunk_studies": CHUNK_STUDIES, "budgets": BUDGETS, "members": members, "member_gpus": MEMBER_GPUS,
            "parity_expected": PARITY_EXPECTED,
            "member_expected": MEMBER_EXPECTED, "member_tol": MEMBER_TOL}
cfg = {"visible_studies": VISIBLE_STUDIES,
       "wheels": [str(w) for w in sorted(ctx.mounts[WHEELS_DS].glob("*.whl")) if not w.name.startswith("opencv")],
       "pins": PINS, "site_dir": str(SITE_DIR), "own_spec": own_spec, "own_budget_s": OWN_BUDGET_S,
       "weights": WEIGHTS, "ckpt_sha256": {p: sha for m in ckpts for p, sha in ckpts[m].items()},
       "n_hidden_studies": N_HIDDEN_STUDIES,
       "hidden_fallback": HIDDEN_FALLBACK,
       "own_only": {"test_csv": str(comp / "test.csv"), "sample_submission": str(comp / "sample_submission.csv")}
       if OWN_ONLY else None}
blend_main(ctx, T0, cfg, None if OWN_ONLY else pipeline.run_public)
