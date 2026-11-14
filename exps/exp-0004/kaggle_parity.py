"""Kaggle-image parity of the preprocessing: re-preprocess the reference series from raw DICOM, compare fingerprints.

    uv run --project kaggle python kaggle/run_kernel.py --exp exp-0004 --script kaggle_parity --accelerator none \
        --docker-image <exp-0001's pinned image> --ds bebra3222/rsna-knee-wheels-cp313

Pass 1: the image's own numpy/cv2 plus our pinned pydicom and decoders (installed offline).
Pass 2, only for the series that mismatch: a subprocess with our cv2 5.0.0.93 installed into a
separate --target dir, so the image's packages stay untouched. Everything goes to the log
(kernel output files cannot be fetched here); the run fails if pass 1 has any mismatch.
EXPECTED comes from parity_ref.py run REF_RUN.
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
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

from utils.ctx import get_ctx  # noqa: E402

WHEELS_DS = "bebra3222/rsna-knee-wheels-cp313"
PINS = {"pydicom": "3.0.2", "pylibjpeg": "2.1.0", "pylibjpeg-libjpeg": "2.4.0", "pylibjpeg-openjpeg": "2.6.0",
        "python-gdcm": "3.2.6"}
CV2_PIN = {"opencv-python-headless": "5.0.0.93"}
REF_RUN = "20261017t120738-561-0468"
EXPECTED = json.loads(r"""[{"split":"train","study":"1.2.826.0.1.3680043.8.498.72342425990274562150756501476989691572","series":"1.2.826.0.1.3680043.8.498.75517303748731445791260989633300329985","ts":"1.2.840.10008.1.2.1","fp":{"shape":[30,412,412],"sha_vol":"06c324b3b319f4db","sha_pos":"0fe9e9567b4a22cc","sum":358666392,"lo":0.0,"hi":1895.555521850586}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.77626578713099735143745396709359607445","series":"1.2.826.0.1.3680043.8.498.25389456937397263689131413956785890271","ts":"1.2.840.10008.1.2.1","fp":{"shape":[30,450,450],"sha_vol":"e9eaf386a7a0f49d","sha_pos":"01354ed5fa48e4d9","sum":340667212,"lo":2.725925922393799,"hi":1955.5266302490231}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.58604600600114899516493668210902251434","series":"1.2.826.0.1.3680043.8.498.82017491166711395173161427403216048751","ts":"1.2.840.10008.1.2.1","fp":{"shape":[36,500,453],"sha_vol":"7bf8efacaa12d66f","sha_pos":"88e54eeb4a46fbce","sum":389706939,"lo":0.0,"hi":348.24706024169876}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.10120841513315425967980213160839994882","series":"1.2.826.0.1.3680043.8.498.65844740287774520398711186000932761578","ts":"1.2.840.10008.1.2.1","fp":{"shape":[11,400,400],"sha_vol":"d3edefc5282f1d7d","sha_pos":"82969e37299ea796","sum":185154874,"lo":0.0,"hi":1819.519775390625}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.11136723251983227230447257874027263286","series":"1.2.826.0.1.3680043.8.498.28279973769786623823757306247809336663","ts":"1.2.840.10008.1.2.1","fp":{"shape":[35,400,400],"sha_vol":"6c71e93789e426a2","sha_pos":"a62ede005c078d0a","sum":402903839,"lo":0.0,"hi":1258.669445800781}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.58280111882249056715877316026191844548","series":"1.2.826.0.1.3680043.8.498.23335855705981483938261974657979091605","ts":"1.2.840.10008.1.2.1","fp":{"shape":[24,450,450],"sha_vol":"342b2346ccadb155","sha_pos":"93dbf11cab184c4f","sum":328059449,"lo":0.0,"hi":444.9077342224117}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.88899519473612920868099376939780201067","series":"1.2.826.0.1.3680043.8.498.67610333890666165583980840002133197075","ts":"1.2.840.10008.1.2.1","fp":{"shape":[40,400,400],"sha_vol":"1659df6bca7dd2e8","sha_pos":"d60925ebb62a9bad","sum":353049743,"lo":0.0,"hi":1164.0242089843741}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.77626578713099735143745396709359607445","series":"1.2.826.0.1.3680043.8.498.60143347930555299752247887975033513736","ts":"1.2.840.10008.1.2.1","fp":{"shape":[28,450,450],"sha_vol":"b1859963d18f1210","sha_pos":"a9f165eb1364dad3","sum":317074484,"lo":0.0,"hi":1905.702168579101}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.68158735450883179427440578190038002606","series":"1.2.826.0.1.3680043.8.498.58434302343002978719224489580240432515","ts":"1.2.840.10008.1.2.1","fp":{"shape":[69,375,375],"sha_vol":"75ec525a0ad8885f","sha_pos":"14d1d5ae42821a1b","sum":439275522,"lo":0.0,"hi":646.6015502929693}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.13353409340862821090624127486679097099","series":"1.2.826.0.1.3680043.8.498.96432067833669175866385142045415223186","ts":"1.2.840.10008.1.2.1","fp":{"shape":[30,412,412],"sha_vol":"51fd5baec11612fe","sha_pos":"4b61c383f3f7043d","sum":445978186,"lo":0.0,"hi":2504.5091992187536}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.58604600600114899516493668210902251434","series":"1.2.826.0.1.3680043.8.498.11331025145329479934244925872126820705","ts":"1.2.840.10008.1.2.1","fp":{"shape":[42,350,350],"sha_vol":"b01a627b2a25b93f","sha_pos":"7d48ea90b2efc684","sum":208471180,"lo":0.0,"hi":499.11432647704925}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.11583098598455610459898849451117846552","series":"1.2.826.0.1.3680043.8.498.10815465141351709268596769656034066608","ts":"1.2.840.10008.1.2.1","fp":{"shape":[24,450,450],"sha_vol":"70e836bdf8921308","sha_pos":"4faf12c4a972d7c6","sum":235164215,"lo":0.0,"hi":775.5869143676757}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035","series":"1.2.826.0.1.3680043.8.498.10354958299917644989297776650748117510","ts":"1.2.840.10008.1.2.1","fp":{"shape":[37,375,375],"sha_vol":"0ef1387e7a9eabe7","sha_pos":"dba60d82147545dc","sum":379581495,"lo":2.13372802734375,"hi":914.8466259765617}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035","series":"1.2.826.0.1.3680043.8.498.38875735530891460298737350997173050019","ts":"1.2.840.10008.1.2.1","fp":{"shape":[39,425,425],"sha_vol":"301ccec5a732b47a","sha_pos":"05cad15be6c320b2","sum":369912398,"lo":0.0,"hi":2075.61658203125}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035","series":"1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348","ts":"1.2.840.10008.1.2.1","fp":{"shape":[36,400,400],"sha_vol":"7e5743e169bfc127","sha_pos":"a1b604744d81e902","sum":296339886,"lo":0.75,"hi":955.0}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035","series":"1.2.826.0.1.3680043.8.498.56454085218863048442970283058276928820","ts":"1.2.840.10008.1.2.1","fp":{"shape":[37,425,425],"sha_vol":"7ac2465967c8a762","sha_pos":"05fdd2e92c37dbb8","sum":376916925,"lo":0.0,"hi":1913.5134130859371}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035","series":"1.2.826.0.1.3680043.8.498.64923993132617839685637225229996868277","ts":"1.2.840.10008.1.2.1","fp":{"shape":[39,425,425],"sha_vol":"429c855f86627efc","sha_pos":"1cc23c785903bb24","sum":439826778,"lo":0.0,"hi":814.4417547607418}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680","series":"1.2.826.0.1.3680043.8.498.24027939485299249865408996288050986555","ts":"1.2.840.10008.1.2.1","fp":{"shape":[34,400,400],"sha_vol":"b8aaa8bcab89b3ca","sha_pos":"9545acdd25afb410","sum":474489509,"lo":0.0,"hi":222.804718170166}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680","series":"1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703","ts":"1.2.840.10008.1.2.1","fp":{"shape":[38,400,400],"sha_vol":"cd564f16054c2918","sha_pos":"7d803d9f79436072","sum":475750769,"lo":0.0,"hi":233.11538696289062}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680","series":"1.2.826.0.1.3680043.8.498.33502970026548879077669041950507260816","ts":"1.2.840.10008.1.2.1","fp":{"shape":[34,400,400],"sha_vol":"0577f66a310e671c","sha_pos":"1edaa22e0fe93133","sum":442896316,"lo":0.0,"hi":679.4615478515625}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680","series":"1.2.826.0.1.3680043.8.498.44371997195817413982467477951770244117","ts":"1.2.840.10008.1.2.1","fp":{"shape":[38,400,400],"sha_vol":"2e96c7fac8d86ad0","sha_pos":"0678fa49a60557a8","sum":460986494,"lo":0.0,"hi":877.9881665039061}},{"split":"train","study":"1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680","series":"1.2.826.0.1.3680043.8.498.87813801633306918004512820977976658354","ts":"1.2.840.10008.1.2.1","fp":{"shape":[42,400,400],"sha_vol":"9dbfdfa829f82059","sha_pos":"9f1b321fe04c113d","sum":488818591,"lo":0.0,"hi":225.04145751953092}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191","series":"1.2.826.0.1.3680043.8.498.11580656442259111255675562605155903947","ts":"1.2.840.10008.1.2.1","fp":{"shape":[34,375,375],"sha_vol":"c4457295bc2f1860","sha_pos":"67e4e2ab1bb79d11","sum":368441360,"lo":2.9319445800781248,"hi":2090.8626831054685}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191","series":"1.2.826.0.1.3680043.8.498.17811502614030631664517622518906646132","ts":"1.2.840.10008.1.2.1","fp":{"shape":[30,375,375],"sha_vol":"7fd5a12a4cd8e25c","sha_pos":"93a2cb77c1a92d6d","sum":345524607,"lo":3.41015625,"hi":1376.033203125}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191","series":"1.2.826.0.1.3680043.8.498.30565395595045942404081022062489758495","ts":"1.2.840.10008.1.2.1","fp":{"shape":[30,375,375],"sha_vol":"24b43404ae4ef526","sha_pos":"08f7395a2a3c8bce","sum":375753594,"lo":6.120652856826782,"hi":2226.6173461914054}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191","series":"1.2.826.0.1.3680043.8.498.32856494541816845805947850304653662081","ts":"1.2.840.10008.1.2.1","fp":{"shape":[30,375,375],"sha_vol":"8a6cc2e34f6dd0dc","sha_pos":"c01ac7bc9fdbffc7","sum":287809716,"lo":1.0344692993164062,"hi":458.2434936523432}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191","series":"1.2.826.0.1.3680043.8.498.44334485654554877495800965096767429689","ts":"1.2.840.10008.1.2.1","fp":{"shape":[160,400,800],"sha_vol":"6b12ba8e8a21658d","sha_pos":"bb9012c3d0d0979c","sum":1386525482,"lo":0.0,"hi":253.765625}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776","series":"1.2.826.0.1.3680043.8.498.10543708582570884867685900269888836671","ts":"1.2.840.10008.1.2.1","fp":{"shape":[24,450,450],"sha_vol":"96be98367570256c","sha_pos":"a29d3c2566ca5038","sum":277759769,"lo":-17.5576171875,"hi":5909.636879882809}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776","series":"1.2.826.0.1.3680043.8.498.10818107180155454609278933308651876447","ts":"1.2.840.10008.1.2.1","fp":{"shape":[33,375,375],"sha_vol":"df2fe1cfe24e0f29","sha_pos":"e618015f99f75b3f","sum":185178554,"lo":-0.7348284912109375,"hi":3889.8841894531233}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776","series":"1.2.826.0.1.3680043.8.498.34851887739080902195133258975769749352","ts":"1.2.840.10008.1.2.1","fp":{"shape":[24,450,450],"sha_vol":"f90a7fbcb0ddd208","sha_pos":"a29d3c2566ca5038","sum":205284587,"lo":-6.3341677856445315,"hi":4368.035205078124}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776","series":"1.2.826.0.1.3680043.8.498.49057071886338907232646925625200929717","ts":"1.2.840.10008.1.2.1","fp":{"shape":[24,450,450],"sha_vol":"869dd8d40815d947","sha_pos":"a29d3c2566ca5038","sum":298966235,"lo":-7.626600444316865,"hi":7718.3652783203115}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776","series":"1.2.826.0.1.3680043.8.498.49397095733611495516760005725956804191","ts":"1.2.840.10008.1.2.1","fp":{"shape":[30,500,400],"sha_vol":"6711492cb4d55a67","sha_pos":"d2d78c39ac9f84f4","sum":265144398,"lo":0.0,"hi":3983.4005566406213}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402","series":"1.2.826.0.1.3680043.8.498.10927312244766790279817592893017048392","ts":"1.2.840.10008.1.2.1","fp":{"shape":[24,425,425],"sha_vol":"8225b92ab15a7083","sha_pos":"8ca850c6aaa26d99","sum":484910035,"lo":0.0,"hi":1861.0731622314443}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402","series":"1.2.826.0.1.3680043.8.498.12207733879434320501816063789123909030","ts":"1.2.840.10008.1.2.1","fp":{"shape":[24,425,425],"sha_vol":"592fac6f7d8434a1","sha_pos":"8ca850c6aaa26d99","sum":333085423,"lo":0.0,"hi":752.7015048217768}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402","series":"1.2.826.0.1.3680043.8.498.15858543841290540180647614348005539746","ts":"1.2.840.10008.1.2.1","fp":{"shape":[40,425,425],"sha_vol":"9bf42f0f9321a962","sha_pos":"858413852242ebc1","sum":350165666,"lo":0.0,"hi":1141.4560821533196}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402","series":"1.2.826.0.1.3680043.8.498.19236602294493686533219656323303746544","ts":"1.2.840.10008.1.2.1","fp":{"shape":[25,450,450],"sha_vol":"3a7ff6c6fb31fad0","sha_pos":"0d2200bb0a7599df","sum":391749146,"lo":0.0,"hi":1213.1723657226562}},{"split":"test","study":"1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402","series":"1.2.826.0.1.3680043.8.498.66069370990398014317043644514003034312","ts":"1.2.840.10008.1.2.1","fp":{"shape":[25,450,450],"sha_vol":"355eb5ee637200af","sha_pos":"0d2200bb0a7599df","sum":333545594,"lo":0.0,"hi":678.3313427734365}}]""")

ctx = get_ctx()
from src.kernel_env import env_report, install_wheels, volume_fingerprint  # noqa: E402

if not EXPECTED:
    raise ValueError("EXPECTED is empty: paste the 'series' list of parity_ref.json")
wheels = sorted(ctx.mounts[WHEELS_DS].glob("*.whl"))
install_wheels([w for w in wheels if not w.name.startswith("opencv")], PINS)
print("PARITY_ENV " + json.dumps(env_report()), flush=True)

from src.knee_volume import SeriesError, load_series_dir  # noqa: E402

comp = ctx.mounts["rsna-knee-abnormality-detection"]


def fingerprint(e):
    """(fingerprint or {'error': code}, seconds) of one reference series preprocessed from raw DICOM."""
    t = time.time()
    try:
        got = volume_fingerprint(*load_series_dir(comp / f"{e['split']}_series" / e["study"] / e["series"]))
    except SeriesError as err:
        got = {"error": err.code}
    return got, time.time() - t


bad = []
for e in EXPECTED:
    got, sec = fingerprint(e)
    match = got == e["fp"]
    print("PARITY1 " + json.dumps({"series": e["series"], "split": e["split"], "ts": e["ts"], "match": match,
                                   "sec": round(sec, 2), **({} if match else {"want": e["fp"], "got": got})}),
          flush=True)
    if not match:
        bad.append(e)
print(f"PARITY1_SUMMARY {len(EXPECTED) - len(bad)}/{len(EXPECTED)} match", flush=True)

if bad:
    target = Path("/kaggle/tmp/cv2_pinned")
    install_wheels([w for w in wheels if w.name.startswith("opencv")], CV2_PIN, target=target)
    todo = ctx.output_dir / "parity_pass2.json"
    todo.write_text(json.dumps(bad))
    code = (
        "import json, sys, cv2\n"
        "from src.kernel_env import volume_fingerprint\n"
        "from src.knee_volume import SeriesError, load_series_dir\n"
        f"comp = {str(comp)!r}\n"
        f"print('PARITY2_CV2', cv2.__version__, flush=True)\n"
        f"for e in json.load(open({str(todo)!r})):\n"
        "    try:\n"
        "        got = volume_fingerprint(*load_series_dir(f\"{comp}/{e['split']}_series/{e['study']}/{e['series']}\"))\n"
        "    except SeriesError as err:\n"
        "        got = {'error': err.code}\n"
        "    print('PARITY2 ' + json.dumps({'series': e['series'], 'match': got == e['fp'], 'got': got}), flush=True)\n"
    )
    env = {**os.environ, "PYTHONPATH": f"{target}{os.pathsep}{ctx.sources_dir}"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    raise RuntimeError(f"PARITY FAILED: {len(bad)}/{len(EXPECTED)} series differ with the image's own cv2/numpy "
                       "(PARITY2 lines: the same series with our pinned cv2)")
print("PARITY OK", flush=True)
