"""Stages of public notebook A v7 (heliosli, LB 0.958): Raptor 384 / 224, R6 compact, stack blend, goodpjw reader.

Source: `this-is-best-public-dataset-new-version-0-9586.ipynb` v7, cells 2-9. The cell code is moved as-is;
what the move from cells to functions requires (globals -> arguments, `display` -> `print`, `find_mount` /
`_o_find` -> mount paths from the caller) is done silently, every other deviation is marked `# [port]`.
Competition constants (weights, slugs, arm, bundle id, deadlines) live in the experiment script.
"""

import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VENDORED = Path(__file__).resolve().parent / "vendored"

# variant -> (vendored script, output stem, name in errors, stage marker); cells 3 / 5
_RAPTOR = {
    "384": ("raptor384_run.py", "raptor", "Raptor", "RAPTOR"),
    "224": ("raptor224_run.py", "raptor224", "Raptor-224", "RAPTOR224"),
}


def _vendored_sha():
    """vendored/sha256.json: {relative name without .txt: sha256} (equals A's SOURCE_SHA / _PB_SHA)."""
    return json.loads((VENDORED / "sha256.json").read_text(encoding="utf-8"))


def materialize(name, dest_dir):
    """Copy one vendored file (e.g. "compact/probe_dual.py") to dest_dir/<file name>, sha256-checked."""
    expected = _vendored_sha()[name]
    dest = Path(dest_dir) / Path(name).name
    dest.write_bytes((VENDORED / (name + ".txt")).read_bytes())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert digest == expected, ("vendored file differs from sha256.json", name, digest, expected)
    return dest


def sha256_files(paths):
    """sha256 of each file -> {absolute path: hex digest} (provenance of the inputs, nothing is compared)."""
    out = {}
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        out[str(p)] = h.hexdigest()
    return out


def run_raptor(variant, work_dir, out_dir, timeout_s):
    """Cells 3 / 5: run the vendored Raptor 384 or 224 script as its own process -> raptor{,224}_{sub,raw}.csv."""
    script, stem, name, marker = _RAPTOR[variant]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # [port] A writes the script to /tmp/raptor_run.py / /tmp/raptor224_run.py; here work_dir/raptor{384,224}_run.py
    # (the scripts use their own path only when /kaggle/working is missing)
    path = materialize(script, work_dir)
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', str(path)], timeout=timeout_s).returncode
    if rc != 0:
        raise RuntimeError('%s stage failed with exit code %d' % (name, rc))
    # the vendored scripts write to /kaggle/working themselves: out_dir must be that folder
    for f in (stem + '_sub.csv', stem + '_raw.csv'):
        assert os.path.isfile(os.path.join(str(out_dir), f)), f + ' missing'
    seconds = time.time() - t
    print('%s_STAGE_DONE seconds %.0f' % (marker, seconds), flush=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False)
    return {"seconds": seconds, "rc": rc}


def run_compact_r6(data_dir, code_src, bundle_src, tmp_dir, out_dir, t0, arm, bundle_id, deadline_hours, flip_tta,
                   code_sha, probe_class, w_src):
    """Cells 6-7: Helios R6 compact model (code/bundle sha checks, pinned wheels, probe_dual.py) -> ours_sub.csv."""
    t_stage = time.time()
    ARM, BUNDLE_ID, DEADLINE_HOURS, FLIP_TTA, CODE_SHA, PROBE_CLASS = arm, bundle_id, deadline_hours, flip_tta, code_sha, probe_class
    T0 = t0   # [blend] the 8.3 h deadline counts from the notebook start (the Raptor stage ran first)
    WORK = Path(out_dir)
    TMP = Path(tmp_dir)
    subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], check=False)
    # [port] A: df -h /dev/shm /kaggle/working /tmp; here out_dir and the parent of tmp_dir (TMP is created below)
    subprocess.run(['df', '-h', '/dev/shm', str(WORK), str(TMP.parent)], check=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    TMP.mkdir(parents=True, exist_ok=True)

    CODE_SRC = Path(code_src)
    assert (CODE_SRC / 'receipt.json').is_file(), 'code dataset: no receipt.json in %s' % CODE_SRC   # A: find_mount(CODE_SLUG, 'receipt.json')
    DATA = Path(data_dir)
    assert (DATA / 'test.csv').is_file(), 'competition data: no test.csv in %s' % DATA   # A: find_mount(COMPETITION, 'test.csv')

    # mainline decode + rendering code: byte-identical to what built the training caches
    CODE = TMP / 'mainline_code'
    if (CODE_SRC / 'cloud_code.zip').is_file():
        with zipfile.ZipFile(CODE_SRC / 'cloud_code.zip') as z:
            z.extractall(CODE)
    else:
        shutil.copytree(CODE_SRC / 'cloud_code', CODE, dirs_exist_ok=True)
    for name, digest in CODE_SHA.items():
        assert hashlib.sha256((CODE / 'training' / name).read_bytes()).hexdigest() == digest, ('mainline module differs', name)

    # pinned wheels (same versions as the runs that produced the training caches)
    WHEELS = TMP / 'wheels'
    WHEELS.mkdir(exist_ok=True)
    if (CODE_SRC / 'dependencies.zip').is_file():
        with zipfile.ZipFile(CODE_SRC / 'dependencies.zip') as z:
            for n in z.namelist():
                if n.endswith('.whl'):
                    (WHEELS / Path(n).name).write_bytes(z.read(n))
    else:
        for p in CODE_SRC.rglob('*.whl'):
            shutil.copy(p, WHEELS / p.name)
    wheels = sorted(str(p) for p in WHEELS.glob('*.whl'))
    assert wheels, 'no wheels found'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)

    SRC = TMP / 'probe_src'
    SRC.mkdir(exist_ok=True)
    # [port] A writes its SOURCES dict and checks SOURCE_SHA; here the same 7 files come from vendored/compact/
    # (materialize checks them against vendored/sha256.json, which equals A's SOURCE_SHA)
    SOURCE_SHA = {}
    for key, digest in sorted(_vendored_sha().items()):
        if key.startswith('compact/'):
            materialize(key, SRC)
            SOURCE_SHA[Path(key).name] = digest
    BUNDLE = TMP / 'bundle'
    BUNDLE.mkdir(exist_ok=True)

    M_SRC = Path(bundle_src)
    assert (M_SRC / 'manifest.json').is_file(), 'R6 bundle dataset: no manifest.json in %s' % M_SRC   # A: find_mount(C_SLUG, 'manifest.json')
    manifest = json.loads((M_SRC / 'manifest.json').read_text())
    assert manifest['bundle_id'] == BUNDLE_ID, (manifest['bundle_id'], BUNDLE_ID)
    for name in manifest['models'] + (manifest.get('models2') or []):
        assert hashlib.sha256((M_SRC / name).read_bytes()).hexdigest() == manifest['sha256'][name], name
        shutil.copy(M_SRC / name, BUNDLE / name)
    if manifest.get('state_spec'):
        name = manifest['state_spec']
        assert hashlib.sha256((M_SRC / name).read_bytes()).hexdigest() == manifest['sha256'][name], name
        shutil.copy(M_SRC / name, BUNDLE / name)
    if manifest.get('side_model'):
        name = manifest['side_model']
        assert hashlib.sha256((M_SRC / name).read_bytes()).hexdigest() == manifest['sha256'][name], name
        shutil.copy(M_SRC / name, BUNDLE / name)
    if manifest.get('fluid_stacker'):
        name = manifest['fluid_stacker']
        assert hashlib.sha256((M_SRC / name).read_bytes()).hexdigest() == manifest['sha256'][name], name
        shutil.copy(M_SRC / name, BUNDLE / name)
    if manifest.get('state_stacker'):
        name = manifest['state_stacker']
        assert hashlib.sha256((M_SRC / name).read_bytes()).hexdigest() == manifest['sha256'][name], name
        shutil.copy(M_SRC / name, BUNDLE / name)
    WORKER = 'compact_infer.py'
    WORKER_ARGS = ['--backbone', manifest['backbone'], '--size', str(manifest['size']), '--crop', str(manifest['crop']),
                   '--eval-stride', str(manifest.get('eval_stride', 3)), '--views', manifest.get('views', 'v1')]
    if manifest.get('state_spec'):
        WORKER_ARGS += ['--state-spec', str(BUNDLE / manifest['state_spec'])]
    if manifest.get('side_model'):
        # image side check: studies without one trustworthy laterality tag are re-rendered when the orientation classifier
        # says they are mirrored (same rule as the S5 training-cache fix)
        WORKER_ARGS += ['--side-model', str(BUNDLE / manifest['side_model']), '--side-threshold', str(manifest.get('side_threshold', 2.0)),
                        '--side-position-mm', str(manifest.get('side_position_mm', 20.0)), '--side-max-x', str(manifest.get('side_max_x', 0.0))]
    WORKER_ARGS += ['--slices', str(manifest.get('slices', 3)), '--phys-mm', str(manifest.get('phys_mm', 0.0))]
    if manifest.get('models2'):
        # second architecture on the same windows (one decode per study), per-class weights fixed in the bundle
        WORKER_ARGS += ['--models2', str(BUNDLE), '--arm2', manifest['arm2'], '--backbone2', manifest['backbone2'],
                        '--w2', ','.join(str(x) for x in manifest['w2'])]
        if manifest.get('crop2') is not None:
            # v3: the second group's own centre crop of the same decoded windows
            WORKER_ARGS += ['--crop2', str(manifest['crop2'])]
    if manifest.get('trainer_md5'):
        # v3: the embedded compact_train.py must be the trainer this bundle was built for (and the worker re-checks what it imports)
        assert hashlib.md5((SRC / 'compact_train.py').read_bytes()).hexdigest() == manifest['trainer_md5'], 'pinned trainer differs from the bundle manifest (md5)'
        assert SOURCE_SHA['compact_train.py'] == manifest['src_sha256']['compact_train.py'], 'pinned trainer differs from the bundle manifest (sha256)'
        WORKER_ARGS += ['--trainer-md5', manifest['trainer_md5']]
    if manifest.get('compile'):
        # v3: torch.compile of the fold models' encoders (checked against eager inside the worker; eager fallback, logged)
        # [port] A's comment on this line is garbled (a W_RAPTOR_C literal pasted into it); the code is unchanged
        WORKER_ARGS += ['--compile']
    if manifest.get('fluid_stacker'):
        # v4b: D2 fluid volume measured on the rendered rows; the classes the json adopts get the stacker's logit
        WORKER_ARGS += ['--fluid-stacker', str(BUNDLE / manifest['fluid_stacker'])]
    if manifest.get('flip_tta') or FLIP_TTA:
        # v4c: every fold model also scores the mirrored windows; the two passes are averaged in logit space (MCL: plain pass)
        # [d957] FLIP_TTA: this notebook's switch, at the manifest key's position (the R6 manifest predates flip TTA: no flip_tta key)
        WORKER_ARGS += ['--flip-tta']
    if manifest.get('state_stacker'):
        # v4c: the classes the json adopts get b0 + coef . [z, state probabilities, binaries (, lv, miss)]
        WORKER_ARGS += ['--state-stacker', str(BUNDLE / manifest['state_stacker'])]
    if manifest.get('views') in ('v3k', 'v4'):
        # knee-centred views: the anchors are rebuilt from the test DICOM headers + the SKM-TEA checkpoint of the code dataset
        assert hashlib.sha256((CODE_SRC / 'anatomy.pt').read_bytes()).hexdigest() == manifest['anchor_weights_sha256'], 'anatomy.pt differs from the training anchor checkpoint'
        WORKER_ARGS += ['--anchor-weights', str(CODE_SRC / 'anatomy.pt')]
    if manifest['backbone'] == 'ofl_l':
        # the fold models hold only the trained tail; rebuild the frozen part from the public OFL-L weights
        # [port] A: W_SRC = find_mount('rsna-knee-ofl-ssl-weights', 'weight_parts.json'); here w_src from the caller
        # (A does not attach that dataset: the R6 bundle is not ofl_l, so the caller passes None)
        assert w_src is not None, 'bundle backbone ofl_l needs the OFL-L weights dataset (rsna-knee-ofl-ssl-weights): pass w_src'
        W_SRC = Path(w_src)
        assert (W_SRC / 'weight_parts.json').is_file(), 'no weight_parts.json in %s' % W_SRC
        parts = json.loads((W_SRC / 'weight_parts.json').read_text())
        h = hashlib.sha256()
        with (BUNDLE / 'model.safetensors').open('wb') as dst:
            for item in parts['parts']:
                blob = (W_SRC / item['name']).read_bytes()
                assert len(blob) == item['bytes'] and hashlib.sha256(blob).hexdigest() == item['sha256'], item['name']
                dst.write(blob)
                h.update(blob)
        assert h.hexdigest() == parts['assembled_sha256'] == manifest['init_weights_sha256'], 'encoder weights differ'
        del blob
        WORKER_ARGS += ['--init', str(BUNDLE / 'model.safetensors'), '--freeze-blocks', str(manifest['freeze_blocks'])]
    print('ASSETS_READY', str(CODE_SRC), str(M_SRC), str(DATA), 'flip_tta', '--flip-tta' in WORKER_ARGS, 'seconds %.1f' % (time.time() - T0), flush=True)

    # ---- cell 7 ----
    deadline = T0 + DEADLINE_HOURS * 3600
    env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
    pred = TMP / 'prediction'
    rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                         '--code', str(CODE), '--bundle', str(BUNDLE), '--arm', ARM, '--output', str(pred),
                         '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                         '--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS),
                         '--max-fallback-fraction', '1.0'], env=env).returncode   # [blend] no 3 % stop here: stage 3 gives a study we did not score Raptor's rank
    for log in sorted(pred.glob('worker*/log.txt')):
        shutil.copyfile(log, WORK / (log.parent.name + '_log.txt'))
    if rc != 0:
        raise RuntimeError('controller failed with exit code %d' % rc)
    shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')

    if PROBE_CLASS:
        # single-class probe: every column except PROBE_CLASS is a constant, so public LB = (11 * 0.5 + AUC_c) / 12
        _df = pd.read_csv(WORK / 'ours_sub.csv')
        assert PROBE_CLASS in _df.columns, PROBE_CLASS
        for _c in _df.columns:
            if _c not in ('StudyInstanceUID', PROBE_CLASS):
                _df[_c] = 0.5
        _df.to_csv(WORK / 'ours_sub.csv', index=False)
        print('SINGLE_CLASS_PROBE', PROBE_CLASS, 'other 11 columns set to 0.5', flush=True)
    receipt = json.loads((pred / 'submission_receipt.json').read_text())
    receipt['seconds_including_notebook_setup'] = time.time() - T0
    receipt['bundle_id'] = BUNDLE_ID
    receipt['arm'] = ARM
    (WORK / 'submission_receipt.json').write_text(json.dumps(receipt, indent=1))
    print('OURS_STAGE_COMPLETE', json.dumps({k: receipt.get(k) for k in ('status', 'arm', 'rows', 'scored', 'fallback_studies',
                                                                     'seconds', 'per_study_mean', 'gpus')}), flush=True)
    print(pd.read_csv(WORK / 'ours_sub.csv').head(), flush=True)
    return {"receipt": receipt, "pred_dir": pred, "seconds": time.time() - t_stage}


def read_stack_inputs(data_dir, work_dir):
    """Cell 8 (reading part): test ids, submission columns, labels and the five member CSVs as A reads them."""
    DATA, WORK = Path(data_dir), Path(work_dir)
    test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
    ss = DATA / 'sample_submission.csv'
    ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
    cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
    labels = cols[1:]
    members = {
        'ours': ours,
        'raptor_raw': pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str}),
        'raptor_sub': pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str}),
        'raptor224_raw': pd.read_csv(WORK / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str}),     # [d957]
        'raptor224_sub': pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str}),
    }
    return test_ids, cols, labels, members


def blend_stack(test_ids, labels, cols, members, pred_dir, w_raptor, w_ours, w_raptor_c, a224, raptor_seconds,
                raptor224_seconds, t0):
    """Cell 8: per-finding percentile-rank blend of Raptor 384/224 and R6 with fail masks -> (submission, receipt, masks)."""
    W_RAPTOR, W_OURS = w_raptor, w_ours
    A224 = a224   # [d957] share of the Raptor half given to the 224-crop model
    assert 0.0 < A224 < 1.0
    W_RAPTOR_C = w_raptor_c   # [V2] per-finding Raptor weight (ours = 1 - it); every other finding uses W_RAPTOR
    assert all(0.0 <= w_ <= 1.0 for w_ in (W_RAPTOR, W_OURS) + tuple(W_RAPTOR_C.values())), 'blend weights outside [0, 1]'   # [v3] a per-finding Raptor weight may be exactly 1.0 (ours 0)
    RAPTOR_SECONDS, RAPTOR224_SECONDS, NB_T0 = raptor_seconds, raptor224_seconds, t0
    pred = Path(pred_dir)
    ours, raw, rsub = members['ours'], members['raptor_raw'], members['raptor_sub']
    raw224, rsub224 = members['raptor224_raw'], members['raptor224_sub']
    for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub'), (raw224, 'raptor224_raw'), (rsub224, 'raptor224_sub')):
        assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'
        assert set(labels) <= set(df.columns), nm + ': columns differ'
    ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]
    raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]
    rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]
    raw224 = raw224.set_index('StudyInstanceUID').loc[test_ids, labels]
    rsub224 = rsub224.set_index('StudyInstanceUID').loc[test_ids, labels]
    n = len(test_ids)
    r384_fail = (raw.values == 0.5).all(axis=1)            # Raptor leaves a failed study at 0.5 in every column
    r224_fail = (raw224.values == 0.5).all(axis=1)         # [d957] the same for the 224 model
    rap_fail = r384_fail & r224_fail                        # [d957] the Raptor half is missing only when both failed
    scored = set()                                          # ours fills a failed study with the column mean: read what was scored
    for f in pred.glob('worker*/predictions.csv'):
        with open(f, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                try:
                    v = [float(r[L]) for L in labels]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(0. <= x <= 1. for x in v):
                    scored.add(r['StudyInstanceUID'])
    our_fail = np.array([u not in scored for u in test_ids])
    if r384_fail.sum() > 0.03 * n:
        raise RuntimeError('Raptor failed on %d of %d studies' % (r384_fail.sum(), n))
    if r224_fail.sum() > 0.03 * n:                          # [d957]
        raise RuntimeError('Raptor-224 failed on %d of %d studies' % (r224_fail.sum(), n))
    if our_fail.sum() > 0.5 * n:                            # [v3] no silent Raptor-only submission under the blend's name
        raise RuntimeError('our stage did not score %d of %d studies (more than 50 %%)' % (our_fail.sum(), n))

    def pct(v):
        return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

    def pct_on(v, ok):
        # [v3] percentile rank among the rows in ok only (one model's non-failed studies); the other rows are NaN and are
        # replaced below by the other model's rank (or 0.5), so a mean-filled failed row no longer shifts the scored rows' ranks
        r = np.full(len(v), np.nan)
        if ok.any():
            r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
        return r

    out = np.zeros((n, len(labels)))
    rho_rr, rho_ro, rho_rr224, rho_224_384 = {}, {}, {}, {}
    for j, c in enumerate(labels):
        r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)   # [d957] each over its own non-failed rows
        rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))   # [d957] NaN only where both failed
        ro = pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows
        wr = W_RAPTOR_C.get(c, W_RAPTOR)
        b = wr * rr + (1.0 - wr) * ro
        b[rap_fail] = ro[rap_fail]
        b[our_fail] = rr[our_fail]
        b[rap_fail & our_fail] = 0.5
        out[:, j] = b
        if n >= 3:
            rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
            rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)
            rho_rr224[c] = round(float(pd.Series(raw224[c].values).corr(pd.Series(rsub224[c].values), method='spearman')), 4)
            rho_224_384[c] = round(float(pd.Series(raw224[c].values).corr(pd.Series(raw[c].values), method='spearman')), 4)
    if rho_rr and min(rho_rr.values()) < 0.99:
        raise RuntimeError('Raptor raw dump does not match its own submission ranks: %s' % rho_rr)
    if rho_rr224 and min(rho_rr224.values()) < 0.99:       # [d957]
        raise RuntimeError('Raptor-224 raw dump does not match its own submission ranks: %s' % rho_rr224)
    sub = pd.DataFrame(out.astype(np.float32), columns=labels)
    sub.insert(0, 'StudyInstanceUID', test_ids)
    sub = sub[cols]
    assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
    assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
    blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'raptor_per_class': W_RAPTOR_C, 'a224': A224}, 'rows': n, 'raptor_failed': int(rap_fail.sum()),
             'raptor384_failed': int(r384_fail.sum()), 'raptor224_failed': int(r224_fail.sum()), 'raptor224_seconds': round(RAPTOR224_SECONDS, 1),
             'spearman_raptor224_raw_vs_sub': rho_rr224, 'spearman_raptor224_vs_raptor384': rho_224_384,
             'raptor_half': '(1 - A224) rank(384) + A224 rank(224), each over its own non-failed rows',
             'ours_failed': int(our_fail.sum()), 'raptor_seconds': round(RAPTOR_SECONDS, 1),
             'total_seconds': round(time.time() - NB_T0, 1), 'spearman_raptor_raw_vs_sub': rho_rr,
             'spearman_raptor_vs_ours': rho_ro,
             'mode': 'blend', 'both_failed': int((rap_fail & our_fail).sum()),   # [v3] (> 50 % ours unscored stops above)
             'rank_basis': 'each model over its own non-failed rows'}
    if r384_fail.any() or r224_fail.any() or our_fail.any():   # [v3] loud: a fallback changed what this submission is
        print('BLEND_FALLBACK ours_failed=%d raptor384_failed=%d raptor224_failed=%d both_failed=%d of %d' % (
            int(our_fail.sum()), int(r384_fail.sum()), int(r224_fail.sum()), int((rap_fail & our_fail).sum()), n), flush=True)
    print('NOTEBOOK_COMPLETE', json.dumps(blend), flush=True)
    print(sub.head(), flush=True)
    # [port] A's cell writes submission.csv / blend_receipt.json here; the caller does. The fail masks are returned
    # in addition to A's outputs, for save_members.
    return sub, blend, {'raptor384': r384_fail, 'raptor224': r224_fail, 'r6': our_fail}


def run_reader(data_dir, reader_dir, out_dir, ckpt_glob):
    """Cell 9 (inner part): goodpjw2008's 2.5D ConvNeXt reader, infer.py over the fold checkpoints -> out_dir/_own.csv."""
    out_dir = Path(out_dir)
    own_src = out_dir / 'own_src'
    own_src.mkdir(parents=True, exist_ok=True)
    # [port] A writes its _PB_FILES dict and checks _PB_SHA before the 8.2 h check (an error there stops the notebook);
    # here the 3 files come from vendored/reader/ inside this function, so an error lands in the caller's fallback
    names = sorted(k for k in _vendored_sha() if k.startswith('reader/'))
    for key in names:
        materialize(key, own_src)
    print('Part B sources written:', sorted(Path(k).name for k in names), flush=True)

    _o_assets, _o_data = str(reader_dir), str(data_dir)
    if not os.path.exists(os.path.join(_o_assets, 'knee.py')):   # A: _o_find("knee.py")
        raise FileNotFoundError('knee.py not in reader dataset %s' % _o_assets)
    if not os.path.exists(os.path.join(_o_data, 'test_series.csv')):   # A: _o_find("test_series.csv")
        raise FileNotFoundError('test_series.csv not in %s' % _o_data)
    _o_ck = sorted(c for g in ckpt_glob.split() for c in glob.glob(_o_assets + "/" + g))
    assert _o_ck, "no own checkpoints found"
    _o_env = dict(os.environ)
    _o_target = str(out_dir / "_own_env")
    # check=False as in A: in A v7's log this install fails ("No matching distribution found for pylibjpeg-libjpeg")
    # and the leg runs on the image's decoders; a strict install would always end in the fallback
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _o_target,
                    "--find-links", _o_assets, "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls", "python-gdcm"], check=False)
    # [port] A's timm wheel lookup over every Kaggle input, kept as is (it decides which timm the reader imports)
    _o_timm = [w for pat in ["/kaggle/input/*/timm-*.whl", "/kaggle/input/*/*/timm-*.whl", "/kaggle/input/*/*/*/timm-*.whl"]
               for w in glob.glob(pat)]
    if _o_timm:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", _o_target, _o_timm[0]], check=False)
    _o_env["PYTHONPATH"] = _o_target + os.pathsep + _o_env.get("PYTHONPATH", "")
    _o_chk = ("import glob,os,numpy as np,pydicom;A=%r;ref=np.load(A+'/fixtures_ref.npy')\n"
              "for f in sorted(glob.glob(A+'/fixtures_*.dcm')):\n"
              "    try: print('decoder check',os.path.basename(f),'OK' if np.array_equal(pydicom.dcmread(f).pixel_array,ref) else 'MISMATCH',flush=True)\n"
              "    except Exception as e: print('decoder check',os.path.basename(f),'FAIL',repr(e)[:120],flush=True)\n") % _o_assets
    if os.path.exists(_o_assets + "/fixtures_ref.npy"):  # compressed-DICOM decoders must work in the leg's environment
        subprocess.run([sys.executable, "-c", _o_chk], env=_o_env, check=False)
    try:  # run on the GPU with the most free memory (A's own try: a failed pick runs on the default GPU)
        import gc as _o_gc
        import torch as _o_torch
        _o_gc.collect(); _o_torch.cuda.empty_cache()
        _o_free = [_o_torch.cuda.mem_get_info(i)[0] for i in range(_o_torch.cuda.device_count())]
        _o_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_o_free)), key=lambda i: _o_free[i]))
        print("own leg: free GPU memory (GB)", [round(f / 1e9, 1) for f in _o_free], flush=True)
    except Exception as _o_e:
        print("own leg: GPU pick skipped:", repr(_o_e))
    own_csv = out_dir / "_own.csv"
    subprocess.run([sys.executable, str(own_src / "infer.py"), _o_data, "test", str(own_csv)] + _o_ck,
                   check=True, env=_o_env)
    assert own_csv.is_file(), '%s missing after infer.py' % own_csv
    return own_csv


def blend_reader(stack, own, labels, own_w, own_w_per_target, n_ckpt, t0):
    """Part B, mode "end": rank-blend the reader into the stack submission per finding, then re-rank -> final DataFrame."""
    # OWN_MODE "T" / "H" and OWN_P33 > 0 are dead branches in A (OWN_MODE = "end", OWN_P33 = 0.0): dropped.
    # [port] n_ckpt and t0 (taken by the caller before run_reader; A: right before infer.py) only feed A's last print.
    OWN_W, OWN_W_PER_TARGET = own_w, own_w_per_target
    _o_stack = stack
    _o_own = own.set_index("StudyInstanceUID").loc[_o_stack.StudyInstanceUID].reset_index()
    _o_labels = [c for c in _o_stack.columns if c != "StudyInstanceUID"]
    assert _o_labels == list(labels), (_o_labels, labels)
    assert _o_own[_o_labels].notna().all().all(), "own predictions contain NaN"
    assert (_o_own[_o_labels].nunique() > 1).all(), "own predictions are constant"
    _o_rank = lambda d: d[_o_labels].rank(method="average", pct=True)  # noqa: E731
    _o_final = _o_stack.copy()
    _o_rs, _o_ro = _o_rank(_o_stack), _o_rank(_o_own)
    for _o_c in _o_labels:
        _o_w = float(OWN_W_PER_TARGET.get(_o_c, OWN_W)) if OWN_W_PER_TARGET else OWN_W
        _o_final[_o_c] = ((1 - _o_w) * _o_rs[_o_c] + _o_w * _o_ro[_o_c]).to_numpy()
    _o_final[_o_labels] = _o_final[_o_labels].rank(method="average", pct=True)
    if OWN_W_PER_TARGET:
        print("own leg: per-target weights", OWN_W_PER_TARGET)
    print(f"own leg: blended {n_ckpt} checkpoints at weight {OWN_W} in {time.time() - t0:.0f}s; "
          f"mean rank corr with stack {_o_rank(_o_stack).corrwith(_o_rank(_o_own)).mean():.3f}")
    return _o_final


def save_members(out_dir, members, fail_masks, weights):
    """Write members/<name>_prob.csv, members/<name>_rank.csv (per-target pct rank) and members/manifest.json."""
    assert members, 'no members'
    assert set(fail_masks) == set(members), ('fail masks must name exactly the members', sorted(fail_masks), sorted(members))
    d = Path(out_dir) / 'members'
    d.mkdir(parents=True, exist_ok=True)
    first = next(iter(members.values()))
    ids = first['StudyInstanceUID'].astype(str).tolist()
    labels = [c for c in first.columns if c != 'StudyInstanceUID']
    manifest = {'rows': len(ids), 'labels': labels, 'weights': weights, 'members': {}}
    for name, df in members.items():
        assert df['StudyInstanceUID'].astype(str).tolist() == ids, name + ': rows or row order differ from ' + next(iter(members))
        assert list(df.columns) == ['StudyInstanceUID'] + labels, name + ': columns differ'
        mask = np.asarray(fail_masks[name], dtype=bool)
        assert mask.shape == (len(ids),), (name, mask.shape)
        prob = df[['StudyInstanceUID'] + labels]
        rank = prob.copy()
        rank[labels] = prob[labels].rank(method='average', pct=True)
        prob.to_csv(d / (name + '_prob.csv'), index=False)
        rank.to_csv(d / (name + '_rank.csv'), index=False)
        manifest['members'][name] = {'prob': name + '_prob.csv', 'rank': name + '_rank.csv', 'failed': int(mask.sum()),
                                     'failed_ids': [u for u, m in zip(ids, mask) if m]}
    (d / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
