"""Kernel environment for inference: offline wheel install, a report of the preprocessing stack, parity fingerprints,
a per-stage watchdog.

Import this module before anything that imports pydicom: `install_wheels` must run first, and
nothing here imports pydicom at module level.
"""
import contextlib
import faulthandler
import hashlib
import importlib
import importlib.metadata as md
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# transfer syntaxes whose decoder availability is reported (the competition data uses LE, JPEG and JPEG 2000)
TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2.1": "ExplicitVRLittleEndian",
    "1.2.840.10008.1.2": "ImplicitVRLittleEndian",
    "1.2.840.10008.1.2.4.70": "JPEGLosslessSV1",
    "1.2.840.10008.1.2.4.57": "JPEGLossless",
    "1.2.840.10008.1.2.4.50": "JPEGBaseline8Bit",
    "1.2.840.10008.1.2.4.80": "JPEGLSLossless",
    "1.2.840.10008.1.2.4.90": "JPEG2000Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG2000",
    "1.2.840.10008.1.2.5": "RLELossless",
}
REPORTED = ("numpy", "opencv-python", "opencv-python-headless", "opencv-contrib-python", "pydicom", "pylibjpeg",
            "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pylibjpeg-rle", "python-gdcm", "torch", "timm")


def install_wheels(wheels, pins, target=None):
    """pip-installs the given wheel files offline without deps (into `target` if set) and checks versions == `pins`.

    With `target` the packages land in that directory only (put it first on PYTHONPATH of a
    subprocess); the running interpreter is untouched and the version check reads that directory.
    """
    wheels = [Path(w) for w in wheels]
    if not wheels or not all(w.is_file() and w.suffix == ".whl" for w in wheels):
        raise FileNotFoundError(f"expected existing .whl files, got {wheels}")
    if target is None and "pydicom" in sys.modules:
        raise RuntimeError("pydicom is already imported: install_wheels must run before any pydicom import")
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet"]
    if target is not None:
        cmd += ["--target", str(target)]
    subprocess.run(cmd + [str(w) for w in wheels], check=True)
    importlib.invalidate_caches()
    search = None if target is None else [str(target)]
    got = {}
    for p in pins:
        found = [d.version for d in md.distributions(name=p, path=search)] if search else [md.version(p)]
        if not found:
            raise RuntimeError(f"{p} is not installed in {target}")
        got[p] = found[0]
    bad = {p: (got[p], v) for p, v in pins.items() if got[p] != v}
    if bad:
        raise RuntimeError(f"installed{'' if target is None else ' into ' + str(target)} != pinned (got, want): {bad}")
    return got


def env_report():
    """Python and package versions plus pydicom's available pixel-decoder plugins per transfer syntax."""
    import cv2
    import pydicom
    from pydicom.pixels import get_decoder

    versions = {}
    for p in REPORTED:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    decoders = {}
    for uid, name in TRANSFER_SYNTAXES.items():
        dec = get_decoder(pydicom.uid.UID(uid))
        decoders[name] = sorted(dec.available_plugins) if dec.is_available else []
    return {"python": sys.version.split()[0], "versions": versions, "cv2_build": cv2.__version__,
            "numpy_config_simd": np.__config__.CONFIG.get("SIMD Extensions", {}).get("found", None),
            "decoders": decoders}


def volume_fingerprint(vol, pos_mm, meta):
    """Parity fingerprint of one preprocessed series: shape, sha256 (16 hex) of vol and pos_mm, voxel sum, lo/hi."""
    vol = np.ascontiguousarray(vol)
    pos = np.ascontiguousarray(pos_mm, dtype=np.float32)
    return {"shape": list(vol.shape), "sha_vol": hashlib.sha256(vol.tobytes()).hexdigest()[:16],
            "sha_pos": hashlib.sha256(pos.tobytes()).hexdigest()[:16], "sum": int(vol.sum(dtype=np.int64)),
            "lo": float(meta["lo"]), "hi": float(meta["hi"])}


_ACTIVE_STAGE: list = []  # name of the running stage, if any


@contextlib.contextmanager
def stage(name, budget_s):
    """Watchdog for one kernel stage: prints start and end; past `budget_s` seconds it dumps every thread's
    traceback to stderr and exits the process with code 1 (faulthandler's own thread, so a hang inside C
    code or a child wait dies too, loudly, instead of running into the 9 h limit). Stages do not nest."""
    if not budget_s > 0:
        raise ValueError(f"budget_s must be > 0, got {budget_s}")
    if _ACTIVE_STAGE:
        raise RuntimeError(f"stage {name!r} started inside stage {_ACTIVE_STAGE[0]!r}: stages do not nest")
    _ACTIVE_STAGE.append(name)
    t0 = time.time()
    print(f"[stage] {name}: start, budget {budget_s:.0f}s", flush=True)
    faulthandler.dump_traceback_later(budget_s, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        _ACTIVE_STAGE.pop()
    print(f"[stage] {name}: done in {time.time() - t0:.0f}s", flush=True)
