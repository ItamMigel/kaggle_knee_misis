"""Synthetic DICOM geometry and integration with the packaged model's cache reader."""
from __future__ import annotations

import ast
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import cv2
    import numpy as np
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
except ImportError as exc:
    raise unittest.SkipTest("Install requirements-preprocess.txt for DICOM tests") from exc

from src.knee_volume import SeriesError, canon_px_to_lps, preprocess_series
from src.local_preprocessing import (Config, build_study, center_from_lps, collapse_thin,
                                     flags, prepare_cache, render_view)

STUDY = "1.2.826.0.1.3680043.10.123.1"
SERIES = STUDY + ".1"


def dicom(k, study=STUDY, series=SERIES, orientation=None, position=None, missing_geometry=False):
    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = MRImageStorage
    fm.MediaStorageSOPInstanceUID = series + f".{k + 1}"
    ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
    ds.FrameOfReferenceUID = study + ".999"
    ds.Modality = "MR"
    ds.Rows, ds.Columns = 32, 32
    ds.ImageOrientationPatient = orientation or [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = position or [0, 0, 2 * k]
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 2
    ds.InstanceNumber = 100 - k  # InstanceNumber deliberately opposes geometric order.
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.RescaleSlope, ds.RescaleIntercept = 2, -10
    ds.SeriesDescription = "T2 FS"
    y, x = np.indices((32, 32))
    ds.PixelData = (x + 2 * y + 100 * k).astype("<u2").tobytes()
    if missing_geometry:
        del ds.ImagePositionPatient
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def write_study(root, uid=STUDY):
    directory = root / uid / (uid + ".1")
    directory.mkdir(parents=True)
    for k in (1, 2, 0):
        (directory / f"slice-{7-k}.dcm").write_bytes(dicom(k, uid, uid + ".1"))
    return directory.parent


class GeometryTest(unittest.TestCase):
    def test_order_is_physical_and_input_permutation_invariant(self):
        files = [(f"s{k}.dcm", dicom(k)) for k in (1, 0, 2)]
        a = preprocess_series(files, mm_per_px=1)
        b = preprocess_series(files[::-1], mm_per_px=1)
        np.testing.assert_array_equal(a.vol, b.vol)
        self.assertEqual([s["file"] for s in a.slices], ["s2.dcm", "s1.dcm", "s0.dcm"])
        np.testing.assert_allclose(a.pos_mm, [-4, -2, 0])
        np.testing.assert_allclose(canon_px_to_lps(a.meta, a.pos_mm, 1, 2, 3), [3, 2, 2])
        self.assertGreater(float(a.vol[0].mean()), float(a.vol[-1].mean()))

    def test_oblique_physical_coordinates_and_anchor_projection(self):
        angle = math.pi / 6
        c, s = math.cos(angle), math.sin(angle)
        orientation = [c, s, 0, -s, c, 0]
        sv = preprocess_series([(str(k), dicom(k, orientation=orientation)) for k in range(3)],
                               mm_per_px=1)
        point = canon_px_to_lps(sv.meta, sv.pos_mm, 1, 2, 3)
        np.testing.assert_allclose(point, [3*c - 2*s, 3*s + 2*c, 2], atol=1e-6)
        np.testing.assert_allclose(center_from_lps(sv, point), [2, 3], atol=1e-6)

    def test_duplicate_positions_are_logged_and_missing_geometry_fails(self):
        files = [("a", dicom(0)), ("b", dicom(1)), ("duplicate", dicom(1))]
        sv = preprocess_series(files, mm_per_px=1)
        self.assertEqual(len(sv.vol), 2)
        self.assertEqual(sv.dropped[0]["reason"], "duplicate_position")
        with self.assertRaises(SeriesError):
            preprocess_series([("bad", dicom(0, missing_geometry=True))])

    def test_thin_slabs_keep_the_last_slice(self):
        volume = np.broadcast_to(np.arange(9, dtype=np.uint8)[:, None, None], (9, 8, 8)).copy()
        result, pos = collapse_thin(volume, np.arange(9), Config())
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(pos, [1.5, 5.5, 8])
        self.assertEqual(int(result[-1, 0, 0]), 8)

    def test_physical_resize_preserves_aspect_and_pads(self):
        y, x = np.indices((21, 41))
        circle = (((y-10)**2 + ((x-20)*.5)**2) <= 25).astype(np.uint8) * 255
        out = render_view(circle[None], (1, .5), 42, 21, (10, 20))[0]
        rows, cols = np.where(out > 127)
        self.assertLessEqual(abs(np.ptp(rows) - np.ptp(cols)), 1)
        padded = render_view(circle[None], (1, .5), 42, 60, (10, 20))[0]
        self.assertEqual(int(padded[0].sum() + padded[-1].sum()), 0)
        self.assertGreater(int(padded[21, 21]), 0)

    def test_unknown_acquisition_is_not_a_negative_label(self):
        self.assertEqual(flags({}), (-1, -1))
        self.assertEqual(flags({"SeriesDescription": "T2 FS"}), (1, 1))
        self.assertEqual(flags({"SeriesDescription": "T1 NON-FS"}), (0, 0))

    def test_invalid_geometry_and_mixed_uids_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_study(Path(tmp))
            bad = path / SERIES / "bad.dcm"
            bad.write_bytes(dicom(3, orientation=[0, 0, 0, 0, 1, 0]))
            with self.assertRaisesRegex(ValueError, "invalid DICOM geometry"):
                build_study(path, Config(size=32))
            bad.write_bytes(dicom(3, study=STUDY + ".9"))
            with self.assertRaisesRegex(ValueError, "inconsistent StudyInstanceUID"):
                build_study(path, Config(size=32))


class CacheIntegrationTest(unittest.TestCase):
    def test_cache_is_read_by_the_actual_compact_dataset(self):
        # Execute the saved dataset and its window sampler, replacing only the
        # tensor wrapper with NumPy so this data-path test needs no torch install.
        tree = ast.parse((ROOT / "sources/B/runtime/heliosli/compact_train.py").read_text())
        names = {"covering_centers", "phys_step", "window_rows", "Studies"}
        subset = ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in names], type_ignores=[])
        fake_torch = types.SimpleNamespace(utils=types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object)),
                                           from_numpy=lambda x: x)
        namespace = dict(np=np, math=math, json=json, os=os, torch=fake_torch)
        exec(compile(subset, "compact_train_dataset", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "dicoms"
            write_study(data)
            other = STUDY + ".2"
            write_study(data, other)
            cache = root / "cache"
            index = prepare_cache(data, cache, Config(size=32, mm_per_px=1))
            self.assertEqual(len(index["done"]), 2)
            first_end = sum(index["done"][STUDY][:2])
            self.assertEqual(index["done"][other][0], first_end)
            ds = namespace["Studies"](str(cache), [STUDY, other], 4, "eval", 42, eval_stride=1, phys_mm=3.5)
            for i in range(2):
                images, meta, sample_idx = ds[i]
                self.assertEqual(images.shape[1:], (3, 32, 32))
                self.assertEqual(meta.shape, (len(images), 6))
                self.assertEqual(set(meta[:, 1]), {0, 1, 4, 5})
                self.assertTrue(np.isfinite(meta).all())
                self.assertEqual(sample_idx, i)
            train = namespace["Studies"](str(cache), [STUDY], 2, "train", 42, phys_mm=3.5)
            self.assertEqual(len(train[0][0]), 2)
            with self.assertRaises(FileExistsError):
                prepare_cache(data, cache, Config(size=32))
            with self.assertRaises(ValueError):
                prepare_cache(data, root / "other", uids=["../escape"])


if __name__ == "__main__":
    unittest.main()
