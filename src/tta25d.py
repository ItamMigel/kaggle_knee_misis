"""Test-time views for exp-0004's 2.5D reader: the anatomical left-right mirror of a study as a cache wrapper.

The exp-0002 cache is canonical in-plane (src/knee_volume.py CANON, LPS): Sagittal slices run patient
right->left, Coronal and Axial columns run patient right->left. exp-0004 keeps native knee sides (no L/R
canonicalisation), so a whole study mirrored along the patient L-R axis is in-distribution (a right knee
looks like a left one and vice versa): sagittal slice order reversed, coronal and axial columns reversed.
"""
import numpy as np

from src.reader25d import DEPTH_HI, DEPTH_LO

MIRROR_PLANES = ("Sagittal", "Coronal", "Axial")


class MirrorCache:
    """A KneeVolCache as reader25d / infer25d.predict_studies see it, every study mirrored along the patient L-R axis.

    Implements only what predict_studies touches (.series via cache_study_table and cache_depth_rows,
    .meta / .pos_mm via cache_depth_rows, .load via StudyDataset) and passes nothing else through, so a
    new dependency fails here loudly (the pattern of exp-0005's joint_view.JointCache).
    - .series: the cache's table, unchanged (the same object).
    - .load(series, mmap): cache.load, then by meta plane_computed: Sagittal -> vol[::-1], pos_mm ->
      -pos_mm[::-1] (still strictly increasing, same spacing); Coronal / Axial -> vol[:, :, ::-1], pos_mm
      unchanged. Reversed views (of the mmap by default): StudyDataset copies the slices it needs
      (np.asarray(vol[need]) + crop_fn). meta is unchanged, the original geometry, so no exp-0002 geometry
      helper may be applied to a mirrored volume.
    - .meta / .pos_mm: the cache's own, original geometry and slice order. That is safe although .load
      reverses sagittal stacks: joint_depth_fractions restricts only axial stacks, whose slice order the
      mirror keeps (only their columns flip), and gives every other series (DEPTH_LO, DEPTH_HI), which is
      symmetric, so the window range is the same in either slice order. The symmetry is asserted here.
    """

    def __init__(self, cache):
        if abs(DEPTH_LO + DEPTH_HI - 1.0) > 1e-12:
            raise ValueError(f"reader25d depth range ({DEPTH_LO}, {DEPTH_HI}) is not symmetric: the sagittal "
                             "reversal in .load would no longer match the .pos_mm-based depth fractions")
        self.series = cache.series
        self._cache = cache

    def load(self, series, mmap=True):
        """(vol, pos_mm, meta) mirrored along patient L-R: sagittal slice order reversed, coronal / axial columns."""
        vol, pos_mm, meta = self._cache.load(series, mmap=mmap)
        plane = meta["plane_computed"]
        if plane == "Sagittal":
            pos_mm = np.asarray(pos_mm)
            if len(pos_mm) != len(vol) or not (np.diff(pos_mm) > 0).all():
                raise ValueError(f"series {series}: pos_mm is not strictly increasing over {len(vol)} slices")
            return vol[::-1], -pos_mm[::-1], meta
        if plane in ("Coronal", "Axial"):
            return vol[:, :, ::-1], pos_mm, meta
        raise ValueError(f"series {series}: plane_computed {plane!r} is not one of {MIRROR_PLANES}")

    def meta(self, series):
        """The cache's meta of a series (original geometry)."""
        return self._cache.meta(series)

    def pos_mm(self, series):
        """The cache's slice positions of a series (original slice order, see the class docstring)."""
        return self._cache.pos_mm(series)
