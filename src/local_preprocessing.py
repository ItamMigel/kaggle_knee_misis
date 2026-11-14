"""Approximate DICOM-to-MIL cache, shared by local training and inference.

This is a separate numerical pipeline, not a drop-in cloud_code implementation.
The slice-cache layout is compatible with compact_train.Studies, but intensities,
slot semantics, laterality and anatomical anchors differ. Retrain/evaluate models
with this cache; historical checkpoint scores do not carry over to this pipeline.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom

from src.knee_volume import SeriesError, SeriesVolume, preprocess_series, read_series_dir

META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing",
             "extent_130", "fluid", "fatsat", "slice_order", "n_slices", "real_series_idx"]
PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
FORMAT = "local-dicom-views-v1"


@dataclass(frozen=True)
class Config:
    size: int = 320
    mm_per_px: float = 0.6
    local_mm: float = 100.0
    thin_mm: float = 1.75
    slab_mm: float = 3.5
    clip_low: float = 0.5
    clip_high: float = 99.5

    def __post_init__(self):
        if not isinstance(self.size, int) or self.size < 8:
            raise ValueError("size must be an integer >= 8")
        values = [self.mm_per_px, self.local_mm, self.thin_mm, self.slab_mm]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("physical dimensions must be finite and positive")
        if not 0 <= self.clip_low < self.clip_high <= 100:
            raise ValueError("invalid intensity percentiles")


def render_view(volume, spacing_rc, size, extent_mm, center_rc):
    """Square physical FOV, aspect-preserving bilinear sampling, black padding.

    Coordinates address pixel centres. The same mapping applies to every slice.
    No gigantic intermediate crop is allocated for a small image or large FOV.
    """
    sy, sx = spacing_rc
    axis = (np.arange(size, dtype=np.float32) + 0.5 - size / 2) * (extent_mm / size)
    x, y = np.meshgrid(center_rc[1] + axis / sx, center_rc[0] + axis / sy)
    return np.stack([cv2.remap(img, x.astype(np.float32), y.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                     for img in volume])


def collapse_thin(volume, positions, config):
    """Average disjoint thin-slice slabs, retaining the final partial slab."""
    positions = np.asarray(positions, np.float64)
    if len(positions) < 2:
        return volume, positions
    delta = np.diff(positions)
    spacing = float(np.median(delta))
    if not (0 < spacing < config.thin_mm) or np.max(delta) > 1.5 * spacing:
        return volume, positions
    group = max(1, round(config.slab_mm / spacing))
    if group == 1 or len(positions) < 2 * group:
        return volume, positions
    starts = range(0, len(positions), group)
    pixels = [np.rint(volume[i:i + group].astype(np.float32).mean(0)).astype(np.uint8)
              for i in starts]
    pos = [positions[i:i + group].mean() for i in starts]
    return np.stack(pixels), np.asarray(pos)


def flags(header):
    """Conservative acquisition hints; unknown is -1, not a negative label."""
    import re
    text = " ".join(str(header.get(k, "")) for k in
                    ("SeriesDescription", "ProtocolName", "SequenceName", "ScanOptions")).upper()
    fat = 1 if re.search(r"FAT[ _-]?(SAT|SUP)|\b(FS|SPAIR|SPIR|STIR)\b", text) else -1
    if fat == -1 and re.search(r"\b(NON[ _-]?FS|NO[ _-]?FS)\b", text):
        fat = 0
    # Explicit negative tokens take precedence over a matching suffix FS.
    if re.search(r"\b(NON[ _-]?FS|NO[ _-]?FS)\b", text):
        fat = 0
    fluid = 1 if re.search(r"\b(T2|PD|STIR)\b", text) else -1
    if fluid == -1 and re.search(r"\bT1\b", text):
        fluid = 0
    return fluid, fat


def center_from_lps(sv: SeriesVolume, point):
    point = np.asarray(point, np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("joint_lps must be three finite LPS coordinates in mm")
    delta = point - np.asarray(sv.meta["origin_lps"])
    sy, sx = sv.meta["spacing_rc"]
    return (float(delta @ np.asarray(sv.meta["row_dir"]) / sy),
            float(delta @ np.asarray(sv.meta["col_dir"]) / sx))


def make_views(sv: SeriesVolume, config: Config, joint_lps=None):
    """Full FOV plus heuristic joint, anterior and posterior 100-mm views."""
    h, w = sv.vol.shape[-2:]
    sy, sx = sv.meta["spacing_rc"]
    field = ((h - 1) / 2, (w - 1) / 2)
    if joint_lps is not None:
        joint = center_from_lps(sv, joint_lps)
        if not (0 <= joint[0] < h and 0 <= joint[1] < w):
            raise ValueError("joint_lps projects outside the series field")
        anchor_kind = "provided_lps"
    elif sv.meta.get("fg_r0") is not None:
        joint = ((sv.meta["fg_r0"] + sv.meta["fg_r1"] - 1) / 2,
                 (sv.meta["fg_c0"] + sv.meta["fg_c1"] - 1) / 2)
        anchor_kind = "foreground_bbox"
    else:
        joint, anchor_kind = field, "field_center"

    def shifted(lps_offset):
        # Project physical anterior/superior/etc. into the actual oblique plane.
        offset = np.asarray(lps_offset)
        return (joint[0] + float(offset @ sv.meta["row_dir"]) / sy,
                joint[1] + float(offset @ sv.meta["col_dir"]) / sx)

    views = [(0, max(h * sy, w * sx), field)]
    plane = sv.meta["plane_computed"]
    if plane == "Coronal":
        views += [(2, config.local_mm, shifted((-30, 0, 0))),
                  (3, config.local_mm, shifted((30, 0, 0)))]
    else:
        anterior = (0, -40, 25) if plane == "Sagittal" else (0, -40, 0)
        posterior = (0, 35, -5) if plane == "Sagittal" else (0, 35, 0)
        views += [(1, config.local_mm, joint), (4, config.local_mm, shifted(anterior)),
                  (5, config.local_mm, shifted(posterior))]
    return views, anchor_kind


def build_study(study_dir: Path, config: Config, anchors=None, skip_invalid=False):
    """Read <study>/<series>/<dicom>. Returns arrays and an auditable summary."""
    study_dir = Path(study_dir)
    uid = study_dir.name
    pixels, metadata, blocks, diagnostics = [], [], [], []
    offset = 0
    for real_idx, series_dir in enumerate(sorted(p for p in study_dir.iterdir() if p.is_dir())):
        try:
            files = read_series_dir(series_dir)
            # Geometry errors must not turn into NaNs or huge allocations downstream.
            for name, content in files:
                try:
                    ds = pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
                except pydicom.errors.InvalidDicomError:
                    continue  # preprocess_series records non-DICOM files as dropped
                for tag, expected in (("StudyInstanceUID", uid), ("SeriesInstanceUID", series_dir.name)):
                    if ds.get(tag) and str(ds.get(tag)) != expected:
                        raise ValueError(f"{name}: inconsistent {tag}")
                if all(ds.get(k) is not None for k in
                       ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing")):
                    iop = np.asarray(ds.ImageOrientationPatient, float)
                    ipp = np.asarray(ds.ImagePositionPatient, float)
                    ps = np.asarray(ds.PixelSpacing, float)
                    valid = (iop.shape == (6,) and ipp.shape == (3,) and ps.shape == (2,)
                             and np.isfinite(iop).all() and np.isfinite(ipp).all()
                             and np.isfinite(ps).all() and (ps > 0).all())
                    if valid:
                        valid = (abs(np.linalg.norm(iop[:3]) - 1) < .01
                                 and abs(np.linalg.norm(iop[3:]) - 1) < .01
                                 and abs(iop[:3] @ iop[3:]) < .01)
                    if not valid:
                        raise ValueError(f"{name}: invalid DICOM geometry")
            sv = preprocess_series(files, config.mm_per_px, (config.clip_low, config.clip_high))
            if sv.meta["study"] != uid or sv.meta["series"] != series_dir.name:
                raise ValueError(f"DICOM UIDs disagree with directory {uid}/{series_dir.name}")
            header_file = series_dir / sv.slices[0]["file"]
            header = pydicom.dcmread(header_file, stop_before_pixels=True)
            fluid, fat = flags(header)
            plane = PLANES[sv.meta["plane_computed"]]
            slot = plane * 3 + (fat if fat in (0, 1) else 2)
            anchor = (anchors or {}).get(series_dir.name, {}).get("joint_lps")
            views, anchor_kind = make_views(sv, config, anchor)
            vol, positions = collapse_thin(sv.vol, sv.pos_mm, config)
            n = len(vol)
            spacing = float(np.median(np.diff(positions))) if n > 1 else 0.0
            relative = 2 * (positions - positions[0]) / max(float(np.ptp(positions)), 1e-6) - 1
            block = {"series": series_dir.name, "plane": sv.meta["plane_computed"],
                     "n_slices": n, "views": []}
            for view_id, extent, center in views:
                block["views"].append([view_id, offset])
                pixels.append(render_view(vol, sv.meta["spacing_rc"], config.size, extent, center))
                metadata.extend([[len(blocks), slot, plane, view_id, relative[k], np.log1p(spacing),
                                  extent / 130, fluid, fat, k, n, real_idx] for k in range(n)])
                offset += n
            blocks.append(block)
            diagnostics.append({"series": series_dir.name, "input_slices": sv.meta["n_files"],
                                "cache_slices": n, "dropped": sv.dropped,
                                "irregular_spacing": sv.meta["gap_irregular"],
                                "anchor": anchor_kind, "fluid": fluid, "fat": fat})
        except (SeriesError, ValueError) as exc:
            if not skip_invalid:
                raise
            diagnostics.append({"series": series_dir.name, "error": str(exc)})
    if not blocks:
        raise ValueError(f"No usable series in {study_dir}")
    return {"uid": uid, "px": np.concatenate(pixels), "meta": np.asarray(metadata, np.float32),
            "blocks": blocks, "diagnostics": diagnostics}


def prepare_cache(data: Path, output: Path, config=Config(), uids=None, anchors=None, skip_invalid=False):
    """Two-pass disk-backed concatenation. Refuses overwrites and fails on bad studies.

    index.json is written last and marks a complete cache. Peak RAM is one study;
    the second pass uses memmaps. Failed work stays in output for inspection.
    """
    data, output = Path(data), Path(output)
    if not data.is_dir():
        raise FileNotFoundError(data)
    uids = sorted(p.name for p in data.iterdir() if p.is_dir()) if uids is None else list(uids)
    if not uids or len(set(uids)) != len(uids):
        raise ValueError("study list must be nonempty and unique")
    if any(not u or u in (".", "..") or "/" in u or "\\" in u for u in uids):
        raise ValueError("invalid study directory identifier")
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "parts"
    temporary.mkdir()
    index = {"format": FORMAT, "approximate": True, "config": asdict(config),
             "meta_columns": META_COLS, "blocks": {}, "done": {}}
    chunks, offset = [], 0
    for uid in uids:
        item = build_study(data / uid, config, (anchors or {}).get(uid), skip_invalid)
        key = hashlib.sha256(uid.encode()).hexdigest()
        px_path, meta_path = temporary / f"{key}.px.npy", temporary / f"{key}.meta.npy"
        np.save(px_path, item["px"], allow_pickle=False)
        np.save(meta_path, item["meta"], allow_pickle=False)
        n = len(item["px"])
        chunks.append((px_path, meta_path, n))
        for block in item["blocks"]:
            block["views"] = [[v, start + offset] for v, start in block["views"]]
        index["blocks"][uid] = item["blocks"]
        index["done"][uid] = [offset, n, len(item["blocks"])]
        (temporary / f"{key}.json").write_text(json.dumps(item["diagnostics"], indent=2) + "\n")
        offset += n
        del item
    px = np.lib.format.open_memmap(output / "px.u8.npy", mode="w+", dtype=np.uint8,
                                   shape=(offset, config.size, config.size))
    meta = np.lib.format.open_memmap(output / "meta.f32.npy", mode="w+", dtype=np.float32,
                                     shape=(offset, len(META_COLS)))
    start = 0
    for px_path, meta_path, n in chunks:
        px[start:start + n] = np.load(px_path, mmap_mode="r", allow_pickle=False)
        meta[start:start + n] = np.load(meta_path, mmap_mode="r", allow_pickle=False)
        start += n
    px.flush()
    meta.flush()
    del px, meta
    for px_path, meta_path, _ in chunks:
        px_path.unlink()
        meta_path.unlink()
    temporary.rename(output / "diagnostics")
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index
