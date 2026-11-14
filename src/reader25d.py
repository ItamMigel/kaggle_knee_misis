"""2.5D study reader: slot table, physical 3-slice windows, crop hook, GPU views and the KneeNet model.

Port of goodpjw's `knee.py` (goodpjw2008/rsna-knee-2-5d-convnext-reader). Changes against it:
the volume loader, the crop and the per-worker rng are injected; windows take physical neighbours
(`window_slices`); normalisation constants are explicit; gradient checkpointing is a switch.
"""
import functools
import math

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.knee_labels import LABELS
from src.knee_volume import canon_px_to_lps

PLANES = ["Sagittal", "Coronal", "Axial"]
# slot = plane index + 3 * (1 - fat_suppressed): fat-suppressed planes first
N_SLOTS = 6
# goodpjw's share of the stack depth that the window centres spread over
DEPTH_LO, DEPTH_HI = 0.04, 0.96
# where a restricted axial stack took its joint position from (joint_depth_fractions)
DEPTH_SOURCES = ("coronal_fs", "coronal", "stack_middle_no_coronal", "stack_middle_outside")
_DEPTH_ROW_KEYS = {"study", "series", "fat_sup", "meta", "pos_mm"}
_FG_KEYS = ("fg_r0", "fg_r1", "fg_c0", "fg_c1", "fg_z0", "fg_z1")


def slot_of(plane, fat_sup):
    """Slot index 0..5 of a series: plane index + 3 * (1 - Fat_Suppression)."""
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


def build_study_table(series_df, n_slices):
    """{study: [[series ids of slot 0], ...]}: series with <3 slices dropped, candidates largest first.

    series_df: StudyInstanceUID, SeriesInstanceUID, Fat_Suppression, Anatomical_Plane.
    n_slices: dict series -> usable slice count; every series of series_df must be in it (an
    unreadable series is passed as 0 explicitly). Every study of series_df gets an entry, possibly
    with all slots empty.
    """
    missing = [s for s in series_df.SeriesInstanceUID if s not in n_slices]
    if missing:
        raise KeyError(f"n_slices has no entry for {len(missing)} series, e.g. {missing[0]}; "
                       "pass 0 for series without a usable volume")
    table = {}
    for st, sr, fs, pl in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID,
                              series_df.Fat_Suppression, series_df.Anatomical_Plane):
        slots = table.setdefault(st, [[] for _ in range(N_SLOTS)])
        if n_slices[sr] < 3:
            continue
        slots[slot_of(pl, fs)].append(sr)
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda s: -n_slices[s])
    return table


def cache_study_table(cache_series, comp_series, studies):
    """build_study_table for `studies` from a KneeVolCache.series table; failed series count as 0 slices.

    cache_series: KneeVolCache.series (study, series, Z, error_code, Anatomical_Plane, Fat_Suppression).
    comp_series: the competition *_series.csv; every study must have exactly its CSV series in the cache.
    """
    studies = set(studies)
    cs = cache_series[cache_series.study.isin(studies)]
    csv = comp_series[comp_series.StudyInstanceUID.isin(studies)]
    absent = studies - set(cs.study)
    if absent:
        raise KeyError(f"{len(absent)} studies are not in the cache, e.g. {next(iter(absent))}")
    if set(zip(cs.study, cs.series)) != set(zip(csv.StudyInstanceUID, csv.SeriesInstanceUID)):
        raise ValueError("cache series differ from the competition series CSV for the requested studies")
    ok = cs.error_code.isna()
    n_slices = {sr: (int(z) if good else 0) for sr, z, good in zip(cs.series, cs.Z, ok)}
    df = pd.DataFrame({"StudyInstanceUID": cs.study.values, "SeriesInstanceUID": cs.series.values,
                       "Fat_Suppression": cs.Fat_Suppression.values, "Anatomical_Plane": cs.Anatomical_Plane.values})
    return build_study_table(df, n_slices)


def _coronal_joint(rows):
    """(z_joint LPS, series, source) from a study's coronal fg-bbox centre, or None without a usable coronal.

    Candidates: plane_computed == Coronal with an fg bbox; fat-suppressed first, then the largest Z.
    """
    cands = []
    for r in rows:
        m = r["meta"]
        if m["plane_computed"] != "Coronal":
            continue
        present = [m[f] is not None for f in _FG_KEYS]
        if any(present) and not all(present):
            raise ValueError(f"series {r['series']}: partial fg bbox {[m[f] for f in _FG_KEYS]}")
        if all(present):
            cands.append(r)
    if not cands:
        return None
    best = min(cands, key=lambda r: (-int(r["fat_sup"]), -r["meta"]["Z"], r["series"]))
    m = best["meta"]
    kz, rr, cc = ((m[a] + m[b] - 1) / 2 for a, b in (("fg_z0", "fg_z1"), ("fg_r0", "fg_r1"), ("fg_c0", "fg_c1")))
    z = float(canon_px_to_lps(m, best["pos_mm"], kz, rr, cc)[2])
    return z, best["series"], "coronal_fs" if int(best["fat_sup"]) == 1 else "coronal"


def joint_depth_fractions(rows, half_mm, max_extent_mm, lo, hi):
    """Per-series (lo, hi) depth fractions for window_centres: long axial stacks keep only the slices near the joint.

    rows: one dict per usable series with keys study, series, fat_sup (CSV Fat_Suppression), meta
    (knee_volume.META_FIELDS) and pos_mm. An axial stack (plane_computed) longer than max_extent_mm
    keeps the slices whose LPS z (at the image centre) is within half_mm of the joint: the z of the
    fg-bbox centre of the study's coronal series (fat-suppressed first, then the largest Z), or the
    stack middle without a usable coronal or when fewer than 3 slices qualify. Every other series
    gets (lo, hi). half_mm = max_extent_mm = 0 switches the restriction off.
    Returns (frac {series: (lo_frac, hi_frac)}, report: one row per restricted axial series).
    """
    cols = ["study", "series", "n_slices", "extent_mm", "source", "joint_series", "joint_offset_mm",
            "n_kept", "i_min", "i_max", "lo_frac", "hi_frac"]
    if not 0 <= lo < hi <= 1:
        raise ValueError(f"need 0 <= lo < hi <= 1, got {lo}, {hi}")
    if (half_mm == 0) != (max_extent_mm == 0) or half_mm < 0 or max_extent_mm < 0:
        raise ValueError(f"half_mm and max_extent_mm must be both 0 (no restriction) or both > 0, "
                         f"got {half_mm}, {max_extent_mm}")
    rows = list(rows)
    by_study = {}
    for r in rows:
        if set(r) != _DEPTH_ROW_KEYS:
            raise ValueError(f"row keys {sorted(r)} != {sorted(_DEPTH_ROW_KEYS)}")
        if int(r["fat_sup"]) not in (0, 1):
            raise ValueError(f"series {r['series']}: fat_sup {r['fat_sup']} is not 0/1")
        by_study.setdefault(r["study"], []).append(r)
    series = [r["series"] for r in rows]
    if len(set(series)) != len(series):
        raise ValueError("rows contain a series twice")
    frac = {s: (lo, hi) for s in series}
    report = []
    if max_extent_mm == 0:
        return frac, pd.DataFrame(report, columns=cols)
    for study, srows in by_study.items():
        joint = None
        for r in srows:
            m, pos = r["meta"], np.asarray(r["pos_mm"], np.float64)
            n = len(pos)
            if m["plane_computed"] != "Axial" or pos[-1] - pos[0] <= max_extent_mm:
                continue
            if joint is None:
                joint = _coronal_joint(srows) or (None, None, "stack_middle_no_coronal")
            # z is affine in pos_mm: interpolate between the first and the last slice at the image centre
            rc, cc = (m["H"] - 1) / 2, (m["W"] - 1) / 2
            z0, z1 = (float(canon_px_to_lps(m, pos, k, rc, cc)[2]) for k in (0, n - 1))
            z = z0 + (pos - pos[0]) / (pos[-1] - pos[0]) * (z1 - z0)
            z_mid = (z0 + z1) / 2
            z_joint, joint_series, source = joint
            if z_joint is None:
                z_joint = z_mid
            keep = np.nonzero(np.abs(z - z_joint) <= half_mm)[0]
            if len(keep) < 3:
                source = "stack_middle_outside"
                keep = np.nonzero(np.abs(z - z_mid) <= half_mm)[0]
                if len(keep) < 3:
                    raise ValueError(f"series {r['series']}: fewer than 3 slices within {half_mm} mm of the stack "
                                     f"middle (spacing {np.diff(pos).mean():.1f} mm)")
            i_min, i_max = int(keep[0]), int(keep[-1])
            frac[r["series"]] = (i_min / (n - 1), i_max / (n - 1))
            report.append([study, r["series"], n, float(pos[-1] - pos[0]), source, joint_series,
                           z_joint - z_mid, i_max - i_min + 1, i_min, i_max, *frac[r["series"]]])
    return frac, pd.DataFrame(report, columns=cols)


def depth_source_counts(report):
    """{source: n restricted series} over all DEPTH_SOURCES (zeros included) of a joint_depth_fractions report."""
    unknown = set(report.source) - set(DEPTH_SOURCES)
    if unknown:
        raise ValueError(f"unknown depth sources {sorted(unknown)}")
    return {s: int((report.source == s).sum()) for s in DEPTH_SOURCES}


def cache_depth_rows(cache, studies):
    """joint_depth_fractions rows (meta, pos_mm, CSV Fat_Suppression) of every error-free series of `studies` in a KneeVolCache."""
    studies = set(studies)
    absent = studies - set(cache.series.study)
    if absent:
        raise KeyError(f"{len(absent)} studies are not in the cache, e.g. {next(iter(absent))}")
    s = cache.series[cache.series.study.isin(studies) & cache.series.error_code.isna()]
    return [{"study": st, "series": sr, "fat_sup": int(fs), "meta": cache.meta(sr), "pos_mm": cache.pos_mm(sr)}
            for st, sr, fs in zip(s.study, s.series, s.Fat_Suppression)]


def window_centres(n, k, train, rng, lo=DEPTH_LO, hi=DEPTH_HI):
    """k slice indices (centres of 3-slice windows) spread over the series; jittered by `rng` in training."""
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + rng.random(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


def window_slices(pos_mm, centres, gap_mm):
    """int [k, 3] slice indices (below, centre, above): the slices nearest to pos[c] -/+ gap_mm.

    The neighbours are forced to differ from the centre by >= 1 index and clipped to [0, n-1], so
    the result equals goodpjw's (c-1, c, c+1) whenever the slice spacing is >= gap_mm. pos_mm must
    be strictly monotonic (either direction); "below" is always the lower index.
    """
    p = np.asarray(pos_mm, np.float64)
    c = np.asarray(centres, np.int64)
    n = len(p)
    if p.ndim != 1 or n < 2:
        raise ValueError(f"pos_mm must be 1-D with >= 2 slices, got shape {p.shape}")
    if not gap_mm > 0:
        raise ValueError(f"gap_mm must be positive, got {gap_mm}")
    if c.ndim != 1 or c.min() < 0 or c.max() > n - 1:
        raise ValueError(f"centres must be 1-D indices in [0, {n - 1}], got {c}")
    d = np.diff(p)
    if not ((d > 0).all() or (d < 0).all()):
        raise ValueError("pos_mm is not strictly monotonic (duplicate or unsorted slices)")
    if d[0] < 0:
        p = -p
    t_lo, t_hi = p[c] - gap_mm, p[c] + gap_mm
    i_lo = np.clip(np.searchsorted(p, t_lo), 1, n - 1)
    i_hi = np.clip(np.searchsorted(p, t_hi), 1, n - 1)
    # nearest slice; a tie goes to the slice closer to the centre
    lo = np.where(t_lo - p[i_lo - 1] < p[i_lo] - t_lo, i_lo - 1, i_lo)
    hi = np.where(t_hi - p[i_hi - 1] <= p[i_hi] - t_hi, i_hi - 1, i_hi)
    lo = np.clip(np.minimum(lo, c - 1), 0, n - 1)
    hi = np.clip(np.maximum(hi, c + 1), 0, n - 1)
    return np.stack([lo, c, hi], 1)


def _center_crop(slices, meta, train, rng, *, size):
    """crop_fn body of `center_crop`: centred size x size crop, zero padding where smaller."""
    m, h, w = slices.shape
    out = np.zeros((m, size, size), np.uint8)
    ch, cw = min(h, size), min(w, size)
    sy, sx = (h - ch) // 2, (w - cw) // 2
    dy, dx = (size - ch) // 2, (size - cw) // 2
    out[:, dy:dy + ch, dx:dx + cw] = slices[:, sy:sy + ch, sx:sx + cw]
    return out


def center_crop(size):
    """Default crop_fn (C10): centred size x size crop of every slice, zero padding if smaller, no resampling."""
    return functools.partial(_center_crop, size=size)


class StudyDataset(torch.utils.data.Dataset):
    """One item = one study: uint8 windows [N_SLOTS, k, 3, S, S], slot mask, relative slice positions, target.

    load_volume(series) -> (vol uint8 [n, H, W] (mmap is fine), pos_mm [n], meta dict).
    crop_fn(slices uint8 [m, H, W], meta, train, rng) -> uint8 [m, crop_px, crop_px], called once per
    series on the slices its windows need. targets: float [len(studies), 12] or None (y = zeros).
    depth_frac: {series: (lo, hi)} share of the stack the window centres spread over (from
    joint_depth_fractions); required for every candidate series. pos stays relative to the full stack.
    All randomness (candidate series, centre jitter, crop_fn) comes from the per-worker `self.rng`,
    seeded from (seed, worker id) by `seed_worker`.
    """

    def __init__(self, studies, table, load_volume, targets, k, train, crop_fn, gap_mm, crop_px, seed, depth_frac):
        self.studies, self.table, self.load_volume = list(studies), table, load_volume
        self.k, self.train, self.crop_fn, self.gap_mm, self.crop_px = k, train, crop_fn, gap_mm, crop_px
        missing = [s for s in self.studies if s not in table]
        if missing:
            raise KeyError(f"{len(missing)} studies are not in the study table, e.g. {missing[0]}")
        cands = [sr for st in self.studies for slot in table[st] for sr in slot]
        missing = [sr for sr in cands if sr not in depth_frac]
        if missing:
            raise KeyError(f"depth_frac has no entry for {len(missing)} candidate series, e.g. {missing[0]}")
        bad = [sr for sr in cands if not 0 <= depth_frac[sr][0] < depth_frac[sr][1] <= 1]
        if bad:
            raise ValueError(f"depth_frac of {len(bad)} series is not 0 <= lo < hi <= 1, e.g. {bad[0]}: {depth_frac[bad[0]]}")
        self.depth_frac = depth_frac
        if targets is not None:
            targets = np.asarray(targets, np.float32)
            if targets.shape != (len(self.studies), len(LABELS)):
                raise ValueError(f"targets shape {targets.shape} != ({len(self.studies)}, {len(LABELS)})")
        self.targets = targets
        self.seed = seed
        self.set_worker(0)

    def set_worker(self, worker_id):
        """Re-seeds the dataset rng for one DataLoader worker."""
        self.rng = np.random.default_rng([self.seed, worker_id])

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        k, size = self.k, self.crop_px
        x = np.zeros((N_SLOTS, k, 3, size, size), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, k), np.float32)
        for s, cands in enumerate(self.table[st]):
            if not cands:
                continue
            sr = cands[self.rng.integers(len(cands))] if self.train else cands[0]
            vol, pos_mm, meta = self.load_volume(sr)
            n = len(vol)
            if vol.dtype != np.uint8 or vol.ndim != 3 or len(pos_mm) != n or n < 3:
                raise ValueError(f"series {sr}: bad volume {vol.dtype} {vol.shape}, {len(pos_mm)} positions")
            c = window_centres(n, k, self.train, self.rng, *self.depth_frac[sr])
            idx = window_slices(pos_mm, c, self.gap_mm)
            need, inv = np.unique(idx.ravel(), return_inverse=True)
            crop = self.crop_fn(np.asarray(vol[need]), meta, self.train, self.rng)
            if crop.dtype != np.uint8 or crop.shape != (len(need), size, size):
                raise ValueError(f"series {sr}: crop_fn returned {crop.dtype} {crop.shape}, "
                                 f"expected uint8 {(len(need), size, size)}")
            x[s], mask[s], pos[s] = crop[inv.reshape(idx.shape)], True, c / (n - 1)
        y = self.targets[i] if self.targets is not None else np.zeros(len(LABELS), np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(y)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: gives the worker's StudyDataset its own rng from (dataset seed, worker id)."""
    torch.utils.data.get_worker_info().dataset.set_worker(worker_id)


def make_views(x, res, train, group, fov, mean, std):
    """uint8 [N, 3, H, W] -> normalised float [N, 3, res, res]: centred `fov` crop, train-time affine/gamma/gain.

    N = n_series * group; the augmentation is shared within a series. In training adds
    rotation/scale/shift and gain/gamma jitter; normalisation uses the explicit mean/std.
    """
    n = x.shape[0]
    ns = n // group
    dev = x.device
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
        sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
        ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
        gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev)
        sc = torch.full((ns,), fov, device=dev)
        tx = ty = torch.zeros(ns, device=dev)
        gamma = gain = None
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    if x.shape[-1] > 1.5 * res:  # avoid aliasing when shrinking a lot
        x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=0.75, mode="bilinear", antialias=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if train:
        g = gamma.repeat_interleave(group).view(n, 1, 1, 1)
        x = x.clamp_min(1e-4).pow(g) * gain.repeat_interleave(group).view(n, 1, 1, 1)
    mean = torch.tensor(mean, device=dev, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=dev, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class KneeNet(nn.Module):
    """2D backbone on 3-slice windows -> tokens tagged with series slot and slice position ->
    small transformer over all windows of the study -> one attention pooling per finding."""

    def __init__(self, backbone, pretrained, dim=384, depth=2, drop=0.2, drop_path=0.1, *, grad_ckpt):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.enc.set_grad_checkpointing(grad_ckpt)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

    def head(self, feat, mask, pos):
        """feat [B, S, K, D], mask [B, S] bool, pos [B, S, K] in 0..1 -> logits [B, 12]."""
        b, s, k, _ = feat.shape
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)  # a study with no series at all attends to everything
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)  # [B, T, C]
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x, mask, pos, *, res, train, fov, mean, std):
        """x uint8 [B, S, K, 3, H, W], mask [B, S], pos [B, S, K] -> logits [B, 12]."""
        b, s, k = x.shape[:3]
        keep = mask.view(-1)  # run the backbone only on series that exist
        if keep.any():
            xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
            f = self.enc(make_views(xv, res, train, k, fov, mean, std).contiguous(memory_format=torch.channels_last))
            feat = f.new_zeros(b * s, k, f.shape[-1])
            feat[keep] = f.view(-1, k, f.shape[-1])
        else:  # no study of the batch has a series: same zero features, the backbone has nothing to run on
            feat = torch.zeros(b * s, k, self.enc.num_features, device=x.device)
        return self.head(feat.view(b, s, k, -1), mask, pos)
