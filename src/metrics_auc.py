"""Macro ROC AUC over the 12 findings and its study-level bootstrap CI."""
import numpy as np
from sklearn.metrics import roc_auc_score


def _check(y_true, y_score):
    """Validated (int y_true, float y_score) arrays: same [N, C] shape, binary y_true."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    if y_true.ndim != 2 or y_true.shape != y_score.shape:
        raise ValueError(f"y_true {y_true.shape} and y_score {y_score.shape} must be the same [N, C] shape")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must be binary 0/1 (threshold soft labels before scoring)")
    return y_true.astype(int), y_score


def macro_auc(y_true, y_score):
    """Per-class ROC AUC (NaN for single-class columns) and their nanmean; raises if every class is NaN."""
    y, p = _check(y_true, y_score)
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    if np.isnan(aucs).all():
        raise ValueError("every class has a single label value: macro AUC is undefined")
    return float(np.nanmean(aucs)), [float(a) for a in aucs]


def bootstrap_macro_auc(y_true, y_score, n_boot, seed):
    """95% CI of macro_auc by resampling studies with replacement: dict(point, lo, hi, n_valid).

    A resample is valid only if every class that is two-class in the full data stays two-class, so each
    replicate averages over the same classes as the point estimate; n_valid counts valid resamples.
    """
    y, p = _check(y_true, y_score)
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    point, each = macro_auc(y, p)
    defined = ~np.isnan(each)
    rng = np.random.default_rng(seed)
    n = len(y)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pos = y[idx][:, defined].sum(axis=0)
        if ((pos > 0) & (pos < n)).all():
            reps.append(macro_auc(y[idx], p[idx])[0])
    if not reps:
        raise ValueError(f"none of {n_boot} resamples keeps every class two-class")
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_valid": len(reps)}
