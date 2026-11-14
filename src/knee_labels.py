"""Label tables for the knee competition: target names, soft report labels, gold labels, CV folds.

Fold logic is ported as-is from goodpjw's `make_labels.py` (goodpjw2008/rsna-knee-2-5d-convnext-reader).
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

CLIP = (0.01, 0.99)


def load_soft_targets(csv_path, study_ids):
    """Soft targets from a report-label CSV: index = study_ids in that order, LABELS, float32 clipped to [0.01, 0.99]."""
    df = pd.read_csv(csv_path)
    dup = df.StudyInstanceUID[df.StudyInstanceUID.duplicated()]
    if len(dup):
        raise ValueError(f"{csv_path}: {len(dup)} duplicated StudyInstanceUID, e.g. {dup.iloc[0]}")
    missing_cols = [c for c in LABELS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path}: no columns {missing_cols}")
    ids = pd.Index(study_ids, name="StudyInstanceUID")
    if ids.has_duplicates:
        raise ValueError(f"study_ids contain {ids.duplicated().sum()} duplicates")
    df = df.set_index("StudyInstanceUID")
    missing = ids.difference(df.index)
    if len(missing):
        raise ValueError(f"{csv_path}: {len(missing)} of {len(ids)} studies missing, e.g. {missing[0]}")
    soft = df.loc[ids, LABELS].astype(float)
    nan = soft.isna().any(axis=1)
    if nan.any():
        raise ValueError(f"{csv_path}: NaN labels in {nan.sum()} studies, e.g. {soft.index[nan][0]}")
    return soft.clip(*CLIP).astype(np.float32)


def gold_labels(train_df):
    """Official labels of the studies that have all 12 of them: index StudyInstanceUID, LABELS, int."""
    lab = train_df.set_index("StudyInstanceUID")[LABELS]
    n = lab.notna().sum(axis=1)
    partial = (n > 0) & (n < len(LABELS))
    if partial.any():
        raise ValueError(f"{partial.sum()} partially labelled studies, e.g. {lab.index[partial][0]}")
    gold = lab[n == len(LABELS)]
    if not gold.isin([0, 1]).all().all():
        raise ValueError("gold labels contain values other than 0/1")
    return gold.astype(int)


def report_group(text):
    """CV group key of a report: lowercase text with every non-word character removed."""
    return re.sub(r"\W+", "", str(text).lower())


def make_folds(train_df, n_folds):
    """GroupKFold over report_group of the weak studies (goodpjw's folds); gold studies get fold -1.

    No seed: goodpjw permutes the studies with a seed first, but GroupKFold (no shuffle) orders groups
    by np.unique, i.e. by the sorted report text, so that permutation never changes the folds; it is
    dropped and the folds are a function of the report texts only (identical to goodpjw's).
    """
    if train_df.StudyInstanceUID.duplicated().any():
        raise ValueError("train_df has duplicated StudyInstanceUID")
    if train_df.Report.isna().any():
        raise ValueError(f"{train_df.Report.isna().sum()} studies without a Report: their CV group is undefined")
    is_gold = train_df.StudyInstanceUID.isin(gold_labels(train_df).index).values
    weak = ~is_gold
    groups = train_df.Report.map(report_group).values[weak]
    folds = np.zeros(weak.sum(), int)
    for k, (_, va) in enumerate(GroupKFold(n_folds).split(groups, groups=groups)):
        folds[va] = k
    fold = np.full(len(train_df), -1)
    fold[weak] = folds
    return pd.DataFrame({"StudyInstanceUID": train_df.StudyInstanceUID.values, "fold": fold, "is_gold": is_gold})
