"""Deterministic rubric mapping: validated extraction cells -> (soft_target, weight, evidence_state).

A rubric/policy change re-runs only this module, never the LLM. Every value in a policy is
fixed a priori; none is fitted on the 58 gold studies or on the gold-based audits.
"""

from __future__ import annotations

import pandas as pd

from src.report_labels.rubric import TARGETS

UID = "StudyInstanceUID"

# (state, meets_rubric) -> (soft_target, weight, evidence_state) for every combination the
# validator's consistency rules allow, except `not_addressed`, which goes through the
# per-target silence rule below. Values: report_labels.md §4 (a priori, not tuned on gold58).
#   present+yes            explicit finding that meets the rubric: confident positive.
#   present+insufficient   finding stated, but grade/extent/acuity missing: lean positive, half weight.
#   present+no             stated but non-qualifying (low-grade sprain, mild effusion,
#                          intrasubstance degeneration): soft negative, high weight.
#   uncertain              hedged / "cannot exclude": below 0.5, low weight.
#   absent                 explicit negation: confident negative, full weight.
#   conflicting            body vs impression or bilateral attribution disagree: masked.
#
# Silence (`not_addressed`, evidence_state "silent") — clinical reasoning, not gold data:
#   * Cruciate/collateral ligaments, menisci, Baker's cysts and fractures are routinely
#     reported when present; a knee MRI report that does not mention them is close to a
#     negative, though weaker than an explicit "intact"/"no tear" (weight < 1).
#   * Bone marrow contusion and joint effusion are standard trauma-protocol findings and are
#     usually described when moderate or larger; silence is a moderate-weight negative.
#   * Qualifying OA (full-thickness loss over ~1 cm) is usually reported, but cartilage is
#     often summarised or omitted in trauma-focused reports: weak negative.
#   * Synovitis is frequently left out of non-contrast reports even when present; silence
#     carries almost no information -> a weak prior with a very small weight, not evidence.
POLICY_V1: dict = {
    "version": "policy_v1",
    "cells": {
        "present": {
            "yes": (0.95, 1.0, "positive"),
            "insufficient_evidence": (0.60, 0.5, "possible"),
            "no": (0.15, 0.8, "nonqualifying"),
        },
        "uncertain": {
            "insufficient_evidence": (0.35, 0.4, "uncertain"),
            "no": (0.35, 0.4, "uncertain"),
        },
        "absent": {
            "no": (0.03, 1.0, "negative"),
        },
        "conflicting": {
            "insufficient_evidence": (0.50, 0.0, "conflicting"),
            "no": (0.50, 0.0, "conflicting"),
        },
    },
    # target -> (soft_target, weight) for state == not_addressed (meets_rubric == no).
    "silence": {
        "ACL": (0.05, 0.7),
        "MCL": (0.05, 0.7),
        "Medial Meniscus": (0.08, 0.6),
        "Lateral Meniscus": (0.08, 0.6),
        "Medial OA": (0.10, 0.4),
        "Lateral OA": (0.10, 0.4),
        "PF OA": (0.10, 0.4),
        "Effusion": (0.10, 0.5),
        "Synovitis": (0.30, 0.1),
        "Baker's": (0.05, 0.6),
        "Contusion": (0.08, 0.5),
        "Fracture": (0.03, 0.7),
    },
}

# Combinations the validator lets through as valid (consistency rules in contracts.md).
_VALID_CELLS = {
    ("present", "yes"), ("present", "insufficient_evidence"), ("present", "no"),
    ("uncertain", "insufficient_evidence"), ("uncertain", "no"),
    ("absent", "no"),
    ("conflicting", "insufficient_evidence"), ("conflicting", "no"),
}


def check_policy(policy: dict) -> None:
    """Fail loudly unless a policy covers every valid (state, meets_rubric) cell and all 12 silence rules."""
    if not isinstance(policy.get("version"), str) or not policy["version"]:
        raise ValueError("policy needs a non-empty string 'version'")
    cells = {(s, m) for s, row in policy["cells"].items() for m in row}
    if cells != _VALID_CELLS:
        raise ValueError(f"policy cells must be exactly the valid combinations: "
                         f"missing {sorted(_VALID_CELLS - cells)}, unexpected {sorted(cells - _VALID_CELLS)}")
    if set(policy["silence"]) != set(TARGETS):
        raise ValueError(f"policy silence rules must cover exactly {TARGETS}, got {sorted(policy['silence'])}")
    values = [v[:2] for row in policy["cells"].values() for v in row.values()]
    values += list(policy["silence"].values())
    for soft, weight in values:
        if not (0.0 <= soft <= 1.0 and 0.0 <= weight <= 1.0):
            raise ValueError(f"policy value out of [0, 1]: soft={soft}, weight={weight}")


def map_to_targets(records_long: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Map the selected cells of validate_run's long table to `uid, target, soft_target, weight, evidence_state, source`."""
    check_policy(policy)
    sel = records_long[records_long["selected"]]
    per_study = sel.groupby(UID)["target"].agg(lambda s: sorted(s) == sorted(TARGETS))
    if not per_study.all():
        bad = per_study.index[~per_study].tolist()
        raise ValueError(f"{len(bad)} studies do not have exactly one selected row per target, e.g. {bad[:3]}")

    rows = []
    for r in sel.to_dict("records"):
        if not r["valid"]:
            soft, weight, evidence = float("nan"), 0.0, "invalid"
        elif r["state"] == "not_addressed":
            soft, weight = policy["silence"][r["target"]]
            evidence = "silent"
        else:
            soft, weight, evidence = policy["cells"][r["state"]][r["meets_rubric"]]
        rows.append({
            UID: r[UID], "target": r["target"], "soft_target": soft, "weight": weight,
            "evidence_state": evidence,
            "source": f"{r['model']}@{r['revision']}/{r['prompt_version']}/{r['pass']}/{policy['version']}",
        })
    return pd.DataFrame(rows, columns=[UID, "target", "soft_target", "weight", "evidence_state", "source"])


def to_wide(labels_long: pd.DataFrame) -> pd.DataFrame:
    """S4-compatible wide table: uid + 12 soft targets, then 12 `<t>__weight` and 12 `<t>__state` columns."""
    if labels_long.duplicated([UID, "target"]).any():
        raise ValueError("labels_long has duplicate (study, target) rows")
    parts = []
    for col, suffix in (("soft_target", ""), ("weight", "__weight"), ("evidence_state", "__state")):
        w = labels_long.pivot(index=UID, columns="target", values=col)
        missing = [t for t in TARGETS if t not in w.columns]
        if missing or len(w.columns) != len(TARGETS):
            raise ValueError(f"labels_long targets differ from TARGETS: missing {missing}, got {list(w.columns)}")
        w = w[list(TARGETS)]
        if col != "soft_target" and w.isna().any().any():
            raise ValueError(f"wide '{col}' has empty cells: some studies lack a row for some target")
        parts.append(w.rename(columns={t: t + suffix for t in TARGETS}))
    wide = pd.concat(parts, axis=1).reset_index()
    return wide
