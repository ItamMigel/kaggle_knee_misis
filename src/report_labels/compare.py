"""Label-set comparisons: ours vs a reference on unlabeled studies, two extractions cell by cell, the frozen gold58 AUC diagnostic, pilot review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from src.report_labels.rubric import TARGETS

UID = "StudyInstanceUID"


def split_gold(train: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """(gold uids, unlabeled uids) of train.csv from the label NaN mask only — label values are not read."""
    labeled = train[list(TARGETS)].notna()
    partial = labeled.any(axis=1) & ~labeled.all(axis=1)
    if partial.any():
        raise ValueError(f"{int(partial.sum())} train rows are partially labeled, e.g. {train.loc[partial, UID].iloc[0]}")
    mask = labeled.all(axis=1)
    return pd.Index(train.loc[mask, UID]), pd.Index(train.loc[~mask, UID])


def compare_with_reference(ours: pd.DataFrame, ref: pd.DataFrame, uids, *, require_complete_ref: bool) -> pd.DataFrame:
    """Per finding + MACRO: positive rates (>0.5), means, our silent/uncertain/conflicting/invalid rates, agreement, κ, Spearman.

    Per finding, the stats use the cells finite in ours and in ref (`n_compared`). With
    require_complete_ref (a published label set such as S4) a NaN in ref is an error; without
    it (ref = another of our extractions) ref's invalid cells are left out like ours.
    """
    uids = pd.Index(uids)
    if not uids.is_unique or len(uids) == 0:
        raise ValueError("uids must be non-empty and unique")
    o_all, r_all = ours.set_index(UID), ref.set_index(UID)
    for name, df in (("ours", o_all), ("ref", r_all)):
        missing = uids.difference(df.index)
        if len(missing):
            raise ValueError(f"{name} lacks {len(missing)} of the requested studies, e.g. {missing[0]}")
    o_all, r_all = o_all.loc[uids], r_all.loc[uids]
    if require_complete_ref and not np.isfinite(r_all[list(TARGETS)].to_numpy(float)).all():
        raise ValueError("reference has missing scores on the requested studies")

    rows = []
    for t in TARGETS:
        state = o_all[f"{t}__state"]
        valid = o_all[t].notna().to_numpy() & r_all[t].notna().to_numpy()
        o, r = o_all[t].to_numpy(float)[valid], r_all[t].to_numpy(float)[valid]
        rows.append({
            "target": t, "n": len(uids), "n_compared": int(valid.sum()),
            "pos_rate_ours": float((o > 0.5).mean()), "pos_rate_ref": float((r > 0.5).mean()),
            "mean_ours": float(o.mean()), "mean_ref": float(r.mean()),
            **{f"{s}_rate": float((state == s).mean()) for s in ("silent", "uncertain", "conflicting", "invalid")},
            "agreement": float(((o > 0.5) == (r > 0.5)).mean()),
            "kappa": float(cohen_kappa_score(o > 0.5, r > 0.5)),
            "spearman": float(spearmanr(o, r).statistic),
        })
    out = pd.DataFrame(rows)
    macro = out.drop(columns=["target", "n", "n_compared"]).mean()
    macro_row = {"target": "MACRO", "n": len(uids), "n_compared": int(out["n_compared"].min()), **macro.to_dict()}
    return pd.concat([out, pd.DataFrame([macro_row])], ignore_index=True)


# --- gold58 diagnostic: weights()/measure() ported from scratch/scout-labels/analyze_labels.py
# (research/report_labels.md, «AUC measurements»), extended with per-target study masks; with
# every mask full they compute exactly what the scout's code did.

def _weights(yy: np.ndarray, masks: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Study-level multinomial bootstrap weights over all studies; draws lacking a class for a target on its masked studies are dropped."""
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(yy), np.full(len(yy), 1 / len(yy)), size=n_boot)
    m = masks.astype(int)
    counts = w @ (yy * m)
    valid = ((counts > 0) & (counts < w @ m)).all(axis=1)
    return w[valid]


def _measure(n: str, a: np.ndarray, yy: np.ndarray, masks: np.ndarray, w: np.ndarray):
    """Per-target AUC on the target's masked studies + macro (mean of per-target AUCs per draw), percentile CIs; returns (rows, per-target bootstrap AUCs, macro bootstrap)."""
    rr = []
    bb = []
    for j, t in enumerate(TARGETS):
        m = masks[:, j]
        assert np.isfinite(a[m, j]).all(), (n, t)
        yp = m & (yy[:, j] == 1)
        yn = m & (yy[:, j] == 0)
        ap = a[yp, j]
        an = a[yn, j]
        comp = (ap[:, None] > an[None, :]).astype(float) + 0.5 * (ap[:, None] == an[None, :])
        point = comp.mean()
        assert abs(point - roc_auc_score(yy[m, j], a[m, j])) < 1e-12
        wp = w[:, yp]
        wn = w[:, yn]
        b = ((wp @ comp) * wn).sum(axis=1) / (wp.sum(axis=1) * wn.sum(axis=1))
        lo, hi = np.quantile(b, [.025, .975])
        rr.append(dict(source=n, target=t, n=int(m.sum()), positive=int(yp.sum()), auc=point, low=lo, high=hi))
        bb.append(b)
    bm = np.stack(bb, axis=1).mean(axis=1)
    # = roc_auc_score(yy, a, average="macro") when every mask is full
    point = float(np.mean([roc_auc_score(yy[masks[:, j], j], a[masks[:, j], j]) for j in range(len(TARGETS))]))
    lo, hi = np.quantile(bm, [.025, .975])
    rr.append(dict(source=n, target="MACRO", n=int(masks.any(axis=1).sum()), positive=int((yy * masks).sum()),
                   auc=point, low=lo, high=hi))
    return rr, np.stack(bb, axis=1), bm


def gold_diagnostic(scores: dict[str, pd.DataFrame], gold: pd.DataFrame, n_boot: int, seed: int,
                    ref: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """gold58 per-finding + macro AUC with study-bootstrap 95% CIs, and paired deltas of every source vs `ref`.

    Per-target masking: a finding is scored on the gold studies where EVERY source has a finite
    score for it (`n`), the same studies for all sources, so an invalid cell costs only its
    finding. MACRO is the mean of the per-finding AUCs, within each draw for the CI; its `n` is
    the studies used by at least one finding. The draws are study-level over all gold studies,
    shared by all sources and findings; a draw lacking a class for some finding on that finding's
    masked studies is dropped. The row order of `gold` fixes the draws (train.csv order
    reproduces research/report_labels.md).
    """
    if ref not in scores:
        raise ValueError(f"reference {ref!r} not among sources {sorted(scores)}")
    g = gold.set_index(UID)[list(TARGETS)]
    if not g.index.is_unique:
        raise ValueError("gold has duplicate StudyInstanceUID")
    if g.isna().any().any() or not g.isin([0, 1]).all().all():
        raise ValueError("gold labels must be complete 0/1")
    mats, masks = {}, np.ones(g.shape, dtype=bool)
    for name, df in scores.items():
        s = df.set_index(UID)
        if not s.index.is_unique:
            raise ValueError(f"{name}: duplicate StudyInstanceUID")
        s = s.reindex(g.index)[list(TARGETS)]
        vals = s.to_numpy(float)
        fin = vals[np.isfinite(vals)]
        if ((fin < 0) | (fin > 1)).any():
            raise ValueError(f"{name}: scores outside [0, 1]")
        mats[name] = vals
        masks &= np.isfinite(vals)
    yy = g.to_numpy(int)
    for j, t in enumerate(TARGETS):
        pos = int(yy[masks[:, j], j].sum())
        if pos == 0 or pos == int(masks[:, j].sum()):
            raise ValueError(f"{t}: the {int(masks[:, j].sum())} gold studies scored by every source do not have "
                             f"both classes ({pos} positive) — AUC undefined")
    w = _weights(yy, masks, n_boot, seed)
    if len(w) == 0:
        raise ValueError("every bootstrap draw lacks a class for some finding")

    rows, boot = [], {}
    for name, vals in mats.items():
        rr, bb, bm = _measure(name, vals, yy, masks, w)
        rows.extend(rr)
        boot[name] = (np.column_stack([bb, bm]), [r["auc"] for r in rr])
    metrics = pd.DataFrame(rows)
    metrics["n_boot_valid"] = len(w)

    n_used = [*masks.sum(axis=0).tolist(), int(masks.any(axis=1).sum())]
    deltas = []
    for name in scores:
        if name == ref:
            continue
        for j, t in enumerate([*TARGETS, "MACRO"]):
            d = boot[name][0][:, j] - boot[ref][0][:, j]
            lo, hi = np.quantile(d, [.025, .975])
            deltas.append(dict(a=name, b=ref, target=t, n=int(n_used[j]),
                               delta=boot[name][1][j] - boot[ref][1][j], low=lo, high=hi,
                               bootstrap_fraction_positive=float((d > 0).mean())))
    return metrics, pd.DataFrame(deltas, columns=["a", "b", "target", "n", "delta", "low", "high",
                                                   "bootstrap_fraction_positive"])


def compare_models(records_a: pd.DataFrame, records_b: pd.DataFrame) -> pd.DataFrame:
    """Two extractions cell by cell on their common studies, per finding + ALL: validity of each, agreement of (state, meets_rubric) and of state on cells valid in both.

    Inputs are validate_run long tables (or their selected rows, records_long.csv); only selected
    cells are compared. Agreement is NaN where no cell is valid in both.
    """
    cells = []
    for name, df in (("a", records_a), ("b", records_b)):
        sel = df[df["selected"]]
        if sel["valid"].dtype != bool:
            raise ValueError(f"records_{name}: column valid is {sel['valid'].dtype}, not bool")
        if sel.duplicated([UID, "target"]).any():
            raise ValueError(f"records_{name}: more than one selected cell per (study, target)")
        if set(sel["target"]) != set(TARGETS) or (sel.groupby(UID)["target"].nunique() != len(TARGETS)).any():
            raise ValueError(f"records_{name}: every study must have one selected cell per target {TARGETS}")
        cells.append(sel[[UID, "target", "valid", "state", "meets_rubric"]])
    m = cells[0].merge(cells[1], on=[UID, "target"], suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("records_a and records_b have no study in common")

    def row(target: str, d: pd.DataFrame) -> dict:
        v = d[d["valid_a"] & d["valid_b"]]
        same_state = v["state_a"] == v["state_b"]
        return {"target": target, "n_common": len(d),
                "valid_share_a": float(d["valid_a"].mean()), "valid_share_b": float(d["valid_b"].mean()),
                "n_valid_both": len(v),
                "agree_state_meets": float((same_state & (v["meets_rubric_a"] == v["meets_rubric_b"])).mean()),
                "agree_state": float(same_state.mean())}

    return pd.DataFrame([row(t, m[m["target"] == t]) for t in TARGETS] + [row("ALL", m)])


def _md(v) -> str:
    """Markdown table cell: empty for None/NaN, `|` escaped, newlines flattened."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).replace("|", "\\|").replace("\n", " ")


def render_review(records_long: pd.DataFrame, labels_long: pd.DataFrame, reports: pd.DataFrame,
                  ref_wide: pd.DataFrame, uids, out_md: Path) -> None:
    """Markdown review: per study the report, then per target extracted evidence, our soft target and the reference score."""
    rec = records_long[records_long["selected"]].set_index([UID, "target"])
    lab = labels_long.set_index([UID, "target"])
    rep = reports.set_index(UID)["Report"]
    refw = ref_wide.set_index(UID)
    uids = list(uids)
    for name, idx in (("records", rec.index.get_level_values(0)), ("labels", lab.index.get_level_values(0)),
                      ("reports", rep.index), ("reference", refw.index)):
        missing = set(uids) - set(idx)
        if missing:
            raise ValueError(f"render_review: {name} lacks {len(missing)} studies, e.g. {sorted(missing)[0]}")

    policy = sorted({s.rsplit("/", 1)[1] for s in labels_long["source"]})
    lines = [f"# Extraction review — {len(uids)} studies", "",
             f"Policy: {', '.join(policy)}. `≠` = ours and reference fall on different sides of 0.5; "
             "`ours` = soft target (weight, evidence_state); ⟨ws_normalised⟩/⟨folded⟩/⟨repaired⟩/⟨spliced⟩ = quote "
             "matched the report only after whitespace removal / case-diacritic folding / ellipsis-backslash-"
             "mojibake repair / joining ellipsis-separated segments.", ""]
    for k, uid in enumerate(uids, 1):
        r0 = rec.loc[uid]
        n_dis = sum(1 for t in TARGETS if np.isfinite(lab.loc[(uid, t), "soft_target"])
                    and (lab.loc[(uid, t), "soft_target"] > 0.5) != (refw.loc[uid, t] > 0.5))
        lines += [f"## {k}. `{uid}`", "",
                  f"report_side: {_md(r0['report_side'].iloc[0])} · pass: {', '.join(sorted(set(r0['pass'])))} · "
                  f"disagreements with reference: {n_dis}", "",
                  "````text", str(rep.loc[uid]), "````", "",
                  "| target | state | meets_rubric | grade | size_mm | acuity | site | quote | ours | ref | |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in TARGETS:
            c, lb, rv = rec.loc[(uid, t)], lab.loc[(uid, t)], float(refw.loc[uid, t])
            soft = lb["soft_target"]
            quote = _md(c["quote"]) if c["valid"] else f"INVALID: {_md(c['errors'])} — {_md(c['quote'])}"
            if c["quote_tier"] in ("ws_normalised", "folded", "repaired", "spliced"):
                quote += f" ⟨{c['quote_tier']}⟩"
            ours = f"{soft:.2f} ({lb['weight']:.1f}, {lb['evidence_state']})" if np.isfinite(soft) \
                else f"NaN (0, {lb['evidence_state']})"
            flag = "≠" if np.isfinite(soft) and (soft > 0.5) != (rv > 0.5) else ""
            lines.append(f"| {t} | {_md(c['state'])} | {_md(c['meets_rubric'])} | {_md(c['grade'])} | "
                         f"{_md(c['size_mm'])} | {_md(c['acuity'])} | {_md(c['site'])} | {quote} | {ours} | "
                         f"{rv:.2f} | {flag} |")
        lines.append("")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
