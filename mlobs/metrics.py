"""ML observability: data drift, fairness, and model degradation.

Dataset-level analyses (not row-wise evaluators). Pure pandas/numpy,
no hard sklearn dependency.

Drift:        PSI (numeric via quantile bins, categorical via frequencies)
              and the KS statistic for numeric columns.
Fairness:     Demographic Parity difference, Disparate Impact ratio,
              Equal Opportunity difference, Average Odds difference,
              per sensitive attribute.
Degradation:  binary-classification performance (accuracy, precision,
              recall, F1) on baseline vs current, with deltas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------

def psi_numeric(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using baseline quantile bins."""
    base = baseline.dropna().astype(float)
    curr = current.dropna().astype(float)
    if len(base) < 2 or len(curr) < 2:
        return float("nan")
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant column
        edges = np.array([-np.inf, edges[0], np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    b_counts, _ = np.histogram(base, bins=edges)
    c_counts, _ = np.histogram(curr, bins=edges)
    b_pct = np.clip(b_counts / max(b_counts.sum(), 1), EPS, None)
    c_pct = np.clip(c_counts / max(c_counts.sum(), 1), EPS, None)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def psi_categorical(baseline: pd.Series, current: pd.Series) -> float:
    base = baseline.dropna().astype(str)
    curr = current.dropna().astype(str)
    if base.empty or curr.empty:
        return float("nan")
    cats = sorted(set(base.unique()) | set(curr.unique()))
    b_pct = np.clip(base.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy(), EPS, None)
    c_pct = np.clip(curr.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy(), EPS, None)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def ks_statistic(baseline: pd.Series, current: pd.Series) -> float:
    """Two-sample KS statistic (no p-value; scipy-free)."""
    base = np.sort(baseline.dropna().astype(float).to_numpy())
    curr = np.sort(current.dropna().astype(float).to_numpy())
    if len(base) == 0 or len(curr) == 0:
        return float("nan")
    grid = np.concatenate([base, curr])
    cdf_b = np.searchsorted(base, grid, side="right") / len(base)
    cdf_c = np.searchsorted(curr, grid, side="right") / len(curr)
    return float(np.max(np.abs(cdf_b - cdf_c)))


def psi_band(psi: float) -> str:
    if np.isnan(psi):
        return "n/a"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate drift"
    return "significant drift"


def drift_report(baseline: pd.DataFrame, current: pd.DataFrame,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """Column-wise drift table between a baseline and a current dataset."""
    if columns is None:
        columns = [c for c in baseline.columns if c in current.columns]
    rows = []
    for col in columns:
        b, c = baseline[col], current[col]
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(c):
            psi = psi_numeric(b, c)
            ks = ks_statistic(b, c)
            kind = "numeric"
        else:
            psi = psi_categorical(b, c)
            ks = float("nan")
            kind = "categorical"
        rows.append({"column": col, "type": kind, "psi": round(psi, 4) if not np.isnan(psi) else None,
                     "ks": round(ks, 4) if not np.isnan(ks) else None, "assessment": psi_band(psi)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fairness (binary classification, favorable outcome = 1)
# --------------------------------------------------------------------------

def _rate(mask_num: np.ndarray, mask_den: np.ndarray) -> float:
    den = mask_den.sum()
    return float(mask_num.sum() / den) if den else float("nan")


def fairness_report(df: pd.DataFrame, y_true: str, y_pred: str,
                    sensitive: str, favorable=1,
                    privileged=None) -> pd.DataFrame:
    """Group fairness metrics per sensitive-attribute value.

    Reference group = ``privileged`` if given, else the group with the
    highest selection rate. Standard definitions:
      selection_rate = P(pred=1 | group)
      DP difference  = selection_rate(group) - selection_rate(reference)
      DI ratio       = selection_rate(group) / selection_rate(reference)
      TPR/FPR        per group
      EO difference  = TPR(group) - TPR(reference)
      AOD            = 0.5 * [(FPR_g - FPR_ref) + (TPR_g - TPR_ref)]
    """
    yt = (df[y_true] == favorable).to_numpy()
    yp = (df[y_pred] == favorable).to_numpy()
    groups = df[sensitive].astype(str).fillna("Unknown")

    stats = {}
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        sel = _rate(yp & m, m)
        tpr = _rate(yp & yt & m, yt & m)
        fpr = _rate(yp & ~yt & m, ~yt & m)
        stats[g] = {"n": int(m.sum()), "selection_rate": sel, "tpr": tpr, "fpr": fpr}

    if privileged is None or str(privileged) not in stats:
        privileged = max(stats, key=lambda g: (stats[g]["selection_rate"]
                                               if not np.isnan(stats[g]["selection_rate"]) else -1))
    ref = stats[str(privileged)]

    rows = []
    for g, s in stats.items():
        di = (s["selection_rate"] / ref["selection_rate"]
              if ref["selection_rate"] not in (0, None) and not np.isnan(ref["selection_rate"])
              else float("nan"))
        rows.append({
            "group": g,
            "n": s["n"],
            "selection_rate": round(s["selection_rate"], 4),
            "tpr": round(s["tpr"], 4) if not np.isnan(s["tpr"]) else None,
            "fpr": round(s["fpr"], 4) if not np.isnan(s["fpr"]) else None,
            "dp_difference": round(s["selection_rate"] - ref["selection_rate"], 4),
            "di_ratio": round(di, 4) if not np.isnan(di) else None,
            "eo_difference": (round(s["tpr"] - ref["tpr"], 4)
                              if not (np.isnan(s["tpr"]) or np.isnan(ref["tpr"])) else None),
            "aod": (round(0.5 * ((s["fpr"] - ref["fpr"]) + (s["tpr"] - ref["tpr"])), 4)
                    if not any(np.isnan(v) for v in (s["fpr"], ref["fpr"], s["tpr"], ref["tpr"])) else None),
            "is_reference": g == str(privileged),
        })
    return pd.DataFrame(rows)


def fairness_flags(report: pd.DataFrame, di_low: float = 0.8, di_high: float = 1.25,
                   dp_threshold: float = 0.1) -> list[str]:
    """Human-readable flags using the four-fifths rule and a DP threshold."""
    flags = []
    for _, row in report.iterrows():
        if row["is_reference"]:
            continue
        if row["di_ratio"] is not None and not (di_low <= row["di_ratio"] <= di_high):
            flags.append(f"Group '{row['group']}': disparate impact ratio {row['di_ratio']} "
                         f"outside [{di_low}, {di_high}] (four-fifths rule).")
        if abs(row["dp_difference"]) > dp_threshold:
            flags.append(f"Group '{row['group']}': demographic parity difference "
                         f"{row['dp_difference']} exceeds {dp_threshold}.")
    return flags


# --------------------------------------------------------------------------
# Model degradation (binary classification)
# --------------------------------------------------------------------------

def classification_metrics(y_true: pd.Series, y_pred: pd.Series, favorable=1) -> dict:
    yt = (y_true == favorable).to_numpy()
    yp = (y_pred == favorable).to_numpy()
    tp = int((yp & yt).sum()); fp = int((yp & ~yt).sum())
    fn = int((~yp & yt).sum()); tn = int((~yp & ~yt).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) and (tp + fn) and (precision + recall) else float("nan"))
    accuracy = (tp + tn) / max(len(yt), 1)
    return {"n": len(yt), "accuracy": round(accuracy, 4),
            "precision": round(precision, 4) if not np.isnan(precision) else None,
            "recall": round(recall, 4) if not np.isnan(recall) else None,
            "f1": round(f1, 4) if not np.isnan(f1) else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def degradation_report(baseline: pd.DataFrame, current: pd.DataFrame,
                       y_true: str, y_pred: str, favorable=1) -> pd.DataFrame:
    b = classification_metrics(baseline[y_true], baseline[y_pred], favorable)
    c = classification_metrics(current[y_true], current[y_pred], favorable)
    rows = []
    for metric in ("accuracy", "precision", "recall", "f1"):
        delta = (round(c[metric] - b[metric], 4)
                 if b[metric] is not None and c[metric] is not None else None)
        rows.append({"metric": metric, "baseline": b[metric], "current": c[metric],
                     "delta": delta,
                     "degraded": bool(delta is not None and delta < -0.05)})
    return pd.DataFrame(rows)
