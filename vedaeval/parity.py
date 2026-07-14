"""Segment Parity Report.

Compares every score column across the groups of a segment column
(e.g. bias_gender, bias_age_band): per-group means for numeric scores,
per-group rates for boolean flags, with gap flags when groups diverge
beyond thresholds. Pure pandas group-by over scores the engine already
computed - no models, no API.

Thresholds (documented in the metric block, configurable here):
- numeric: flag on the STANDARDIZED gap (Cohen's d style): (max group
  mean - min group mean) divided by the pooled standard deviation of
  the score. Flag when d >= d_threshold (default 0.8, the literature's
  "large effect" convention). This is scale-independent and immune to
  the near-zero-mean problem of relative thresholds (a 0.15 sentiment
  gap on a -1..1 scale no longer flags just because the overall mean
  is small).
- boolean rates: flag when the rate difference exceeds
  rate_gap_threshold (default 0.10 = 10 percentage points).
- groups smaller than min_rows (default 5) are shown but excluded from
  gap computation (small-cell noise, same reasoning as the intake
  checks).

Part of the LLM product (NOT the deletable mlobs add-on): this measures
LLM application outputs; mlobs measures classic ML predictions.
"""

from __future__ import annotations

import pandas as pd

INPUT_LIKE = {"request", "response", "ground_truth", "context", "timestamp"}


# ---------------------------------------------------------------------------
# statistical certainty (p-values) - magnitude and significance are
# DIFFERENT questions: effect size d answers "is the gap big?", the
# p-value answers "could a gap this size appear by chance?". We report
# both; see docs 16/16b.
# ---------------------------------------------------------------------------

def _norm_sf(z: float) -> float:
    """Survival function of the standard normal (1 - CDF), scipy-free."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _welch_p(a, b) -> float | None:
    """Two-sided Welch's t-test p-value for two samples of numbers.

    Uses scipy's exact t-distribution when installed; otherwise a normal
    approximation (adequate for n around 30+, slightly optimistic below).
    """
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = (va / len(a) + vb / len(b)) ** 0.5
    if se == 0:
        return 1.0 if abs(a.mean() - b.mean()) < 1e-12 else 0.0
    t = abs(a.mean() - b.mean()) / se
    try:
        from scipy import stats
        df = (va / len(a) + vb / len(b)) ** 2 / (
            (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
        return float(2 * stats.t.sf(t, df))
    except Exception:
        return float(2 * _norm_sf(t))


def _two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float | None:
    """Two-sided two-proportion z-test p-value (rates comparison)."""
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = (pool * (1 - pool) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return 1.0 if abs(p1 - p2) < 1e-12 else 0.0
    z = abs(p1 - p2) / se
    return float(2 * _norm_sf(z))


def _is_boolish(series: pd.Series) -> bool:
    vals = set(series.dropna().unique())
    return len(vals) > 0 and vals <= {True, False, 0, 1} and series.dtype != float


def segment_parity(df: pd.DataFrame, segment_col: str,
                   score_cols: list[str] | None = None,
                   min_rows: int = 5,
                   d_threshold: float = 0.8,
                   rate_gap_threshold: float = 0.10,
                   alpha: float = 0.05,
                   abs_floor: float = 1e-9):
    """Returns (summary_df, detail_df).

    detail_df: one row per (score column x segment value): n and value
               (mean or rate).
    summary_df: one row per score column: per-segment values, the gap,
                and whether it is flagged.
    """
    seg = df[segment_col].astype("string").fillna("Unknown").str.strip()
    seg = seg.replace("", "Unknown")

    if score_cols is None:
        score_cols = [c for c in df.columns
                      if c not in INPUT_LIKE and c != segment_col
                      and (pd.api.types.is_numeric_dtype(df[c]) or _is_boolish(df[c]))]

    detail_rows, summary_rows = [], []
    groups = [g for g in seg.unique() if g != "Unknown"]

    for col in score_cols:
        series = df[col]
        boolish = _is_boolish(series)
        if not (boolish or pd.api.types.is_numeric_dtype(series)):
            continue
        per_group = {}
        for g in groups:
            vals = series[seg == g].dropna()
            if boolish:
                value = float(vals.astype(bool).mean()) if len(vals) else None
            else:
                value = float(vals.mean()) if len(vals) else None
            per_group[g] = {"n": int(len(vals)), "value": value}
            detail_rows.append({"score column": col, "segment": g,
                                "n": int(len(vals)),
                                "value": round(value, 4) if value is not None else None,
                                "type": "rate" if boolish else "mean"})

        eligible = {g: v for g, v in per_group.items()
                    if v["value"] is not None and v["n"] >= min_rows}
        if len(eligible) < 2:
            summary_rows.append({"score column": col,
                                 **{f"{g}": (round(v["value"], 4) if v["value"] is not None else None)
                                    for g, v in per_group.items()},
                                 "gap": None, "effect size": None,
                                 "p value": None,
                                 "significant": None,
                                 "flagged": False,
                                 "note": "not enough data per group"})
            continue

        values = [v["value"] for v in eligible.values()]
        gap = max(values) - min(values)
        # the max-vs-min pair drives both the gap and the p-value; with
        # more than 2 groups this is noted (a full ANOVA is roadmap)
        g_lo = min(eligible, key=lambda g: eligible[g]["value"])
        g_hi = max(eligible, key=lambda g: eligible[g]["value"])
        lo_vals = series[seg == g_lo].dropna()
        hi_vals = series[seg == g_hi].dropna()
        note = "max-vs-min pair tested" if len(eligible) > 2 else ""

        if boolish:
            flagged = gap > rate_gap_threshold
            effect = None
            p = _two_prop_p(int(hi_vals.astype(bool).sum()), len(hi_vals),
                            int(lo_vals.astype(bool).sum()), len(lo_vals))
        else:
            # standardized gap (Cohen's d flavor): gap / pooled std.
            # Scale-independent; a near-zero overall mean cannot inflate it.
            pooled_std = float(series.dropna().astype(float).std(ddof=0))
            denom = max(pooled_std, abs_floor)
            effect = gap / denom
            flagged = effect >= d_threshold
            p = _welch_p(hi_vals.astype(float), lo_vals.astype(float))
        summary_rows.append({"score column": col,
                             **{f"{g}": (round(v["value"], 4) if v["value"] is not None else None)
                                for g, v in per_group.items()},
                             "gap": round(gap, 4),
                             "effect size": (round(effect, 2) if effect is not None else None),
                             "p value": (round(p, 4) if p is not None else None),
                             "significant": (bool(p < alpha) if p is not None else None),
                             "flagged": bool(flagged),
                             "note": note})

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def parity_flags(summary_df: pd.DataFrame, segment_col: str) -> list[str]:
    """Human-readable flag sentences for the UI."""
    out = []
    if summary_df.empty:
        return out
    group_cols = [c for c in summary_df.columns
                  if c not in ("score column", "gap", "effect size",
                               "p value", "significant",
                               "flagged", "note")]
    for _, row in summary_df[summary_df["flagged"]].iterrows():
        vals = {g: row[g] for g in group_cols if row[g] is not None}
        if not vals:
            continue
        lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
        out.append(f"'{row['score column']}' differs across {segment_col}: "
                   f"{hi} = {vals[hi]} vs {lo} = {vals[lo]} "
                   f"(gap {row['gap']}). Investigate before trusting "
                   f"aggregate scores.")
    return out
