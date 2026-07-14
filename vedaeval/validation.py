"""Pre-flight dataset validation.

Mirrors the validation stages an enterprise eval platform runs before
any metric is computed: health statistics, duplicate detection,
conflicting duplicates, RAG-leakage detection, and PII scanning.
Every check returns row indices so the UI can offer audit-trailed
row exclusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    check: str
    severity: str  # "info" | "warning" | "error"
    message: str
    rows: list[int] = field(default_factory=list)


@dataclass
class ValidationReport:
    n_rows: int
    health: dict
    issues: list[ValidationIssue]

    @property
    def flagged_rows(self) -> set[int]:
        out: set[int] = set()
        for issue in self.issues:
            if issue.severity in ("warning", "error"):
                out.update(issue.rows)
        return out


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------

def health_stats(df) -> dict:
    stats: dict = {"rows": int(len(df))}
    for col in ("request", "response", "ground_truth", "context"):
        if col in df.columns:
            series = df[col].astype("string")
            lengths = series.str.len()
            stats[col] = {
                "missing": int(series.isna().sum() + (series.fillna("").str.strip() == "").sum()),
                "avg_len": float(lengths.mean()) if len(lengths.dropna()) else 0.0,
                "max_len": int(lengths.max()) if len(lengths.dropna()) else 0,
            }
    return stats


def check_min_rows(df, minimum: int = 60) -> list[ValidationIssue]:
    if len(df) < minimum:
        return [ValidationIssue(
            check="dataset_size",
            severity="warning",
            message=f"{len(df)} rows; at least {minimum} recommended for statistical significance.",
        )]
    return []


def check_missing_required(df) -> list[ValidationIssue]:
    issues = []
    for col in ("request", "response"):
        if col not in df.columns:
            issues.append(ValidationIssue(
                check="required_columns", severity="error",
                message=f"Required column '{col}' is missing.",
            ))
            continue
        series = df[col].astype("string").fillna("")
        empty = df.index[series.str.strip() == ""].tolist()
        if empty:
            issues.append(ValidationIssue(
                check="missing_values", severity="warning",
                message=f"{len(empty)} rows have empty '{col}'.",
                rows=[int(i) for i in empty],
            ))
    return issues


def check_exact_duplicates(df) -> list[ValidationIssue]:
    cols = [c for c in ("request", "response", "ground_truth", "context") if c in df.columns]
    dup_mask = df.duplicated(subset=cols, keep="first")
    rows = df.index[dup_mask].tolist()
    if rows:
        return [ValidationIssue(
            check="exact_duplicates", severity="warning",
            message=f"{len(rows)} exact duplicate rows (all evaluation columns identical).",
            rows=[int(i) for i in rows],
        )]
    return []


def check_conflicting_duplicates(df) -> list[ValidationIssue]:
    """Same request appearing with different ground truths."""
    if "request" not in df.columns or "ground_truth" not in df.columns:
        return []
    sub = df[["request", "ground_truth"]].astype("string").fillna("")
    grouped = sub.groupby("request")["ground_truth"].nunique()
    conflicted = set(grouped[grouped > 1].index)
    rows = df.index[sub["request"].isin(conflicted)].tolist()
    if rows:
        return [ValidationIssue(
            check="conflicting_duplicates", severity="warning",
            message=(f"{len(conflicted)} questions appear with more than one distinct "
                     f"ground truth ({len(rows)} rows affected)."),
            rows=[int(i) for i in rows],
        )]
    return []


def check_rag_leakage(df, min_len: int = 30) -> list[ValidationIssue]:
    """Ground truth text found verbatim inside the retrieved context.

    Verbatim containment of a long ground-truth string inside context
    inflates faithfulness/accuracy scores without measuring the model.
    """
    if "ground_truth" not in df.columns or "context" not in df.columns:
        return []
    rows = []
    gt = df["ground_truth"].astype("string").fillna("")
    ctx = df["context"].astype("string").fillna("")
    for idx in df.index:
        g = gt.loc[idx].strip()
        if len(g) >= min_len and g.lower() in ctx.loc[idx].lower():
            rows.append(int(idx))
    if rows:
        return [ValidationIssue(
            check="rag_leakage", severity="warning",
            message=(f"{len(rows)} rows where the ground truth appears verbatim in the "
                     f"context (possible leakage; scores may be inflated)."),
            rows=rows,
        )]
    return []


# regex fallback engine for PII (always available)
_PII_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE_NUMBER": re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)[-.\s]?)?\d{3,4}[-.\s]\d{3,4}(?:[-.\s]\d{2,4})?(?!\d)"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def scan_pii_regex(df, columns: tuple[str, ...] = ("request", "response", "context")) -> list[ValidationIssue]:
    hits: dict[str, list[int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype("string").fillna("")
        for idx in df.index:
            text = series.loc[idx]
            for label, pattern in _PII_PATTERNS.items():
                if pattern.search(text):
                    hits.setdefault(label, []).append(int(idx))
    issues = []
    for label, rows in hits.items():
        rows = sorted(set(rows))
        issues.append(ValidationIssue(
            check=f"pii_{label.lower()}", severity="warning",
            message=f"{len(rows)} rows contain {label} (regex engine).",
            rows=rows,
        ))
    return issues


def scan_pii_presidio(df, columns: tuple[str, ...] = ("request", "response", "context"),
                      score_threshold: float = 0.5) -> list[ValidationIssue] | None:
    """Presidio-based PII scan. Returns None when presidio is unavailable."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
    except Exception:
        return None
    try:
        analyzer = AnalyzerEngine()
    except Exception:
        return None
    hits: dict[str, list[int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype("string").fillna("")
        for idx in df.index:
            text = series.loc[idx]
            if not text:
                continue
            try:
                results = analyzer.analyze(text=str(text), language="en")
            except Exception:
                return None
            for r in results:
                if r.score >= score_threshold:
                    hits.setdefault(r.entity_type, []).append(int(idx))
    issues = []
    for label, rows in hits.items():
        rows = sorted(set(rows))
        issues.append(ValidationIssue(
            check=f"pii_{label.lower()}", severity="warning",
            message=f"{len(rows)} rows contain {label} (Presidio engine).",
            rows=rows,
        ))
    return issues


def _bias_columns(df) -> list[str]:
    """Columns that look like bias/segmentation attributes (name contains
    'bias', 'gender', 'age', or 'segment')."""
    hits = []
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in ("bias", "gender", "age", "segment")):
            if df[col].dtype == object or str(df[col].dtype) == "string" or df[col].nunique() <= 20:
                hits.append(col)
    return hits


def check_segment_sizes(df, min_size: int = 10) -> list[ValidationIssue]:
    """Each value of a bias column should have at least min_size rows,
    otherwise per-segment metrics (fairness, per-group scores) are noise."""
    issues = []
    for col in _bias_columns(df):
        counts = df[col].astype("string").fillna("Unknown").value_counts()
        small = {str(v): int(c) for v, c in counts.items()
                 if v != "Unknown" and c < min_size}
        if small:
            listed = ", ".join(f"'{v}' ({c} rows)" for v, c in small.items())
            issues.append(ValidationIssue(
                check=f"segment_size_{_norm_name(col)}", severity="warning",
                message=(f"Column '{col}': segments below {min_size} rows: "
                         f"{listed}. Per-segment results will be unreliable."),
            ))
    return issues


def check_diversity(df, min_distinct: int = 2) -> list[ValidationIssue]:
    """A bias column with fewer than 2 distinct values cannot support any
    group comparison (the 'all rows male >=65' failure mode)."""
    issues = []
    for col in _bias_columns(df):
        values = set(df[col].astype("string").fillna("").str.strip())
        values.discard("")
        values.discard("Unknown")
        if len(values) < min_distinct:
            found = ", ".join(sorted(values)) if values else "none"
            issues.append(ValidationIssue(
                check=f"diversity_{_norm_name(col)}", severity="warning",
                message=(f"Column '{col}' has only {len(values)} distinct "
                         f"value(s) ({found}); at least {min_distinct} are "
                         f"needed for balanced group evaluation."),
            ))
    return issues


def _norm_name(col: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_")


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def validate_dataset(df, pii_engine: str = "auto") -> ValidationReport:
    """Run the full validation pipeline on a canonical dataframe.

    pii_engine: "auto" (presidio if installed, else regex), "presidio",
    "regex", or "off".
    """
    issues: list[ValidationIssue] = []
    issues += check_min_rows(df)
    issues += check_missing_required(df)
    issues += check_exact_duplicates(df)
    issues += check_conflicting_duplicates(df)
    issues += check_rag_leakage(df)
    issues += check_segment_sizes(df)
    issues += check_diversity(df)

    if pii_engine != "off":
        pii_issues = None
        if pii_engine in ("auto", "presidio"):
            pii_issues = scan_pii_presidio(df)
        if pii_issues is None and pii_engine in ("auto", "regex"):
            pii_issues = scan_pii_regex(df)
        issues += pii_issues or []

    return ValidationReport(n_rows=len(df), health=health_stats(df), issues=issues)
