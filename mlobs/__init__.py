"""mlobs - classic ML observability add-on for VedaEval.

Fully OPTIONAL and fully ISOLATED by design: nothing in the core
`vedaeval` package or the LLM evaluation flow imports anything from this
folder. Delete this folder and the rest of the application works
unchanged (the landing page simply stops offering the classic-ML option).

Contents:
    metrics.py - PSI/KS drift, group fairness (DP/DI/EO/AOD), degradation
    ui.py      - the Streamlit page (fairness / drift / degradation tabs)
    run_ml_checks.py - standalone tests for this add-on
"""

from mlobs.metrics import (
    psi_numeric, psi_categorical, ks_statistic, psi_band, drift_report,
    fairness_report, fairness_flags, classification_metrics,
    degradation_report,
)

__all__ = [
    "psi_numeric", "psi_categorical", "ks_statistic", "psi_band",
    "drift_report", "fairness_report", "fairness_flags",
    "classification_metrics", "degradation_report",
]
