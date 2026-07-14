"""VedaEval - an open LLM evaluation engine.

Batch evaluation of LLM request/response datasets: data validation,
a registry of evaluators (deterministic + model-backed), and reporting.
"""

__version__ = "0.1.0"

from vedaeval.schema import CANONICAL_FIELDS, auto_map_columns, apply_mapping
from vedaeval.engine import run_evaluation
from vedaeval.evaluators import REGISTRY, available_evaluators

__all__ = [
    "CANONICAL_FIELDS",
    "auto_map_columns",
    "apply_mapping",
    "run_evaluation",
    "REGISTRY",
    "available_evaluators",
]
