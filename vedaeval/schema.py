"""Canonical dataset schema and column auto-mapping.

VedaEval normalizes every uploaded dataset to a small canonical schema
before evaluation. Column aliases are auto-detected, and the user can
override the mapping in the UI.

Canonical fields:
    request       - the full raw prompt/question sent to the LLM (required)
    response      - the raw LLM output (required)
    ground_truth  - expected/reference answer (optional; enables overlap metrics)
    context       - retrieved documents for RAG (optional; enables faithfulness)
    timestamp     - event time (optional)
"""

from __future__ import annotations

import re

CANONICAL_FIELDS = ["request", "response", "ground_truth", "context",
                    "history", "timestamp"]

REQUIRED_FIELDS = ["request", "response"]

# lowercase alias -> canonical field
COLUMN_ALIASES: dict[str, str] = {
    # request
    "request": "request",
    "question": "request",
    "prompt": "request",
    "query": "request",
    "input": "request",
    "user_query": "request",
    "instruction": "request",
    # response
    "response": "response",
    "answer": "response",
    "output": "response",
    "completion": "response",
    "model_response": "response",
    "model_answer": "response",
    "generated": "response",
    "prediction": "response",
    # ground truth
    "ground_truth": "ground_truth",
    "groundtruth": "ground_truth",
    "golden_response": "ground_truth",
    "expected": "ground_truth",
    "expected_answer": "ground_truth",
    "reference": "ground_truth",
    "target": "ground_truth",
    "label": "ground_truth",
    "gt": "ground_truth",
    # context
    "context": "context",
    "contexts": "context",
    "documents": "context",
    "retrieved_context": "context",
    "retrieved_documents": "context",
    "source_documents": "context",
    "passages": "context",
    # timestamp
    "timestamp": "timestamp",
    "time": "timestamp",
    "event_ts": "timestamp",
    "created_at": "timestamp",
    "date": "timestamp",
    "history": "history",
    "conversation": "history",
    "chat_history": "history",
    "messages": "history",
    "dialog": "history",
}


def _norm(name: str) -> str:
    """Normalize a column name for alias lookup."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def auto_map_columns(columns: list[str]) -> dict[str, str | None]:
    """Suggest a mapping {canonical_field: source_column or None}.

    First alias match wins; a source column is never mapped twice.
    """
    mapping: dict[str, str | None] = {f: None for f in CANONICAL_FIELDS}
    used: set[str] = set()
    for col in columns:
        alias = COLUMN_ALIASES.get(_norm(col))
        if alias and mapping[alias] is None and col not in used:
            mapping[alias] = col
            used.add(col)
    return mapping


def apply_mapping(df, mapping: dict[str, str | None]):
    """Return a copy of df with canonical column names.

    Unmapped canonical fields are absent. Non-mapped source columns are
    kept as metadata columns (prefixed ``meta_`` only if they collide
    with a canonical name).
    """
    import pandas as pd  # local import keeps schema importable without pandas

    out = pd.DataFrame(index=df.index)
    mapped_sources = set()
    for field, src in mapping.items():
        if src is not None and src in df.columns:
            values = df[src]
            # context may arrive as a LIST of passages (common in JSONL
            # exports) or a single string; normalize lists by joining the
            # passages so every downstream check sees one string.
            if field == "context":
                values = values.apply(
                    lambda v: "\n\n".join(str(p) for p in v)
                    if isinstance(v, (list, tuple)) else v)
            out[field] = values
            mapped_sources.add(src)
    for col in df.columns:
        if col in mapped_sources:
            continue
        name = col if col not in out.columns else f"meta_{col}"
        out[name] = df[col]
    return out


def validate_required(mapping: dict[str, str | None]) -> list[str]:
    """Return list of missing required canonical fields."""
    return [f for f in REQUIRED_FIELDS if not mapping.get(f)]
