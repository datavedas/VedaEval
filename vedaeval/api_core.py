"""Core logic behind the REST API - plain functions, no web framework.

Deliberately separated from api.py (the FastAPI wrapper) so that:
- every function here is testable without a server or fastapi installed;
- the API layer stays a thin translator: JSON in -> these functions ->
  JSON out. Same engine as the app; the API is just the third face.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# guardrail 1: safety (single text -> per-category scores)
# ---------------------------------------------------------------------------

def safety_scores(text: str) -> dict[str, float]:
    """Score one text across harm categories using the local classifier.

    Returns {"fdl_<category>": probability} - flat dict, mirroring the
    response shape of commercial guardrail endpoints.
    Raises RuntimeError if the local model dependencies are missing.
    """
    from vedaeval.evaluators.safety import SafetyClassifier, _get_model

    ev = SafetyClassifier()
    ok, why = ev.available()
    if not ok:
        raise RuntimeError(f"safety model unavailable: {why}")
    raw = _get_model().predict([text if text else " "])
    out = {}
    for src in ev.CATEGORIES:
        if src in raw:
            out[f"fdl_{src}"] = round(float(raw[src][0]), 4)
    if out:
        out["fdl_max_risk_prob"] = round(max(out.values()), 4)
    return out


# ---------------------------------------------------------------------------
# guardrail 2: faithfulness (response + context -> one score)
# ---------------------------------------------------------------------------

def faithfulness_score(response: str, context: str) -> dict[str, float]:
    """Mean entailment of the response's sentences against the context.

    Returns {"fdl_faithful_score": 0..1} (higher = more faithful).
    Raises RuntimeError if the local NLI dependencies are missing.
    """
    from vedaeval.evaluators.faithfulness import (
        NLIFaithfulness, _sentences, _entailment_probs)

    ev = NLIFaithfulness()
    ok, why = ev.available()
    if not ok:
        raise RuntimeError(f"faithfulness model unavailable: {why}")
    sents = _sentences(response or "")
    if not sents or not (context or "").strip():
        return {"fdl_faithful_score": None}
    probs = _entailment_probs(context, sents)
    return {"fdl_faithful_score": round(sum(probs) / len(probs), 4)}


# ---------------------------------------------------------------------------
# guardrail 3: sensitive information (single text -> PII spans)
# ---------------------------------------------------------------------------

_REGEX_CONFIDENCE = 0.85  # fixed confidence for pattern-based hits

def pii_spans(text: str) -> list[dict[str, Any]]:
    """Detect PII in one text. Presidio when installed, regex fallback.

    Returns a list of {"score", "label", "start", "end", "text"} spans,
    mirroring the shape of commercial sensitive-information endpoints.
    """
    text = text or ""
    # preferred engine: Presidio
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        return [{"score": round(float(r.score), 4), "label": r.entity_type,
                 "start": r.start, "end": r.end,
                 "text": text[r.start:r.end]} for r in results]
    except Exception:
        pass
    # fallback engine: the same regex patterns validation uses
    from vedaeval.validation import _PII_PATTERNS
    spans = []
    for label, pattern in _PII_PATTERNS.items():
        for m in pattern.finditer(text):
            spans.append({"score": _REGEX_CONFIDENCE, "label": label,
                          "start": m.start(), "end": m.end(),
                          "text": m.group(0)})
    return sorted(spans, key=lambda s: s["start"])


# ---------------------------------------------------------------------------
# batch evaluation (rows + metric selection -> score table)
# ---------------------------------------------------------------------------

def evaluate_rows(rows: list[dict], metrics: list[str],
                  configs: dict | None = None) -> dict:
    """Run the evaluation engine over submitted rows.

    rows: [{"request": ..., "response": ..., "ground_truth"?, "context"?}]
    Returns {"rows": [...records with score columns...],
             "ran": [...], "skipped": {...}, "n_rows": int}
    """
    import numpy as np
    import pandas as pd

    from vedaeval.engine import run_evaluation

    df = pd.DataFrame(rows)
    result = run_evaluation(df, metrics, configs or {})
    scores = result.scores.replace({np.nan: None})
    return {"rows": scores.to_dict(orient="records"),
            "ran": result.ran, "skipped": result.skipped,
            "n_rows": int(len(scores))}


def availability() -> dict[str, dict]:
    """Which evaluators this deployment can run right now."""
    from vedaeval.evaluators import REGISTRY
    out = {}
    for key, ev in REGISTRY.items():
        ok, why = ev.available()
        out[key] = {"available": ok,
                    "needs_llm_key": ev.info.needs_llm,
                    **({} if ok else {"install": why})}
    return out
