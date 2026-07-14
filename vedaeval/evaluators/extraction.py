"""Information Extraction metrics - for tasks where the
model pulls STRUCTURED fields out of documents.

Data contract: response and ground_truth each hold the extracted fields
as JSON objects ({"name": "...", "dob": "..."}) or key: value lines.
All metrics are plain set/equality logic - transparent, always on.
"""

from __future__ import annotations

import json
import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


def parse_fields(text: str) -> dict[str, str]:
    """JSON object preferred; 'key: value' lines as fallback."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {str(k).strip().lower(): str(v).strip()
                    for k, v in obj.items()}
    except Exception:
        pass
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip() and v.strip():
                fields[k.strip().lower()] = v.strip()
    return fields


def _norm_val(v: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", v.lower())).strip()


class ExtractionAccuracy(Evaluator):
    info = EvaluatorInfo(
        key="extraction", name="Extraction Accuracy Suite", category="quality",
        inputs=["response", "ground_truth"], optional_inputs=["context"],
        description="For structured extraction tasks (fields as JSON or "
                    "key: value lines): precision/recall/F1 over extracted "
                    "fields, field-level accuracy, spurious extraction rate "
                    "(invented fields - IE's hallucination), and span "
                    "grounding (values verbatim in the source) when context "
                    "exists.",
    )

    def evaluate(self, df, config=None):
        p_l, r_l, f_l, acc_l, spur_l, span_l = [], [], [], [], [], []
        has_ctx = "context" in df.columns
        for idx in df.index:
            gt = parse_fields(str(df["ground_truth"].loc[idx])
                              if "ground_truth" in df.columns else "")
            pred = parse_fields(str(df["response"].loc[idx]))
            if not gt:
                p_l.append(None); r_l.append(None); f_l.append(None)
                acc_l.append(None); spur_l.append(None); span_l.append(None)
                continue
            # value-correct = same key present AND normalized values equal
            correct_keys = [k for k in pred
                            if k in gt and _norm_val(pred[k]) == _norm_val(gt[k])]
            tp = len(correct_keys)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(gt)
            f = (2 * p * r / (p + r)) if (p + r) else 0.0
            # field-level accuracy over the GT schema
            acc = sum(1 for k in gt
                      if k in pred and _norm_val(pred[k]) == _norm_val(gt[k])) / len(gt)
            # spurious = keys the model invented (not in the GT schema)
            spur = (sum(1 for k in pred if k not in gt) / len(pred)) if pred else 0.0
            p_l.append(round(p, 4)); r_l.append(round(r, 4))
            f_l.append(round(f, 4)); acc_l.append(round(acc, 4))
            spur_l.append(round(spur, 4))
            # span grounding: extracted values appearing verbatim in context
            if has_ctx and pred:
                ctx_norm = _norm_val(str(df["context"].loc[idx]))
                grounded = sum(1 for v in pred.values()
                               if _norm_val(v) and _norm_val(v) in ctx_norm)
                span_l.append(round(grounded / len(pred), 4))
            else:
                span_l.append(None)
        return {"extraction_precision": p_l, "extraction_recall": r_l,
                "extraction_f1": f_l, "field_accuracy": acc_l,
                "spurious_extraction_rate": spur_l,
                "span_grounding_rate": span_l}
