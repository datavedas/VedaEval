"""RAG depth, second wave.

Four evaluators:
- FaithfulnessJudge [F1+F3]: the judge variant of grounding. Holistic
  High/Medium/Low with the worst unsupported claim quoted. Deliberately
  overlaps the local NLI faithfulness and the per-claim QAG so a run
  can cross-check calculation against opinion (the established
  answer/context-relevance pattern).
- NoiseSensitivity [F1+F2]: with mixed retrieval, how much does the
  answer lean on chunks that do NOT support the ground truth? Reuses
  the per-passage grounding machinery (USU pattern) crossed with
  chunk-vs-GT support. Our operationalization of the RAGAS idea -
  documented deviation.
- CitationPrecision / CitationRecall [F1]: for responses that cite
  their sources with [n] / [docN] markers (config regex): precision =
  of the citations made, how many point at a chunk that supports the
  citing sentence (NLI); recall = how many sentences carry a citation
  at all. Split per the catalog (v2 split them deliberately).
"""

from __future__ import annotations

import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo
from vedaeval.evaluators.judge import _JudgeBase
from vedaeval.evaluators.safety_novel import _passages


def _nli_available():
    import importlib.util
    if (importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None):
        return False, "pip install torch transformers (large)"
    return True, ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip())
            if len(s.strip()) >= 3]


class FaithfulnessJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="faithfulness_judge", name="Faithfulness (LLM judge)",
        category="quality", inputs=["response", "context"], needs_llm=True,
        description="The judge's opinion on grounding: is everything the "
                    "response asserts supported by the retrieved documents? "
                    "High/Medium/Low with the worst unsupported claim "
                    "quoted. Cross-checks the local NLI faithfulness "
                    "(calculation vs opinion). Needs an API key.",
    )
    OUTPUT_PREFIX = "faithfulness_judge"
    PROMPT = (
        "You are auditing whether an AI answer is grounded in its source "
        "documents.\nDocuments: {context}\nAnswer: {response}\n\n"
        "Is every claim in the answer supported by the documents? High = "
        "fully supported; Medium = minor unsupported detail; Low = a "
        "material claim is unsupported.\nReply with ONLY a JSON object: "
        '{{"rating": "High"|"Medium"|"Low", '
        '"reason": "<the worst unsupported claim, or \'fully supported\'>"}}'
    )


class NoiseSensitivity(Evaluator):
    info = EvaluatorInfo(
        key="noise_sensitivity", name="Noise Sensitivity",
        category="quality", inputs=["response", "context", "ground_truth"],
        description="With mixed retrieval, how much of the answer's "
                    "support comes from chunks that do NOT back the "
                    "correct answer? High = the model is distracted by "
                    "retrieval noise. Per-passage NLI grounding crossed "
                    "with passage-vs-ground-truth support. Local models, "
                    "no key.",
    )

    def available(self):
        return _nli_available()

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs
        tau = float((config or {}).get("support_tau", 0.5))
        scores, noise_shares = [], []
        for idx in df.index:
            resp = str(df["response"].loc[idx] or "")
            ctx = str(df["context"].loc[idx] or "")
            gt = str(df["ground_truth"].loc[idx] or "")
            passages = _passages(ctx)
            r_sents = _sentences(resp)
            gt_sents = _sentences(gt)
            if not passages or not r_sents or not gt_sents:
                scores.append(None); noise_shares.append(None)
                continue
            # which passages support the ground truth at all
            supporting = []
            for p in passages:
                probs = _entailment_probs(p, gt_sents)
                supporting.append(max(float(x) for x in probs) >= tau)
            noise_shares.append(round(
                sum(1 for s in supporting if not s) / len(passages), 4))
            if all(supporting):
                scores.append(0.0)   # no noise in view; nothing to be
                continue             # distracted by
            # response grounding weight per passage (USU pattern)
            weights = []
            for p in passages:
                probs = _entailment_probs(p, r_sents)
                weights.append(sum(float(x) for x in probs) / len(probs))
            total = sum(weights)
            if total < 1e-6:
                scores.append(None)
                continue
            noise_w = sum(w for w, s in zip(weights, supporting) if not s)
            scores.append(round(noise_w / total, 4))
        return {"noise_sensitivity": scores,
                "noise_passage_share": noise_shares}


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

DEFAULT_CITE_RE = r"\[(?:doc\s*)?(\d+)\]"


def _parse_citations(text: str, pattern: str) -> list[tuple[str, list[int]]]:
    """[(sentence_without_markers, [passage indices 1-based]), ...]"""
    rx = re.compile(pattern, re.IGNORECASE)
    out = []
    for sent in _sentences(text):
        idxs = [int(m) for m in rx.findall(sent)]
        clean = rx.sub("", sent).strip()
        out.append((clean, idxs))
    return out


class CitationPrecision(Evaluator):
    info = EvaluatorInfo(
        key="citation_precision", name="Citation Precision",
        category="quality", inputs=["response", "context"],
        description="Of the citations the response makes ([1], [doc2] "
                    "markers; config cite_pattern overrides), how many "
                    "point at a context chunk that actually supports the "
                    "citing sentence (NLI). None when the response cites "
                    "nothing. Local model, no key.",
    )

    def available(self):
        return _nli_available()

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs
        pattern = (config or {}).get("cite_pattern", DEFAULT_CITE_RE)
        tau = float((config or {}).get("support_tau", 0.5))
        precisions, bad = [], []
        for idx in df.index:
            resp = str(df["response"].loc[idx] or "")
            passages = _passages(str(df["context"].loc[idx] or ""))
            pairs = _parse_citations(resp, pattern)
            cites = [(s, i) for s, idxs in pairs for i in idxs]
            if not cites or not passages:
                precisions.append(None); bad.append("")
                continue
            ok = 0; misses = []
            for sent, i in cites:
                if 1 <= i <= len(passages) and sent:
                    p = float(_entailment_probs(passages[i - 1], [sent])[0])
                    if p >= tau:
                        ok += 1
                        continue
                misses.append(f"[{i}] {sent[:50]}")
            precisions.append(round(ok / len(cites), 4))
            bad.append("; ".join(misses)[:200])
        return {"citation_precision": precisions,
                "citation_unsupported": bad}


class CitationRecall(Evaluator):
    info = EvaluatorInfo(
        key="citation_recall", name="Citation Recall",
        category="quality", inputs=["response", "context"],
        description="Of the response's sentences, how many carry a "
                    "citation marker at all ([1], [doc2]; config "
                    "cite_pattern overrides). Pairs with citation "
                    "precision: recall = claims cited, precision = "
                    "citations correct. None when the response has no "
                    "sentences. Always available.",
    )

    def evaluate(self, df, config=None):
        pattern = (config or {}).get("cite_pattern", DEFAULT_CITE_RE)
        recalls = []
        for resp in df["response"].astype("string").fillna(""):
            pairs = _parse_citations(resp, pattern)
            if not pairs:
                recalls.append(None)
                continue
            cited = sum(1 for _, idxs in pairs if idxs)
            recalls.append(round(cited / len(pairs), 4))
        return {"citation_recall": recalls}
