"""Faithfulness evaluator - NLI (entailment) based, local, no API.

Idea (see docs/06): treat the CONTEXT as the premise and each sentence of
the RESPONSE as a hypothesis, then ask a small NLI model "does the premise
entail this?". The faithfulness score for a row is the mean entailment
probability across its response sentences; a low score means the response
asserts things the context does not support (possible hallucination).

Model: cross-encoder/nli-deberta-v3-xsmall (~70MB) - small enough to be
practical on modest machines, well-known in open-source faithfulness
metrics. Downloaded once from the Hugging Face library and cached.

Outputs:
    faithful_score     float 0-1 (mean entailment; higher = more faithful)
    faithful           bool (score >= threshold, default 0.5)
    unsupported_count  int (# response sentences with entailment < threshold)
"""

from __future__ import annotations

import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo

_NLI = None  # (tokenizer, model, entail_idx) singleton

_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"


def _get_nli():
    global _NLI
    if _NLI is None:
        import torch  # noqa: F401  (ensures torch present before load)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        model.eval()
        # find which output position means "entailment" from the config
        label2id = {v.lower(): k for k, v in model.config.id2label.items()}
        entail_idx = label2id.get("entailment", 1)
        _NLI = (tokenizer, model, entail_idx)
    return _NLI


def _sentences(text: str) -> list[str]:
    """Light sentence splitter (period/question/exclamation boundaries)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def _entailment_probs(premise: str, hypotheses: list[str]) -> list[float]:
    import torch

    tokenizer, model, entail_idx = _get_nli()
    probs = []
    with torch.no_grad():
        for hyp in hypotheses:
            enc = tokenizer(premise, hyp, truncation=True, max_length=512,
                            return_tensors="pt")
            logits = model(**enc).logits[0]
            p = torch.softmax(logits, dim=-1)[entail_idx].item()
            probs.append(float(p))
    return probs


class NLIFaithfulness(Evaluator):
    info = EvaluatorInfo(
        key="faithfulness", name="Faithfulness (NLI, local)", category="quality",
        inputs=["response", "context"],
        description="Checks whether the response is supported by the "
                    "context using a local entailment model. Low scores "
                    "suggest hallucination. First run downloads a ~70MB "
                    "model. Needs a context column (RAG datasets).",
    )

    def available(self):
        # existence check only - importing torch here would stall fast paths
        # like the API's /health endpoint for 30+ seconds.
        import importlib.util
        if (importlib.util.find_spec("torch") is None
                or importlib.util.find_spec("transformers") is None):
            return False, "pip install torch transformers (large)"
        return True, ""

    def evaluate(self, df, config=None):
        threshold = float((config or {}).get("threshold", 0.5))
        scores, flags, unsupported = [], [], []

        responses = df["response"].astype("string").fillna("").tolist()
        contexts = df["context"].astype("string").fillna("").tolist()

        for resp, ctx in zip(responses, contexts):
            sents = _sentences(resp)
            if not sents or not ctx.strip():
                scores.append(None)
                flags.append(None)
                unsupported.append(None)
                continue
            probs = _entailment_probs(ctx, sents)
            mean_p = sum(probs) / len(probs)
            scores.append(round(mean_p, 4))
            flags.append(mean_p >= threshold)
            unsupported.append(sum(1 for p in probs if p < threshold))

        return {"faithful_score": scores, "faithful": flags,
                "unsupported_count": unsupported}
