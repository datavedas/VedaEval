"""Topic classification - zero-shot, reusing the NLI faithfulness model.

"Zero-shot" = the model has seen zero examples of the user's categories
and classifies anyway. Trick: for each user-defined label we ask the NLI
entailment model "does this text support the statement 'This text is
about <label>'?" - the label with the strongest entailment wins.

Config: {"topics": ["billing", "coverage", ...], "column": "request"}
Outputs: topic (winning label), topic_confidence (its normalized share).
Local and free; needs the same torch/transformers install as
faithfulness (~70MB model, shared and cached).
"""

from __future__ import annotations

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


class TopicClassification(Evaluator):
    info = EvaluatorInfo(
        key="topic_classification", name="Topic Classification (zero-shot)",
        category="text_stats", inputs=["request"],
        description="Sorts rows into YOUR topic labels with no training, "
                    "using the same local entailment model as faithfulness "
                    "(config: comma-separated topics). Adds the winning "
                    "topic and its confidence per row.",
    )

    def available(self):
        import importlib.util
        if (importlib.util.find_spec("torch") is None
                or importlib.util.find_spec("transformers") is None):
            return False, "pip install torch transformers (large)"
        return True, ""

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs

        cfg = config or {}
        topics = [t.strip() for t in cfg.get("topics", []) if t.strip()]
        column = cfg.get("column", "request")
        if column not in df.columns:
            column = "response"
        if not topics:
            return {"topic": [None] * len(df),
                    "topic_confidence": [None] * len(df)}

        winners, confidences = [], []
        texts = df[column].astype("string").fillna("").tolist()
        hypotheses = [f"This text is about {t}." for t in topics]
        for text in texts:
            if not text.strip():
                winners.append(None)
                confidences.append(None)
                continue
            # premise = the row's text; one entailment score per label
            probs = _entailment_probs(text, hypotheses)
            total = sum(probs) or 1.0
            best = max(range(len(topics)), key=lambda i: probs[i])
            winners.append(topics[best])
            confidences.append(round(probs[best] / total, 4))
        return {"topic": winners, "topic_confidence": confidences}
