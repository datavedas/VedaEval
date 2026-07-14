"""Response consistency - the base metric for Consistency Parity
(variance for equivalent / paraphrased inputs).

Idea: ask the same question a few different ways and a reliable model
gives you the same answer each time. This evaluator measures the
AGREEMENT across a row's answers to paraphrased versions of its
question. Feed the resulting response_consistency column into the
Segment Parity Report and you get Consistency Parity - does the tool
answer some user segments more stably than others - exactly the way
the refusal detector feeds refusal parity and readability feeds
comprehension-burden parity.

Data contract: a `response_variants` column holding the model's
answers to paraphrased versions of the same request, '||'-separated
(the offline analogue of a perturbation runner: the paraphrasing and
re-asking happen upstream, the same Mode-B pattern the counterfactual
suite uses). Rows without it are not applicable.

Scoring is deterministic and always available (token-F1 agreement, so
it runs on any log and golden-tests exactly). A `semantic` config
switches to bidirectional NLI entailment when torch/transformers are
installed, for paraphrase-tolerant agreement.
"""

from __future__ import annotations

from itertools import combinations

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


class ResponseConsistency(Evaluator):
    info = EvaluatorInfo(
        key="response_consistency", name="Response Consistency",
        category="quality", inputs=["response", "response_variants"],
        description="How stable the answer is when the same question is "
                    "paraphrased: mean pairwise agreement across the "
                    "row's answers to reworded versions of its request "
                    "(needs a response_variants column, '||'-separated). "
                    "Feed it into Segment Parity for Consistency Parity "
                    "(who gets less stable answers). Deterministic "
                    "token agreement by default; config semantic=true "
                    "uses local NLI when installed.",
    )

    def available(self):
        return True, ""   # token agreement always runs; NLI is an upgrade

    @staticmethod
    def _token_f1(a: str, b: str) -> float:
        # duplicate-aware token F1 (same convention as reference.py)
        from vedaeval.evaluators.reference import ExactMatch
        return ExactMatch._token_f1(a, b)

    def evaluate(self, df, config=None):
        cfg = config or {}
        semantic = bool(cfg.get("semantic", False))
        entail = None
        if semantic:
            import importlib.util
            if (importlib.util.find_spec("torch") is not None
                    and importlib.util.find_spec("transformers") is not None):
                from vedaeval.evaluators.faithfulness import _entailment_probs
                entail = _entailment_probs

        def sim(a: str, b: str) -> float:
            if entail is not None:
                # bidirectional entailment mean (paraphrase-tolerant)
                fwd = float(entail(a, [b])[0])
                bwd = float(entail(b, [a])[0])
                return (fwd + bwd) / 2
            return self._token_f1(a, b)

        scores, weakest = [], []
        resp = df["response"].astype("string").fillna("")
        var = df["response_variants"].astype("string").fillna("")
        for r, v in zip(resp, var):
            variants = [s.strip() for s in v.split("||") if s.strip()]
            texts = [r.strip()] + variants if r.strip() else variants
            texts = [t for t in texts if t]
            if len(texts) < 2:
                scores.append(None); weakest.append(None)
                continue
            pair_sims = [sim(a, b) for a, b in combinations(texts, 2)]
            scores.append(round(sum(pair_sims) / len(pair_sims), 4))
            weakest.append(round(min(pair_sims), 4))
        return {"response_consistency": scores,
                "response_consistency_weakest_pair": weakest}
