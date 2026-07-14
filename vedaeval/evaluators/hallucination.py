"""Hallucination / factuality metrics.

Four implementations + two documented deferrals:
- SummaC-ZS style consistency [NLI, local]: per response sentence, MAX
  entailment over individual CONTEXT SENTENCES (finer-grained than our
  whole-context faithfulness; Laban et al. 2022 zero-shot variant).
- QAG support ratio [LLM judge]: one judge call lists the response's
  factual claims and marks each supported/unsupported by the context.
- Sample consistency (SelfCheckGPT-style) [NLI, local]: needs extra
  response samples in a 'response_samples' column ('||'-separated);
  offline datasets rarely have them - not-applicable otherwise.
- Summary coverage + compression [always-on stats]: reference-free
  summarization signals (NOT a BLANC implementation - see deferrals).
"""

from __future__ import annotations

import json
import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo
from vedaeval.evaluators.judge import _JudgeBase, _call_llm, _err_detail, _parse_json


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def _nli_available():
    import importlib.util
    if (importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None):
        return False, "pip install torch transformers (large)"
    return True, ""


class SummaCConsistency(Evaluator):
    info = EvaluatorInfo(
        key="summac", name="SummaC Consistency (NLI)", category="quality",
        inputs=["response", "context"],
        description="Sentence-granular factual consistency: each response "
                    "sentence must be entailed by SOME context sentence. "
                    "Strict on paraphrases - cross-check low scores with "
                    "QAG Support Ratio. Local, no key.",
    )

    def available(self):
        return _nli_available()

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs
        threshold = float((config or {}).get("threshold", 0.5))
        scores, weakest = [], []
        for resp, ctx in zip(df["response"].astype("string").fillna(""),
                             df["context"].astype("string").fillna("")):
            r_sents, c_sents = _sentences(resp), _sentences(ctx)
            if not r_sents or not c_sents:
                scores.append(None); weakest.append(None)
                continue
            per_sent = []
            for rs in r_sents:
                probs = _entailment_probs(" ".join(c_sents), [rs]) \
                    if len(c_sents) > 20 else \
                    [max(_entailment_probs(cs, [rs])[0] for cs in c_sents)]
                per_sent.append(probs[0] if isinstance(probs[0], float) else probs[0])
            per_sent = [float(p) for p in per_sent]
            scores.append(round(sum(1 for p in per_sent if p >= threshold) / len(per_sent), 4))
            weakest.append(round(min(per_sent), 4))
        return {"summac_consistency": scores, "summac_weakest_sentence": weakest}


class QagSupportRatio(_JudgeBase):
    info = EvaluatorInfo(
        key="qag_support", name="QAG Support Ratio (LLM judge)",
        category="quality", inputs=["response", "context"], needs_llm=True,
        description="A judge lists the response's factual claims and marks "
                    "each supported or unsupported by the context; the score "
                    "is the supported share. Question-answer-generation "
                    "style verification in one call. Needs an API key.",
    )
    OUTPUT_PREFIX = "qag"
    PROMPT = (
        "You are auditing an AI answer against its source documents.\n"
        "Documents: {context}\n"
        "Answer: {response}\n\n"
        "List every distinct factual claim the answer makes. For each, "
        "decide if the documents support it.\n"
        'Reply with ONLY JSON: {{"claims": <int>, "supported": <int>, '
        '"unsupported_examples": ["<short quote>", ...]}}'
    )

    def evaluate(self, df, config=None):
        cfg = config or {}
        api_key = cfg.get("api_key", "")
        if not api_key:
            raise RuntimeError("no API key provided")
        ratios, claims_n, examples = [], [], []
        for idx in df.index:
            prompt = self.PROMPT.format(**self._fields(df, idx))
            try:
                verdict = _parse_json(_call_llm(cfg.get("provider", "openai"),
                                                cfg.get("model", ""), api_key,
                                                prompt))
                c = int(verdict.get("claims", 0))
                s = int(verdict.get("supported", 0))
                ratios.append(round(s / c, 4) if c else None)
                claims_n.append(c or None)
                examples.append("; ".join(verdict.get("unsupported_examples", []))[:300])
            except Exception as exc:
                ratios.append(None); claims_n.append(None)
                examples.append(_err_detail(exc))
        return {"qag_support_ratio": ratios, "qag_claims": claims_n,
                "qag_unsupported": examples}


class SampleConsistency(Evaluator):
    info = EvaluatorInfo(
        key="sample_consistency", name="Sample Consistency (SelfCheck-style)",
        category="quality", inputs=["response", "response_samples"],
        description="Hallucination signal WITHOUT ground truth or context: "
                    "if the model is guessing, independent samples disagree. "
                    "Needs a response_samples column ('||'-separated "
                    "alternative generations). Local NLI.",
    )

    def available(self):
        return _nli_available()

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs
        out = []
        for resp, samples in zip(
                df["response"].astype("string").fillna(""),
                df["response_samples"].astype("string").fillna("")):
            alts = [s.strip() for s in samples.split("||") if s.strip()]
            sents = _sentences(resp)
            if not alts or not sents:
                out.append(None)
                continue
            support = []
            for s in sents:
                probs = [_entailment_probs(a, [s])[0] for a in alts]
                support.append(sum(probs) / len(probs))
            out.append(round(sum(support) / len(support), 4))
        return {"sample_consistency": out}


class SummaryStats(Evaluator):
    info = EvaluatorInfo(
        key="summary_stats", name="Summary Coverage + Compression",
        category="quality", inputs=["response", "context"],
        description="Reference-free summarization signals: compression "
                    "ratio (summary length / source length) and keyword "
                    "coverage (share of the source's salient words that "
                    "made it into the summary). Simple, transparent, "
                    "always available.",
    )

    _STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
             "is", "are", "was", "were", "be", "with", "at", "by", "from"}

    def evaluate(self, df, config=None):
        top_k = int((config or {}).get("keywords", 15))
        comp, cov = [], []
        for resp, ctx in zip(df["response"].astype("string").fillna(""),
                             df["context"].astype("string").fillna("")):
            if not ctx.strip() or not resp.strip():
                comp.append(None); cov.append(None)
                continue
            comp.append(round(len(resp.split()) / max(len(ctx.split()), 1), 4))
            freq: dict[str, int] = {}
            for t in re.findall(r"[a-z0-9]+", ctx.lower()):
                if t not in self._STOP and len(t) > 3:
                    freq[t] = freq.get(t, 0) + 1
            keywords = sorted(freq, key=freq.get, reverse=True)[:top_k]
            if not keywords:
                cov.append(None)
                continue
            low = resp.lower()
            cov.append(round(sum(1 for k in keywords if k in low) / len(keywords), 4))
        return {"compression_ratio": comp, "keyword_coverage": cov}
