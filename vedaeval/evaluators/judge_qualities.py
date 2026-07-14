"""Judge-quality metrics - five LLM-as-a-judge evaluators
on the tested judge frame. All need an API key (BYO); all High/Medium/
Low + reason unless stated.
"""

from __future__ import annotations

from vedaeval.evaluators.base import EvaluatorInfo
from vedaeval.evaluators.judge import _JudgeBase, _call_llm, _err_detail, _parse_json


class CompletenessJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="completeness", name="Completeness (LLM judge)",
        category="quality", inputs=["request", "response"], needs_llm=True,
        description="Did the answer address ALL parts of the question? "
                    "Multi-part questions with half answers score Low.",
    )
    OUTPUT_PREFIX = "completeness"
    PROMPT = (
        "You are evaluating an AI assistant's answer.\n"
        "Question: {request}\nAnswer: {response}\n\n"
        "Identify every distinct part of the question. Does the answer "
        "address ALL of them? Partial coverage = Medium; major parts "
        "ignored = Low.\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class HelpfulnessJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="helpfulness", name="Helpfulness (LLM judge)",
        category="quality", inputs=["request", "response"], needs_llm=True,
        description="Overall usefulness to the person asking: actionable, "
                    "clear, appropriately detailed (MT-Bench-style rubric).",
    )
    OUTPUT_PREFIX = "helpfulness"
    PROMPT = (
        "You are evaluating how HELPFUL an AI answer is to the person "
        "asking.\nQuestion: {request}\nAnswer: {response}\n\n"
        "Consider: does it solve their need, is it actionable, is the "
        "level of detail right? Ignore style polish.\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class InstructionAdherenceJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="instruction_adherence", name="Instruction Adherence (LLM judge)",
        category="quality", inputs=["request", "response"], needs_llm=True,
        description="Did the answer follow the EXPLICIT constraints in the "
                    "request (format, length, language, exclusions)? "
                    "IFEval-style.",
    )
    OUTPUT_PREFIX = "instruction_adherence"
    PROMPT = (
        "You are auditing instruction-following.\n"
        "Request: {request}\nResponse: {response}\n\n"
        "List any EXPLICIT instructions in the request (format, length, "
        "language, things to include or avoid). Did the response follow "
        "every one? No explicit instructions at all = High.\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class GEvalJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="geval", name="G-Eval Scored Rubric (LLM judge)",
        category="quality", inputs=["request", "response"], needs_llm=True,
        description="Fine-grained 1-5 scoring with think-then-score "
                    "reasoning against a criterion you set (default: overall "
                    "quality). Config: criterion=...",
    )
    OUTPUT_PREFIX = "geval"

    def evaluate(self, df, config=None):
        cfg = config or {}
        api_key = cfg.get("api_key", "")
        if not api_key:
            raise RuntimeError("no API key provided")
        criterion = cfg.get("criterion",
                            "overall response quality: accuracy, clarity, "
                            "usefulness")
        scores, reasons = [], []
        for idx in df.index:
            f = self._fields(df, idx)
            prompt = (
                "You are scoring an AI response against a criterion, "
                "G-Eval style.\n"
                f"Criterion: {criterion}\n"
                f"Question: {f['request']}\nResponse: {f['response']}\n\n"
                "Think through the criterion step by step, then give an "
                "integer score 1 (very poor) to 5 (excellent).\n"
                'Reply with ONLY JSON: {"score": <1-5>, '
                '"reason": "<one short sentence>"}'
            )
            try:
                v = _parse_json(_call_llm(cfg.get("provider", "openai"),
                                          cfg.get("model", ""), api_key, prompt))
                s = v.get("score")
                scores.append(int(s) if s is not None else None)
                reasons.append(str(v.get("reason", ""))[:300])
            except Exception as exc:
                scores.append(None)
                reasons.append(_err_detail(exc))
        return {"geval_score": scores, "geval_reason": reasons}


class PairwiseJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="pairwise", name="Pairwise Win Rate (LLM judge)",
        category="quality", inputs=["request", "response", "response_b"],
        needs_llm=True,
        description="A vs B on the same question: which response is better? "
                    "Needs a response_b column (the challenger). Outputs "
                    "winner per row; the run summary's counts ARE the win "
                    "rate. Order randomization applied (position-bias "
                    "guard).",
    )
    OUTPUT_PREFIX = "pairwise"

    def evaluate(self, df, config=None):
        import random
        cfg = config or {}
        api_key = cfg.get("api_key", "")
        if not api_key:
            raise RuntimeError("no API key provided")
        rng = random.Random(7)  # deterministic order flips
        winners, reasons = [], []
        for idx in df.index:
            req = str(df["request"].loc[idx])
            a = str(df["response"].loc[idx])
            b = str(df["response_b"].loc[idx])
            flipped = rng.random() < 0.5
            first, second = (b, a) if flipped else (a, b)
            prompt = (
                "Two AI responses to the same question. Judge which is "
                "better overall (accuracy, helpfulness, clarity). A tie is "
                "allowed.\n"
                f"Question: {req}\nResponse 1: {first}\nResponse 2: {second}\n\n"
                'Reply with ONLY JSON: {"winner": "1"|"2"|"tie", '
                '"reason": "<one short sentence>"}'
            )
            try:
                v = _parse_json(_call_llm(cfg.get("provider", "openai"),
                                          cfg.get("model", ""), api_key, prompt))
                w = str(v.get("winner", "")).strip()
                if w == "tie":
                    winners.append("tie")
                elif w in ("1", "2"):
                    # map back through the flip
                    is_first = (w == "1")
                    winners.append("B" if (is_first == flipped) else "A")
                else:
                    winners.append(None)
                reasons.append(str(v.get("reason", ""))[:300])
            except Exception as exc:
                winners.append(None)
                reasons.append(_err_detail(exc))
        return {"pairwise_winner": winners, "pairwise_reason": reasons}
