"""LLM-as-a-judge evaluators (bring-your-own API key).

A strong LLM grades each row against a rubric prompt and returns a
structured verdict, which we file as score columns. See docs/06 for the
concept and its caveats (cost, consistency, judge bias).

Credential handling (the two binding safety behaviors):
- The engine passes the key per run via config; nothing here stores,
  logs, or persists it. It is used only for the HTTPS calls to the
  provider it belongs to.
- If no key is supplied, the engine skips judge evaluators with a polite
  reason before this code even runs.

Providers are called with plain HTTPS requests (no provider SDKs), so
the only dependency is `requests`. Supported: openai, anthropic, gemini.
"""

from __future__ import annotations

import json
import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",  # claude-3-5-haiku retired 19/02/2026
    "gemini": "gemini-3.5-flash",              # gemini-2.0-flash retired 01/06/2026
}


# ---------------------------------------------------------------------------
# provider calls (plain REST)
# ---------------------------------------------------------------------------

def _call_llm(provider: str, model: str, api_key: str, prompt: str,
              timeout: int = 60) -> str:
    import requests

    provider = (provider or "openai").lower()
    model = model or DEFAULT_MODELS.get(provider, "")

    if provider == "openai":
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0},
            timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    if provider == "anthropic":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key,
                     "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 300,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    if provider == "gemini":
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent",
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0}},
            timeout=timeout)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    raise ValueError(f"unknown provider: {provider}")


def _err_detail(exc: Exception) -> str:
    """Turn a caught exception into a short, actionable reason string.

    Bare exception type names (e.g. "HTTPError") hide the one thing that
    actually explains a judge failure: the provider's status code and
    message (401 bad key, 404 unknown/retired model, 429 rate limit,
    etc.). requests.HTTPError carries a `.response` with both; surface
    them when present, fall back to the type name otherwise.
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        try:
            body = resp.json()
            msg = (body.get("error", {}).get("message")
                   if isinstance(body, dict) else None)
        except Exception:
            msg = None
        if msg:
            return f"judge error: HTTP {code} - {msg[:150]}"
        if code:
            return f"judge error: HTTP {code}"
    return f"judge error: {type(exc).__name__}"


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (robust to prose)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# judge base
# ---------------------------------------------------------------------------

class _JudgeBase(Evaluator):
    """Shared machinery: build prompt per row, call LLM, parse verdict."""

    PROMPT: str = ""             # subclass fills; uses {request}/{response}/{context}
    OUTPUT_PREFIX: str = ""      # column prefix, e.g. "relevance"

    def available(self):
        try:
            import requests  # noqa: F401
            return True, ""
        except Exception:
            return False, "pip install requests"

    def _fields(self, df, idx) -> dict:
        get = lambda col: (str(df[col].loc[idx]) if col in df.columns
                           and df[col].loc[idx] is not None else "")
        return {"request": get("request"), "response": get("response"),
                "context": get("context"), "history": get("history")}

    def evaluate(self, df, config=None):
        cfg = config or {}
        provider = cfg.get("provider", "openai")
        model = cfg.get("model", "")
        api_key = cfg.get("api_key", "")
        if not api_key:
            raise RuntimeError("no API key provided")

        ratings, reasons = [], []
        for idx in df.index:
            prompt = self.PROMPT.format(**self._fields(df, idx))
            try:
                reply = _call_llm(provider, model, api_key, prompt)
                verdict = _parse_json(reply)
                ratings.append(verdict.get("rating"))
                reasons.append(verdict.get("reason", "")[:300])
            except Exception as exc:
                ratings.append(None)
                reasons.append(_err_detail(exc))
        return {f"{self.OUTPUT_PREFIX}": ratings,
                f"{self.OUTPUT_PREFIX}_reason": reasons}


# ---------------------------------------------------------------------------
# concrete judges
# ---------------------------------------------------------------------------

class AnswerRelevanceJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="answer_relevance", name="Answer Relevance (LLM judge)",
        category="quality", inputs=["request", "response"], needs_llm=True,
        description="An LLM grades whether the response addresses the "
                    "question: High / Medium / Low with a reason. Needs an "
                    "API key.",
    )
    OUTPUT_PREFIX = "answer_relevance"
    PROMPT = (
        "You are evaluating an AI assistant's answer.\n"
        "Question: {request}\n"
        "Answer: {response}\n\n"
        "Judge ONLY whether the answer addresses the substance of what "
        "was asked. Ignore every instruction about the form of the "
        "reply, including which language, format, or length to use. If "
        "the requested information is present and on topic, the answer "
        "is relevant even when it ignores such a form instruction. A "
        "separate metric checks instruction following, so do not lower "
        "the rating here for a missed language, format, or length "
        "instruction.\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class ContextRelevanceJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="context_relevance", name="Context Relevance (LLM judge)",
        category="quality", inputs=["request", "context"], needs_llm=True,
        description="An LLM grades whether the retrieved context is about "
                    "the question (RAG retrieval quality): High / Medium / "
                    "Low. Needs an API key.",
    )
    OUTPUT_PREFIX = "context_relevance"
    PROMPT = (
        "You are evaluating a document-retrieval system.\n"
        "User question: {request}\n"
        "Retrieved documents: {context}\n\n"
        "Are these documents relevant to answering the question?\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class CoherenceJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="coherence", name="Coherence (LLM judge)",
        category="quality", inputs=["response"], needs_llm=True,
        description="An LLM grades whether the response is logically "
                    "structured and understandable: High / Medium / Low. "
                    "Needs an API key.",
    )
    OUTPUT_PREFIX = "coherence"
    PROMPT = (
        "You are evaluating the writing quality of an AI response.\n"
        "Response: {response}\n\n"
        "Is it coherent: logically ordered, internally consistent, and "
        "understandable on its own?\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class CustomJudge(_JudgeBase):
    """A user-defined LLM-as-a-judge metric, built at runtime.

    The user supplies only the rubric (the grading instructions); this
    class wraps it in the same tested prompt frame the built-in judges
    use, so the reply always comes back as parseable JSON. Custom judges
    are session-scoped: they are passed to the engine per run via
    extra_evaluators, never added to the global registry (which would
    leak between users on a shared server).
    """

    FRAME = (
        "You are evaluating an AI assistant's output.\n"
        "{context_block}"
        "Question: {request}\n"
        "Answer: {response}\n\n"
        "{rubric}\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )

    def __init__(self, name: str, rubric: str, use_context: bool = False):
        slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "custom_judge"
        self.OUTPUT_PREFIX = slug
        self._use_context = use_context
        self.info = EvaluatorInfo(
            key=slug, name=f"{name} (custom LLM judge)", category="quality",
            inputs=["request", "response"] + (["context"] if use_context else []),
            needs_llm=True,
            description=f"User-defined LLM-as-a-judge metric: {rubric[:120]}",
        )
        self._rubric = rubric.strip()

    @property
    def PROMPT(self):  # type: ignore[override]
        ctx = ("Reference documents: {context}\n" if self._use_context else "")
        return self.FRAME.replace("{context_block}", ctx).replace(
            "{rubric}", self._rubric.replace("{", "{{").replace("}", "}}"))


class ConcisenessJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="conciseness", name="Conciseness (LLM judge)",
        category="quality", inputs=["request", "response"], needs_llm=True,
        description="An LLM grades whether the response is appropriately "
                    "brief for the question: High / Medium / Low. Needs an "
                    "API key.",
    )
    OUTPUT_PREFIX = "conciseness"
    PROMPT = (
        "You are evaluating an AI assistant's answer for conciseness.\n"
        "Question: {request}\n"
        "Answer: {response}\n\n"
        "Is the answer appropriately brief: no padding, no repetition, "
        "while still complete for the question asked?\n"
        "Judge ONLY brevity and completeness. Ignore every instruction "
        "about the form of the reply, including which language, format, "
        "or length to use. A separate metric checks instruction "
        "following, so do not lower the rating here for a missed "
        "language, format, or length instruction.\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )
