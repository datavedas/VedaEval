"""Deterministic (no-LLM) evaluators.

All evaluators degrade gracefully: if an optional dependency is missing,
``available()`` returns False and the registry hides the evaluator.
"""

from __future__ import annotations

import json
import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


def _texts(df, col):
    return df[col].astype("string").fillna("").tolist()


# --------------------------------------------------------------------------
# Text statistics (textstat)
# --------------------------------------------------------------------------

class TextStat(Evaluator):
    info = EvaluatorInfo(
        key="textstat", name="Text Statistics", category="text_stats",
        inputs=["response"],
        description="Readability and complexity statistics of the response "
                    "(Flesch-Kincaid grade, reading ease, word/sentence counts).",
    )

    # Default to just three, chosen for interpretability; the full textstat
    # menu stays available via config {"stats": [...]}.
    STATS = ["lexicon_count", "flesch_reading_ease", "flesch_kincaid_grade"]

    def available(self):
        try:
            import textstat  # noqa: F401
            return True, ""
        except Exception:
            return False, "pip install textstat"

    def evaluate(self, df, config=None):
        import textstat
        stats = (config or {}).get("stats", self.STATS)
        out = {s: [] for s in stats}
        for text in _texts(df, "response"):
            for s in stats:
                try:
                    out[s].append(getattr(textstat, s)(text) if text else None)
                except Exception:
                    out[s].append(None)
        return out


# --------------------------------------------------------------------------
# Sentiment (VADER)
# --------------------------------------------------------------------------

class Sentiment(Evaluator):
    info = EvaluatorInfo(
        key="sentiment", name="Sentiment (VADER)", category="text_stats",
        inputs=["response"],
        description="Compound sentiment score and positive/neutral/negative label.",
    )

    def available(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401
            return True, ""
        except Exception:
            return False, "pip install vaderSentiment"

    def evaluate(self, df, config=None):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        compounds, labels = [], []
        for text in _texts(df, "response"):
            c = analyzer.polarity_scores(text)["compound"] if text else 0.0
            compounds.append(c)
            labels.append("positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")
        return {"sentiment_compound": compounds, "sentiment": labels}


# --------------------------------------------------------------------------
# N-gram overlap vs ground truth (BLEU / ROUGE)
# --------------------------------------------------------------------------

class Overlap(Evaluator):
    info = EvaluatorInfo(
        key="overlap", name="BLEU / ROUGE vs Ground Truth", category="quality",
        inputs=["response", "ground_truth"],
        description="N-gram overlap between response and ground truth "
                    "(BLEU, ROUGE-1/2/L). Reference-based accuracy proxy.",
    )

    def available(self):
        try:
            import rouge_score  # noqa: F401
            import nltk  # noqa: F401
            return True, ""
        except Exception:
            return False, "pip install rouge-score nltk"

    def evaluate(self, df, config=None):
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        smooth = SmoothingFunction().method1
        out = {"bleu": [], "rouge1": [], "rouge2": [], "rougeL": []}
        responses = _texts(df, "response")
        refs = _texts(df, "ground_truth")
        for resp, ref in zip(responses, refs):
            if not resp or not ref:
                for k in out:
                    out[k].append(None)
                continue
            scores = scorer.score(ref, resp)
            out["rouge1"].append(round(scores["rouge1"].fmeasure, 4))
            out["rouge2"].append(round(scores["rouge2"].fmeasure, 4))
            out["rougeL"].append(round(scores["rougeL"].fmeasure, 4))
            try:
                bleu = sentence_bleu([ref.split()], resp.split(), smoothing_function=smooth)
            except Exception:
                bleu = None
            out["bleu"].append(round(bleu, 4) if bleu is not None else None)
        return out


# --------------------------------------------------------------------------
# Profanity
# --------------------------------------------------------------------------

class Profanity(Evaluator):
    info = EvaluatorInfo(
        key="profanity", name="Profanity", category="safety",
        inputs=["response"], optional_inputs=["request"],
        description="Flags offensive or inappropriate language in request/response.",
    )

    def available(self):
        try:
            from better_profanity import profanity  # noqa: F401
            return True, ""
        except Exception:
            return False, "pip install better-profanity"

    def evaluate(self, df, config=None):
        from better_profanity import profanity
        profanity.load_censor_words()
        cols = [c for c in ("request", "response") if c in df.columns]
        out = {}
        for col in cols:
            out[f"profanity_{col}"] = [bool(profanity.contains_profanity(t)) if t else False
                                       for t in _texts(df, col)]
        return out


# --------------------------------------------------------------------------
# Banned keywords
# --------------------------------------------------------------------------

class BannedKeywords(Evaluator):
    info = EvaluatorInfo(
        key="banned_keywords", name="Banned Keywords", category="safety",
        inputs=["response"],
        description="Detects user-defined restricted terms (config: keywords=[...]).",
    )

    def evaluate(self, df, config=None):
        keywords = [k.lower() for k in (config or {}).get("keywords", []) if k.strip()]
        flags, matched = [], []
        for text in _texts(df, "response"):
            low = text.lower()
            hits = [k for k in keywords if k in low]
            flags.append(bool(hits))
            matched.append(", ".join(hits) if hits else "")
        return {"banned_keywords": flags, "banned_keywords_matched": matched}


# --------------------------------------------------------------------------
# Regex match
# --------------------------------------------------------------------------

class RegexMatch(Evaluator):
    info = EvaluatorInfo(
        key="regex_match", name="Regex Match", category="validation",
        inputs=["response"],
        description="Matches the response against a user-defined regex "
                    "(config: pattern=...). Output: Match / No Match.",
    )

    def evaluate(self, df, config=None):
        pattern = (config or {}).get("pattern", "")
        try:
            compiled = re.compile(pattern) if pattern else None
        except re.error:
            compiled = None
        results = []
        for text in _texts(df, "response"):
            if compiled is None:
                results.append(None)
            else:
                results.append("Match" if compiled.search(text) else "No Match")
        return {"regex_match": results}


# --------------------------------------------------------------------------
# Token count
# --------------------------------------------------------------------------

class TokenCount(Evaluator):
    info = EvaluatorInfo(
        key="token_count", name="Token Count", category="text_stats",
        inputs=["response"], optional_inputs=["request"],
        description="Token counts (tiktoken if available, whitespace fallback).",
    )

    def _counter(self):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda t: len(enc.encode(t))
        except Exception:
            return lambda t: len(t.split())

    def evaluate(self, df, config=None):
        count = self._counter()
        out = {}
        for col in ("request", "response"):
            if col in df.columns:
                out[f"token_count_{col}"] = [count(t) if t else 0 for t in _texts(df, col)]
        return out


# --------------------------------------------------------------------------
# JSON validation
# --------------------------------------------------------------------------

class JsonValidation(Evaluator):
    info = EvaluatorInfo(
        key="json_validation", name="Response JSON Check", category="validation",
        inputs=["response"],
        description="For structured-output tasks where the MODEL is asked to "
                    "reply in JSON: checks each response is valid JSON "
                    "(optionally against a schema). This checks the model's "
                    "answers, not your dataset file.",
    )

    def evaluate(self, df, config=None):
        schema = (config or {}).get("schema")
        valid, errors = [], []
        for text in _texts(df, "response"):
            try:
                obj = json.loads(text)
            except Exception as exc:
                valid.append(False)
                errors.append(f"invalid json: {exc}")
                continue
            if schema:
                try:
                    import jsonschema
                    jsonschema.validate(obj, schema)
                except Exception as exc:
                    valid.append(False)
                    errors.append(f"schema: {exc}")
                    continue
            valid.append(True)
            errors.append("")
        return {"json_valid": valid, "json_error": errors}


# --------------------------------------------------------------------------
# SQL validation (syntax only)
# --------------------------------------------------------------------------

class SqlValidation(Evaluator):
    info = EvaluatorInfo(
        key="sql_validation", name="SQL Validation", category="validation",
        inputs=["response"],
        description="Validates SQL syntax for a dialect (config: dialect=...). "
                    "Syntax-based; no schema check.",
    )

    def available(self):
        try:
            import sqlglot  # noqa: F401
            return True, ""
        except Exception:
            return False, "pip install sqlglot"

    def evaluate(self, df, config=None):
        import sqlglot
        dialect = (config or {}).get("dialect", None)
        valid, errors = [], []
        for text in _texts(df, "response"):
            try:
                sqlglot.parse_one(text, read=dialect)
                valid.append(True)
                errors.append("")
            except Exception as exc:
                valid.append(False)
                errors.append(str(exc)[:200])
        return {"sql_valid": valid, "sql_error": errors}
