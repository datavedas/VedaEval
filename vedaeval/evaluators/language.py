"""Language detection - which language is each response in?

A cheap sanity check that catches wrong-language responses (an
English-only chatbot answering in Hindi is a real production defect).
Uses the small pure-Python `langdetect` library: letter/word pattern
statistics, milliseconds per row, no models to download.

Outputs: language (ISO code like 'en', 'hi', 'es'), language_confidence.
"""

from __future__ import annotations

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


class LanguageDetection(Evaluator):
    info = EvaluatorInfo(
        key="language_detection", name="Language Detection",
        category="text_stats", inputs=["response"],
        description="Identifies each response's language (en, hi, es, ...) "
                    "with a confidence score. Catches wrong-language "
                    "replies cheaply.",
    )

    def available(self):
        import importlib.util
        if importlib.util.find_spec("langdetect") is None:
            return False, "pip install langdetect"
        return True, ""

    def evaluate(self, df, config=None):
        from langdetect import DetectorFactory, detect_langs
        from langdetect.lang_detect_exception import LangDetectException

        DetectorFactory.seed = 7  # langdetect is randomized; pin for stability

        langs, confs = [], []
        for text in df["response"].astype("string").fillna("").tolist():
            if len(text.strip()) < 3:
                langs.append(None)
                confs.append(None)
                continue
            try:
                best = detect_langs(text)[0]
                langs.append(best.lang)
                confs.append(round(float(best.prob), 4))
            except LangDetectException:
                langs.append(None)
                confs.append(None)
        return {"language": langs, "language_confidence": confs}
