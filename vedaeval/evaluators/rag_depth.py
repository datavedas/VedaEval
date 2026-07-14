"""RAG depth metrics - the retrieval-quality suite.

Five metrics answering, together: did the retriever bring the right
material, and did the answer make correct use of it?

LIBRARY DECISION (rule 1, recorded): the RAGAS library implements
several of these, but it pulls a heavy dependency tree (langchain
wrappers, provider clients) and its API moves fast. We implement the
RAGAS-paper definitions directly on our own tested machinery (the NLI
model from faithfulness + plain token logic), which keeps the
implementations transparent and the dependency set unchanged. Each
block cites the RAGAS definition it follows.

Local-first: three metrics use the NLI model (same ~70MB download as
faithfulness, no API key); two are pure token logic (always available).
"""

from __future__ import annotations

import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo

_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
         "is", "are", "was", "were", "be", "with", "at", "by", "from",
         "it", "its", "this", "that", "as", "per", "your", "my"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _STOP]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def _chunks(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\n+", text or "") if p.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


def _nli_available():
    import importlib.util
    if (importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None):
        return False, "pip install torch transformers (large)"
    return True, ""


# ---------------------------------------------------------------------------
# 1. Retrieval hit rate  [E, always available; needs ground_truth + context]
# ---------------------------------------------------------------------------

class RetrievalHitRate(Evaluator):
    info = EvaluatorInfo(
        key="retrieval_hit_rate", name="Retrieval Hit Rate", category="quality",
        inputs=["ground_truth", "context"],
        description="Did retrieval bring ANY of the needed material? A row "
                    "is a hit when at least half of the ground truth's "
                    "content words appear in the context. The bluntest, most "
                    "explainable RAG metric.",
    )

    def evaluate(self, df, config=None):
        threshold = float((config or {}).get("token_recall_threshold", 0.5))
        hits, recalls = [], []
        gts = df["ground_truth"].astype("string").fillna("").tolist()
        ctxs = df["context"].astype("string").fillna("").tolist()
        for gt, ctx in zip(gts, ctxs):
            gt_toks = set(_tokens(gt))
            if not gt_toks or not ctx.strip():
                hits.append(None)
                recalls.append(None)
                continue
            ctx_toks = set(_tokens(ctx))
            recall = len(gt_toks & ctx_toks) / len(gt_toks)
            recalls.append(round(recall, 4))
            hits.append(recall >= threshold)
        return {"retrieval_hit": hits, "retrieval_token_recall": recalls}


# ---------------------------------------------------------------------------
# 2. Context entity recall  [E, always available]
# ---------------------------------------------------------------------------

class ContextEntityRecall(Evaluator):
    info = EvaluatorInfo(
        key="context_entity_recall", name="Context Entity Recall",
        category="quality", inputs=["ground_truth", "context"],
        description="Share of the ground truth's entities (numbers, names, "
                    "capitalized terms) that the context actually contains. "
                    "Misses here = retrieval lost the specific facts.",
    )

    @staticmethod
    def _entities(text: str) -> set[str]:
        ents = set(re.findall(r"\b\d[\d,.\-]*\b", text or ""))          # numbers
        ents |= {w.lower() for w in re.findall(
            r"\b[A-Z][a-zA-Z]{2,}\b", text or "")
            if w.lower() not in _STOP}   # capitalized terms, minus
                                         # sentence-initial stopwords ("The")
        return ents

    def evaluate(self, df, config=None):
        out = []
        for gt, ctx in zip(df["ground_truth"].astype("string").fillna(""),
                           df["context"].astype("string").fillna("")):
            ents = self._entities(gt)
            if not ents or not ctx.strip():
                out.append(None)
                continue
            ctx_low = ctx.lower()
            found = sum(1 for e in ents if e.lower() in ctx_low)
            out.append(round(found / len(ents), 4))
        return {"context_entity_recall": out}


# ---------------------------------------------------------------------------
# 3. Context recall  [E via NLI; needs ground_truth + context]
# ---------------------------------------------------------------------------

class ContextRecall(Evaluator):
    info = EvaluatorInfo(
        key="context_recall", name="Context Recall (NLI)",
        category="quality", inputs=["ground_truth", "context"],
        description="Share of ground-truth sentences the context can "
                    "support (entailment). Low = retrieval failed to fetch "
                    "the facts the correct answer needs. RAGAS-style, "
                    "computed locally.",
    )

    def available(self):
        return _nli_available()

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs
        threshold = float((config or {}).get("threshold", 0.5))
        out = []
        for gt, ctx in zip(df["ground_truth"].astype("string").fillna(""),
                           df["context"].astype("string").fillna("")):
            sents = _sentences(gt)
            if not sents or not ctx.strip():
                out.append(None)
                continue
            probs = _entailment_probs(ctx, sents)
            out.append(round(sum(1 for p in probs if p >= threshold) / len(sents), 4))
        return {"context_recall": out}


# ---------------------------------------------------------------------------
# 4. Context precision  [E via NLI; needs ground_truth + context]
# ---------------------------------------------------------------------------

class ContextPrecision(Evaluator):
    info = EvaluatorInfo(
        key="context_precision", name="Context Precision (NLI)",
        category="quality", inputs=["ground_truth", "context"],
        description="Share of retrieved context chunks that are USEFUL "
                    "(support some part of the ground truth). Low = the "
                    "retriever padded the prompt with irrelevant material. "
                    "RAGAS-style, computed locally.",
    )

    def available(self):
        return _nli_available()

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs
        threshold = float((config or {}).get("threshold", 0.5))
        out = []
        for gt, ctx in zip(df["ground_truth"].astype("string").fillna(""),
                           df["context"].astype("string").fillna("")):
            chunks = _chunks(ctx)
            if not chunks or not gt.strip():
                out.append(None)
                continue
            useful = 0
            for ch in chunks:
                probs = _entailment_probs(ch, _sentences(gt) or [gt])
                if max(probs) >= threshold:
                    useful += 1
            out.append(round(useful / len(chunks), 4))
        return {"context_precision": out}


# ---------------------------------------------------------------------------
# 5. Answer correctness  [E hybrid: token F1 + bidirectional NLI]
# ---------------------------------------------------------------------------

class AnswerCorrectness(Evaluator):
    info = EvaluatorInfo(
        key="answer_correctness", name="Answer Correctness (hybrid)",
        category="quality", inputs=["response", "ground_truth"],
        description="How correct is the answer against the ground truth: "
                    "half token-overlap F1 (the words match) + half "
                    "bidirectional entailment (the MEANING matches). "
                    "Stronger than overlap alone. Computed locally.",
    )

    def available(self):
        return _nli_available()

    @staticmethod
    def _token_f1(a: str, b: str) -> float:
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return 0.0
        common = set(ta) & set(tb)
        if not common:
            return 0.0
        p = len(common) / len(set(ta))
        r = len(common) / len(set(tb))
        return 2 * p * r / (p + r)

    def evaluate(self, df, config=None):
        from vedaeval.evaluators.faithfulness import _entailment_probs
        w = float((config or {}).get("semantic_weight", 0.5))
        out_score, out_f1, out_sem = [], [], []
        for resp, gt in zip(df["response"].astype("string").fillna(""),
                            df["ground_truth"].astype("string").fillna("")):
            if not resp.strip() or not gt.strip():
                out_score.append(None); out_f1.append(None); out_sem.append(None)
                continue
            f1 = self._token_f1(resp, gt)
            fwd = _entailment_probs(gt, [resp])[0]   # gt supports resp
            bwd = _entailment_probs(resp, [gt])[0]   # resp supports gt
            sem = (fwd + bwd) / 2
            out_f1.append(round(f1, 4))
            out_sem.append(round(sem, 4))
            out_score.append(round((1 - w) * f1 + w * sem, 4))
        return {"answer_correctness": out_score,
                "answer_token_f1": out_f1,
                "answer_semantic_agreement": out_sem}
