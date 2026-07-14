"""Reference-based metrics - comparing response to ground
truth at increasing levels of sophistication:

    exact words -> edit distance -> character n-grams -> synonyms ->
    contextual embeddings -> sentence embeddings

LIBRARY DECISIONS (rule 1, per metric):
- Exact match + token F1: FROM SCRATCH (trivial, transparent).
- Levenshtein similarity: FROM SCRATCH (20-line DP; the pip 'Levenshtein'
  package adds a C dependency for no accuracy gain at our text sizes).
- chrF and TER: sacrebleu library (the MT community's reference
  implementation; light, pure Python).
- METEOR: nltk (canonical implementation; needs the wordnet data pack -
  downloaded on first use).
- BERTScore: bert-score library. Default model downsized to
  distilbert-base-uncased (~260MB) instead of the paper's roberta-large
  (~1.4GB) - deviation documented; configurable via config model_type.
- Embedding cosine: sentence-transformers, all-MiniLM-L6-v2 (~80MB).
The last three are optional heavies: absent -> the metric hides itself.
"""

from __future__ import annotations

import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (text or "").lower())).strip()


def _pairs(df):
    return zip(df["response"].astype("string").fillna("").tolist(),
               df["ground_truth"].astype("string").fillna("").tolist())


# ---------------------------------------------------------------------------
# Exact match + token F1  [always available]
# ---------------------------------------------------------------------------

class ExactMatch(Evaluator):
    info = EvaluatorInfo(
        key="exact_match", name="Exact Match + Token F1", category="quality",
        inputs=["response", "ground_truth"],
        description="Strict QA correctness: exact_match = normalized "
                    "response equals ground truth; token_f1 = word-overlap "
                    "F1 (the SQuAD pair).",
    )

    @staticmethod
    def _token_f1(a: str, b: str) -> float:
        ta, tb = _norm_text(a).split(), _norm_text(b).split()
        if not ta or not tb:
            return 0.0
        common = {}
        for t in ta:
            common[t] = common.get(t, 0) + 1
        overlap = 0
        for t in tb:
            if common.get(t, 0) > 0:
                overlap += 1
                common[t] -= 1
        if overlap == 0:
            return 0.0
        p, r = overlap / len(ta), overlap / len(tb)
        return 2 * p * r / (p + r)

    def evaluate(self, df, config=None):
        em, f1 = [], []
        for resp, gt in _pairs(df):
            if not gt.strip():
                em.append(None); f1.append(None)
                continue
            em.append(_norm_text(resp) == _norm_text(gt))
            f1.append(round(self._token_f1(resp, gt), 4))
        return {"exact_match": em, "token_f1": f1}


# ---------------------------------------------------------------------------
# Levenshtein similarity  [always available]
# ---------------------------------------------------------------------------

class LevenshteinSimilarity(Evaluator):
    info = EvaluatorInfo(
        key="levenshtein", name="Levenshtein Similarity", category="quality",
        inputs=["response", "ground_truth"],
        description="Character-level edit similarity, 0-1: how many "
                    "insertions/deletions/substitutions separate the "
                    "response from the ground truth, normalized by length.",
    )

    @staticmethod
    def _distance(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                                prev[j - 1] + (ca != cb)))
            prev = curr
        return prev[-1]

    def evaluate(self, df, config=None):
        out = []
        for resp, gt in _pairs(df):
            if not gt.strip():
                out.append(None)
                continue
            a, b = _norm_text(resp), _norm_text(gt)
            denom = max(len(a), len(b)) or 1
            out.append(round(1 - self._distance(a, b) / denom, 4))
        return {"levenshtein_similarity": out}


# ---------------------------------------------------------------------------
# chrF + TER (sacrebleu)
# ---------------------------------------------------------------------------

class ChrfTer(Evaluator):
    info = EvaluatorInfo(
        key="chrf_ter", name="chrF + TER (sacrebleu)", category="quality",
        inputs=["response", "ground_truth"],
        description="chrF: character n-gram F-score, 0-1 (robust to "
                    "morphology/typos). TER: translation edit rate, 0 = "
                    "identical, higher = more edits needed (can exceed 1).",
    )

    def available(self):
        import importlib.util
        if importlib.util.find_spec("sacrebleu") is None:
            return False, "pip install sacrebleu"
        return True, ""

    def evaluate(self, df, config=None):
        from sacrebleu.metrics import CHRF, TER
        chrf, ter = CHRF(), TER()
        out_c, out_t = [], []
        for resp, gt in _pairs(df):
            if not gt.strip() or not resp.strip():
                out_c.append(None); out_t.append(None)
                continue
            out_c.append(round(chrf.sentence_score(resp, [gt]).score / 100, 4))
            out_t.append(round(ter.sentence_score(resp, [gt]).score / 100, 4))
        return {"chrf": out_c, "ter": out_t}


# ---------------------------------------------------------------------------
# METEOR (nltk)
# ---------------------------------------------------------------------------

class Meteor(Evaluator):
    info = EvaluatorInfo(
        key="meteor", name="METEOR", category="quality",
        inputs=["response", "ground_truth"],
        description="Overlap score that also credits stems and synonyms "
                    "(via WordNet), 0-1 - kinder than BLEU to legitimate "
                    "rephrasing.",
    )

    def available(self):
        import importlib.util
        if importlib.util.find_spec("nltk") is None:
            return False, "pip install nltk"
        return True, ""

    def evaluate(self, df, config=None):
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        out = []
        for resp, gt in _pairs(df):
            if not gt.strip() or not resp.strip():
                out.append(None)
                continue
            out.append(round(meteor_score([gt.split()], resp.split()), 4))
        return {"meteor": out}


# ---------------------------------------------------------------------------
# BERTScore  [optional heavy]
# ---------------------------------------------------------------------------

class BertScore(Evaluator):
    info = EvaluatorInfo(
        key="bertscore", name="BERTScore", category="quality",
        inputs=["response", "ground_truth"],
        description="Similarity computed on contextual embeddings - words "
                    "match by MEANING in context, not spelling. F1 0-1. "
                    "First run downloads a model (~260MB default).",
    )

    def available(self):
        import importlib.util
        if importlib.util.find_spec("bert_score") is None:
            return False, "pip install bert-score (large: includes torch)"
        return True, ""

    def evaluate(self, df, config=None):
        from bert_score import score as bs_score
        model_type = (config or {}).get("model_type", "distilbert-base-uncased")
        resps, gts, idx = [], [], []
        for i, (resp, gt) in enumerate(_pairs(df)):
            if gt.strip() and resp.strip():
                resps.append(resp); gts.append(gt); idx.append(i)
        out = [None] * len(df)
        if resps:
            _, _, f1 = bs_score(resps, gts, model_type=model_type, lang="en",
                                verbose=False)
            for i, v in zip(idx, f1.tolist()):
                out[i] = round(float(v), 4)
        return {"bertscore_f1": out}


# ---------------------------------------------------------------------------
# Embedding cosine similarity  [optional heavy]
# ---------------------------------------------------------------------------

class EmbeddingSimilarity(Evaluator):
    info = EvaluatorInfo(
        key="embedding_similarity", name="Embedding Cosine Similarity",
        category="quality", inputs=["response", "ground_truth"],
        description="Whole-sentence semantic closeness via sentence "
                    "embeddings (MiniLM, ~80MB download), 0-1-ish. The "
                    "'same meaning, different words' detector.",
    )

    def available(self):
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            return False, "pip install sentence-transformers (large)"
        return True, ""

    def evaluate(self, df, config=None):
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer(
            (config or {}).get("model", "all-MiniLM-L6-v2"))
        out = []
        for resp, gt in _pairs(df):
            if not gt.strip() or not resp.strip():
                out.append(None)
                continue
            e = model.encode([resp, gt])
            out.append(round(float(util.cos_sim(e[0], e[1])[0][0]), 4))
        return {"embedding_similarity": out}
