"""Quality, diversity, structure and operations metrics.

Six evaluators:
- MoverSimilarity [F2]: MoverScore-STYLE semantic distance to the
  reference over sentence embeddings with greedy matching. A documented
  adaptation (the original library is unmaintained and heavy), never
  claimed as the paper implementation.
- DiversitySuite [F0]: distinct-1/2 and type-token ratio per response,
  plus a self-similarity column (mean token-F1 of the row's response
  against a seeded sample of other rows - the Self-BLEU stand-in, named
  as such, same convention as the BLANC stand-in).
- IntentMatch [F0, local model]: the deterministic cousin of answer
  relevance - embedding similarity between request and response.
- MarkdownValidity [F0]: structure screen for markdown-shaped output:
  balanced code fences, well-formed links, sane headers. Live-URL
  checking stays OFF by default (network).
- LatencyCost [F0 + op columns]: per-row latency and cost readings when
  the log carries them, normalized numeric so they flow into segment
  parity (latency/cost parity is an escalation-parity cousin).
- CodeExecution [F2, SECURITY-SENSITIVE, opt-in only]: run generated
  code against per-row tests in a hard sandbox (subprocess, -I isolated
  mode, timeout, temp dir). Deliberately excluded from the recommended
  matrix; explicit opt-in from the all-metrics list only.
"""

from __future__ import annotations

import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


def _st_available():
    import importlib.util
    if importlib.util.find_spec("sentence_transformers") is None:
        return False, "pip install sentence-transformers (large)"
    return True, ""


_ST_MODEL = None


def _embed(texts: list[str]):
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL.encode(texts, normalize_embeddings=True)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip())
            if len(s.strip()) >= 3]


class MoverSimilarity(Evaluator):
    info = EvaluatorInfo(
        key="mover_similarity", name="Mover Similarity (adaptation)",
        category="quality", inputs=["response", "ground_truth"],
        description="MoverScore-STYLE semantic distance to the reference: "
                    "sentence embeddings greedily matched between response "
                    "and ground truth, matched similarities averaged. A "
                    "documented adaptation of the earth-mover idea, not "
                    "the paper implementation. Needs "
                    "sentence-transformers.",
    )

    def available(self):
        return _st_available()

    def evaluate(self, df, config=None):
        scores = []
        for resp, gt in zip(df["response"].astype("string").fillna(""),
                            df["ground_truth"].astype("string").fillna("")):
            r_s, g_s = _sentences(resp), _sentences(gt)
            if not r_s or not g_s:
                scores.append(None)
                continue
            embs = _embed(r_s + g_s)
            R, G = embs[:len(r_s)], embs[len(r_s):]
            sims = R @ G.T                      # cosine (normalized)
            # greedy one-to-one matching, response side
            import numpy as np
            sims = np.array(sims, dtype=float)
            matched = []
            work = sims.copy()
            for _ in range(min(work.shape)):
                i, j = divmod(int(work.argmax()), work.shape[1])
                matched.append(work[i, j])
                work[i, :] = -1; work[:, j] = -1
            scores.append(round(float(sum(matched) / len(matched)), 4))
        return {"mover_similarity": scores}


class DiversitySuite(Evaluator):
    info = EvaluatorInfo(
        key="diversity", name="Diversity Suite",
        category="quality", inputs=["response"],
        description="Lexical diversity per response: distinct-1 and "
                    "distinct-2 (unique n-gram share), type-token ratio, "
                    "and self_similarity (mean token overlap with a "
                    "seeded sample of OTHER rows' responses - high means "
                    "templated/generic output; the documented Self-BLEU "
                    "stand-in). Always available.",
    )

    @staticmethod
    def _toks(t: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", (t or "").lower())

    @staticmethod
    def _token_f1(a: str, b: str) -> float:
        from vedaeval.evaluators.reference import ExactMatch
        return ExactMatch._token_f1(a, b)

    def evaluate(self, df, config=None):
        import random
        rng = random.Random(7)
        texts = df["response"].astype("string").fillna("").tolist()
        d1, d2, ttr, selfsim = [], [], [], []
        n = len(texts)
        for i, t in enumerate(texts):
            toks = self._toks(t)
            if not toks:
                d1.append(None); d2.append(None); ttr.append(None)
                selfsim.append(None)
                continue
            d1.append(round(len(set(toks)) / len(toks), 4))
            bg = list(zip(toks, toks[1:]))
            d2.append(round(len(set(bg)) / len(bg), 4) if bg else None)
            ttr.append(round(len(set(toks)) / len(toks), 4))
            others = [j for j in range(n) if j != i and texts[j].strip()]
            if not others:
                selfsim.append(None)
                continue
            sample = rng.sample(others, min(5, len(others)))
            sims = [self._token_f1(t, texts[j]) for j in sample]
            selfsim.append(round(sum(sims) / len(sims), 4))
        return {"distinct_1": d1, "distinct_2": d2,
                "type_token_ratio": ttr, "self_similarity": selfsim}


class IntentMatch(Evaluator):
    info = EvaluatorInfo(
        key="intent_match", name="Intent Match",
        category="quality", inputs=["request", "response"],
        description="The deterministic cousin of answer relevance: "
                    "embedding similarity between the request and the "
                    "response. Low = the answer is off-intent, whatever "
                    "its fluency. Flag below the dial (default 0.3). "
                    "Needs sentence-transformers, no key.",
    )

    def available(self):
        return _st_available()

    def evaluate(self, df, config=None):
        thr = float((config or {}).get("intent_threshold", 0.3))
        reqs = df["request"].astype("string").fillna("").tolist()
        resps = df["response"].astype("string").fillna("").tolist()
        scores, flags = [], []
        pairs = [(q, r) for q, r in zip(reqs, resps)]
        texts = [t for p in pairs for t in p]
        embs = _embed(texts)
        for k, (q, r) in enumerate(pairs):
            if not q.strip() or not r.strip():
                scores.append(None); flags.append(None)
                continue
            sim = float(embs[2 * k] @ embs[2 * k + 1])
            scores.append(round(sim, 4))
            flags.append(bool(sim < thr))
        return {"intent_match": scores, "intent_mismatch_flag": flags}


class MarkdownValidity(Evaluator):
    info = EvaluatorInfo(
        key="markdown_validity", name="Markdown / Link Validity",
        category="validation", inputs=["response"],
        description="Structure screen for markdown-shaped responses: "
                    "balanced code fences, well-formed `[text](url)` links "
                    "(scheme + host for web links), sane headers. None "
                    "when the response contains no markdown constructs. "
                    "Live URL checking stays off (no network). Always "
                    "available.",
    )

    _LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")

    def evaluate(self, df, config=None):
        from urllib.parse import urlparse
        valid, issues = [], []
        for text in df["response"].astype("string").fillna(""):
            has_md = ("```" in text or self._LINK.search(text)
                      or re.search(r"^#{1,6}\s", text, re.MULTILINE)
                      or "http://" in text or "https://" in text)
            if not has_md:
                valid.append(None); issues.append("")
                continue
            probs = []
            if text.count("```") % 2 != 0:
                probs.append("unbalanced code fence")
            for m in self._LINK.finditer(text):
                label, url = m.group(1), m.group(2).strip()
                if not url:
                    probs.append("empty link target")
                elif url.startswith(("http://", "https://")):
                    u = urlparse(url)
                    if not u.netloc or " " in url:
                        probs.append(f"malformed url: {url[:40]}")
            for m in re.finditer(r"^(#{1,6})([^#\s])", text, re.MULTILINE):
                probs.append("header missing space after #")
            for m in re.finditer(r"https?://\S+", text):
                u = urlparse(m.group(0).rstrip(".,)"))
                if not u.netloc:
                    probs.append(f"malformed bare url: {m.group(0)[:40]}")
            valid.append(not probs)
            issues.append("; ".join(dict.fromkeys(probs))[:200])
        return {"markdown_valid": valid, "markdown_issues": issues}


class LatencyCost(Evaluator):
    info = EvaluatorInfo(
        key="latency_cost", name="Latency / Cost Readings",
        category="text_stats", inputs=["response"],
        description="Per-row latency and cost when the log carries them "
                    "(columns: latency_ms/latency/response_time_ms and "
                    "cost/cost_usd). Normalized numeric so they flow into "
                    "segment parity - latency parity across groups is an "
                    "escalation-parity cousin. None when the log has no "
                    "operational columns.",
    )

    _LAT = ("latency_ms", "latency", "response_time_ms", "response_time")
    _COST = ("cost_usd", "cost", "price_usd")

    def evaluate(self, df, config=None):
        import pandas as pd
        lat_col = next((c for c in self._LAT if c in df.columns), None)
        cost_col = next((c for c in self._COST if c in df.columns), None)
        n = len(df)
        def numeric(col):
            if col is None:
                return [None] * n
            s = pd.to_numeric(df[col], errors="coerce")
            return [round(float(v), 4) if pd.notna(v) else None for v in s]
        return {"latency_ms": numeric(lat_col),
                "cost_usd": numeric(cost_col)}


class CodeExecution(Evaluator):
    """SECURITY-SENSITIVE: sandboxed, opt-in only, never in
    the recommended set. The sandbox: a fresh subprocess with python -I
    (isolated mode), a hard timeout, a temp working directory, and a
    minimal environment. This bounds, but cannot perfectly eliminate,
    what executed code can do - which is why the metric is opt-in and
    the block says so plainly."""

    info = EvaluatorInfo(
        key="code_execution", name="Code Execution pass@1 (opt-in)",
        category="validation", inputs=["response", "ground_truth"],
        description="Runs the response as Python code followed by the "
                    "ground-truth test snippet in a sandboxed subprocess "
                    "(isolated mode, 5s timeout, temp dir, minimal env). "
                    "pass@1 per row. SECURITY-SENSITIVE: opt-in only, "
                    "never pre-ticked; run only on code you are prepared "
                    "to execute.",
    )

    TIMEOUT = 5

    def evaluate(self, df, config=None):
        import subprocess, sys, tempfile, os
        timeout = int((config or {}).get("timeout", self.TIMEOUT))
        passed, errors = [], []
        for code, tests in zip(df["response"].astype("string").fillna(""),
                               df["ground_truth"].astype("string").fillna("")):
            if not code.strip() or not tests.strip():
                passed.append(None); errors.append("")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "candidate.py")
                with open(path, "w") as f:
                    f.write(code + "\n\n" + tests + "\n")
                try:
                    proc = subprocess.run(
                        [sys.executable, "-I", path],
                        capture_output=True, text=True, timeout=timeout,
                        cwd=tmp, env={"PATH": ""})
                    ok = proc.returncode == 0
                    passed.append(bool(ok))
                    errors.append("" if ok else
                                  (proc.stderr or "nonzero exit")[-200:])
                except subprocess.TimeoutExpired:
                    passed.append(False); errors.append("timeout")
                except Exception as exc:
                    passed.append(None)
                    errors.append(f"sandbox error: {type(exc).__name__}")
        return {"code_pass": passed, "code_error": errors}
