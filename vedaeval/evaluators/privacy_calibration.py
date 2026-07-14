"""Privacy + calibration metrics.

Two implementations; three documented deferrals (perplexity - needs
token logprobs closed APIs rarely expose; canary leakage - needs
planted canaries in a training corpus we don't have; perturbation
robustness runner - the counterfactual Mode A machinery already covers
the pattern, a dedicated typo-perturbation runner is a later add).
"""

from __future__ import annotations

import re

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo


class VerbatimCopyRate(Evaluator):
    info = EvaluatorInfo(
        key="verbatim_copy", name="Verbatim Copy Rate", category="quality",
        inputs=["response", "context"],
        description="How much of the response is copied word-for-word from "
                    "the context: share of response tokens inside copied "
                    "runs of 5+ consecutive words. High = extractive "
                    "(sometimes fine, sometimes lazy); also the offline "
                    "cousin of regurgitation checks.",
    )

    def evaluate(self, df, config=None):
        n = int((config or {}).get("min_run", 5))
        out = []
        for resp, ctx in zip(df["response"].astype("string").fillna(""),
                             df["context"].astype("string").fillna("")):
            r_toks = re.findall(r"[a-z0-9]+", resp.lower())
            c_text = " ".join(re.findall(r"[a-z0-9]+", ctx.lower()))
            if not r_toks or not c_text:
                out.append(None)
                continue
            copied = [False] * len(r_toks)
            for i in range(len(r_toks) - n + 1):
                if " ".join(r_toks[i:i + n]) in c_text:
                    for j in range(i, i + n):
                        copied[j] = True
            out.append(round(sum(copied) / len(r_toks), 4))
        return {"verbatim_copy_rate": out}


class Calibration(Evaluator):
    info = EvaluatorInfo(
        key="calibration", name="Calibration (ECE)", category="quality",
        inputs=["response", "ground_truth"],
        description="Does stated confidence match actual correctness? Uses "
                    "a 'confidence' column (0-1) if present, else extracts "
                    "verbalized confidence from the response ('90% sure'). "
                    "Outputs per-row confidence + correctness; the Expected "
                    "Calibration Error appears in the run summary via the "
                    "calibration_gap column.",
    )

    _PCT = re.compile(r"(\d{1,3})\s*(?:%|percent)", re.IGNORECASE)
    _WORDS = {"certain": 0.95, "confident": 0.85, "likely": 0.7,
              "probably": 0.7, "possibly": 0.4, "unsure": 0.3,
              "not sure": 0.3, "uncertain": 0.3}

    def _confidence(self, df, idx):
        if "confidence" in df.columns:
            try:
                v = float(df["confidence"].loc[idx])
                return v / 100 if v > 1 else v
            except Exception:
                pass
        text = str(df["response"].loc[idx]).lower()
        m = self._PCT.search(text)
        if m:
            return min(int(m.group(1)), 100) / 100
        for w, v in self._WORDS.items():
            if w in text:
                return v
        return None

    @staticmethod
    def _correct(resp: str, gt: str) -> float:
        from vedaeval.evaluators.reference import ExactMatch
        return 1.0 if ExactMatch._token_f1(resp, gt) >= 0.5 else 0.0

    def evaluate(self, df, config=None):
        confs, corrs, gaps = [], [], []
        for idx in df.index:
            gt = str(df["ground_truth"].loc[idx]) if "ground_truth" in df.columns else ""
            resp = str(df["response"].loc[idx])
            conf = self._confidence(df, idx)
            if conf is None or not gt.strip():
                confs.append(None); corrs.append(None); gaps.append(None)
                continue
            corr = self._correct(resp, gt)
            confs.append(round(conf, 4))
            corrs.append(corr)
            gaps.append(round(abs(conf - corr), 4))
        return {"stated_confidence": confs, "answer_correct": corrs,
                "calibration_gap": gaps}
