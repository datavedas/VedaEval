"""Dependency-free smoke test for VedaEval core (LLM evaluation).

Runs without pytest and without any optional evaluator dependency.
Covers: schema mapping, validation checks, engine skip-don't-crash
behavior, and Phase 2 wiring. The classic-ML add-on has its OWN checks
in mlobs/run_ml_checks.py - nothing here imports it, by design.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd

from vedaeval.schema import auto_map_columns, apply_mapping, validate_required
from vedaeval.validation import validate_dataset
from vedaeval.engine import run_evaluation
from vedaeval.evaluators import REGISTRY, recommended_for

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


print("== schema ==")
raw = pd.read_csv(pathlib.Path(__file__).parent.parent / "sample_data" / "qa_rag_demo.csv")
mapping = auto_map_columns(list(raw.columns))
check("aliases detected", mapping["request"] == "Question" and mapping["response"] == "Answer"
      and mapping["ground_truth"] == "Expected" and mapping["context"] == "Documents")
check("required satisfied", validate_required(mapping) == [])
df = apply_mapping(raw, mapping)
check("metadata kept", "Bias Gender" in df.columns and "Bias Age" in df.columns)

print("== validation ==")
report = validate_dataset(df, pii_engine="regex")
checks = {i.check for i in report.issues}
check("exact duplicate found", "exact_duplicates" in checks)
check("conflicting duplicate found", "conflicting_duplicates" in checks)
check("rag leakage found", "rag_leakage" in checks)
check("pii email found", "pii_email_address" in checks)
check("size warning", "dataset_size" in checks)
check("flagged rows valid", report.flagged_rows and max(report.flagged_rows) < len(df))

print("== engine ==")
result = run_evaluation(df, ["token_count", "banned_keywords", "regex_match",
                             "json_validation", "textstat", "sentiment", "overlap",
                             "profanity", "does_not_exist"],
                        configs={"banned_keywords": {"keywords": ["guarantee"]},
                                 "regex_match": {"pattern": r"\d+ (dollars|USD)"}})
check("always-available metrics ran",
      {"token_count", "banned_keywords", "regex_match"} <= set(result.ran))
check("unknown evaluator skipped cleanly",
      result.skipped.get("does_not_exist") == "unknown evaluator")
check("no evaluator crashed",
      all("failed" not in r for r in result.skipped.values()))
check("token counts positive", result.scores["token_count_response"].gt(0).all())
check("regex matched copay rows", (result.scores["regex_match"] == "Match").sum() >= 3)

print("== phase 2 wiring ==")
check("new evaluators registered",
      {"safety", "faithfulness", "answer_relevance", "context_relevance",
       "coherence", "conciseness"} <= set(REGISTRY))
check("judges marked needs_llm",
      all(REGISTRY[k].info.needs_llm for k in
          ("answer_relevance", "context_relevance", "coherence", "conciseness")))
r2 = run_evaluation(df, ["answer_relevance"], configs={})
check("judge skipped without key",
      "api key" in r2.skipped.get("answer_relevance", "").lower())
r3 = run_evaluation(df, ["safety", "faithfulness"])
for k in ("safety", "faithfulness"):
    reason = r3.skipped.get(k, "RAN")
    check(f"{k} ran or cleanly unavailable",
          reason == "RAN" or reason.startswith("unavailable"))
skew = df.copy()
skew["Bias Gender"] = "M"
checks2 = {i.check for i in validate_dataset(skew, pii_engine="off").issues}
check("diversity check fires", any(c.startswith("diversity_") for c in checks2))
checks3 = {i.check for i in validate_dataset(df, pii_engine="off").issues}
check("segment size check fires", any(c.startswith("segment_size_") for c in checks3))
ldf = pd.DataFrame({"Question": ["q"], "Answer": ["a"],
                    "Documents": [["passage one", "passage two"]]})
lm = apply_mapping(ldf, auto_map_columns(list(ldf.columns)))
check("list context joined", lm["context"].iloc[0] == "passage one\n\npassage two")
check("recommendation logic", "overlap" in recommended_for("qa", "rai", True))

print("== isolation ==")
import vedaeval.schema, vedaeval.validation, vedaeval.engine, vedaeval.evaluators
core_srcs = []
for mod in ("schema", "validation", "engine"):
    core_srcs.append((pathlib.Path(__file__).parent.parent / "vedaeval" / f"{mod}.py").read_text(encoding="utf-8"))
check("core package never imports mlobs",
      all("mlobs" not in src for src in core_srcs))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
