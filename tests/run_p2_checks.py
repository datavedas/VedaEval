"""Phase 2 wiring checks (dependency-free)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import pandas as pd

from vedaeval.schema import auto_map_columns, apply_mapping
from vedaeval.validation import validate_dataset
from vedaeval.engine import run_evaluation
from vedaeval.evaluators import REGISTRY, recommended_for

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok   {name}")
    else: FAIL += 1; print(f"  FAIL {name}")

csv = pathlib.Path(__file__).parent.parent / "sample_data" / "qa_rag_demo.csv"
raw = pd.read_csv(csv)
df = apply_mapping(raw, auto_map_columns(list(raw.columns)))

check("new evaluators registered",
      {"safety", "faithfulness", "answer_relevance", "context_relevance",
       "coherence", "conciseness"} <= set(REGISTRY))
check("judges marked needs_llm",
      all(REGISTRY[k].info.needs_llm for k in
          ("answer_relevance", "context_relevance", "coherence", "conciseness")))

r = run_evaluation(df, ["answer_relevance"], configs={})
check("judge skipped without key",
      "api key" in r.skipped.get("answer_relevance", "").lower())

r2 = run_evaluation(df, ["safety", "faithfulness"])
for k in ("safety", "faithfulness"):
    reason = r2.skipped.get(k, "RAN")
    check(f"{k} ran or cleanly unavailable",
          reason == "RAN" or reason.startswith("unavailable"))

skew = df.copy(); skew["Bias Gender"] = "M"
checks2 = {i.check for i in validate_dataset(skew, pii_engine="off").issues}
check("diversity check fires", any(c.startswith("diversity_") for c in checks2))
checks3 = {i.check for i in validate_dataset(df, pii_engine="off").issues}
check("segment size check fires on 20-row demo",
      any(c.startswith("segment_size_") for c in checks3))

ldf = pd.DataFrame({"Question": ["q"], "Answer": ["a"],
                    "Documents": [["passage one", "passage two"]]})
lm = apply_mapping(ldf, auto_map_columns(list(ldf.columns)))
check("list context joined", lm["context"].iloc[0] == "passage one\n\npassage two")

rec = recommended_for("qa", "rai", True)
check("recommendations include new metrics",
      {"safety", "faithfulness", "answer_relevance", "context_relevance"} <= set(rec))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
