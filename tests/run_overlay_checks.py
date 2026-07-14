"""Dependency-free checks for the task-aware selection overlay.

Run from the VedaEval folder:  python tests/run_overlay_checks.py

Covers: the tier rule, Table A wiring, admissibility (Table B), the
feasibility gap logic (a RAG dataset without a context column must
name "context capture"), maturity coverage, and the registry drift
alarm. No pandas, no optional dependency needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedaeval import overlay
from vedaeval.evaluators import REGISTRY

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


print("1) Tier rule (Layer 1): five triggers, three tiers")
t = overlay.governance_tier
# order: member_facing, decision_influence, regulated_data,
#        automation_at_scale, human_oversight
check("internal, low impact, human in loop -> Tier 0",
      t(False, False, False, False, True) == 0)
check("member-facing, low impact -> Tier 1",
      t(True, False, False, False, True) == 1)
check("regulated data alone -> Tier 2",
      t(False, False, True, False, True) == 2)
check("decision influence alone -> Tier 2",
      t(False, True, False, False, True) == 2)
check("automation at scale alone -> Tier 2",
      t(False, False, False, True, True) == 2)
check("no effective human oversight -> Tier 2",
      t(False, False, False, False, False) == 2)
check("member-facing AND high impact stays Tier 2 (no double count)",
      t(True, True, True, True, False) == 2)

print("2) Table A: tier -> required dimensions")
check("Tier 0 = performance + safety",
      overlay.TIER_DIMENSIONS[0] == ["performance", "safety"])
check("Tier 1 adds bias and privacy",
      overlay.TIER_DIMENSIONS[1] == ["performance", "safety",
                                     "bias_fairness", "privacy"])
check("Tier 2 = all five",
      overlay.TIER_DIMENSIONS[2] == list(overlay.DIMENSIONS))

print("3) Registry drift alarm: every encoded key exists in REGISTRY")
encoded = set(overlay.ALWAYS_ON)
for task, dims in overlay.TASK_MATRIX.items():
    for keys in dims.values():
        encoded.update(keys)
unknown = sorted(k for k in encoded if k not in REGISTRY)
check("no unknown keys in Table B / Table C", not unknown, str(unknown))
unknown2 = sorted(k for k in list(overlay.FEASIBILITY) + list(overlay.MATURITY)
                  if k not in REGISTRY)
check("no unknown keys in feasibility / maturity tables", not unknown2,
      str(unknown2))
check("every matrix key has a feasibility label",
      all(k in overlay.FEASIBILITY for k in encoded))
check("every matrix key has a maturity label",
      all(k in overlay.MATURITY for k in encoded))
check("maturity levels are only mature/emerging/experimental",
      set(overlay.MATURITY.values()) <= {"mature", "emerging", "experimental"})

print("4) Admissibility: recommend() returns only the task's row")
cols_full = ["request", "response", "ground_truth", "context"]
for task in overlay.TASKS:
    rec = overlay.recommend(2, task, cols_full, judge_key=True)
    admissible = overlay.admissible_keys(task)
    returned = {m.key for d in rec.dimensions for m in d.metrics}
    stray = returned - admissible
    check(f"{task}: no inadmissible metric returned", not stray, str(stray))
rec_cqa = overlay.recommend(2, "closed_qa", cols_full, judge_key=True)
ret_cqa = {m.key for d in rec_cqa.dimensions for m in d.metrics}
check("closed_qa: faithfulness is inadmissible and absent",
      "faithfulness" not in ret_cqa)
check("closed_qa: exact_match present and runnable",
      any(m.key == "exact_match" and m.runnable
          for d in rec_cqa.dimensions for m in d.metrics))

print("5) Feasibility gaps: RAG dataset WITHOUT a context column")
rec_gap = overlay.recommend(2, "rag", ["request", "response", "ground_truth"])
by_key = {m.key: m for d in rec_gap.dimensions for m in d.metrics}
for k in ("faithfulness", "summac", "context_precision", "verbatim_copy",
          "unsafe_source_utilization", "plan_grounded"):
    m = by_key[k]
    check(f"{k} is a documented gap naming 'context capture'",
          (not m.runnable) and "context capture" in m.missing,
          f"runnable={m.runnable} missing={m.missing}")
check("gap skip reason reuses engine vocabulary",
      by_key["faithfulness"].skip_reason
      == "not applicable (missing columns: context)")
check("judge metrics without a key name 'judge key'",
      "judge key" in by_key["answer_relevance"].missing)
check("judge skip reason reuses engine vocabulary",
      by_key["answer_relevance"].skip_reason
      == "needs an LLM API key (none provided)")
check("F0 metric still runnable without context",
      by_key["safety"].runnable)
check("documented_gaps() lists every non-runnable required metric",
      {m.key for _, m in rec_gap.documented_gaps()}
      == {k for k, m in by_key.items() if not m.runnable})

print("6) Feasibility gaps: RAG dataset WITH context, judge key present")
rec_ok = overlay.recommend(2, "rag",
                           ["request", "response", "ground_truth", "context"],
                           judge_key=True)
ok_keys = set(rec_ok.runnable_keys())
for k in ("faithfulness", "answer_relevance", "context_relevance",
          "retrieval_hit_rate"):
    check(f"{k} runnable once enabled", k in ok_keys)
check("sample_consistency still gapped (needs response_samples column)",
      "sample_consistency" not in ok_keys)

print("7) Task reconciliation: legacy dropdown values -> canonical tasks")
ctask = overlay.canonical_task
check("qa + rag toggle -> rag", ctask("qa", rag=True) == "rag")
check("qa + ground truth -> closed_qa",
      ctask("qa", rag=False, has_ground_truth=True) == "closed_qa")
check("qa without ground truth -> open_qa",
      ctask("qa", rag=False, has_ground_truth=False) == "open_qa")
check("structured_output -> extraction",
      ctask("structured_output") == "extraction")
check("text_to_sql -> extraction", ctask("text_to_sql") == "extraction")
check("canonical values pass through",
      all(ctask(k) == k for k in overlay.TASKS))

print("8) Always-on layer (Table C)")
check("always-on = the five descriptive metrics",
      overlay.ALWAYS_ON == ["textstat", "token_count", "sentiment",
                            "language_detection", "topic_classification"])
check("always-on metrics all F0",
      all(overlay.FEASIBILITY[k] == "F0" for k in overlay.ALWAYS_ON))

print("9) Informational: transcribed F-labels vs declared inputs")
for line in overlay.feasibility_input_mismatches():
    print(f"  note  {line}")
print("  (label mismatches are matrix-revision candidates; runnability "
      "always follows the declared inputs, so none can cause a wrong run)")

print()
print(f"{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
