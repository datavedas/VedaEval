"""Task-aware metric selection overlay (the governance framework, as code).

This module encodes the framework as DATA plus PURE FUNCTIONS - no UI,
no side effects. The chain it implements, one direction only:

    risk context  -> governance tier        (five intake triggers)
    tier          -> required dimensions    (Table A)
    task x dimension -> admissible metrics  (Table B, registry keys)
    metric        -> feasibility F0-F3      (data dependency ladder)
    metric        -> maturity               (mature / emerging / experimental)

recommend() joins those layers with the dataset's actual columns.
Every admissible metric for a required dimension comes back either
RUNNABLE today or as a DOCUMENTED GAP with its enablement need named
(context capture, ground truth, judge key). Nothing is silently
dropped, and the engine's skip-don't-crash reason vocabulary is
reused so the same words appear at selection time and at run time.

Two moves are forbidden by design and have no code path here:
a task never determines a tier, and a tier never selects a metric
directly.

The mapping tables are VERSIONED (see MATRIX_VERSION) and transcribed
from the framework's implementation matrix v1.0. They change only
with a versioned matrix revision. Every metric key
is verified against the evaluator registry at import time; an unknown
key raises immediately (catches matrix-vs-registry drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vedaeval.evaluators import REGISTRY

MATRIX_VERSION = "1.2"
# 1.1: five feasibility-label corrections.
# 1.2: safety / RAG second-wave / quality-ops metrics wired (15 new) +
#      response_consistency wired, fixing its orphan status.
#      code_execution stays OUT of the matrix by design
#      (security opt-in only, reachable from the all-metrics list).

# ------------------------------------------------------------------
# Layer 1: risk context -> governance tier
# ------------------------------------------------------------------

# The five intake triggers, in presentation order. These are questions
# about the USE CASE and its impact context, never about the model or
# the technology stack.
TRIGGERS: list[tuple[str, str]] = [
    ("member_facing",
     "Member-facing: do people outside the operating team see its outputs?"),
    ("decision_influence",
     "Decision influence: do outputs influence or automate decisions "
     "about individuals?"),
    ("regulated_data",
     "Regulated data: does regulated or high-sensitivity data (for example "
     "PHI or PFI) enter the workflow?"),
    ("automation_at_scale",
     "Automation at scale: does it operate autonomously at scale?"),
    ("human_oversight",
     "Human oversight: is a human effectively in the loop before outputs "
     "take effect?"),
]

TIER_NAMES = {0: "Tier 0 (low risk)", 1: "Tier 1 (moderate risk)",
              2: "Tier 2 (high risk)"}

TIER_CHARACTER = {0: "Qualitative, sampled",
                  1: "Structured, periodic",
                  2: "Continuous, thresholds and alerts"}


def governance_tier(member_facing: bool, decision_influence: bool,
                    regulated_data: bool, automation_at_scale: bool,
                    human_oversight: bool) -> int:
    """Classify a use case into governance Tier 0, 1 or 2.

    High impact = outputs influence or automate decisions about
    individuals, OR regulated data enters the workflow, OR the system
    operates autonomously at scale, OR no human is effectively in the
    loop. High impact is Tier 2 regardless of audience. Otherwise a
    member-facing use case is Tier 1 and an internal one is Tier 0.

    Note: a sixth trigger operates at RUNTIME rather than at intake -
    monitoring evidence of material disparities escalates the tier
    after deployment. That escalation is a monitoring decision, so it
    is documented here but not computed here.
    """
    impact_high = (decision_influence or regulated_data
                   or automation_at_scale or not human_oversight)
    if impact_high:
        return 2
    return 1 if member_facing else 0


# ------------------------------------------------------------------
# Layer 2: tier -> required dimensions (Table A)
# ------------------------------------------------------------------

DIMENSIONS = {
    "performance": "Performance (incl. faithfulness)",
    "safety": "Safety",
    "bias_fairness": "Bias & Fairness",
    "privacy": "Privacy & Compliance",
    "robustness": "Robustness & Control",
}

TIER_DIMENSIONS: dict[int, list[str]] = {
    0: ["performance", "safety"],
    1: ["performance", "safety", "bias_fairness", "privacy"],
    2: ["performance", "safety", "bias_fairness", "privacy", "robustness"],
}

# Table A qualifiers, shown next to the dimension at that tier.
TIER_DIMENSION_NOTES: dict[tuple[int, str], str] = {
    (0, "performance"): "spot checks",
    (0, "safety"): "basic checks",
    (1, "performance"): "incl. faithfulness where context-based",
    (1, "privacy"): "where sensitive data is present",
    (2, "robustness"): "continuous",
}

# ------------------------------------------------------------------
# The seven canonical tasks, and the mapping from the app's legacy
# task_type values. The legacy dropdown values keep working; the
# overlay translates instead of renaming.
# ------------------------------------------------------------------

TASKS = {
    "closed_qa": "Closed QA (one known correct answer)",
    "open_qa": "Open QA (no single reference answer)",
    "summarization": "Summarization",
    "rag": "RAG (answers grounded in retrieved documents)",
    "extraction": "Information extraction",
    "classification": "Classification / intent",
    "chat_agentic": "Chat & agentic (multi-turn)",
}

# Legacy Step 3 values -> canonical task. "qa" is ambiguous on its own,
# so canonical_task() resolves it with the RAG toggle and ground truth.
TASK_ALIASES = {
    "qa": "closed_qa",
    "summarization": "summarization",
    "structured_output": "extraction",
    "text_to_sql": "extraction",
}


def canonical_task(task_type: str, rag: bool = False,
                   has_ground_truth: bool = True) -> str:
    """Resolve any known task value (canonical or legacy) to canonical."""
    if task_type in TASKS:
        return task_type
    if task_type == "qa":
        if rag:
            return "rag"
        return "closed_qa" if has_ground_truth else "open_qa"
    return TASK_ALIASES.get(task_type, "open_qa")


# ------------------------------------------------------------------
# Layer 3: task x dimension -> admissible metrics (Table B)
#
# Cell entries are REGISTRY KEYS only. A metric absent from a task's
# cell is INADMISSIBLE for that task, not merely unrecommended.
# Report-level features (Fairness tab, validation scans, classic-ML
# observability) are not registry evaluators; they live in
# REPORT_FEATURES below so they can still be surfaced as guidance.
# ------------------------------------------------------------------

TASK_MATRIX: dict[str, dict[str, list[str]]] = {
    "closed_qa": {
        "performance": ["exact_match", "levenshtein", "answer_correctness",
                        "chrf_ter", "calibration", "mover_similarity",
                        "intent_match"],
        "safety": ["safety", "profanity", "banned_keywords", "refusal",
                   "deflection", "jailbreak_detection", "harm_taxonomy",
                   "refusal_correctness", "moderation_screen"],
        "bias_fairness": [],
        "privacy": ["phi_echo", "phi_entities"],
        "robustness": ["sample_consistency", "regex_match",
                       "response_consistency", "latency_cost"],
    },
    "open_qa": {
        "performance": ["answer_relevance", "completeness", "helpfulness",
                        "coherence", "conciseness", "geval", "meteor",
                        "bertscore", "embedding_similarity", "calibration",
                        "mover_similarity", "intent_match", "diversity"],
        "safety": ["safety", "profanity", "banned_keywords", "refusal",
                   "deflection", "jailbreak_detection", "harm_taxonomy",
                   "refusal_correctness", "moderation_screen"],
        "bias_fairness": [],
        "privacy": ["phi_echo", "phi_entities"],
        "robustness": ["sample_consistency", "instruction_adherence",
                       "response_consistency", "markdown_validity",
                       "latency_cost"],
    },
    "summarization": {
        "performance": ["overlap", "meteor", "bertscore", "summary_stats",
                        "faithfulness", "summac", "qag_support",
                        "faithfulness_judge", "mover_similarity",
                        "diversity"],
        "safety": ["safety", "toxicity_preservation", "profanity",
                   "banned_keywords", "harm_taxonomy", "moderation_screen"],
        "bias_fairness": [],
        "privacy": ["phi_echo", "verbatim_copy", "phi_entities"],
        "robustness": ["sample_consistency", "textstat",
                       "markdown_validity", "latency_cost"],
    },
    "rag": {
        "performance": ["faithfulness", "summac", "qag_support",
                        "retrieval_hit_rate", "context_entity_recall",
                        "context_recall", "context_precision",
                        "answer_correctness", "answer_relevance",
                        "context_relevance", "plan_grounded",
                        "faithfulness_judge", "noise_sensitivity",
                        "citation_precision", "citation_recall",
                        "intent_match"],
        "safety": ["safety", "unsafe_source_utilization", "refusal",
                   "deflection", "jailbreak_detection", "harm_taxonomy",
                   "refusal_correctness", "moderation_screen"],
        "bias_fairness": [],
        "privacy": ["phi_echo", "verbatim_copy", "phi_entities"],
        "robustness": ["sample_consistency", "instruction_adherence",
                       "response_consistency", "latency_cost"],
    },
    "extraction": {
        "performance": ["extraction", "json_validation", "sql_validation"],
        "safety": ["safety", "banned_keywords", "jailbreak_detection"],
        "bias_fairness": [],
        "privacy": ["phi_echo", "phi_entities"],
        "robustness": ["json_validation", "sql_validation", "regex_match",
                       "latency_cost"],
    },
    "classification": {
        "performance": ["exact_match", "topic_classification"],
        "safety": ["safety"],
        "bias_fairness": [],
        "privacy": ["phi_entities"],
        "robustness": ["latency_cost"],
    },
    "chat_agentic": {
        "performance": ["role_adherence", "conversation_relevancy",
                        "knowledge_retention", "helpfulness", "coherence",
                        "pairwise", "geval", "intent_match", "diversity"],
        "safety": ["safety", "profanity", "refusal", "deflection",
                   "banned_keywords", "jailbreak_detection",
                   "harm_taxonomy", "refusal_correctness",
                   "moderation_screen"],
        "bias_fairness": [],
        "privacy": ["phi_echo", "phi_entities"],
        "robustness": ["instruction_adherence", "language_detection",
                       "response_consistency", "markdown_validity",
                       "latency_cost"],
    },
}
# Deliberate matrix exclusion: code_execution is
# SECURITY-SENSITIVE (it runs generated code) and must never be
# pre-ticked. It stays out of TASK_MATRIX and is reachable only from
# the all-metrics list, with sandbox + opt-in documented in its block.

# Report-level features per task x dimension. These are Fairness-tab
# reports, validation scans or classic-ML observability, not Step 3
# checkboxes. needs_segment: unlocked by a segment column in the data.
# needs_context: unlocked by a context column. planned: approved
# roadmap, not yet built.
REPORT_FEATURES: dict[str, dict[str, list[dict]]] = {
    "closed_qa": {
        "bias_fairness": [
            {"name": "Segment parity over score columns", "needs_segment": True},
            {"name": "Refusal + deflection parity", "needs_segment": True},
            {"name": "Counterfactual testing (Fairness tab)"},
            {"name": "Benchmark battery (Fairness tab)"},
        ],
        "privacy": [{"name": "PII scan (validation layer)"}],
    },
    "open_qa": {
        "bias_fairness": [
            {"name": "Segment parity over score columns", "needs_segment": True},
            {"name": "Counterfactual testing (Fairness tab)"},
            {"name": "Benchmark battery (Fairness tab)"},
        ],
        "privacy": [{"name": "PII scan (validation layer)"}],
    },
    "summarization": {
        "bias_fairness": [
            {"name": "Segment parity over score columns", "needs_segment": True},
            {"name": "Comprehension-burden (readability) parity",
             "needs_segment": True},
            {"name": "Counterfactual testing (Fairness tab)"},
        ],
        "privacy": [{"name": "PII scan (validation layer)"}],
    },
    "rag": {
        "safety": [
            {"name": "Source bias amplification (Fairness tab)",
             "needs_context": True},
        ],
        "bias_fairness": [
            {"name": "Segment parity over score columns", "needs_segment": True},
            {"name": "Retrieval fairness score (Fairness tab)",
             "needs_segment": True, "needs_context": True},
            {"name": "Counterfactual testing (Fairness tab)"},
            {"name": "Benchmark battery (Fairness tab)"},
        ],
        "privacy": [{"name": "PII scan (validation layer)"}],
    },
    "extraction": {
        "bias_fairness": [
            {"name": "Segment parity over extraction score columns",
             "needs_segment": True},
        ],
        "privacy": [{"name": "PII scan (validation layer)"}],
    },
    "classification": {
        "performance": [
            {"name": "Classic-ML classification metrics (ML branch, "
                     "needs predictions and actuals)"},
        ],
        "bias_fairness": [
            {"name": "Classic-ML fairness suite: DP / DI / EO / AOD + "
                     "four-fifths rule (ML branch)", "needs_segment": True},
            {"name": "Segment parity over score columns", "needs_segment": True},
        ],
        "privacy": [{"name": "PII scan (validation layer)"}],
        "robustness": [
            {"name": "PSI / KS drift (ML branch, needs a baseline window)"},
        ],
    },
    "chat_agentic": {
        "bias_fairness": [
            {"name": "Segment parity over score columns", "needs_segment": True},
            {"name": "Escalation / deflection parity", "needs_segment": True},
            {"name": "Counterfactual testing (Fairness tab)"},
            {"name": "Benchmark battery (Fairness tab)"},
        ],
        "privacy": [{"name": "PII scan (validation layer)"}],
        "robustness": [
            {"name": "Consistency parity (segment parity over "
                     "response_consistency)", "needs_segment": True},
        ],
    },
}

# Table C: the always-on descriptive layer. Any task, any tier. These
# inform reading and segment slicing and carry no governance weight.
ALWAYS_ON = ["textstat", "token_count", "sentiment", "language_detection",
             "topic_classification"]

# ------------------------------------------------------------------
# Layer 4: metric -> feasibility (the F0-F3 data dependency ladder)
#
# F0 = response only, runs on any log. F1 = needs retrieved context.
# F2 = needs ground truth. F3 = needs a governed LLM judge key.
# Combined levels mean the metric needs both enablements.
# These labels are transcribed from the matrix; runnability itself is
# computed from each evaluator's declared inputs, so the two can be
# cross-checked (see feasibility_input_mismatches).
# ------------------------------------------------------------------

FEASIBILITY: dict[str, str] = {
    # F0: response only
    "safety": "F0", "profanity": "F0", "banned_keywords": "F0",
    "refusal": "F0", "deflection": "F0", "phi_echo": "F0",
    "regex_match": "F0", "json_validation": "F0", "sql_validation": "F0",
    "textstat": "F0", "token_count": "F0", "sentiment": "F0",
    "language_detection": "F0", "topic_classification": "F0",
    # F0 with a special column: needs response_samples (k generations)
    "sample_consistency": "F0",
    # later additions
    "jailbreak_detection": "F0", "harm_taxonomy": "F0",
    "phi_entities": "F0", "response_consistency": "F0",
    "diversity": "F0", "intent_match": "F0", "markdown_validity": "F0",
    "latency_cost": "F0",
    "citation_recall": "F1", "citation_precision": "F1",
    "faithfulness_judge": "F1+F3", "noise_sensitivity": "F1+F2",
    "mover_similarity": "F2", "code_execution": "F2",
    "refusal_correctness": "F3", "moderation_screen": "F3",
    # F1: needs retrieved context
    "faithfulness": "F1", "summac": "F1", "summary_stats": "F1",
    "toxicity_preservation": "F1", "verbatim_copy": "F1",
    "unsafe_source_utilization": "F1", "plan_grounded": "F1",
    "context_precision": "F1",
    # F1+F2 / F1+F3 combinations
    "context_recall": "F1+F2", "qag_support": "F1+F3",
    # F2: needs ground truth
    "exact_match": "F2", "levenshtein": "F2", "answer_correctness": "F2",
    "chrf_ter": "F2", "calibration": "F2", "meteor": "F2",
    "bertscore": "F2", "embedding_similarity": "F2", "overlap": "F2",
    "retrieval_hit_rate": "F2", "context_entity_recall": "F2",
    "extraction": "F2",
    # F3: needs a governed LLM judge key
    "answer_relevance": "F3", "completeness": "F3", "helpfulness": "F3",
    "coherence": "F3", "conciseness": "F3", "geval": "F3",
    "instruction_adherence": "F3", "context_relevance": "F3",
    "role_adherence": "F3", "conversation_relevancy": "F3",
    "knowledge_retention": "F3", "pairwise": "F3",
}

# Short per-metric usage notes surfaced in the UI.
METRIC_NOTES: dict[str, str] = {
    "sample_consistency": "needs a response_samples column "
                          "(k generations per prompt)",
    "response_consistency": "needs a response_variants column "
                            "(answers to paraphrased questions)",
    "latency_cost": "needs latency_ms and/or cost columns in the log",
    "moderation_screen": "needs an API key (moderation endpoint)",
    "code_execution": "SECURITY-SENSITIVE: runs generated code in a "
                      "sandbox; opt-in only, never pre-ticked",
    "intent_match": "needs sentence-transformers installed",
    "mover_similarity": "needs sentence-transformers installed",
    "pairwise": "needs a response_b column (a second model's answers)",
    "plan_grounded": "benefits-plan domain metric",
    "extraction": "span grounding additionally uses the context column "
                  "when present",
}

# ------------------------------------------------------------------
# Layer 5: metric -> technique maturity (Table E)
#
# Maturity is a property of the TECHNIQUE, orthogonal to feasibility:
# it governs how much authority a reported number carries, not whether
# the metric can run. Mature = report as authoritative. Emerging =
# reliable with the technique's known caveats. Experimental =
# directional, for triage and within-run comparison; includes all
# LLM-as-a-judge metrics and this program's original metrics.
# ------------------------------------------------------------------

MATURITY: dict[str, str] = {
    # mature
    "exact_match": "mature", "overlap": "mature", "levenshtein": "mature",
    "chrf_ter": "mature", "meteor": "mature", "textstat": "mature",
    "token_count": "mature", "sentiment": "mature",
    "language_detection": "mature", "json_validation": "mature",
    "sql_validation": "mature", "regex_match": "mature",
    "banned_keywords": "mature", "profanity": "mature",
    # emerging
    "bertscore": "emerging", "embedding_similarity": "emerging",
    "faithfulness": "emerging", "summac": "emerging",
    "context_recall": "emerging", "context_precision": "emerging",
    "retrieval_hit_rate": "emerging", "context_entity_recall": "emerging",
    "answer_correctness": "emerging", "safety": "emerging",
    "calibration": "emerging", "extraction": "emerging",
    "topic_classification": "emerging", "summary_stats": "emerging",
    "verbatim_copy": "emerging",
    # experimental: all LLM-as-a-judge metrics ...
    "answer_relevance": "experimental", "context_relevance": "experimental",
    "coherence": "experimental", "conciseness": "experimental",
    "completeness": "experimental", "helpfulness": "experimental",
    "instruction_adherence": "experimental", "geval": "experimental",
    "pairwise": "experimental", "qag_support": "experimental",
    "role_adherence": "experimental", "conversation_relevancy": "experimental",
    "knowledge_retention": "experimental",
    # ... plus sample-based consistency and the program originals
    "sample_consistency": "experimental",
    "refusal": "experimental", "deflection": "experimental",
    "phi_echo": "experimental", "plan_grounded": "experimental",
    "toxicity_preservation": "experimental",
    "unsafe_source_utilization": "experimental",
    # later additions
    "phi_entities": "mature",            # span detection lineage
    "diversity": "mature",               # distinct-n / TTR are textbook
    "markdown_validity": "mature",       # structural checks
    "latency_cost": "mature",            # operational readings
    "jailbreak_detection": "emerging",
    "harm_taxonomy": "emerging",
    "moderation_screen": "emerging",
    "noise_sensitivity": "emerging",
    "citation_precision": "emerging", "citation_recall": "emerging",
    "mover_similarity": "emerging",      # adaptation of an established idea
    "intent_match": "emerging",
    "code_execution": "emerging",        # pass@k is established; our sandbox is v1
    "response_consistency": "experimental",   # original pairing
    "refusal_correctness": "experimental",    # judge
    "faithfulness_judge": "experimental",     # judge
}

MATURITY_REPORTING_RULE = {
    "mature": "Report as authoritative; a gap is stated plainly.",
    "emerging": "Reliable; report with the technique's known caveats.",
    "experimental": "Directional; use for triage and within-run comparison. "
                    "A gap is a signal to investigate, not a verdict.",
}

# ------------------------------------------------------------------
# Enablement naming: which missing thing unblocks a metric. The three
# framework names are context capture, ground truth and judge key;
# special columns are named directly.
# ------------------------------------------------------------------

ENABLEMENT_BY_COLUMN = {
    "context": "context capture",
    "ground_truth": "ground truth",
    "response_samples": "sampled generations (a response_samples column)",
    "response_b": "a second response column (response_b)",
    "history": "a conversation history column",
    "response_variants": "paraphrase re-asks (a response_variants column)",
}

JUDGE_ENABLEMENT = "judge key"


# ------------------------------------------------------------------
# recommend(): join the layers with the dataset's actual columns
# ------------------------------------------------------------------

@dataclass
class MetricAdvice:
    """One admissible metric, tagged for the UI."""
    key: str
    name: str
    description: str
    feasibility: str            # F0 / F1 / F2 / F3 / F1+F2 / F1+F3
    maturity: str               # mature / emerging / experimental
    runnable: bool              # feasible with the columns we have now
    missing: list[str] = field(default_factory=list)   # enablement names
    skip_reason: str = ""       # engine skip vocabulary, when not runnable
    note: str = ""


@dataclass
class DimensionAdvice:
    """One required dimension with its admissible metrics."""
    dimension: str              # canonical key, e.g. "bias_fairness"
    display: str                # e.g. "Bias & Fairness"
    tier_note: str              # Table A qualifier for this tier, if any
    metrics: list[MetricAdvice] = field(default_factory=list)
    report_features: list[dict] = field(default_factory=list)

    @property
    def gaps(self) -> list[MetricAdvice]:
        return [m for m in self.metrics if not m.runnable]


@dataclass
class Recommendation:
    tier: int
    task: str
    dimensions: list[DimensionAdvice] = field(default_factory=list)
    always_on: list[MetricAdvice] = field(default_factory=list)

    def runnable_keys(self) -> list[str]:
        """Admissible-and-runnable registry keys, deduped, in order."""
        seen: set[str] = set()
        out: list[str] = []
        for dim in self.dimensions:
            for m in dim.metrics:
                if m.runnable and m.key not in seen:
                    seen.add(m.key)
                    out.append(m.key)
        for m in self.always_on:
            if m.runnable and m.key not in seen:
                seen.add(m.key)
                out.append(m.key)
        return out

    def documented_gaps(self) -> list[tuple[str, MetricAdvice]]:
        """(dimension display, advice) for every required-but-infeasible
        metric. Deduped per metric within a dimension only, because a
        gap belongs to the dimension whose requirement it leaves open."""
        out = []
        for dim in self.dimensions:
            for m in dim.gaps:
                out.append((dim.display, m))
        return out


def _advise(key: str, available_columns: list[str],
            judge_key: bool) -> MetricAdvice:
    """Tag one metric runnable or gap, from its declared inputs."""
    ev = REGISTRY[key]
    cols = set(available_columns)
    missing_cols = [c for c in ev.info.inputs if c not in cols]
    missing = [ENABLEMENT_BY_COLUMN.get(c, f"a {c} column")
               for c in missing_cols]
    skip = ""
    if missing_cols:
        # same words the engine uses when it skips at run time
        skip = f"not applicable (missing columns: {', '.join(missing_cols)})"
    if ev.info.needs_llm and not judge_key:
        missing.append(JUDGE_ENABLEMENT)
        skip = skip or "needs an LLM API key (none provided)"
    return MetricAdvice(
        key=key, name=ev.info.name, description=ev.info.description,
        feasibility=FEASIBILITY.get(key, "?"),
        maturity=MATURITY.get(key, "?"),
        runnable=not missing, missing=missing, skip_reason=skip,
        note=METRIC_NOTES.get(key, ""))


def recommend(tier: int, task: str, available_columns: list[str],
              judge_key: bool = False) -> Recommendation:
    """The framework recommendation for one use case.

    tier: governance tier from the intake triggers (0, 1 or 2).
    task: canonical task key (see TASKS); legacy values are accepted.
    available_columns: the canonical dataframe's columns after mapping.
    judge_key: whether an LLM judge credential is present this session.

    Returns, per required dimension, the admissible metrics - each
    tagged runnable or documented gap with the missing enablement
    named - plus the always-on descriptive layer. Metrics stay
    suggestions: the caller decides what actually runs.
    """
    task = canonical_task(task,
                          rag="context" in available_columns,
                          has_ground_truth="ground_truth" in available_columns)
    if task not in TASK_MATRIX:
        raise ValueError(f"unknown task: {task!r}")
    if tier not in TIER_DIMENSIONS:
        raise ValueError(f"unknown governance tier: {tier!r}")

    rec = Recommendation(tier=tier, task=task)
    row = TASK_MATRIX[task]
    features = REPORT_FEATURES.get(task, {})
    for dim in TIER_DIMENSIONS[tier]:
        advice = DimensionAdvice(
            dimension=dim, display=DIMENSIONS[dim],
            tier_note=TIER_DIMENSION_NOTES.get((tier, dim), ""),
            report_features=features.get(dim, []))
        for key in row.get(dim, []):
            advice.metrics.append(_advise(key, available_columns, judge_key))
        rec.dimensions.append(advice)
    for key in ALWAYS_ON:
        rec.always_on.append(_advise(key, available_columns, judge_key))
    return rec


def admissible_keys(task: str) -> set[str]:
    """Every registry key admissible for a task, across all dimensions."""
    task = canonical_task(task)
    return {k for keys in TASK_MATRIX[task].values() for k in keys}


def feasibility_input_mismatches() -> list[str]:
    """Cross-check the transcribed F-labels against each evaluator's
    declared inputs. Returns human-readable mismatch lines (empty when
    the matrix and the code agree). Used by tests and matrix reviews;
    runnability always follows the declared inputs, so a label mismatch
    can never make a metric run without its data."""
    lines = []
    for key, label in FEASIBILITY.items():
        ev = REGISTRY.get(key)
        if ev is None:
            continue
        needs_ctx = "context" in ev.info.inputs
        needs_gt = "ground_truth" in ev.info.inputs
        if needs_ctx and "F1" not in label:
            lines.append(f"{key}: labeled {label} but declares a context input")
        if needs_gt and "F2" not in label:
            lines.append(f"{key}: labeled {label} but declares a "
                         f"ground_truth input")
        if ev.info.needs_llm and "F3" not in label:
            lines.append(f"{key}: labeled {label} but needs an LLM judge")
    return lines


def _validate_tables() -> None:
    """Fail loudly at import if any encoded key is unknown to the
    registry, or if a matrix key lacks a feasibility or maturity entry.
    This is the drift alarm between the written matrix and the code."""
    problems: list[str] = []
    encoded: set[str] = set()
    for task, dims in TASK_MATRIX.items():
        for dim, keys in dims.items():
            if dim not in DIMENSIONS:
                problems.append(f"unknown dimension {dim!r} in task {task!r}")
            for key in keys:
                encoded.add(key)
                if key not in REGISTRY:
                    problems.append(
                        f"unknown registry key {key!r} in {task}/{dim}")
    for key in ALWAYS_ON:
        encoded.add(key)
        if key not in REGISTRY:
            problems.append(f"unknown registry key {key!r} in ALWAYS_ON")
    for key in list(FEASIBILITY) + list(MATURITY):
        if key not in REGISTRY:
            problems.append(
                f"unknown registry key {key!r} in FEASIBILITY/MATURITY")
    for key in encoded:
        if key not in FEASIBILITY:
            problems.append(f"matrix key {key!r} has no feasibility label")
        if key not in MATURITY:
            problems.append(f"matrix key {key!r} has no maturity label")
    if problems:
        raise ValueError("overlay tables out of sync with the registry:\n  "
                         + "\n  ".join(problems))


_validate_tables()

__all__ = ["MATRIX_VERSION", "TRIGGERS", "TIER_NAMES", "TIER_CHARACTER",
           "governance_tier", "DIMENSIONS", "TIER_DIMENSIONS",
           "TIER_DIMENSION_NOTES", "TASKS", "TASK_ALIASES", "canonical_task",
           "TASK_MATRIX", "REPORT_FEATURES", "ALWAYS_ON", "FEASIBILITY",
           "MATURITY", "MATURITY_REPORTING_RULE", "METRIC_NOTES",
           "MetricAdvice", "DimensionAdvice", "Recommendation", "recommend",
           "admissible_keys", "feasibility_input_mismatches"]
