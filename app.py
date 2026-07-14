"""VedaEval - Streamlit UI.

Structure:
    Landing         - choose evaluation type: LLM application or classic ML model
    ML branch       - fairness / drift / degradation reports (standalone)
    LLM branch      - 5-step wizard:
        1. Upload   - load a CSV/Excel/JSON/JSONL dataset
        2. Validate - pick your DOMAIN, map columns, health checks
                      (domain decides which scans run, e.g. Healthcare -> PII/PHI)
        3. Configure- pick evaluation metrics (task-aware recommendations)
        4. Run      - execute the evaluation engine
        5. Results  - readable summaries first, charts on demand, export

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from vedaeval.schema import CANONICAL_FIELDS, auto_map_columns, apply_mapping, validate_required
from vedaeval.validation import validate_dataset
from vedaeval.engine import run_evaluation
from vedaeval.evaluators import REGISTRY, available_evaluators, recommended_for
from vedaeval import overlay
from vedaeval.evaluators.judge import DEFAULT_MODELS

# Classic-ML observability is an OPTIONAL, fully isolated add-on: the
# mlobs/ folder can be deleted and everything below still works - the
# landing page simply stops offering the classic-ML option.
try:
    from mlobs.ui import render_ml_page
    HAS_MLOBS = True
except Exception:
    HAS_MLOBS = False

st.set_page_config(page_title="VedaEval", page_icon="📊", layout="wide")

# --- DataVedas fonts (IBM Plex) ------------------------------------------
# The .streamlit/config.toml theme sets the DARK colour palette. Streamlit's
# theme file cannot load a custom web font, so this small style block pulls
# IBM Plex from Google Fonts and points the app's text at it: IBM Plex Sans
# for body text, IBM Plex Mono for headings and buttons (the DataVedas
# convention). This block only changes appearance - delete it to revert.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Body text -> IBM Plex Sans. Set at the app root so it cascades, but
       do NOT target bare div/span - that overrides Streamlit's icon font
       and turns glyphs into literal words (upload, keyboard_arrow_right). */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp input,
    .stApp textarea, .stApp select, [data-testid="stMarkdownContainer"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif;
    }

    /* Headings and buttons -> IBM Plex Mono (the DataVedas display font) */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp .stButton button {
        font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace;
        letter-spacing: -0.01em;
    }

    /* Top-of-page headers (st.header -> h2, e.g. "Step 3 - Configure
       metrics") in DataVedas orange, so they stand apart from the section
       sub-headers (h3) and body text. */
    .stApp h2 {
        color: #ff5c2b;
    }

    /* Protect Streamlit's Material icons (upload, expander arrows, etc.)
       so they keep rendering as glyphs, not as literal ligature text. */
    [data-testid="stIconMaterial"],
    span[class*="material-icons"],
    span[class*="material-symbols"],
    span[data-baseweb="icon"], span[data-baseweb="icon"] i {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                     'Material Icons' !important;
    }

    /* Pill-shaped buttons, matching the DataVedas .btn style */
    .stApp .stButton button {
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LLM_STEPS = ["1. Upload", "2. Validate", "3. Configure", "4. Run", "5. Results",
             "6. Compare runs"]


def load_domains() -> dict:
    """Domain packs are JSON config files, not code: each file sets the
    scanning behavior, recommended metrics, keyword defaults, and demo
    dataset for one domain. Add a domain = add a file."""
    import json as _json
    import pathlib as _pl
    packs = {}
    ddir = _pl.Path(__file__).parent / "domains"
    if ddir.exists():
        for f in sorted(ddir.glob("*.json")):
            try:
                p = _json.loads(f.read_text(encoding="utf-8"))
                packs[p["name"]] = p
            except Exception:
                continue
    if not packs:  # fallback if the folder is missing
        packs = {"Generic": {"name": "Generic", "order": 0, "pii": "optional",
                             "note": "No domain-specific scanning.",
                             "recommended_extra": [],
                             "banned_keywords_default": "guarantee, lawsuit, refund",
                             "demo_dataset": "sample_data/qa_rag_demo.csv"}}
    return dict(sorted(packs.items(), key=lambda kv: kv[1].get("order", 99)))


DOMAINS = load_domains()

# Plain-language explanations for score columns (shown in Results)
METRIC_HELP = {
    "jailbreak_flag": "The request contains a jailbreak pattern (ignore-instructions, persona override, prompt extraction, encoded payload).",
    "jailbreak_signal": "The matched jailbreak pattern.",
    "injection_in_context_flag": "A retrieved context chunk contains instruction-like text (indirect prompt injection).",
    "injection_in_context_signal": "The matched injection text in the context.",
    "harm_taxonomy_flag": "The response matches a named harm lexicon (self-harm, violence instruction, illegal facilitation, medical misinformation markers). Markers, not verdicts.",
    "harm_taxonomy_signal": "Which harm category and phrase matched.",
    "refusal_correctness": "Judge verdict on whether refusing (or answering) was the right call: correct_refusal / over_refusal / under_refusal / correct_answer.",
    "moderation_flag": "The provider moderation endpoint flagged this response (alternative safety engine, cross-checks the local classifier).",
    "moderation_categories": "Which moderation categories flagged.",
    "moderation_max_score": "The highest moderation category score.",
    "phi_count_request": "PHI entities detected in the request (IDs, DOBs, phones, emails).",
    "phi_count_response": "PHI entities detected in the response.",
    "phi_count_context": "PHI entities detected in the retrieved context.",
    "phi_entity_types": "Which PHI entity types appear in the row.",
    "phi_present": "Any PHI anywhere in the row.",
    "faithfulness_judge": "Judge opinion on grounding (High/Medium/Low) - cross-checks the local NLI faithfulness score.",
    "noise_sensitivity": "Share of the answer's support drawn from context chunks that do NOT back the correct answer. High = distracted by retrieval noise.",
    "noise_passage_share": "Share of context chunks that do not support the ground truth (how noisy retrieval was).",
    "citation_precision": "Of the citations made ([1], [doc2]), the share pointing at a chunk that supports the citing sentence.",
    "citation_unsupported": "Citations whose chunk does not support the sentence - the audit trail.",
    "citation_recall": "Share of the response's sentences that carry a citation at all.",
    "mover_similarity": "Semantic closeness to the reference via greedily matched sentence embeddings (MoverScore-style adaptation).",
    "distinct_1": "Share of unique words in the response (lexical diversity).",
    "distinct_2": "Share of unique word pairs (bigram diversity).",
    "type_token_ratio": "Unique words / total words.",
    "self_similarity": "How similar this response is to OTHER rows' responses - high means templated or generic output (Self-BLEU stand-in).",
    "intent_match": "Embedding similarity between request and response - low means the answer is off-intent regardless of fluency.",
    "intent_mismatch_flag": "intent_match fell below the dial (default 0.3).",
    "markdown_valid": "Markdown structure is sane (balanced fences, well-formed links, proper headers). None = the response has no markdown.",
    "markdown_issues": "What is structurally wrong with the markdown.",
    "latency_ms": "Per-row latency reading from the log's operational columns. Feed into Segment Parity for latency parity.",
    "cost_usd": "Per-row cost reading from the log's operational columns.",
    "code_pass": "The generated code passed its test snippet in the sandbox (pass@1). SECURITY-SENSITIVE: opt-in only.",
    "code_error": "Why the code failed (stderr tail, or timeout).",
    "response_consistency": "How stable the answer is when the same question is paraphrased (mean agreement across the row's answers to reworded versions). Needs a response_variants column. Feed it into Segment Parity for Consistency Parity - who gets less stable answers.",
    "response_consistency_weakest_pair": "The least-agreeing pair of paraphrase answers in the row - the sharpest evidence of instability.",
    "phi_echo_rate": "Share of the request's PHI entities repeated in the response. None = the request carried no PHI.",
    "phi_echoed_count": "How many of the user's PHI entities the response repeated back.",
    "phi_introduced_count": "PHI entities in the response that the user never supplied - possibly someone else's. Always worth reading.",
    "phi_echo_flag": "True when the echo rate crosses the dial (default 0.5).",
    "phi_introduced_flag": "True when the response introduced ANY PHI the user did not supply. No dial - introduction is always reportable.",
    "pgc_claims": "Numeric entitlement claims found in the response (amounts, counts, percentages).",
    "pgc_grounded": "How many of those numbers appear in the plan document near a shared anchor word.",
    "pgc_grounded_share": "Grounded claims / total claims. Below 1.0 means the response stated a number the plan document does not support at that spot.",
    "pgc_ungrounded": "The ungrounded numbers, quoted with their sentence - the audit trail.",
    "deflection": "True when the response redirects ('call member services') without answering and without refusing. Distinct from the contact-center 'deflection rate' KPI. Feed into Segment Parity for who gets brushed off more often.",
    "deflection_signal": "The redirect phrase that triggered the flag.",
    "tp_max_regression": "Largest safety-category INCREASE from source to response. Above the epsilon dial = the transformation added harm (safety regression).",
    "tp_max_sanitization": "Largest safety-category DECREASE from source to response. Sanitization is sometimes desired (support summaries) and sometimes a defect (moderation evidence) - task policy decides.",
    "tp_regression_flag": "True when a safety category rose past epsilon between source and response.",
    "tp_sanitization_flag": "True when a safety category fell past epsilon. Informational, not automatically bad.",
    "tp_regression_category": "Which safety category drove the regression flag.",
    "usu_ratio": "Unsafe grounding share / unsafe context share. Reads like disparate impact: 1 = the model uses unsafe passages in proportion to their presence; above 1 = preferential use. None = no unsafe passage in the context (the healthy case).",
    "usu_unsafe_grounding_share": "Share of the response's support that comes from unsafe passages.",
    "usu_unsafe_context_share": "Share of the context's passages that are unsafe.",
    "usu_flag": "True when usu_ratio crosses the flag dial (default 1.5).",
    "lexicon_count": "Word count of the response.",
    "flesch_reading_ease": "How easy the text is to read, 0-100. Higher = easier. 60+ reads like plain English.",
    "flesch_kincaid_grade": "US school grade needed to read the text. 8 means an 8th grader could follow it.",
    "sentiment_compound": "Overall tone from -1 (very negative) to +1 (very positive). Near 0 = neutral.",
    "sentiment": "Tone label: positive, neutral, or negative.",
    "bleu": "Word overlap with the ground truth, 0-1. Higher = closer wording to the reference answer.",
    "rouge1": "Single-word overlap with ground truth, 0-1.",
    "rouge2": "Two-word-phrase overlap with ground truth, 0-1.",
    "rougeL": "Longest matching sequence with ground truth, 0-1.",
    "token_count_request": "Length of the prompt in tokens (the units LLMs are billed in).",
    "token_count_response": "Length of the response in tokens.",
    "profanity_response": "True if the response contains offensive language.",
    "profanity_request": "True if the request contains offensive language.",
    "banned_keywords": "True if the response contains any of your restricted terms.",
    "banned_keywords_matched": "Which restricted terms were found.",
    "regex_match": "Whether the response matches your pattern.",
    "json_valid": "True if the response is valid JSON (for structured-output tasks).",
    "json_error": "Why the response failed the JSON check.",
    "sql_valid": "True if the response is valid SQL syntax.",
    "sql_error": "Why the response failed the SQL check.",
    "max_risk_prob": "Worst safety category score for this row, 0-1. Above 0.5 = flagged.",
    "safety_flag": "True if any safety category crossed the threshold.",
    "safety_toxicity": "Probability the response is toxic, 0-1.",
    "safety_severe_toxicity": "Probability of severe toxicity, 0-1.",
    "safety_obscene": "Probability of obscene content, 0-1.",
    "safety_threat": "Probability the response contains a threat, 0-1.",
    "safety_insult": "Probability the response contains an insult, 0-1.",
    "safety_identity_attack": "Probability of an attack on identity (race, religion, gender...), 0-1.",
    "safety_sexual_explicit": "Probability of sexually explicit content, 0-1.",
    "faithful_score": "How well the response is supported by the context, 0-1. Low = possible hallucination.",
    "faithful": "True if the response stayed faithful to the context overall.",
    "unsupported_count": "How many sentences of the response are NOT supported by the context.",
    "answer_relevance": "LLM judge: does the answer address the question? High / Medium / Low.",
    "answer_relevance_reason": "The judge's one-line reason.",
    "context_relevance": "LLM judge: were the retrieved documents about the question? High / Medium / Low.",
    "context_relevance_reason": "The judge's one-line reason.",
    "coherence": "LLM judge: is the response logically structured? High / Medium / Low.",
    "coherence_reason": "The judge's one-line reason.",
    "conciseness": "LLM judge: is the response appropriately brief? High / Medium / Low.",
    "conciseness_reason": "The judge's one-line reason.",
    "topic": "Best-matching topic label for the row (from your own list, zero-shot).",
    "topic_confidence": "How confidently the winning topic beat the others, 0-1.",
    "language": "Detected language of the response (ISO code: en, hi, es...).",
    "language_confidence": "Confidence of the language detection, 0-1.",
    "refusal": "True if the model declined to help (refusal phrase opening the answer, no alternative offered).",
    "refusal_signal": "Which refusal phrase was detected.",
    "retrieval_hit": "True if the context contains at least half of the ground truth's content words - retrieval brought the needed material.",
    "retrieval_token_recall": "Share of ground-truth content words present in the context, 0-1.",
    "context_entity_recall": "Share of the ground truth's specific facts (numbers, names) present in the context, 0-1.",
    "context_recall": "Share of ground-truth sentences the context can support, 0-1. Low = retrieval missed needed facts.",
    "context_precision": "Share of retrieved chunks that were actually useful, 0-1. Low = the retriever padded the prompt.",
    "answer_correctness": "Combined correctness vs ground truth: half word overlap, half meaning agreement, 0-1.",
    "answer_token_f1": "Word-overlap F1 between response and ground truth, 0-1.",
    "answer_semantic_agreement": "Bidirectional meaning agreement between response and ground truth, 0-1.",
    "exact_match": "True when the normalized response equals the ground truth exactly (strict QA).",
    "token_f1": "Word-overlap F1 between response and ground truth, 0-1 (the SQuAD metric).",
    "levenshtein_similarity": "Character-level edit similarity to the ground truth, 0-1.",
    "chrf": "Character n-gram F-score vs ground truth, 0-1; robust to typos and word forms.",
    "ter": "Translation Edit Rate: edits needed to match the ground truth; 0 = identical, higher is worse.",
    "meteor": "Overlap score crediting stems and synonyms, 0-1; kinder to legitimate rephrasing.",
    "bertscore_f1": "Meaning-level similarity via contextual embeddings, 0-1.",
    "embedding_similarity": "Whole-sentence semantic closeness via embeddings; 'same meaning, different words' detector.",
}


def init_state():
    defaults = {
        "mode": None,                      # None -> landing; "llm" | "ml"
        "step": 0, "raw_df": None, "mapping": None, "canonical_df": None,
        "report": None, "excluded": set(), "selected_metrics": [],
        "configs": {}, "result": None, "task_type": "qa",
        "rag": True, "exclusion_log": [], "domain": "Generic",
        "ml_df": None, "intake_skipped": False, "governance": None,
        "sel_mode": "framework", "sel_ctx": None, "rec_fp_applied": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()

# Metric-checkbox state must survive reruns even when a widget is not
# rendered that run (filtered out by the Manual search, or the user is
# on another step). Streamlit garbage-collects unrendered widget state;
# re-asserting the keys on every run keeps every tick alive.
for _k in REGISTRY:
    st.session_state[f"cb_{_k}"] = bool(
        st.session_state.get(f"cb_{_k}", False))

def _fill_sidebar_count(slot):
    """Write the live metric-selection count into the sidebar slot.

    Called once while the sidebar renders and again at the end of Step 3,
    after Apply/reset may have changed the ticks. The sidebar draws before
    the page body runs, so without the second fill the count showed one
    rerun behind.
    """
    n_sel = sum(bool(st.session_state.get(f"cb_{k}")) for k in REGISTRY)
    if n_sel:
        n_j = sum(1 for k in REGISTRY
                  if st.session_state.get(f"cb_{k}")
                  and REGISTRY[k].info.needs_llm)
        slot.caption(f"Metrics selected: {n_sel}"
                     + (f" - {n_j} LLM-as-a-Judge" if n_j else ""))
    else:
        slot.empty()


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("VedaEval")
    st.caption("Open evaluation engine · v0.2")
    if st.session_state.mode is None:
        st.markdown("Choose an evaluation type to begin.")
    elif st.session_state.mode == "llm":
        step = st.radio("LLM evaluation", LLM_STEPS, index=st.session_state.step)
        st.session_state.step = LLM_STEPS.index(step)
        _sb_count_slot = st.empty()
        _fill_sidebar_count(_sb_count_slot)
    else:
        st.markdown("**ML model observability**")
        st.caption("Fairness · Drift · Degradation")
    if st.session_state.mode is not None:
        st.divider()
        if st.button("Start over / switch type"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ---------------------------------------------------------------- landing

if st.session_state.mode is None:
    st.header("What do you want to evaluate?")
    st.markdown("")
    # Text row and button row are separate column sets so the buttons stay
    # aligned regardless of how long each card's text is.
    n_cards = 2 if HAS_MLOBS else 1
    text_cols = st.columns(n_cards)
    with text_cols[0]:
        st.subheader("An LLM application")
        st.markdown(
            "Chatbots, Q&A systems, RAG applications, summarizers. "
            "Upload prompts and responses; get safety, quality, "
            "faithfulness and relevance scores."
        )
    if HAS_MLOBS:
        with text_cols[1]:
            st.subheader("A classic ML model")
            st.markdown(
                "Classification models with predictions and actuals. "
                "Get fairness reports across demographic groups, data drift "
                "against a baseline, and performance degradation."
            )
    btn_cols = st.columns(n_cards)
    with btn_cols[0]:
        if st.button("Evaluate an LLM application", type="primary",
                     use_container_width=True):
            st.session_state.mode = "llm"
            st.rerun()
    if HAS_MLOBS:
        with btn_cols[1]:
            if st.button("Analyze a classic ML model", use_container_width=True):
                st.session_state.mode = "ml"
                st.rerun()
    st.stop()

# ================================================================ ML branch

if st.session_state.mode == "ml":
    if not HAS_MLOBS:
        st.error("The classic-ML add-on (mlobs folder) is not present in "
                 "this copy of VedaEval.")
        st.stop()
    render_ml_page(st, st.session_state)
    st.stop()

# ================================================================ LLM branch

# ---------------------------------------------------------------- step 1

if st.session_state.step == 0:
    st.header("Step 1 - Upload dataset")
    st.markdown(
        "Accepted formats: **CSV, Excel (.xlsx), JSON, JSONL**. Each row is one "
        "interaction: the prompt sent to the LLM and the answer it gave "
        "(optionally with a known correct answer and retrieved context)."
    )
    uploaded = st.file_uploader("Drop a file here", type=["csv", "xlsx", "json", "jsonl"])

    st.markdown("...or load a built-in demo dataset:")
    import pathlib
    demo_cols = st.columns(max(len(DOMAINS), 1))
    for i, (dname, pack) in enumerate(DOMAINS.items()):
        demo_path = pathlib.Path(__file__).parent / pack.get("demo_dataset", "")
        if pack.get("demo_dataset") and demo_path.exists() and dname != "Generic":
            if demo_cols[i].button(f"{dname} demo", use_container_width=True,
                                   key=f"demo_{dname}"):
                st.session_state.raw_df = pd.read_csv(demo_path)
                st.session_state.domain = dname
    if uploaded is not None:
        try:
            name = uploaded.name.lower()
            if name.endswith(".csv"):
                st.session_state.raw_df = pd.read_csv(uploaded)
            elif name.endswith(".xlsx"):
                st.session_state.raw_df = pd.read_excel(uploaded)
            elif name.endswith(".jsonl"):
                raw_bytes = uploaded.getvalue()
                st.session_state.jsonl_bytes = raw_bytes
                st.session_state.jsonl_name = uploaded.name
                st.session_state.raw_df = pd.read_json(uploaded, lines=True)
            else:
                st.session_state.raw_df = pd.read_json(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

    # JSONL submissions get the intake FILE check right here in Step 1,
    # before row-level validation (Step 2) and metrics (Steps 3-5).
    if st.session_state.get("jsonl_bytes") and st.session_state.raw_df is not None:
        from vedaeval.ui_jsonl import render_jsonl_report
        with st.expander("Dataset file check (JSONL intake report)", expanded=True):
            render_jsonl_report(st, st.session_state.jsonl_bytes,
                                st.session_state.get("jsonl_name", "dataset.jsonl"))

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.success(f"Loaded {len(df)} rows x {len(df.columns)} columns.")
        st.dataframe(df.head(10), use_container_width=True)
        if st.button("Continue to validation ->", type="primary"):
            st.session_state.mapping = auto_map_columns(list(df.columns))
            st.session_state.step = 1
            st.rerun()

# ---------------------------------------------------------------- step 2

elif st.session_state.step == 1:
    st.header("Step 2 - Validate")
    if st.session_state.raw_df is None:
        st.warning("Upload a dataset first (Step 1).")
        st.stop()
    df = st.session_state.raw_df

    st.subheader("Your domain")
    domain = st.selectbox("What kind of data is this?", list(DOMAINS),
                          index=list(DOMAINS).index(st.session_state.domain))
    st.session_state.domain = domain
    st.caption(DOMAINS[domain]["note"])

    st.subheader("Column mapping")
    st.markdown(
        "VedaEval standardizes your columns so every metric knows where to "
        "look. Map whichever fields your file has - only *request* and "
        "*response* are required, and common column names are picked up "
        "automatically."
    )
    with st.expander("What each field means"):
        st.markdown(
            "- **request** (required) - the prompt or question sent to "
            "the model.\n"
            "- **response** (required) - the model's answer, the text "
            "being evaluated.\n"
            "- **ground_truth** (optional) - the known correct answer. "
            "Unlocks accuracy-style metrics (exact match, BLEU/ROUGE and "
            "similar).\n"
            "- **context** (optional) - the retrieved passages a RAG "
            "system handed the model. Unlocks faithfulness and retrieval "
            "metrics.\n"
            "- **history** (optional) - the earlier turns of the "
            "conversation, as one text block per row. The multi-turn "
            "LLM-as-a-Judge metrics (role adherence, conversation "
            "relevancy, knowledge retention) read it; without it they show "
            "up as a named gap, never an error. "
            "sample_data/conversation_demo.csv shows the expected shape.\n"
            "- **timestamp** (optional) - when the exchange happened. "
            "Carried through to saved runs and exports for traceability.\n"
        )
    mapping = st.session_state.mapping or auto_map_columns(list(df.columns))
    cols = st.columns(len(CANONICAL_FIELDS))
    options = ["(none)"] + list(df.columns)
    for i, field in enumerate(CANONICAL_FIELDS):
        current = mapping.get(field)
        idx = options.index(current) if current in options else 0
        pick = cols[i].selectbox(field, options, index=idx, key=f"map_{field}")
        mapping[field] = None if pick == "(none)" else pick
    st.session_state.mapping = mapping

    missing = validate_required(mapping)
    if missing:
        st.error(f"Required fields not mapped yet: {', '.join(missing)}")
        st.stop()

    canonical = apply_mapping(df, mapping)
    st.session_state.canonical_df = canonical

    st.subheader("Data quality checks")
    if DOMAINS[domain]["pii"] == "on":
        pii_engine = "auto"
        st.caption(f"{domain} domain: PII/PHI scan is on automatically.")
    else:
        pii_engine = st.selectbox("PII scan", ["off", "auto"],
                                  help="Optional for the generic domain.")
    if st.button("Run validation", type="primary"):
        st.session_state.report = validate_dataset(canonical, pii_engine=pii_engine)

    report = st.session_state.report
    if report:
        health = report.health
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", health["rows"])
        if "request" in health:
            c2.metric("Empty requests", health["request"]["missing"])
        if "response" in health:
            c3.metric("Empty responses", health["response"]["missing"])

        if not report.issues:
            st.success("No issues found.")
        for issue in report.issues:
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "🛑"}[issue.severity]
            with st.expander(f"{icon} {issue.check}: {issue.message}"):
                if issue.rows:
                    st.dataframe(canonical.loc[issue.rows].head(20), use_container_width=True)

        flagged = sorted(report.flagged_rows)
        if flagged:
            st.subheader("Row exclusion (audit-trailed)")
            to_exclude = st.multiselect("Rows to exclude from evaluation", flagged,
                                        default=sorted(st.session_state.excluded) or flagged)
            reason = st.text_input("Exclusion reason (recorded in the audit log)",
                                   "Flagged by validation (duplicates / PII / leakage)")
            if st.button("Apply exclusions"):
                st.session_state.excluded = set(to_exclude)
                st.session_state.exclusion_log.append(
                    {"rows": to_exclude, "reason": reason,
                     "ts": pd.Timestamp.now().isoformat()})
                st.success(f"{len(to_exclude)} rows will be excluded.")

        if st.button("Continue to metric configuration ->", type="primary"):
            st.session_state.step = 2
            st.rerun()

# ---------------------------------------------------------------- step 3

elif st.session_state.step == 2:
    st.header("Step 3 - Configure metrics")
    if st.session_state.canonical_df is None:
        st.warning("Complete Steps 1-2 first.")
        st.stop()

    # -------- selection mode + use case intake (governance overlay) --------
    _MODE_LABELS = {
        "framework": "Framework (recommended) - five questions, a governance "
                     "tier, and a recommendation you apply explicitly",
        "quick": "Quick mode - task-based suggestions on request, no "
                 "governance framing",
        "manual": "Manual - nothing is ever suggested or pre-ticked",
    }
    sel_mode = st.radio(
        "How do you want to pick metrics?", list(_MODE_LABELS),
        format_func=lambda m: _MODE_LABELS[m],
        index=list(_MODE_LABELS).index(st.session_state.sel_mode))
    st.session_state.sel_mode = sel_mode
    st.session_state.intake_skipped = (sel_mode != "framework")

    # ---- shared vocabulary: descriptive labels instead of codes ----
    FEAS_TEXT = {"F0": "runs on any log", "F1": "needs a context column",
                 "F2": "needs ground truth", "F3": "needs your judge key"}
    TIER_COLORS = {0: "#2e7d32", 1: "#b26a00", 2: "#c62828"}
    TRIGGER_REASONS = {
        "member_facing": "member-facing",
        "decision_influence": "influences decisions about individuals",
        "regulated_data": "regulated data in the workflow",
        "automation_at_scale": "operates autonomously at scale",
        "human_oversight": "no human oversight",
    }

    def _feas_words(code):
        return " + ".join(FEAS_TEXT.get(p, p) for p in str(code).split("+"))

    def _is_judge(key):
        return REGISTRY[key].info.needs_llm

    def _metric_label(key, name):
        """Bold, color-coded metric name: violet = LLM-as-a-Judge,
        green = formula. Color reinforces the labeled chips, never
        replaces them (color-blind safety)."""
        color = "violet" if _is_judge(key) else "green"
        return f":{color}[**{name}**]"

    def _fired_reasons(triggers):
        fired = []
        for k, _lbl in overlay.TRIGGERS:
            v = triggers.get(k)
            if (k == "human_oversight" and v is False) or \
               (k != "human_oversight" and v is True):
                fired.append(TRIGGER_REASONS[k])
        return fired

    def _tier_badge(tier_val, triggers=None):
        if tier_val is None:
            st.markdown(
                "<span style='background:#9e9e9e;color:white;padding:4px "
                "14px;border-radius:14px;font-weight:600'>Not assessed"
                "</span>", unsafe_allow_html=True)
            return
        st.markdown(
            f"<span style='background:{TIER_COLORS[tier_val]};color:white;"
            f"padding:4px 14px;border-radius:14px;font-weight:600'>"
            f"{overlay.TIER_NAMES[tier_val]}</span>",
            unsafe_allow_html=True)
        if triggers is not None:
            fired = _fired_reasons(triggers)
            if fired:
                st.caption("Because you answered: " + "; ".join(fired) + ".")
            else:
                st.caption("No risk triggers fired: internal, informational, "
                           "human in the loop.")

    with st.expander("How this works, and what the words mean"):
        st.markdown(
            "The chain runs one way: your **use case** decides a governance "
            "tier, the tier decides which **dimensions** must be evaluated, "
            "the **task type** decides which metrics are admissible for each "
            "dimension, and your **dataset's columns** decide what can run "
            "today. Anything required but not runnable is a **documented "
            "gap**, never silently dropped.\n\n"
            "**Judge key** - some checks need judgment, not a formula "
            "(\"was this refusal appropriate?\"). Those send each row to a "
            "large model acting as a judge, using your own API key. The key "
            "lives only in this session's memory and is sent only to its "
            "provider.\n\n"
            "**LLM-as-a-Judge vs formula** - every metric is labeled one "
            "or the other: formula metrics run locally and cost nothing; "
            "LLM-as-a-Judge metrics call an API on your key, one call per "
            "row.\n\n"
            "**Data needs** - each metric says what it needs in plain "
            "words: runs on any log, needs a context column, needs ground "
            "truth, or needs your judge key. (The papers call these F0 to "
            "F3, in that order.)\n\n"
            "**Maturity** - mature results can be reported as "
            "authoritative; emerging results are reliable with known "
            "caveats; experimental results (all LLM-as-a-Judge metrics and the "
            "original metrics) are directional signals for triage.\n\n"
            "**Governance evidence** - only metrics admissible for your "
            "task count toward tier requirements; anything else you tick "
            "still runs and reports, but carries no governance weight.")

    avail = available_evaluators()
    selected = []
    tier = None

    def _reset_selections_if_ctx_changed(ctx):
        """Selections never survive a mode or task switch silently."""
        prev = st.session_state.sel_ctx
        if prev != ctx:
            for _k in REGISTRY:
                st.session_state[f"cb_{_k}"] = False
            st.session_state.sel_ctx = ctx
            st.session_state.rec_fp_applied = None
            if prev is not None:
                st.info("Task or mode changed: metric selections were "
                        "cleared so nothing carries over silently.")

    st.divider()

    if sel_mode == "quick":
        # ---------------- quick mode: suggestions only on request ----------------
        task_keys = list(overlay.TASKS)
        st.session_state.governance = {
            "mode": "quick", "note": "task-based suggestions, no governance framing"}
        ct = st.selectbox("Task type", task_keys,
                          format_func=lambda k: overlay.TASKS[k],
                          key="quick_task")
        st.session_state.task_type = ct
        st.session_state.rag = (ct == "rag")
        _reset_selections_if_ctx_changed(("quick", ct))
        st.caption(
            "Quick mode carries no governance framing. Click the button to "
            "pre-tick a sensible set for this task; nothing ticks itself.")
        _QUICK_MAP = {
            "closed_qa": ("qa", False), "open_qa": ("qa", False),
            "rag": ("qa", True), "summarization": ("summarization", False),
            "extraction": ("structured_output", False),
            "classification": ("structured_output", False),
            "chat_agentic": ("qa", False),
        }
        qb1, qb2, _ = st.columns([1.4, 1.2, 3])
        if qb1.button("Suggest for this task", type="primary"):
            _t, _r = _QUICK_MAP[ct]
            rec = recommended_for(_t, _r)
            rec = list(dict.fromkeys(
                rec + DOMAINS.get(st.session_state.domain, {}).get(
                    "recommended_extra", [])))
            for key in REGISTRY:
                if avail[key][0]:
                    st.session_state[f"cb_{key}"] = key in rec
            st.rerun()
        if qb2.button("Deselect all"):
            for key in REGISTRY:
                st.session_state[f"cb_{key}"] = False
            st.rerun()
        for key, ev in REGISTRY.items():
            ok, why = avail[key]
            if not ok:
                st.checkbox(f"~~{ev.info.name}~~ (install: {why})", value=False,
                            disabled=True, key=f"cb_{key}")
                continue
            if st.checkbox(f"{_metric_label(key, ev.info.name)} - {ev.info.description}",
                           value=False, key=f"cb_{key}"):
                selected.append(key)

    elif sel_mode == "manual":
        # ---------------- manual mode: the user does everything ----------------
        st.session_state.governance = {"mode": "manual selection"}
        _reset_selections_if_ctx_changed(("manual",))
        st.caption(
            "Manual mode never suggests or pre-ticks anything: every tick is "
            "your own. The governance record will say 'manual selection'.")
        search = st.text_input("Search metrics", "",
                               placeholder="e.g. toxicity, faithfulness, drift")
        _picked = sorted(REGISTRY[k].info.name for k in REGISTRY
                         if st.session_state.get(f"cb_{k}"))
        if _picked:
            st.caption(f"**Selected so far ({len(_picked)}):** "
                       + ", ".join(_picked)
                       + ". Searching never drops these; they run even "
                         "while hidden by the filter.")
        if st.button("Deselect all"):
            for key in REGISTRY:
                st.session_state[f"cb_{key}"] = False
            st.rerun()
        groups = {}
        for key, ev in REGISTRY.items():
            hay = (ev.info.name + " " + ev.info.description).lower()
            if search and search.lower().strip() not in hay:
                continue
            groups.setdefault(ev.info.category, []).append((key, ev))
        if not groups:
            st.caption("No metric matches that search.")
        for cat in sorted(groups):
            st.markdown(f"**{cat}**")
            for key, ev in groups[cat]:
                ok, why = avail[key]
                if not ok:
                    st.checkbox(f"~~{ev.info.name}~~ (install: {why})",
                                value=False, disabled=True, key=f"cb_{key}")
                    continue
                if st.checkbox(f"{_metric_label(key, ev.info.name)} - {ev.info.description}",
                               value=False, key=f"cb_{key}"):
                    selected.append(key)
        # ticked metrics hidden by the current filter still count
        for key in REGISTRY:
            if (st.session_state.get(f"cb_{key}") and key not in selected
                    and avail[key][0]):
                selected.append(key)

    else:
        # ---------------- framework intake: assess-gated ----------------
        st.subheader("Use case intake")
        st.markdown(
            "Five yes/no questions about the **use case** (not the model or "
            "the technology) place it in a governance tier. Nothing below "
            "changes until you click **Assess my use case**, and metric "
            "selections change only when you click **Apply recommended**.")
        answers_now = {}
        unanswered = []
        for trig_key, trig_label in overlay.TRIGGERS:
            pick = st.radio(trig_label, ["Yes", "No"], index=None,
                            horizontal=True, key=f"itk_{trig_key}")
            if pick is None:
                unanswered.append(trig_label)
            else:
                answers_now[trig_key] = (pick == "Yes")
        if st.button("Assess my use case", type="primary",
                     disabled=bool(unanswered)):
            _tier = overlay.governance_tier(**answers_now)
            st.session_state.governance = {
                "triggers": dict(answers_now), "tier": _tier,
                "matrix_version": overlay.MATRIX_VERSION,
                "assessed_at": pd.Timestamp.now().isoformat()}
            st.rerun()
        gov = st.session_state.governance
        if gov is None or "tier" not in gov:
            if unanswered:
                st.caption("Answer all five questions to enable the "
                           "assessment.")
            _tier_badge(None)
            st.info("No tier and no recommendation yet: answer the five "
                    "questions and click **Assess my use case**, or switch "
                    "to Quick mode or Manual above.")
        else:
            if not unanswered and dict(answers_now) != gov["triggers"]:
                st.warning("Your answers changed since the last assessment: "
                           "click **Assess my use case** again. Everything "
                           "below still reflects the previous answers.")
            _t = gov["tier"]
            _tier_badge(_t, gov["triggers"])
            dim_bits = []
            for d in overlay.TIER_DIMENSIONS[_t]:
                note = overlay.TIER_DIMENSION_NOTES.get((_t, d), "")
                dim_bits.append(overlay.DIMENSIONS[d]
                                + (f" ({note})" if note else ""))
            st.caption(f"This tier requires: {'; '.join(dim_bits)}. "
                       f"Evidence character: {overlay.TIER_CHARACTER[_t]}. "
                       "A sixth trigger operates at runtime: monitoring "
                       "evidence of material disparities escalates the tier "
                       "after deployment.")

    if sel_mode == "framework" and "tier" in (st.session_state.governance or {}):
        tier = st.session_state.governance["tier"]
        st.session_state.governance["mode"] = "framework"
        # ---------------- framework path: overlay.recommend() ----------------
        canonical_cols = list(st.session_state.canonical_df.columns)
        default_ct = overlay.canonical_task(
            st.session_state.task_type, st.session_state.rag,
            has_ground_truth="ground_truth" in canonical_cols)
        task_keys = list(overlay.TASKS)
        fc1, fc2 = st.columns([2, 1])
        ct = fc1.selectbox("Task type", task_keys,
                           index=task_keys.index(default_ct),
                           format_func=lambda k: overlay.TASKS[k])
        st.session_state.task_type = ct
        st.session_state.rag = (ct == "rag")
        with fc2:
            _tier_badge(tier)
        _reset_selections_if_ctx_changed(("framework", ct))

        judge_ready = bool(st.session_state.get("judge_credentials", {}).get("api_key"))
        rec_res = overlay.recommend(tier, ct, canonical_cols, judge_key=judge_ready)
        rec_fp = (tier, ct, tuple(sorted(canonical_cols)), judge_ready,
                  tuple(sorted(st.session_state.governance["triggers"].items())))
        rec_applied = st.session_state.rec_fp_applied
        rec_stale = rec_applied is not None and rec_applied != rec_fp
        from vedaeval.validation import _bias_columns
        seg_cols = _bias_columns(st.session_state.canonical_df)
        has_ctx = "context" in canonical_cols

        # ---- the recommendation as a matrix slice (the paper's table) ----
        _tbl_rows, _seen_tbl = [], set()
        for dim_adv in rec_res.dimensions:
            for m in dim_adv.metrics:
                if m.key in _seen_tbl:
                    continue
                _seen_tbl.add(m.key)
                _tbl_rows.append({
                    "Dimension": dim_adv.display,
                    "Metric": m.name,
                    "What it checks": m.description,
                    "Data status": ("Ready" if m.runnable
                                    else "Gap - needs " + ", ".join(m.missing)),
                    "Engine": "LLM-as-a-Judge" if _is_judge(m.key) else "formula",
                    "Maturity": m.maturity,
                })
        st.markdown(
            f"**The recommendation for {overlay.TASKS[ct]} at "
            f"{overlay.TIER_NAMES[tier]}** - this is the framework matrix "
            "row for your task, checked against your dataset. Nothing is "
            "ticked until you click **Apply recommended**; after that, "
            "fine-tune freely. The framework recommends, you decide.")
        st.dataframe(pd.DataFrame(_tbl_rows), use_container_width=True,
                     hide_index=True)
        if rec_stale:
            st.warning("The recommendation changed (tier, task, dataset, "
                       "or judge key). Your current selections are "
                       "untouched: click the button below to apply the "
                       "new recommended set.")
        _apply_label = ("Recommendation changed - apply again" if rec_stale
                        else ("Apply recommended (applied)"
                              if rec_applied == rec_fp
                              else "Apply recommended"))
        _apply_type = "secondary" if rec_applied == rec_fp else "primary"
        sel_r, sel_b, _ = st.columns([2, 1.2, 2.6])
        if sel_r.button(_apply_label, type=_apply_type):
            runnable = set(rec_res.runnable_keys())
            for key in REGISTRY:
                if avail[key][0]:
                    st.session_state[f"cb_{key}"] = key in runnable
            st.session_state.rec_fp_applied = rec_fp
            st.session_state.governance["rec_applied_at"] = (
                pd.Timestamp.now().isoformat())
            st.rerun()
        if sel_b.button("Deselect all"):
            for key in REGISTRY:
                st.session_state[f"cb_{key}"] = False
            st.rerun()

        rendered = set()

        def _metric_checkbox(m):
            ok, why = avail[m.key]
            if not ok:
                st.checkbox(f"~~{m.name}~~ (install: {why})", value=False,
                            disabled=True, key=f"cb_{m.key}")
                return
            _chip = (":violet[LLM-as-a-Judge]" if _is_judge(m.key)
                     else ":green[formula]")
            label = (f"{_metric_label(m.key, m.name)} "
                     f"\\[{_feas_words(m.feasibility)} · {_chip} · "
                     f"{m.maturity}] - {m.description}")
            # Nothing ever ticks itself. Apply recommended is the only
            # path that sets these true in bulk.
            picked = st.checkbox(label, value=False, key=f"cb_{m.key}")
            notes = []
            if not m.runnable:
                notes.append("documented gap - needs: " + ", ".join(m.missing))
            if m.note:
                notes.append(m.note)
            if notes:
                st.caption("&nbsp;&nbsp;&nbsp;&nbsp;" + "  ·  ".join(notes))
            if picked:
                selected.append(m.key)

        def _dim_count(dim_adv):
            keys = {m.key for m in dim_adv.metrics}
            n = sum(bool(st.session_state.get(f"cb_{k}")) for k in keys)
            return f"{n} of {len(keys)} selected"

        with st.expander("Advanced: fine-tune the selection (per-metric "
                         "tick boxes, grouped by dimension)"):
            for dim_adv in rec_res.dimensions:
                title = f"{dim_adv.display} - {_dim_count(dim_adv)}"
                if dim_adv.tier_note:
                    title += f" ({dim_adv.tier_note})"
                st.markdown(f"**{title}**")
                for m in dim_adv.metrics:
                    if m.key in rendered:
                        st.caption(f"{m.name}: also admissible here "
                                   "(listed above).")
                        continue
                    rendered.add(m.key)
                    _metric_checkbox(m)
                for feat in dim_adv.report_features:
                    if feat.get("planned"):
                        status = "planned, not yet built"
                    elif feat.get("needs_segment") and not seg_cols:
                        status = ("needs a segment column (name containing "
                                  "bias / gender / age / segment)")
                    elif feat.get("needs_context") and not has_ctx:
                        status = "needs context capture"
                    else:
                        status = ("available - see the Fairness tab / "
                                  "validation report")
                    st.caption(f"Report-level: {feat['name']} - {status}.")

            st.markdown("**Always-on descriptive layer** - any task, any "
                        "tier. These inform reading and segment slicing and "
                        "carry no governance weight by themselves.")
            for m in rec_res.always_on:
                if m.key in rendered:
                    continue
                rendered.add(m.key)
                _metric_checkbox(m)

        gaps = rec_res.documented_gaps()
        st.session_state.governance["gaps_open"] = [
            m.name for _d, m in gaps]
        if gaps:
            by_need = {}
            for dim_name, m in gaps:
                need = ", ".join(m.missing) or "unknown enablement"
                by_need.setdefault(need, []).append((dim_name, m))
            with st.expander(
                    f"Documented gaps ({len(gaps)}) - required at Tier "
                    f"{tier}, not feasible with this dataset yet. Grouped "
                    "by the action that unlocks them.", expanded=True):
                st.markdown(
                    "The framework never drops a required metric silently. "
                    f"Your {len(gaps)} gaps come down to "
                    f"{len(by_need)} action"
                    f"{'s' if len(by_need) > 1 else ''}:")
                _any_judge_gap = False
                for need, items in sorted(by_need.items(),
                                          key=lambda kv: -len(kv[1])):
                    st.markdown(f"**{need} -> unlocks {len(items)} "
                                f"metric{'s' if len(items) > 1 else ''}**")
                    st.caption(", ".join(f"{m.name} ({d})"
                                         for d, m in items))
                    if "judge" in need.lower():
                        _any_judge_gap = True
                if _any_judge_gap:
                    st.markdown("**Unlock the judge metrics right here:**")
                    gk1, gk2, gk3 = st.columns([1, 1, 2])
                    g_prov = gk1.selectbox(
                        "Provider ", ["openai", "anthropic", "gemini"],
                        key="gap_provider")
                    g_model = gk2.text_input(
                        "Model (blank = default)", key="gap_model")
                    g_key = gk3.text_input(
                        "Judge API key (session memory only, never "
                        "saved)", type="password", key="gap_api_key")
                    st.caption(
                        f"Leave Model blank to use the default for "
                        f"{g_prov}: {DEFAULT_MODELS.get(g_prov, '')}. "
                        "Type a model name to override it.")
                    if g_key:
                        st.session_state.judge_credentials = {
                            "provider": g_prov, "model": g_model.strip(),
                            "api_key": g_key}
                        st.warning(
                            "Key registered, but the judge metrics are not "
                            "in your run yet. Scroll up and click Apply "
                            "recommended again to add them (the button now "
                            "flags that the recommendation changed). Until "
                            "you do, the run stays formula-only.")
                st.caption("This list, and what you did about it, goes "
                           "into the governance record.")

        others = [k for k in REGISTRY if k not in rendered]
        if others:
            with st.expander(
                    f"All other metrics ({len(others)}) - not in this task's "
                    "row, so they do not count as governance evidence for it. "
                    "Still selectable."):
                for key in others:
                    ok, why = avail[key]
                    ev = REGISTRY[key]
                    if not ok:
                        st.checkbox(f"~~{ev.info.name}~~ (install: {why})",
                                    value=False, disabled=True, key=f"cb_{key}")
                        continue
                    if st.checkbox(f"{_metric_label(key, ev.info.name)} - {ev.info.description}",
                                   value=False, key=f"cb_{key}"):
                        selected.append(key)

    configs = {}
    if "banned_keywords" in selected:
        default_kw = DOMAINS.get(st.session_state.domain, {}).get(
            "banned_keywords_default", "guarantee, lawsuit, refund")
        kw = st.text_input("Banned keywords (comma-separated)", default_kw)
        configs["banned_keywords"] = {"keywords": [k.strip() for k in kw.split(",")]}
    if "regex_match" in selected:
        pattern = st.text_input("Regex pattern to look for in responses", r"\d+ (dollars|USD)")
        configs["regex_match"] = {"pattern": pattern}
    if "sql_validation" in selected:
        configs["sql_validation"] = {"dialect": st.selectbox(
            "SQL dialect", ["", "mysql", "postgres", "snowflake", "bigquery"]) or None}
    if "topic_classification" in selected:
        topics_text = st.text_input(
            "Topics (comma-separated, your own labels)",
            "billing, coverage, claims, appointments",
            help="Zero-shot: no training needed. Each row gets the best-"
                 "matching label and a confidence.")
        configs["topic_classification"] = {
            "topics": [t.strip() for t in topics_text.split(",") if t.strip()]}

    # ------- build your own LLM-as-a-judge metric -------
    st.session_state.setdefault("custom_judges", [])
    with st.expander("Build your own LLM-as-a-judge metric"):
        st.markdown(
            "Write the **grading instructions** (the rubric) and VedaEval "
            "wraps them in the same tested prompt frame the built-in judges "
            "use, so replies come back as clean High/Medium/Low grades. "
            "Examples of sharp rubrics:\n"
            "- *Does the answer align with standard clinical guidance? "
            "Penalize any specific dosage advice.*\n"
            "- *Is the tone consistent with a professional insurance "
            "brand: helpful, plain, never salesy?*"
        )
        cj1, cj2 = st.columns([1, 2])
        cj_name = cj1.text_input("Metric name", "", placeholder="e.g. Medical Accuracy")
        cj_rubric = cj2.text_input("Grading instructions (the rubric)", "",
                                   placeholder="e.g. Does the answer align with clinical guidance?")
        cj_ctx = st.checkbox("Judge should also see the context column", value=False)
        if st.button("Add this judge", disabled=not (cj_name.strip() and cj_rubric.strip())):
            st.session_state.custom_judges.append(
                {"name": cj_name.strip(), "rubric": cj_rubric.strip(),
                 "use_context": cj_ctx})
            st.success(f"Added '{cj_name.strip()}'. It runs with this evaluation "
                       f"(needs the API key below, like any judge).")
        if st.session_state.custom_judges:
            st.write("Custom judges for this session: " +
                     ", ".join(f"**{j['name']}**" for j in st.session_state.custom_judges))
            if st.button("Remove all custom judges"):
                st.session_state.custom_judges = []
                st.rerun()

    # ------- LLM-as-a-judge credentials (bring your own key) -------
    judge_keys = [k for k in selected if REGISTRY[k].info.needs_llm]
    if st.session_state.custom_judges:
        judge_keys = judge_keys + ["__custom__"]  # custom judges need the key too
    if judge_keys:
        st.subheader("LLM judge settings")
        st.markdown(
            "The metrics marked as *LLM judge* send each row to a large "
            "language model for grading, using **your own API key**. The key "
            "is held only in this session's memory: it is never saved, never "
            "logged, and is sent only to the provider it belongs to. Close "
            "the tab and it is gone. Without a key, judge metrics are simply "
            "skipped and everything else still runs."
        )
        _jc = st.session_state.get("judge_credentials", {}) or {}
        _provs = ["openai", "anthropic", "gemini"]
        jc1, jc2, jc3 = st.columns(3)
        provider = jc1.selectbox(
            "Provider", _provs,
            index=_provs.index(_jc.get("provider", "openai"))
            if _jc.get("provider") in _provs else 0)
        model = jc2.text_input("Model (blank = sensible default)",
                               _jc.get("model", ""))
        jc2.caption(f"blank uses {DEFAULT_MODELS.get(provider, '')}")
        api_key = jc3.text_input("API key", type="password",
                                 value=_jc.get("api_key", ""),
                                 help="Held in session memory only.")
        for k in judge_keys:
            if k != "__custom__":
                configs[k] = {"provider": provider, "model": model,
                              "api_key": api_key}
        st.session_state.judge_credentials = {
            "provider": provider, "model": model, "api_key": api_key}
        if not api_key:
            st.info("No key entered: the judge metrics will be skipped; "
                    "all other metrics run normally.")

    st.session_state.selected_metrics = selected
    st.session_state.configs = configs

    # ---------------- the framework matrix explorer (any mode) ----------------
    with st.expander("Explore the framework matrix (read-only: does not "
                     "touch your selections)"):
        st.caption("Pick any task and tier to see what the framework would "
                   "require, assuming ideal data (all columns present, "
                   "judge key available). This is the full matrix from the "
                   "framework paper, live from the same module the "
                   "recommendation uses, so it can never go stale.")
        mx1, mx2 = st.columns(2)
        mx_task = mx1.selectbox("Task ", list(overlay.TASKS),
                                format_func=lambda k: overlay.TASKS[k],
                                key="mx_task")
        mx_tier = mx2.selectbox("Tier ", [0, 1, 2],
                                format_func=lambda t: overlay.TIER_NAMES[t],
                                key="mx_tier")
        _ideal_cols = list(dict.fromkeys(
            ["request", "response", "context", "ground_truth"]
            + list(getattr(overlay, "ENABLEMENT_BY_COLUMN", {}))))
        mx_res = overlay.recommend(mx_tier, mx_task, _ideal_cols,
                                   judge_key=True)
        def _why_here(dim_adv, m, task_key):
            why = (f"In the {overlay.TASKS[task_key]} row of "
                   f"{dim_adv.display}")
            if dim_adv.tier_note:
                why += f", which this tier requires ({dim_adv.tier_note})"
            else:
                why += ", which this tier requires"
            if m.note:
                why += ". " + m.note
            return why

        _mx_rows, _mx_seen = [], set()
        for dim_adv in mx_res.dimensions:
            for m in dim_adv.metrics:
                if m.key in _mx_seen:
                    continue
                _mx_seen.add(m.key)
                _mx_rows.append({
                    "Dimension": dim_adv.display, "Metric": m.name,
                    "Needs": _feas_words(m.feasibility),
                    "Engine": "LLM-as-a-Judge" if _is_judge(m.key) else "formula",
                    "Maturity": m.maturity,
                    "Why offered here": _why_here(dim_adv, m, mx_task),
                })
        st.dataframe(pd.DataFrame(_mx_rows), use_container_width=True,
                     hide_index=True)

        @st.cache_data
        def _full_matrix_download():
            import io
            import re as _re
            frames = {}
            for _t in (0, 1, 2):
                for _task in overlay.TASKS:
                    rr = overlay.recommend(_t, _task, _ideal_cols,
                                           judge_key=True)
                    for dim_adv in rr.dimensions:
                        for m in dim_adv.metrics:
                            frames.setdefault(dim_adv.display, []).append({
                                "Tier": overlay.TIER_NAMES[_t],
                                "Task": overlay.TASKS[_task],
                                "Metric": m.name,
                                "Needs": _feas_words(m.feasibility),
                                "Engine": ("LLM-as-a-Judge"
                                           if _is_judge(m.key)
                                           else "formula"),
                                "Maturity": m.maturity,
                                "Why offered here": _why_here(
                                    dim_adv, m, _task),
                            })
            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    for dim, rws in frames.items():
                        sheet = _re.sub(r"[\\/*?:\[\]]", "-", dim)[:31]
                        pd.DataFrame(rws).to_excel(xw, sheet_name=sheet,
                                                   index=False)
                return (buf.getvalue(), "vedaeval_framework_matrix.xlsx",
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet")
            except Exception:
                flat = [dict(r, Dimension=dim)
                        for dim, rws in frames.items() for r in rws]
                return (pd.DataFrame(flat).to_csv(index=False).encode(),
                        "vedaeval_framework_matrix.csv", "text/csv")

        # Build-on-click: the workbook covers every task x tier combination,
        # so the first build takes up to half a minute. Building it eagerly
        # froze the page on first open; now it builds only
        # when asked, behind a spinner, and st.cache_data keeps later builds
        # instant for the rest of the server session.
        if st.session_state.get("mx_bundle") is None:
            if st.button("Prepare the full matrix download (every task, "
                         "every tier)", key="mx_prepare"):
                with st.spinner("Building the full matrix workbook - "
                                "one-time, up to 30 seconds..."):
                    st.session_state.mx_bundle = _full_matrix_download()
                st.rerun()
            st.caption("One-time build, up to 30 seconds. After that the "
                       "download is instant.")
        else:
            _mx_data, _mx_name, _mx_mime = st.session_state.mx_bundle
            st.download_button("Download the full matrix (every task, every "
                               "tier)", _mx_data, file_name=_mx_name,
                               mime=_mx_mime)

    # ---------------- review card: the governance record header ----------------
    if selected:
        st.divider()
        st.subheader("Review before running")
        _rows_eff = (len(st.session_state.canonical_df)
                     - len(st.session_state.excluded))
        _judge_sel = [k for k in selected if _is_judge(k)]
        _n_judges = len(_judge_sel) + len(st.session_state.custom_judges)
        _gov = st.session_state.governance or {}
        _lines = []
        if sel_mode == "framework" and "tier" in _gov:
            fired = _fired_reasons(_gov.get("triggers", {}))
            _lines.append(
                f"- **Governance:** {overlay.TIER_NAMES[_gov['tier']]}"
                + (f" (because: {', '.join(fired)})" if fired
                   else " (no risk triggers fired)"))
            _lines.append(
                f"- **Task:** {overlay.TASKS[st.session_state.task_type]}")
            _open_gaps = _gov.get("gaps_open", [])
            if _open_gaps:
                _lines.append(f"- **Acknowledged gaps:** {len(_open_gaps)} "
                              "required metrics not feasible yet (named in "
                              "the gaps list above; they go into the "
                              "record)")
        else:
            _lines.append(f"- **Mode:** {_gov.get('mode', sel_mode)} - no "
                          "governance framing; the record says so")
            _lines.append(f"- **Task:** {st.session_state.task_type}")
        _lines.append(f"- **Metrics:** {len(selected)} selected, "
                      f"{_n_judges} of them LLM-as-a-Judge")
        if _n_judges:
            _lines.append(f"- **Judge cost:** {_n_judges} LLM-as-a-Judge "
                          f"metric{'s' if _n_judges > 1 else ''} x "
                          f"{_rows_eff} rows = about "
                          f"{_n_judges * _rows_eff} API calls on your key")
        else:
            _lines.append("- **Cost:** formula metrics only - runs "
                          "locally, no API calls")
        st.markdown("\n".join(_lines))
        _gov["review"] = {
            "mode": _gov.get("mode", sel_mode),
            "task": st.session_state.task_type,
            "metrics_selected": len(selected),
            "ai_judged": _n_judges,
            "rows": _rows_eff,
            "estimated_judge_calls": _n_judges * _rows_eff,
            "acknowledged_gaps": _gov.get("gaps_open", []),
        }
        st.session_state.governance = _gov

    if selected and st.button("Confirm and continue to run ->",
                              type="primary"):
        st.session_state.step = 3
        st.rerun()

    # Second fill of the sidebar count, now that Apply recommended / reset /
    # manual ticks in this run are all in session state (see the helper's
    # docstring: the sidebar itself rendered before any of that happened).
    if st.session_state.mode == "llm":
        _fill_sidebar_count(_sb_count_slot)

# ---------------------------------------------------------------- step 4

elif st.session_state.step == 3:
    st.header("Step 4 - Run evaluation")
    canonical = st.session_state.canonical_df
    if canonical is None or not st.session_state.selected_metrics:
        st.warning("Complete Steps 1-3 first.")
        st.stop()

    excluded = st.session_state.excluded
    run_df = canonical.drop(index=[i for i in excluded if i in canonical.index])

    # rebuild session-scoped custom judges (never in the global registry)
    extra, extra_keys = {}, []
    creds = st.session_state.get("judge_credentials", {})
    for spec in st.session_state.get("custom_judges", []):
        from vedaeval.evaluators.judge import CustomJudge
        judge = CustomJudge(spec["name"], spec["rubric"],
                            use_context=spec.get("use_context", False))
        extra[judge.info.key] = judge
        extra_keys.append(judge.info.key)
        st.session_state.configs[judge.info.key] = dict(creds)

    all_metrics = st.session_state.selected_metrics + extra_keys
    st.markdown(f"Evaluating **{len(run_df)}** rows "
                f"({len(excluded)} excluded) with "
                f"**{len(all_metrics)}** metrics: {', '.join(all_metrics)}")

    if st.button("Run now", type="primary"):
        with st.spinner("Running evaluators..."):
            st.session_state.result = run_evaluation(
                run_df, all_metrics, st.session_state.configs,
                extra_evaluators=extra)
        st.session_state.step = 4
        st.rerun()

# ---------------------------------------------------------------- step 5

elif st.session_state.step == 4:
    st.header("Step 5 - Results")
    result = st.session_state.result
    if result is None:
        st.warning("Run an evaluation first (Step 4).")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows evaluated", len(result.scores))
    c2.metric("Metrics run", len(result.ran))
    c3.metric("Metrics skipped", len(result.skipped))
    if result.skipped:
        with st.expander("Why were metrics skipped?"):
            for key, reason in result.skipped.items():
                st.write(f"- **{key}**: {reason}")

    # F5 (13/07/2026): a judge metric whose rows all errored still counts
    # as "run", so errors could hide behind "0 skipped". Surface them.
    _err_cols = {}
    for _c in result.scores.columns:
        if result.scores[_c].dtype == object:
            _n_err = result.scores[_c].astype(str).str.startswith(
                "judge error").sum()
            if _n_err:
                _err_cols[_c] = int(_n_err)
    if _err_cols:
        _tot = sum(_err_cols.values())
        st.warning(
            f"{_tot} judge error{'s' if _tot > 1 else ''} in "
            f"{len(_err_cols)} column{'s' if len(_err_cols) > 1 else ''}: "
            + ", ".join(f"{c} ({n} row{'s' if n > 1 else ''})"
                        for c, n in _err_cols.items())
            + ". These rows were not scored - the reason columns carry "
              "the provider's error message (a wrong key or model is the "
              "usual cause).")

    scores = result.scores
    input_cols = [c for c in CANONICAL_FIELDS if c in scores.columns]
    base_cols = set(st.session_state.canonical_df.columns)
    score_cols = [c for c in scores.columns
                  if c not in input_cols and c not in base_cols]

    # ---------------- readable summary, one line per score column ----------------
    st.subheader("What the scores say")
    st.markdown("One line per metric output. *Avg* is over evaluated rows.")
    summary_rows = []
    for col in score_cols:
        series = scores[col]
        # boolean check FIRST: pandas treats bools as numeric, which made
        # refusal display as "avg 0.167" instead of "4 of 24 rows flagged"
        if series.notna().any() and set(series.dropna().unique()) <= {True, False}:
            flagged = int(series.fillna(False).sum())
            value = f"{flagged} of {series.notna().sum()} rows flagged"
        elif pd.api.types.is_numeric_dtype(series) and series.notna().any():
            value = f"avg {series.mean():.3f} (min {series.min():.3f}, max {series.max():.3f})"
        else:
            top = series.dropna().astype(str).replace("", pd.NA).dropna().value_counts()
            value = ", ".join(f"{v}: {c}" for v, c in top.head(3).items()) if len(top) else "-"
        summary_rows.append({
            "metric output": col,
            "result": value,
            "what it means": METRIC_HELP.get(col, ""),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True,
                 height=min(420, 42 + 35 * len(summary_rows)))

    # ---------------- full table + one chart on demand ----------------
    with st.expander("Full row-by-row score table"):
        st.dataframe(scores[input_cols + score_cols], use_container_width=True, height=380)

    numeric_scores = [c for c in score_cols if pd.api.types.is_numeric_dtype(scores[c])
                      and scores[c].notna().any()]
    if numeric_scores:
        with st.expander("Distribution chart (pick one metric)"):
            pick = st.selectbox("Metric column", numeric_scores)
            if METRIC_HELP.get(pick):
                st.caption(METRIC_HELP[pick])
            st.bar_chart(scores[pick].value_counts(
                bins=10 if scores[pick].nunique() > 10 else None).sort_index())

    # ---------------- export ----------------
    # ---------------- fairness (LLM outputs across segments) ----------------
    st.subheader("Fairness")
    tab_par, tab_cf, tab_rfs, tab_sba, tab_bench = st.tabs(
        ["Segment parity", "Counterfactual", "Retrieval fairness",
         "Bias amplification", "Benchmark battery"])
    with tab_par:
        st.markdown(
            "Compares every score column across the groups of a segment "
            "column (e.g. gender, age band). Large gaps mean the "
            "application behaves differently for different groups - "
            "investigate before trusting aggregate scores."
        )
        from vedaeval.validation import _bias_columns
        from vedaeval.parity import segment_parity, parity_flags
        seg_candidates = _bias_columns(scores)
        if not seg_candidates:
            st.info("No segment columns found (looking for columns with "
                    "bias/gender/age/segment in the name). Add them to "
                    "your dataset to unlock parity analysis.")
        else:
            seg_pick = st.selectbox("Segment column", seg_candidates)
            with st.expander("Thresholds (policy dials - defaults are the documented conventions)"):
                t1, t2, t3, t4 = st.columns(4)
                d_thr = t1.number_input(
                    "Effect size flag (Cohen's d)", 0.1, 3.0, 0.8, 0.1,
                    help="0.8 = Cohen's 'large effect' convention. Lower = more sensitive.")
                r_thr = t2.number_input(
                    "Rate gap flag (points)", 0.01, 0.50, 0.10, 0.01,
                    help="Flag boolean-rate gaps above this many percentage points (0.10 = 10 points).")
                mrows = t3.number_input(
                    "Min rows per group", 2, 100, 5, 1,
                    help="Groups smaller than this are shown but never flagged (small-cell guard).")
                alpha_v = t4.number_input(
                    "Significance level (alpha)", 0.001, 0.20, 0.05, 0.005,
                    help="The 'significant' column uses p < alpha. With many score "
                         "columns consider alpha / number-of-columns (Bonferroni).")
            par_summary, par_detail = segment_parity(
                scores, seg_pick, score_cols, min_rows=int(mrows),
                d_threshold=float(d_thr), rate_gap_threshold=float(r_thr),
                alpha=float(alpha_v))
            if par_summary.empty:
                st.info("No comparable score columns for this run.")
            else:
                flags = parity_flags(par_summary, seg_pick)
                if flags:
                    for f in flags:
                        st.warning(f)
                else:
                    st.success("No parity flags at the standard thresholds "
                               "for this segment.")
                st.dataframe(par_summary, use_container_width=True)
                with st.expander("Per-segment detail"):
                    st.dataframe(par_detail, use_container_width=True)
    with tab_cf:
        st.markdown(
            "Create a **twin** of each row where only the demographic "
            "signal changes (husband -> wife, he -> she, gender label "
            "flipped), then compare pair by pair. Two different "
            "experiments - be clear which one you are running:\n\n"
            "- **Mode A - evaluator-bias check (instant):** re-scores the "
            "swapped TEXT with the same metrics. The model never runs, so "
            "any change means OUR METRICS treat demographics differently.\n"
            "- **Mode B - model-bias check (two-step):** download the "
            "swapped prompts, have your LLM answer them offline, then "
            "compare that twin dataset here. Changes then mean THE MODEL "
            "treats demographics differently."
        )
        from vedaeval.counterfactual import (
            DEFAULT_PAIRS, generate_swapped, counterfactual_compare, row_diffs)

        pairs_text = st.text_input(
            "Swap pairs (comma-separated a<->b; edit or add domain/name pairs)",
            ", ".join(f"{a}<->{b}" for a, b in DEFAULT_PAIRS))
        pairs = []
        for chunk in pairs_text.split(","):
            if "<->" in chunk:
                a, b = chunk.split("<->", 1)
                if a.strip() and b.strip():
                    pairs.append((a.strip(), b.strip()))

        run_df_cf = st.session_state.canonical_df.drop(
            index=[i for i in st.session_state.excluded
                   if i in st.session_state.canonical_df.index])
        twin = generate_swapped(run_df_cf, pairs)

        with st.expander("Preview the swapped twin (review before trusting)"):
            prev = row_diffs(run_df_cf, twin, ["request", "response"], max_rows=50)
            if prev.empty:
                st.info("No rows changed - your data may not contain the "
                        "swap terms. Edit the pairs above (e.g. add name "
                        "pairs like rajesh<->jennifer).")
            else:
                st.dataframe(prev, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Mode A - run the evaluator-bias check now**")
            if st.button("Re-score swapped text with the same metrics"):
                with st.spinner("Scoring the twin..."):
                    from vedaeval.engine import run_evaluation as _run
                    twin_res = _run(twin, st.session_state.selected_metrics,
                                    st.session_state.configs)
                cmp_df = counterfactual_compare(scores, twin_res.scores,
                                                score_cols)
                st.markdown("Interpretation: near-zero diffs and flip rates "
                            "= the metrics pass their own fairness test.")
                st.dataframe(cmp_df, use_container_width=True)
                verdicts = cmp_df[cmp_df["type"] == "verdict"]
                for _, vr in verdicts.iterrows():
                    if vr["flip rate"] and vr["flip rate"] > 0:
                        st.warning(f"'{vr['score column']}' flipped on "
                                   f"{vr['flips']} rows ({vr['flip rate']:.0%}) "
                                   f"purely from the demographic swap.")
        with colB:
            st.markdown("**Mode B - test the model (two-step)**")
            st.download_button(
                "1) Download swapped prompts (answer them with your LLM)",
                twin.drop(columns=[c for c in ("response",) if c in twin.columns])
                    .to_csv(index=False).encode(),
                "swapped_prompts.csv", "text/csv")
            up_twin = st.file_uploader(
                "2) Upload the answered twin (same rows, same order)",
                type=["csv"], key="cf_twin")
            if up_twin is not None:
                try:
                    twin_answered = pd.read_csv(up_twin)
                    with st.spinner("Scoring the answered twin..."):
                        from vedaeval.engine import run_evaluation as _run
                        twin_res2 = _run(twin_answered,
                                         st.session_state.selected_metrics,
                                         st.session_state.configs)
                    cmp2 = counterfactual_compare(scores, twin_res2.scores,
                                                  score_cols)
                    st.dataframe(cmp2, use_container_width=True)
                    st.caption("Gaps/flips here reflect the MODEL's response "
                               "to the demographic change (plus any metric "
                               "noise - run Mode A first as the control).")
                except Exception as exc:
                    st.error(f"Could not process the twin: {exc}")
    with tab_rfs:
        st.markdown(
            "**Retrieval Fairness Score (RFS)** - does the RETRIEVER serve "
            "different-quality context across segments? Fairness upstream "
            "of the answer: the model can be innocent while the retrieval "
            "stage discriminates. Needs a per-row context-quality signal."
        )
        from vedaeval.novel import quality_from_ratings, retrieval_fairness
        from vedaeval.validation import _bias_columns as _segcols
        if "context" not in scores.columns:
            st.info("This dataset has no context column - RFS applies to "
                    "RAG datasets.")
        else:
            q_options = ["(pick a quality column)"] + [
                c for c in scores.columns
                if c not in ("request", "response", "context", "ground_truth")]
            q_pick = st.selectbox(
                "Context-quality column (judge rating High/Medium/Low or any "
                "numeric score; e.g. a context_relevance column)", q_options)
            seg_rfs = st.selectbox("Segment column ",
                                   _segcols(scores) or ["(none found)"])
            if q_pick != "(pick a quality column)" and seg_rfs != "(none found)":
                q_series = scores[q_pick]
                if not pd.api.types.is_numeric_dtype(q_series):
                    q_series = quality_from_ratings(q_series)
                    st.caption("Ratings mapped High=1, Medium=0.5, Low=0.")
                rfs_sum, rfs_det, excl = retrieval_fairness(
                    scores, seg_rfs, q_series)
                if excl:
                    st.caption(f"{excl} rows without context excluded.")
                if rfs_sum.empty:
                    st.info("Not enough data per segment.")
                else:
                    r0 = rfs_sum.iloc[0]
                    if r0["flagged"]:
                        st.warning(
                            f"RFS FLAG: context quality differs across "
                            f"{seg_rfs} (effect size {r0['effect size']}, "
                            f"p {r0['p value']}). The retrieval stage is "
                            f"serving groups unequally - investigate before "
                            f"blaming the model.")
                    else:
                        st.success("No retrieval-fairness flag: context "
                                   "quality is comparable across segments.")
                    st.dataframe(rfs_sum, use_container_width=True)

    with tab_sba:
        st.markdown(
            "**Source Bias Amplification (SBA)** - does the response contain "
            "MORE bias-relevant content than the sources it was built from? "
            "Positive SBA = the model ADDS it (model problem). Negative or "
            "zero = the sources carried it (corpus problem). Separating the "
            "two changes what you fix."
        )
        from vedaeval.novel import source_bias_amplification, get_bias_scorer
        if "context" not in scores.columns:
            st.info("This dataset has no context column - SBA applies to "
                    "RAG/summarization datasets.")
        else:
            e1, e2, e3 = st.columns(3)
            engine = e1.selectbox("Bias property",
                                  ["identity_attack", "toxicity",
                                   "demo_lexicon"],
                                  help="identity_attack/toxicity = local "
                                       "detoxify model (heavy install). "
                                       "demo_lexicon = transparent wordlist "
                                       "for demos and tests.")
            sba_thr = e2.number_input("Flag threshold", 0.01, 1.0, 0.10, 0.01)
            sba_alpha = e3.number_input("Alpha ", 0.001, 0.20, 0.05, 0.005)
            _, eng_ok, eng_note = get_bias_scorer(engine)
            if not eng_ok:
                st.info(f"Engine unavailable here: {eng_note}. Use "
                        "demo_lexicon or install locally.")
            elif st.button("Compute SBA", type="primary"):
                with st.spinner("Scoring responses and their sources..."):
                    res, per_row = source_bias_amplification(
                        scores, engine=engine, threshold=float(sba_thr),
                        alpha=float(sba_alpha))
                m1, m2, m3 = st.columns(3)
                m1.metric("SBA (mean amplification)", res["sba"])
                m2.metric("p value", res["p value"])
                m3.metric("rows compared", res["n rows"])
                if res["flagged"]:
                    st.warning("FLAG: the model adds bias-relevant content "
                               "its sources do not contain (model problem).")
                elif res["sba"] is not None and res["sba"] <= 0:
                    st.success("Model cleared: responses carry no more than "
                               "their sources (any bias present is a corpus "
                               "problem).")
                else:
                    st.success("No amplification flag at these thresholds.")
                st.caption(f"Engine: {res['engine note']}.")
                if not per_row.empty:
                    with st.expander("Per-row amplification (largest first)"):
                        st.dataframe(per_row.sort_values(
                            "amplification", ascending=False),
                            use_container_width=True)

    with tab_bench:
        st.markdown(
            "Published bias benchmarks probe the MODEL with constructed "
            "items instead of observing your logs. This works like "
            "counterfactual Mode B: download a prompt pack, run it "
            "through your LLM offline (add a `response` column), upload "
            "the answered file here, and get the benchmark's scores. "
            "Built-in packs are compact demo sets - the scorers also "
            "accept the full official datasets in the same schema."
        )
        from vedaeval.battery import PACKS, prompt_pack, score_pack
        pack_key = st.selectbox(
            "Benchmark pack", list(PACKS.keys()),
            format_func=lambda k: PACKS[k][0])
        pack_df = prompt_pack(pack_key)
        st.caption(f"{len(pack_df)} items, axes: "
                   f"{', '.join(sorted(pack_df['axis'].unique()))}")
        st.download_button(
            "Download prompt pack (CSV)",
            pack_df.to_csv(index=False).encode(),
            file_name=f"battery_{pack_key}_prompts.csv", mime="text/csv")
        answered_file = st.file_uploader(
            "Upload the ANSWERED pack (same columns + response)",
            type=["csv"], key=f"battery_{pack_key}")
        if answered_file is not None:
            answered = pd.read_csv(answered_file)
            result = score_pack(pack_key, answered)
            if "error" in result:
                st.error(result["error"])
            else:
                per_axis = result.pop("per_axis", None)
                st.json(result)
                if per_axis:
                    st.markdown("**Per-axis breakdown**")
                    st.json(per_axis)
                if pack_key == "bbq_style":
                    st.caption(
                        "Read: disambig_accuracy should be HIGH (the "
                        "context states the answer); ambig_bias_rate "
                        "should be LOW (ambiguity answered with "
                        "stereotypes is the failure).")
                elif pack_key == "stereoset_style":
                    st.caption(
                        "Read: stereo_selection_rate near 0.5 = no "
                        "systematic preference, and refusals to "
                        "generalize are a good sign. Note this is the "
                        "response-based adaptation of a logprob metric.")
                elif pack_key == "discrimeval_style":
                    st.caption(
                        "Read: approval_gap near 0 and no paired flips. "
                        "A flip means the SAME scenario got a different "
                        "decision when only the demographic detail "
                        "changed.")

    # ---------------- save this run for later comparison ----------------
    st.subheader("Save this run")
    st.markdown(
        "Saving keeps the scores on this computer so you can compare runs "
        "later - e.g. model A vs model B on the same dataset, or before vs "
        "after a prompt change. Saved runs never leave your machine."
    )
    sv1, sv2 = st.columns([3, 1])
    run_name = sv1.text_input("Run name", "", placeholder="e.g. model-A baseline")
    if sv2.button("Save run", type="primary", disabled=not run_name.strip()):
        from vedaeval.runstore import save_run
        rid = save_run(scores, run_name.strip(), {
            "domain": st.session_state.domain,
            "task_type": st.session_state.task_type,
            "metrics": result.ran,
            "governance": st.session_state.get("governance"),
        })
        st.success(f"Saved as {rid}. Find it under '6. Compare runs'.")

    st.subheader("Export")
    csv_bytes = scores.to_csv(index=False).encode()
    st.download_button("Download scores (CSV)", csv_bytes, "vedaeval_scores.csv", "text/csv")
    audit = {
        "run_summary": result.summary(),
        "exclusions": st.session_state.exclusion_log,
        "domain": st.session_state.domain,
        "task_type": st.session_state.task_type,
        "rag": st.session_state.rag,
        "governance": st.session_state.get("governance"),
    }
    st.download_button("Download audit log (JSON)",
                       json.dumps(audit, indent=2, default=str).encode(),
                       "vedaeval_audit.json", "application/json")

# ---------------------------------------------------------------- step 6

elif st.session_state.step == 5:
    st.header("Step 6 - Compare runs")
    st.markdown(
        "The most common evaluation question is not *how good is my model?* "
        "but *is version B better than version A?* Pick two saved runs on "
        "the same dataset - each shared metric is shown side by side with "
        "the difference."
    )
    from vedaeval.runstore import list_runs, load_run, compare_runs

    runs = list_runs()
    if len(runs) < 2:
        st.info("You need at least two saved runs to compare. Run an "
                "evaluation and use 'Save this run' on the Results step "
                f"(saved so far: {len(runs)}).")
        st.stop()

    labels = {f"{r['name']}  ({r['saved_at']}, {r.get('rows', '?')} rows)": r["run_id"]
              for r in runs}
    keys = list(labels)
    ca, cb = st.columns(2)
    pick_a = ca.selectbox("Run A (baseline)", keys, index=min(1, len(keys) - 1))
    pick_b = cb.selectbox("Run B (candidate)", keys, index=0)
    if labels[pick_a] == labels[pick_b]:
        st.warning("Pick two different runs.")
        st.stop()

    df_a, df_b = load_run(labels[pick_a]), load_run(labels[pick_b])
    cmp_df = compare_runs(df_a, df_b, "Run A", "Run B")
    if cmp_df.empty:
        st.warning("These runs share no metric columns to compare.")
        st.stop()

    st.subheader("Side-by-side metrics")
    st.markdown("For numeric metrics, *delta* = Run B minus Run A: positive "
                "means B scored higher on that metric (whether higher is "
                "better depends on the metric - e.g. higher BLEU is better, "
                "higher toxicity is worse).")
    st.dataframe(cmp_df, use_container_width=True,
                 height=min(500, 42 + 35 * len(cmp_df)))

    num = cmp_df[cmp_df["type"] == "numeric"]
    if len(num):
        improved = int((num["delta (B - A)"] > 0).sum())
        st.caption(f"{improved} of {len(num)} numeric metrics are higher in "
                   f"Run B. Read direction per metric before concluding.")
