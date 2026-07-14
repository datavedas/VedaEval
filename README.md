# VedaEval

An open LLM evaluation engine: upload a dataset of LLM requests and
responses, validate it, run a configurable panel of evaluation metrics,
and get scores, charts, fairness/drift reports, and an audit log.

## Quick start (local)

```bash
# 1. Create and activate a virtual environment (one-time)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then click "load the built-in demo dataset" on Step 1 and walk the wizard.

## Run the tests

```bash
python tests/run_core_checks.py   # dependency-free smoke test
pytest tests/ -v                  # full suite (needs requirements installed)
```

## Structure

- `app.py` - Streamlit app: LLM evaluation wizard (Upload, Validate, Configure, Run, Results)
- `vedaeval/schema.py` - canonical fields + column auto-mapping
- `vedaeval/validation.py` - quality checks: duplicates, conflicts, RAG leakage, PII scan, segments
- `vedaeval/jsonl_check.py` - JSONL intake file check (report shown at upload)
- `vedaeval/engine.py` - evaluation runner
- `vedaeval/evaluators/` - metric registry (deterministic + model-backed + LLM-judge)
- `mlobs/` - OPTIONAL classic-ML observability add-on (drift, fairness, degradation).
  Fully isolated: delete this folder and the LLM application works unchanged
- `sample_data/` - synthetic demo datasets (CSV + JSONL) with seeded issues

## Roadmap

Phase 1 (done): deterministic metrics + validation + UI + ML observability.
Phase 2: model-based metrics - safety classification, faithfulness
detection, and LLM-as-a-judge scoring (bring-your-own API key).
Phase 3: REST API service and public hosting.
