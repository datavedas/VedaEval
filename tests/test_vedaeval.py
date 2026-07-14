"""Unit tests for the VedaEval engine, schema, and validation.

The classic-ML add-on has its own checks in mlobs/run_ml_checks.py -
nothing here imports it, so the mlobs folder stays deletable.
"""

import numpy as np
import pandas as pd
import pytest

from vedaeval.schema import auto_map_columns, apply_mapping, validate_required
from vedaeval.validation import validate_dataset
from vedaeval.engine import run_evaluation
from vedaeval.evaluators import REGISTRY, recommended_for


@pytest.fixture
def demo_df():
    import pathlib
    csv = pathlib.Path(__file__).parent.parent / "sample_data" / "qa_rag_demo.csv"
    raw = pd.read_csv(csv)
    mapping = auto_map_columns(list(raw.columns))
    return apply_mapping(raw, mapping)


# ---------------------------------------------------------------- schema

def test_auto_mapping_detects_aliases():
    mapping = auto_map_columns(["Question", "Answer", "Expected", "Documents", "timestamp"])
    assert mapping["request"] == "Question"
    assert mapping["response"] == "Answer"
    assert mapping["ground_truth"] == "Expected"
    assert mapping["context"] == "Documents"
    assert mapping["timestamp"] == "timestamp"
    assert validate_required(mapping) == []


def test_metadata_columns_preserved(demo_df):
    assert "Bias Gender" in demo_df.columns
    assert "Bias Age" in demo_df.columns


# ---------------------------------------------------------------- validation

def test_validation_finds_seeded_issues(demo_df):
    report = validate_dataset(demo_df, pii_engine="regex")
    checks = {i.check for i in report.issues}
    assert "exact_duplicates" in checks          # row 10 duplicates row 0
    assert "conflicting_duplicates" in checks    # specialist copay 40 vs 45
    assert "rag_leakage" in checks               # gold plan row
    assert "pii_email_address" in checks         # john.smith82@example.com
    assert "dataset_size" in checks              # 20 rows < 60
    assert report.n_rows == 20


def test_flagged_rows_are_indices(demo_df):
    report = validate_dataset(demo_df, pii_engine="regex")
    flagged = report.flagged_rows
    assert flagged and all(isinstance(i, int) for i in flagged)
    assert max(flagged) < len(demo_df)


# ---------------------------------------------------------------- engine

def test_engine_runs_deterministic_suite(demo_df):
    keys = ["textstat", "sentiment", "overlap", "token_count",
            "banned_keywords", "regex_match", "profanity"]
    result = run_evaluation(demo_df, keys, configs={
        "banned_keywords": {"keywords": ["guarantee", "lawsuit"]},
        "regex_match": {"pattern": r"\d+ (dollars|USD)"},
    })
    df = result.scores
    ran = set(result.ran)
    # these have no optional deps and must always run
    assert {"token_count", "banned_keywords", "regex_match"} <= ran
    assert "token_count_response" in df.columns
    assert df["token_count_response"].gt(0).all()
    # no evaluator may crash the run
    for key, reason in result.skipped.items():
        assert "failed" not in reason, f"{key}: {reason}"


def test_overlap_scores_bounded(demo_df):
    result = run_evaluation(demo_df, ["overlap"])
    if "overlap" in result.ran:
        vals = result.scores["rougeL"].dropna()
        assert ((vals >= 0) & (vals <= 1)).all()


def test_unknown_evaluator_is_skipped(demo_df):
    result = run_evaluation(demo_df, ["does_not_exist"])
    assert result.skipped["does_not_exist"] == "unknown evaluator"


def test_recommendations():
    rec = recommended_for("qa", "rai", rag=True)
    assert "overlap" in rec and "profanity" in rec


# ML observability tests moved to mlobs/run_ml_checks.py (isolated add-on).
