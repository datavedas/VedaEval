"""Run history: save, list, and load evaluation runs locally.

Every saved run = two files in the runs/ folder (which is gitignored -
run results are the user's data, never published):
    <run_id>.csv   - the full score table
    <run_id>.json  - metadata: name, timestamp, domain, task, metrics run

This enables the most common real evaluation question: "is version B
better than version A?" - compare two saved runs side by side.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "runs"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return s[:40] or "run"


def save_run(scores_df, name: str, meta: dict | None = None) -> str:
    """Save a run; returns its run_id."""
    RUNS_DIR.mkdir(exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{_slug(name)}"
    scores_df.to_csv(RUNS_DIR / f"{run_id}.csv", index=False)
    payload = {"run_id": run_id, "name": name,
               "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "rows": int(len(scores_df)), **(meta or {})}
    (RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return run_id


def list_runs() -> list[dict]:
    """Newest first."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for meta_file in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def load_run(run_id: str):
    import pandas as pd
    return pd.read_csv(RUNS_DIR / f"{run_id}.csv")


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare_runs(df_a, df_b, label_a: str = "Run A", label_b: str = "Run B"):
    """Side-by-side comparison of two score tables.

    Returns a DataFrame: one row per shared score column.
    Numeric columns -> mean A, mean B, delta (B - A).
    Categorical/boolean -> top-value counts A vs B.
    """
    import pandas as pd

    input_like = {"request", "response", "ground_truth", "context", "timestamp"}
    shared = [c for c in df_a.columns
              if c in df_b.columns and c not in input_like]
    rows = []
    for col in shared:
        a, b = df_a[col], df_b[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) \
                and a.notna().any() and b.notna().any():
            ma, mb = float(a.mean()), float(b.mean())
            rows.append({"metric": col, "type": "numeric",
                         label_a: round(ma, 4), label_b: round(mb, 4),
                         "delta (B - A)": round(mb - ma, 4)})
        else:
            va = a.dropna().astype(str).value_counts()
            vb = b.dropna().astype(str).value_counts()
            fmt = lambda v: ", ".join(f"{k}: {c}" for k, c in v.head(3).items()) or "-"
            rows.append({"metric": col, "type": "categorical",
                         label_a: fmt(va), label_b: fmt(vb),
                         "delta (B - A)": ""})
    return pd.DataFrame(rows)
