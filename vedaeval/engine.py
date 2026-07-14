"""Evaluation engine: run selected evaluators over a canonical dataframe."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vedaeval.evaluators import REGISTRY


@dataclass
class RunResult:
    scores: "object"                    # DataFrame: input columns + score columns
    ran: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)   # key -> reason
    timings: dict[str, float] = field(default_factory=dict)  # key -> seconds

    def summary(self) -> dict:
        return {"evaluators_run": self.ran, "skipped": self.skipped,
                "rows": int(len(self.scores)), "timings_sec": self.timings}


def run_evaluation(df, evaluator_keys: list[str],
                   configs: dict[str, dict] | None = None,
                   extra_evaluators: dict | None = None) -> RunResult:
    """Run each selected evaluator; join score columns onto a copy of df.

    extra_evaluators: session-scoped evaluators (e.g. user-built custom
    LLM judges) merged over the global registry for THIS run only - they
    are never registered globally, so they cannot leak between users on
    a shared server.

    Evaluators that are unavailable (missing dependency) or inapplicable
    (missing input columns) are skipped with a recorded reason, never
    a crash.
    """
    configs = configs or {}
    lookup = {**REGISTRY, **(extra_evaluators or {})}
    out = df.copy()
    result = RunResult(scores=out)

    for key in evaluator_keys:
        ev = lookup.get(key)
        if ev is None:
            result.skipped[key] = "unknown evaluator"
            continue
        ok, reason = ev.available()
        if not ok:
            result.skipped[key] = f"unavailable ({reason})"
            continue
        ok, reason = ev.applicable(df)
        if not ok:
            result.skipped[key] = f"not applicable ({reason})"
            continue
        # Judge-style evaluators need a credential; skip politely without one.
        # (Binding safety behavior: the key is passed per run in config only,
        # never stored - see project docs.)
        if ev.info.needs_llm and not (configs.get(key) or {}).get("api_key"):
            result.skipped[key] = "needs an LLM API key (none provided)"
            continue
        start = time.time()
        try:
            cols = ev.evaluate(df, configs.get(key))
        except Exception as exc:  # never let one metric kill the run
            result.skipped[key] = f"failed ({type(exc).__name__}: {exc})"
            continue
        for name, values in cols.items():
            out[name] = values
        result.ran.append(key)
        result.timings[key] = round(time.time() - start, 3)

    result.scores = out
    return result
