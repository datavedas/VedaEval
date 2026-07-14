"""Standalone checks for the mlobs add-on (dependency-free).

Run:  python mlobs/run_ml_checks.py
These live INSIDE the mlobs folder on purpose: delete the folder and no
test elsewhere references it.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from mlobs import metrics as mlobs

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


rng = np.random.default_rng(7)

s = pd.Series(rng.normal(size=500))
check("psi identical ~ 0", abs(mlobs.psi_numeric(s, s)) < 1e-6)
check("psi shift detected",
      mlobs.psi_numeric(pd.Series(rng.normal(0, 1, 1000)),
                        pd.Series(rng.normal(1.5, 1, 1000))) > 0.25)
ks = mlobs.ks_statistic(pd.Series(rng.normal(0, 1, 300)),
                        pd.Series(rng.normal(3, 1, 300)))
check("ks in bounds", 0.5 < ks <= 1.0)

n = 200
gender = np.where(rng.random(n) < 0.5, "M", "F")
y_true = rng.integers(0, 2, n)
y_pred = np.where(gender == "M", 1, (rng.random(n) < 0.2).astype(int))
fdf = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "gender": gender})
rep = mlobs.fairness_report(fdf, "y_true", "y_pred", "gender")
check("fairness reference unique", rep["is_reference"].sum() == 1)
check("fairness flags fire on bias", len(mlobs.fairness_flags(rep)) > 0)

y = rng.integers(0, 2, 500)
good = np.where(rng.random(500) < 0.9, y, 1 - y)
bad = np.where(rng.random(500) < 0.7, y, 1 - y)
deg = mlobs.degradation_report(pd.DataFrame({"y": y, "p": good}),
                               pd.DataFrame({"y": y, "p": bad}), "y", "p")
acc = deg[deg.metric == "accuracy"].iloc[0]
check("degradation detected", bool(acc["degraded"]))

drift = mlobs.drift_report(pd.DataFrame({"a": rng.normal(0, 1, 300)}),
                           pd.DataFrame({"a": rng.normal(2, 1, 300)}))
check("drift table built", set(drift.columns) >= {"column", "psi", "assessment"})

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
