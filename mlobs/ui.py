"""Streamlit page for the classic-ML observability add-on.

Rendered only when the mlobs folder exists; app.py imports this lazily
inside a try/except, so deleting the folder removes the feature cleanly.
"""

from __future__ import annotations

import pandas as pd

from mlobs import metrics as mlm


def render_ml_page(st, session):
    st.header("Classic ML observability")
    st.markdown(
        "Upload a scoring dataset: one row per prediction, with at least an "
        "**actual label** column and a **predicted label** column. A "
        "demographic column (e.g. gender) unlocks the fairness report."
    )
    up = st.file_uploader("Scoring dataset (CSV)", type=["csv"], key="ml_up")
    if up is not None:
        try:
            session.ml_df = pd.read_csv(up)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
    mdf = session.get("ml_df")
    if mdf is None:
        st.stop()

    st.success(f"Loaded {len(mdf)} rows x {len(mdf.columns)} columns.")
    st.dataframe(mdf.head(8), use_container_width=True)

    cols = list(mdf.columns)
    c1, c2, c3 = st.columns(3)
    yt = c1.selectbox("Actual label column", cols)
    yp = c2.selectbox("Predicted label column", cols,
                      index=min(1, len(cols) - 1))
    fav = c3.text_input("Favorable outcome value", "1",
                        help="The label that counts as the 'positive' outcome, "
                             "e.g. 1 = approved.")
    try:
        fav_cast = type(mdf[yt].dropna().iloc[0])(fav)
    except Exception:
        fav_cast = fav

    tab_fair, tab_drift, tab_deg = st.tabs(
        ["Fairness report", "Drift report", "Degradation report"])

    with tab_fair:
        st.markdown(
            "Compares outcomes across the groups of a sensitive attribute: "
            "selection rate, true/false positive rates, demographic parity "
            "difference, disparate impact ratio (four-fifths rule), equal "
            "opportunity difference, and average odds difference."
        )
        sens = st.selectbox("Sensitive attribute (e.g. gender)", ["(pick)"] + cols)
        if sens != "(pick)":
            try:
                rep = mlm.fairness_report(mdf, yt, yp, sens, favorable=fav_cast)
                st.dataframe(rep, use_container_width=True)
                flags = mlm.fairness_flags(rep)
                if flags:
                    for f in flags:
                        st.warning(f)
                else:
                    st.success("No fairness flags at the standard thresholds.")
            except Exception as exc:
                st.error(f"Could not compute fairness: {exc}")

    with tab_drift:
        st.markdown(
            "Compares this dataset's column distributions against a "
            "**baseline** file (an earlier sample of the same data). PSI "
            "under 0.10 = stable; 0.10-0.25 = moderate drift; above 0.25 = "
            "significant drift."
        )
        base_up = st.file_uploader("Baseline dataset (CSV, same columns)",
                                   type=["csv"], key="ml_base")
        if base_up is not None:
            try:
                bdf = pd.read_csv(base_up)
                common = [c for c in bdf.columns if c in mdf.columns]
                st.dataframe(mlm.drift_report(bdf, mdf, common),
                             use_container_width=True)
            except Exception as exc:
                st.error(f"Could not compute drift: {exc}")

    with tab_deg:
        st.markdown(
            "Compares model performance (accuracy, precision, recall, F1) "
            "between a **baseline** period and this dataset. A drop larger "
            "than 5 points is flagged as degradation."
        )
        base_up2 = st.file_uploader(
            "Baseline scored dataset (CSV with the same actual/predicted columns)",
            type=["csv"], key="ml_base2")
        if base_up2 is not None:
            try:
                bdf2 = pd.read_csv(base_up2)
                rep = mlm.degradation_report(bdf2, mdf, yt, yp, favorable=fav_cast)
                st.dataframe(rep, use_container_width=True)
                bad = rep[rep["degraded"]]
                if len(bad):
                    st.warning(f"Degradation detected in: {', '.join(bad['metric'])}")
                else:
                    st.success("No degradation beyond the 5-point threshold.")
            except Exception as exc:
                st.error(f"Could not compute degradation: {exc}")
