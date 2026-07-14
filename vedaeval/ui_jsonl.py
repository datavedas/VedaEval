"""Streamlit section: JSONL intake file check (part of LLM flow Step 1).

When the uploaded dataset is a JSONL file, this renders the intake report
inline: status banner,
errors / warnings / passed checks, statistics cards, field coverage,
segment distributions, and a JSON structure sketch.
"""

from __future__ import annotations

import json

from vedaeval.jsonl_check import check_jsonl


def render_jsonl_report(st, raw_bytes: bytes, file_name: str):
    """Render the intake file check for an uploaded JSONL's raw bytes."""
    st.subheader("Dataset file check (JSONL intake report)")
    st.markdown(
        "Because this is a JSONL submission file, VedaEval first checks the "
        "**file itself**: format, required fields, coverage, segment "
        "balance, data quality. Row-level checks (duplicates, leakage) come "
        "next in Step 2, and the metrics after that."
    )
    strict = st.toggle("Strict mode (warnings become failures)", value=False,
                       key="jsonl_strict")

    report = check_jsonl(raw_bytes, file_name, strict=strict)

    # ---- status banner ----
    if report.passed:
        st.success(f"FILE CHECK PASSED - {file_name}")
    else:
        st.error(f"FILE CHECK FAILED - {file_name}")

    # ---- errors / warnings / passed checks ----
    if report.errors:
        st.markdown(f"**Errors ({len(report.errors)})**")
        for e in report.errors:
            st.error(e)
    if report.warnings:
        st.markdown(f"**Warnings ({len(report.warnings)})**")
        for w in report.warnings:
            st.warning(w)
    with st.expander(f"Passed checks ({len(report.info)})",
                     expanded=not report.errors):
        for i in report.info:
            st.write(f"- {i}")

    # ---- statistics cards ----
    s = report.stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total records", s.get("total_records", 0))
    c2.metric("File size", f"{s.get('file_size_bytes', 0) / 1024:.1f} KB")
    c3.metric("Unique intents", f"~{s.get('estimated_unique_intents', 0)}")
    c4.metric("Avg request len", f"{s.get('avg_request_length', 0):.0f} chars")
    c5.metric("Avg response len", f"{s.get('avg_response_length', 0):.0f} chars")

    # ---- field coverage ----
    if s.get("field_coverage"):
        st.markdown("**Field coverage**")
        rows = [{"field": f, "count": f"{d['count']}/{s.get('total_records', 0)}",
                 "percent": f"{d['percent']}%"}
                for f, d in s["field_coverage"].items()]
        st.table(rows)

    # ---- segment distributions ----
    g, a = s.get("gender_distribution"), s.get("age_distribution")
    if g or a:
        st.markdown("**Segment distribution**")
        d1, d2 = st.columns(2)
        if g:
            d1.markdown("Gender")
            d1.table([{"value": k, "records": v} for k, v in g.items()])
        if a:
            d2.markdown("Age band")
            d2.table([{"value": k, "records": v} for k, v in a.items()])

    # ---- fields + structure ----
    if s.get("all_fields"):
        st.markdown("**All fields present:** " +
                    ", ".join(f"`{f}`" for f in s["all_fields"]))
    if s.get("json_structure"):
        with st.expander("JSON structure (from first record)"):
            st.json(s["json_structure"])

    # ---- export ----
    payload = {
        "validation_status": "PASSED" if report.passed else "FAILED",
        "errors": report.errors, "warnings": report.warnings,
        "info": report.info, "statistics": report.stats,
    }
    st.download_button("Download file-check report (JSON)",
                       json.dumps(payload, indent=2, default=str).encode(),
                       f"{file_name}_report.json", "application/json")
    return report
