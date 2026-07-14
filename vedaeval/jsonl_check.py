"""JSONL dataset file checker.

Standalone intake validation for JSONL submission files: is the FILE
well-formed and complete enough to be worth evaluating? This is distinct
from the row-level validation inside the LLM wizard (duplicates, leakage)
and from the metrics engine: this checks the file you were handed.

Check set: required fields, flexible age field, coverage
minimums for ground_truth/context (string OR list), record-count floors,
segment sizes, diversity, unique intents, data quality, unrecognized
fields, and a JSON structure sketch of the first record.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

REQUIRED_FIELDS = ["request", "response"]
HIGH_COVERAGE_FIELDS = ["ground_truth", "context"]
RECOGNIZED_FIELDS = {
    "request", "response", "ground_truth", "context", "bias_gender",
    "bias_age", "bias_age_band", "timestamp", "request_id", "topic",
    "model", "rationale", "query_intent", "post_processed_output",
    "reference_documents",
}
MIN_RECORDS = 20
RECOMMENDED_RECORDS = 100
MIN_COVERAGE_PERCENT = 80
MIN_SEGMENT_SIZE = 10
MIN_UNIQUE_INTENTS = 10


@dataclass
class JsonlReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str, strict: bool):
        if strict:
            self.errors.append(msg)
            self.passed = False
        else:
            self.warnings.append(msg)


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _has_content(record: dict, fld: str) -> bool:
    v = record.get(fld)
    if isinstance(v, list):
        return len(v) > 0
    if isinstance(v, str):
        return len(v.strip()) > 0
    return v is not None and v != {}


def _analyze_structure(obj: Any, depth: int = 0, max_depth: int = 5) -> dict:
    if depth > max_depth:
        return {"type": "...", "note": "max depth reached"}
    if obj is None:
        return {"type": "null"}
    if isinstance(obj, bool):
        return {"type": "boolean"}
    if isinstance(obj, int):
        return {"type": "integer"}
    if isinstance(obj, float):
        return {"type": "float"}
    if isinstance(obj, str):
        return {"type": "string", "sample_length": len(obj)}
    if isinstance(obj, list):
        if not obj:
            return {"type": "array", "items": "empty"}
        item_types = set()
        sample = None
        for item in obj[:3]:
            a = _analyze_structure(item, depth + 1, max_depth)
            item_types.add(a.get("type", "unknown"))
            if sample is None:
                sample = a
        return {"type": "array", "length": len(obj),
                "item_types": sorted(item_types), "sample_item": sample}
    if isinstance(obj, dict):
        return {"type": "object",
                "properties": {k: _analyze_structure(v, depth + 1, max_depth)
                               for k, v in obj.items()}}
    return {"type": type(obj).__name__}


def check_jsonl(raw_bytes: bytes, file_name: str = "dataset.jsonl",
                strict: bool = False) -> JsonlReport:
    """Run the full intake check on raw JSONL file bytes."""
    rep = JsonlReport()
    rep.stats["file_name"] = file_name
    rep.stats["file_size_bytes"] = len(raw_bytes)

    # ---- format ----
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
            rep.warnings.append("File is not UTF-8; decoded as latin-1.")
        except Exception as exc:
            rep.errors.append(f"Could not decode file: {exc}")
            rep.passed = False
            return rep

    records: list[dict] = []
    for line_num, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            rep.warnings.append(f"Empty line at line {line_num}")
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            rep.errors.append(f"Line {line_num}: invalid JSON ({exc})")
            rep.passed = False
            return rep
        if not isinstance(obj, dict):
            rep.errors.append(f"Line {line_num}: record is not a JSON object")
            rep.passed = False
            return rep
        records.append(obj)

    rep.info.append(f"Valid JSONL format with {len(records)} records")
    rep.stats["total_records"] = len(records)
    if not records:
        rep.errors.append("File contains no records.")
        rep.passed = False
        return rep

    # ---- record count (volume issue -> warning unless strict) ----
    n = len(records)
    if n < MIN_RECORDS:
        rep.fail(f"Insufficient records: {n} (minimum: {MIN_RECORDS})", strict)
    elif n < RECOMMENDED_RECORDS:
        rep.fail(f"Record count below recommended: {n} "
                 f"(recommended: {RECOMMENDED_RECORDS})", strict)
    else:
        rep.info.append(f"Record count: {n} (meets recommendations)")

    # ---- required fields ----
    missing_required: dict[str, list[int]] = defaultdict(list)
    missing_gender: list[int] = []
    for idx, record in enumerate(records, 1):
        for fld in REQUIRED_FIELDS:
            if fld not in record or not record[fld]:
                missing_required[fld].append(idx)
        if not record.get("bias_gender"):
            missing_gender.append(idx)
    if missing_required:
        for fld, rows in missing_required.items():
            rep.errors.append(
                f"Missing required field '{fld}' in {len(rows)} records "
                f"(e.g. lines: {rows[:5]})")
        rep.passed = False
    else:
        rep.info.append(f"All records have required fields: "
                        f"{', '.join(REQUIRED_FIELDS)}")
    if missing_gender:
        rep.fail(f"Missing 'bias_gender' in {len(missing_gender)} records "
                 f"(e.g. lines: {missing_gender[:5]})", strict)
    else:
        rep.info.append("All records have 'bias_gender' field")

    # ---- age field (flexible: bias_age_band preferred) ----
    has_band = any(r.get("bias_age_band") for r in records)
    has_age = any(r.get("bias_age") for r in records)
    if has_band:
        age_field = "bias_age_band"
        if has_age:
            rep.info.append("Both 'bias_age' and 'bias_age_band' present; "
                            "using 'bias_age_band'.")
    elif has_age:
        age_field = "bias_age"
        rep.warnings.append("Found 'bias_age' but expected 'bias_age_band'; "
                            "consider renaming.")
    else:
        age_field = None
        rep.fail("Missing age field: need 'bias_age' or 'bias_age_band'", strict)
    if age_field:
        rep.info.append(f"Using age field: '{age_field}'")

    # ---- coverage (context counts string OR non-empty list) ----
    coverage = {}
    for fld in HIGH_COVERAGE_FIELDS:
        count = sum(1 for r in records if _has_content(r, fld))
        pct = count / n * 100
        coverage[fld] = {"count": count, "percent": round(pct, 1)}
        if pct < MIN_COVERAGE_PERCENT:
            rep.fail(f"Low coverage for '{fld}': {pct:.1f}% "
                     f"(minimum: {MIN_COVERAGE_PERCENT}%)", strict)
        else:
            rep.info.append(f"Good coverage for '{fld}': {pct:.1f}%")
    rep.stats["field_coverage"] = coverage

    # ---- segment distribution + diversity ----
    gender_dist: Counter = Counter()
    age_dist: Counter = Counter()
    for r in records:
        gender_dist[str(r.get("bias_gender") or "Unknown")] += 1
        age_dist[str(r.get(age_field) or "Unknown") if age_field else "Unknown"] += 1
    rep.stats["gender_distribution"] = dict(gender_dist)
    rep.stats["age_distribution"] = dict(age_dist)

    for name, dist in (("gender", gender_dist), ("age band", age_dist)):
        smalls = {v: c for v, c in dist.items()
                  if v != "Unknown" and c < MIN_SEGMENT_SIZE}
        for v, c in smalls.items():
            rep.fail(f"Insufficient data for {name} '{v}': {c} records "
                     f"(minimum: {MIN_SEGMENT_SIZE})", strict)
        if not smalls and any(v != "Unknown" for v in dist):
            rep.info.append(f"All {name} segments meet minimum size")
        distinct = {v for v in dist if v != "Unknown"}
        if len(distinct) < 2:
            rep.fail(f"Insufficient {name} diversity: only {len(distinct)} "
                     f"distinct value(s); at least 2 recommended for "
                     f"balanced bias evaluation", strict)
        else:
            rep.info.append(f"Good {name} diversity: {len(distinct)} distinct values")

    # ---- unique intents ----
    has_intent = any("query_intent" in r for r in records)
    intents = set()
    if has_intent:
        for r in records:
            v = r.get("query_intent")
            if v and v != "Unknown":
                intents.add(str(v).lower().strip())
        rep.info.append(f"Using explicit 'query_intent' field: "
                        f"{len(intents)} unique intents")
    else:
        for r in records:
            intents.add(_as_str(r.get("request", "")).lower().strip()[:50])
        rep.info.append(f"Using 50-character heuristic for intent detection "
                        f"(~{len(intents)} unique). For accuracy, add a "
                        f"'query_intent' field.")
    rep.stats["estimated_unique_intents"] = len(intents)
    if len(intents) < MIN_UNIQUE_INTENTS:
        rep.fail(f"Insufficient unique intents: ~{len(intents)} "
                 f"(minimum: {MIN_UNIQUE_INTENTS})", strict)
    else:
        rep.info.append(f"Sufficient unique intents: ~{len(intents)}")

    # ---- data quality ----
    empty_req, empty_resp, short_req, short_resp = [], [], [], []
    structured_responses = 0
    for idx, r in enumerate(records, 1):
        req = _as_str(r.get("request", ""))
        resp_raw = r.get("response", "")
        resp = _as_str(resp_raw)
        is_structured = isinstance(resp_raw, (dict, list))
        if is_structured:
            structured_responses += 1
        if not req.strip():
            empty_req.append(idx)
        elif len(req.strip()) < 10:
            short_req.append(idx)
        if not resp.strip():
            empty_resp.append(idx)
        elif len(resp.strip()) < (30 if is_structured else 10):
            short_resp.append(idx)
    if empty_req:
        rep.errors.append(f"{len(empty_req)} records with empty requests")
        rep.passed = False
    if empty_resp:
        rep.errors.append(f"{len(empty_resp)} records with empty responses")
        rep.passed = False
    if short_req:
        rep.warnings.append(f"{len(short_req)} records with very short "
                            f"requests (<10 chars)")
    if short_resp:
        rep.warnings.append(f"{len(short_resp)} records with very short responses")
    if not empty_req and not empty_resp:
        rep.info.append("All records have non-empty request and response")
    if structured_responses:
        rep.info.append(f"Structured data detected: {structured_responses}/{n} "
                        f"responses are dict/list objects (valid)")

    # ---- fields inventory ----
    all_fields: set[str] = set()
    for r in records:
        all_fields.update(r.keys())
    unrecognized = sorted(all_fields - RECOGNIZED_FIELDS)
    rep.stats["all_fields"] = sorted(all_fields)
    if unrecognized:
        rep.info.append(f"Unrecognized (extra) fields present: "
                        f"{', '.join(unrecognized)} - extra fields are fine")

    # ---- averages + structure sketch ----
    rep.stats["avg_request_length"] = round(
        sum(len(_as_str(r.get("request", ""))) for r in records) / n, 1)
    rep.stats["avg_response_length"] = round(
        sum(len(_as_str(r.get("response", ""))) for r in records) / n, 1)
    rep.stats["json_structure"] = _analyze_structure(records[0])

    return rep
