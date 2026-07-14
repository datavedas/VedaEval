"""VedaEval REST API - the engine's third face (programs, not humans).

Run locally:
    pip install fastapi uvicorn
    uvicorn api:app --reload
Then the API listens at http://localhost:8000 and interactive docs are
auto-generated at http://localhost:8000/docs (FastAPI writes those for
free - open it in a browser and you can try every endpoint).

Authentication: if the environment variable VEDAEVAL_API_TOKEN is set,
every request must carry the header "Authorization: Bearer <token>".
If it is not set (local learning mode), the API is open.

Endpoints (the first three mirror the request/response shapes of
commercial guardrail APIs, which is what makes provider swapping a
configuration change):
    POST /v3/guardrails/ftl-safety                  {"data": {"input": text}}
    POST /v3/guardrails/ftl-response-faithfulness   {"data": {"response":..., "context":...}}
    POST /v3/guardrails/sensitive-information       {"data": {"input": text}}
    POST /v1/evaluate      {"rows":[...], "metrics":[...], "configs":{...}}
    GET  /health           availability of every evaluator
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field

from vedaeval import api_core

app = FastAPI(
    title="VedaEval API",
    description="Open LLM evaluation engine - REST interface",
    version="0.3",
)


# ---------------------------------------------------------------------------
# auth (bearer token when VEDAEVAL_API_TOKEN is set)
# ---------------------------------------------------------------------------

def require_token(authorization: str | None = Header(default=None)):
    expected = os.environ.get("VEDAEVAL_API_TOKEN", "")
    if not expected:
        return  # open mode (local development)
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401,
                            detail="missing or invalid bearer token")


# ---------------------------------------------------------------------------
# request models (FastAPI validates incoming JSON against these)
# ---------------------------------------------------------------------------

class SafetyData(BaseModel):
    input: str


class SafetyRequest(BaseModel):
    data: SafetyData


class FaithfulnessData(BaseModel):
    response: str
    context: str


class FaithfulnessRequest(BaseModel):
    data: FaithfulnessData


class EvaluateRequest(BaseModel):
    rows: list[dict] = Field(..., description="one dict per interaction: "
                             "request, response, optional ground_truth/context")
    metrics: list[str] = Field(..., description="evaluator keys to run")
    configs: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "evaluators": api_core.availability()}


@app.post("/v3/guardrails/ftl-safety", dependencies=[Depends(require_token)])
def guardrail_safety(payload: SafetyRequest):
    try:
        return api_core.safety_scores(payload.data.input)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/v3/guardrails/ftl-response-faithfulness",
          dependencies=[Depends(require_token)])
def guardrail_faithfulness(payload: FaithfulnessRequest):
    try:
        return api_core.faithfulness_score(payload.data.response,
                                           payload.data.context)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/v3/guardrails/sensitive-information",
          dependencies=[Depends(require_token)])
def guardrail_pii(payload: SafetyRequest):
    return {"fdl_sensitive_information_scores":
            api_core.pii_spans(payload.data.input)}


@app.post("/v1/evaluate", dependencies=[Depends(require_token)])
def evaluate(payload: EvaluateRequest):
    if not payload.rows:
        raise HTTPException(status_code=422, detail="rows must not be empty")
    if not payload.metrics:
        raise HTTPException(status_code=422, detail="metrics must not be empty")
    return api_core.evaluate_rows(payload.rows, payload.metrics,
                                  payload.configs)
