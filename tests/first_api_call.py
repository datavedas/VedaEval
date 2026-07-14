"""Your first API call.

Prerequisite: the API must be running. In ANOTHER terminal (with your
.venv active, in the VedaEval folder):

    pip install fastapi uvicorn
    uvicorn api:app --reload

Then run this script here:  python tests/first_api_call.py
It sends three letters to your own server and prints the replies.
"""

import requests

BASE = "http://localhost:8000"

print("1) Asking the server how it feels...")
# generous timeout: on a cold server the first request can be slow
r = requests.get(f"{BASE}/health", timeout=120)
print("   status code:", r.status_code)
avail = r.json()["evaluators"]
ready = [k for k, v in avail.items() if v["available"]]
print(f"   evaluators ready on this machine: {len(ready)} ->", ", ".join(sorted(ready)[:8]), "...")

print()
print("2) Sending a text to the sensitive-information guardrail...")
payload = {"data": {"input": "Call Jane at 312-555-7890 or jane.doe@example.com"}}
r = requests.post(f"{BASE}/v3/guardrails/sensitive-information",
                  json=payload, timeout=60)
print("   status code:", r.status_code)
for span in r.json()["fdl_sensitive_information_scores"]:
    print(f"   found {span['label']!r}: {span['text']!r} (confidence {span['score']})")

print()
print("3) Sending two rows to the batch evaluate endpoint...")
payload = {
    "rows": [
        {"request": "What is the copay?", "response": "The copay is 40 dollars.",
         "ground_truth": "40 dollars copay."},
        {"request": "Is the plan good?", "response": "It is a decent plan overall."},
    ],
    "metrics": ["token_count", "sentiment", "overlap"],
}
r = requests.post(f"{BASE}/v1/evaluate", json=payload, timeout=120)
print("   status code:", r.status_code)
body = r.json()
print("   metrics that ran:", body["ran"], "| skipped:", body["skipped"])
row0 = body["rows"][0]
scores_only = {k: v for k, v in row0.items()
               if k not in ("request", "response", "ground_truth")}
print("   row 1 scores:", scores_only)

print()
print("That was it: three requests to a server, three JSON replies.")
print("You have now called an API - and it was your own.")
