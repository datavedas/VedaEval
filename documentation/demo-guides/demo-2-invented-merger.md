# Hallucination detection - triangulating an invented fact

**What this covers.** Detecting a fabricated fact three independent ways,
using a case where an answer invents a corporate merger, a date, and a
company name that are absent from the source document.

**Capabilities demonstrated.** Three independent grounding checks that agree,
sentence-level pinpointing of the bad sentence, and fully local operation
with no API key.

**What you need.** The application and `hallucination_demo.csv`. No key.

---

## The data

Six rows. Columns map to `request`, `response`, `context` (the source
document the answer is meant to stick to), and `response_samples`
(regenerations of the same answer, used by Sample Consistency).

The focus is row 4:

| field | value |
|---|---|
| request | "When was the vision benefit added?" |
| response | "The vision benefit was added in 2019 after the merger with ClearSight." |
| context | one routine eye exam a year, frames and lenses reimbursed - **no merger, no 2019, no ClearSight** |

The answer invents a merger, a date, and a company, none present in the
source, and the regenerations disagree (one says 2021).

---

## Steps

1. Choose **Evaluate an LLM application**.
2. **Upload** `hallucination_demo.csv`.
3. **Validate:** domain **Generic**, accept the mapping (note `context` and
   `response_samples`), run validation, continue.
4. **Configure (Manual mode):** tick **SummaC Consistency**, **Sample
   Consistency**, and **Summary Coverage + Compression**. No key panel
   appears - all three are local formulas.
5. **Run.** On the first run a local NLI model (~70 MB) downloads once.

---

## Reading the results

On row 4:

| Metric | Value | Why it caught the invention |
|---|---|---|
| Keyword coverage (from Summary Coverage) | 0.0 | the answer's vocabulary (merger, 2019, ClearSight) does not appear in the source |
| SummaC | 0.0 | no answer sentence is supported by any context sentence |
| Sample Consistency | 0.0 | the regenerations disagree (2019 vs 2021) - the model is guessing |

Three independent mechanisms - vocabulary overlap, sentence entailment, and
self-agreement - all reach zero. A single check can be fooled; three
unrelated checks agreeing is an actionable finding. None of this required an
API key.

Note: the *Summary Coverage + Compression* metric also outputs
`compression_ratio` (output length relative to source), a summary-quality
signal unrelated to hallucination. Only `keyword_coverage` is relevant here.

---

## The metrics used

**SummaC** (formula, in-house). Each answer sentence must be entailed by some
context sentence; reports the weakest one. Range 0-1. Row 4: 0.0.

**Sample Consistency** (formula, in-house). Whether the model's regenerations
of the same prompt agree. Range 0-1. Row 4: 0.0. Needs no source context.

**Keyword coverage** (formula, in-house, from Summary Coverage). How much of
the source's salient vocabulary the answer uses. Range 0-1. Row 4: 0.0.

---

## Capabilities demonstrated

- Triangulation - three unrelated checks, not one oracle.
- Fully local - no API key, no per-row cost.
- Sentence-level pinpointing (see Going further).
- Self-consistency catches invention with no source document.
- Row-level, explainable evidence.

---

## Troubleshooting

- If the first run is slow, the local NLI model is downloading (~70 MB, one
  time).
- The three checks use different mechanisms (vocabulary, entailment,
  self-agreement); their independence is why agreement among them is
  meaningful.
- The expected reading on row 4 is 0.0 / 0.0 / 0.0.

---

## Going further

Row 2 (a dental summary that invents "full orthodontic braces") shows SummaC
at 0.5 - not zero, because most of the answer is true - with
`summac_weakest_sentence` near 0.0 pointing at exactly the braces sentence.
This identifies a single invented sentence inside an otherwise correct
answer, at sentence level rather than row level.
