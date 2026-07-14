# Relevance vs. Instruction Adherence - the language-mismatch case

**What this covers.** How the engine separates *whether the question was
answered* from *whether the requested form was followed*, using a case where
an assistant is asked to reply in French and answers correctly - but in
English. It also compares two candidate answers head-to-head.

**Capabilities demonstrated.** Bring-your-own-key judge metrics, up-front
cost estimation, the "one metric, one question" design, A-vs-B comparison,
and a written reason attached to every judge verdict.

**What you need.** The application, the file `judge_qualities_demo.csv`, and
a judge API key (about 18 calls).

---

## The data

Six member-services questions and answers. Columns map to `request` (the
question), `response` (answer A), `response_b` (answer B, a second candidate
answer used only by Pairwise Win Rate), and `ground_truth`.

The focus is row 4:

| field | value |
|---|---|
| request | "Reply in French: what is the generic drug copay?" |
| response (A) | "The generic drug copay is 10 USD at retail pharmacies." *(English)* |
| response_b (B) | "Le copaiement pour les medicaments generiques est de 10 USD en pharmacie." *(French)* |

Answer A gives the correct copay but ignores the instruction to reply in
French. Answer B is correct and in French.

---

## Steps

1. On the start screen, choose **Evaluate an LLM application**. A six-step
   wizard appears (Upload, Validate, Configure, Run, Results, Compare).
2. **Upload** `judge_qualities_demo.csv`.
3. **Validate:** keep the domain **Generic**, accept the column auto-mapping,
   run validation, and continue. (On healthcare data this step also runs the
   PHI/PII screen; this file is clean.)
4. **Configure (Manual mode):** tick three metrics - **Answer Relevance**,
   **Instruction Adherence**, and **Pairwise Win Rate**. All three are judge
   metrics, so a provider/key panel appears; Pairwise uses the `response_b`
   column.
5. Enter the provider (for example anthropic), leave the model field blank
   for the default, and provide the API key. An estimate of about 18 calls
   (3 metrics x 6 rows) is shown before the run. The key is held for the
   session only - never saved, never logged.
6. **Run.** Results appear as one row per input, one column per metric.

---

## Reading the results

On row 4:

| Metric | Verdict | Reason (in the `_reason` column) |
|---|---|---|
| Answer Relevance | High | the copay question was answered; 10 USD is correct |
| Instruction Adherence | Low | the instruction to reply in French was ignored (answer A is in English) |
| Pairwise Win Rate | B wins | answer B is correct and in French, so it followed the instruction A missed |

The same row, in a single run, is both correct and non-compliant. A single
overall score would either call this a good answer or average the two into an
uninformative middle. Two named metrics instead state precisely what
happened - right answer, wrong form - each with its own reason. Pairwise then
identifies the better of the two candidate answers.

---

## The metrics used

**Answer Relevance** (judge, in-house). Does the response address the
substance of the question; format and language are out of scope. Reads
High / Medium / Low, with a reason. Row 4: High.

**Instruction Adherence** (judge, in-house). Were the explicit constraints
obeyed - format, length, language, exclusions. Reads High / Medium / Low.
Row 4: Low (the French instruction was ignored).

**Pairwise Win Rate** (judge, in-house). Given two answers to the same
question (A = `response`, B = `response_b`), which is better overall. The
judge sees them in a randomized order each row to guard against position
bias. Row 4: B.

---

## Capabilities demonstrated

- Judge metrics run on your own key; data and spend stay with you.
- The call count is shown before the run.
- One metric, one question - form and substance are scored separately.
- A-vs-B comparison selects the better of two answers.
- Every judge verdict ships a written reason.
- Results are row-level, not a single aggregate.

---

## Troubleshooting

- The expected reading on row 4 is Answer Relevance High, Instruction
  Adherence Low, Pairwise B.
- All three metrics require a key. Without one, they are unavailable.
- Judge verdicts are opinions: expect the same overall pattern, not identical
  numbers on every rerun. A decision should not rest on one judge, one row,
  or one run.

---

## Going further

Adding **Completeness** and **Coherence** and re-running surfaces a second
pattern on row 3 ("How do I file a claim, and what is the deadline?",
answered with only the deadline): Completeness reads Medium/Low - only one of
the two parts was answered - while Coherence stays High - well-formed but
incomplete. Two different findings, one sentence.
