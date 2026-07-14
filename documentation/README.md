# VedaEval - Documentation

User-facing documentation for VedaEval, shipped with the application. Every
file in this folder is written to be shared publicly.

## Contents

### `demo-guides/`
Step-by-step guides for running representative evaluations end to end. Each
guide covers one scenario - what it shows, the data, the exact steps, how to
read the results, the metrics involved, and troubleshooting.

- **demo-1-french-question** - Relevance vs. instruction adherence: an answer
  that is correct but ignores a "reply in French" instruction; also compares
  two candidate answers.
- **demo-2-invented-merger** - Hallucination detection: a fabricated fact
  caught three independent ways, fully local (no key).
- **demo-3-jailbreak** - Jailbreak detection: separating attack detection from
  whether the refusal was the right response.
- **demo-4-poisoned-document** - Indirect prompt injection: an attack hidden
  in the retrieved context, not the user's prompt.
- **demo-5-counterfactual-fairness** - Counterfactual fairness: Mode A (is a
  metric biased?) and Mode B (is the model biased?), with an in-depth
  appendix on Mode A.
- **demo-6-segment-refusal-parity** - Segment and refusal parity: comparing
  how often the model declines across demographic groups.

## Planned

- **user-guide/** - the full user guide: the three ways to use VedaEval (app,
  Python package, REST API), the governed metric-selection framework, and the
  complete metric reference.
- **overview/** - a presentation / overview of the engine.

## Notes

- Guides assume the built-in sample datasets that ship in `sample_data/`.
