# Conversation compression benchmark

Run from the repository root after `python -m pip install -e ".[dev]"`:

```bash
python -m benchmarks.run
python -m benchmarks.run validate
```

The default command validates the dataset, calibrates compaction on the separate
development split using local fake models, then runs all 28 evaluation cases in
five modes. It makes no network requests and needs no credentials. The fake
downstream returns `{}`, so zero answer accuracy is expected. This run checks
the experiment machinery and supports no conclusions about model quality.
`validate` checks the dataset and split invariants without running comparisons.

| Mode | Context sent to the downstream model |
| --- | --- |
| `original` | Full conversation and final question |
| `hygiene` | `PromptHygiene()` output |
| `compact` | `Compactor` output |
| `verified` | `Compactor(..., verify=True)` output |
| `pipeline` | `PromptHygiene()` followed by `Compactor`, without verification |

Compaction can be `not_triggered`, `accepted`, `rejected`, or `error`. Rejected
summaries fall back to the compactor's input; for `pipeline`, this is the
already cleaned context. `keep_recent` includes the final question, which must
survive unchanged. One configured downstream model instance and its settings
are shared across modes; the summarizer is configured separately.

## Bounded real-model runs

Copy [config.example.json](config.example.json) to a local configuration, replace
both model IDs with IDs available to you, and keep credentials only in the
environment variables named by `api_key_env`. The example contains placeholders,
not model recommendations. Choose `provider: "openai"` for a compatible Chat
Completions API, or `provider: "anthropic"` with an appropriate environment
variable such as `ANTHROPIC_API_KEY`. Adjust settings to the chosen API. The
optional `base_url` must not contain credentials or query parameters.

After configuring the example file, these commands explicitly make real API
requests; they are separate from offline smoke:

```bash
python -m benchmarks.run calibrate-live --config benchmarks/config.example.json --limit 4 --output artifacts/calibration
python -m benchmarks.run live --config benchmarks/config.example.json --calibration artifacts/calibration/calibration.json --limit 3 --output artifacts/live
```

Use your copied configuration path in both commands if you keep the example
unchanged. Every real-model command requires a positive explicit `--limit` in
cases, not calls. It selects a deterministic prefix; small evaluation subsets
can omit categories, and the four-case development split covers only numbers,
exceptions, updated facts and task status.
Each evaluation case makes at most nine calls across the five modes: five
downstream answers, three summaries, and one verification. Calibration runs
three candidates across `compact`, `verified`, and `pipeline`, with at most
seven calls per development case per candidate. The limits above therefore
bound calibration to 84 calls and evaluation to 27 calls. These are call-count
bounds, not a dollar budget.

Live calibration uses four development cases disjoint from the 28 evaluation
cases. It chooses among the three parameter sets in `run.py`, requiring accepted
compaction and no errors, then ranking answer accuracy, lexical context recall,
and accepted-compaction count. The chosen parameters apply to all compaction
modes in the evaluation. Fake calibration checks this selection mechanism only.
Live evaluation requires a live calibration matching the dataset fingerprint,
code/scoring hash, model configurations, response limit and token counter.
Keep the complete configuration and
calibration artifact together to reproduce a run; use a new output directory for
each experiment because files in an existing directory are overwritten.

## Measurements and artifacts

The default output is `artifacts/benchmark/`:

- `results.jsonl` and `results.csv`: every case/mode, retained context and answer,
  reference answer, errors, recall, token counts, call records, and timings.
- `summary.json`: totals and breakdowns by mode and by mode/category.
- `metadata.json`: dataset version and SHA-256, scoring/code SHA-256, Git commit
  and dirty state, Python version, case IDs, configuration, and calibration.
- `report.md`: comparison tables and examples of quality regressions relative
  to correct full-history answers. All incorrect answers remain in the results.
- `development/calibration.json` and `development/candidate-{0,1,2}/`: selected
  parameters and complete candidate results, summaries, metadata, and reports.
  `calibrate-live` writes these directly under its `--output` directory instead.

Local context sizes use the configured counter and estimated message framing;
they are separate from billable provider usage. The default counter estimates
four characters per token. For a compatible tokenizer, install the optional
`tiktoken` extra and select `{"kind": "tiktoken", "encoding": "o200k_base"}`;
these counts still need not equal provider accounting.

Provider usage is captured by the transport and normalized into input, output,
cached-input, and cache-write-input tokens. Timings and costs include summary,
verification, and downstream calls, including discarded summaries and failed
calls. Missing provider usage remains `null`. Both example `prices` are `null`,
so costs remain unknown until you supply USD-per-million-token rates for
`input_per_million` and `output_per_million`, plus separate
`cached_input_per_million` and `cache_write_input_per_million` rates when those
tokens occur. Missing usage or required rates produce `cost_usd: null`, never a
free call; `known_cost_usd` is only the known subtotal. Set a rate to zero only
when it actually is free. No API keys, authorization headers, or raw provider
response envelopes are logged; results intentionally include benchmark contexts
and answers. Errors retain their exception type rather than provider text.

Usage fields follow the [OpenAI usage schema](https://github.com/openai/openai-python/blob/main/src/openai/types/completion_usage.py)
and [Anthropic cache accounting](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
Cache-write pricing is one configured effective rate; configure it for the cache
lifetime used by your provider. This harness does not enable tools or retries.

## Dataset and interpretation

`data/cases.jsonl` contains evaluation inputs, while `data/labels.jsonl` contains
the separate reference answers, required facts, and forbidden claims. The
development split uses `dev_cases.jsonl` and `dev_labels.jsonl`. Labels are used
only for scoring, never inserted into model prompts. Evaluation has 14 Russian
and 14 English cases, four per category: `numbers_units`,
`negations_exceptions`, `updated_facts`, `entity_ids`, `sources`, `code_json`,
and `task_status`. Each synthetic history has six messages: two establish the
target facts and four provide unrelated conversational distractors, followed by
the final question. Validation checks that required facts are in the older
history eligible for summarization.

Answer correctness requires the exact annotated JSON object with string values;
required-fact recall counts exact field/value matches. Invalid JSON, extra
fields, changed units, or a semantically similar spelling can fail correctness.
Forbidden-claim counts cover only the annotated claims. Context recall uses
lexical regular expressions and can miss correct paraphrases or match text
without proving its meaning. Full-history answers are a comparison baseline;
the separate annotations are the gold answers.

This small synthetic dataset tests specific retention failures rather than
production quality. Read individual failures and compare quality, context size,
latency, and complete call cost together. Provider behavior and outputs can vary
between runs even with temperature zero. No paid run or model-quality result is
included by the offline smoke.

## Summary budget and instruction study

`python -m benchmarks.study` runs an offline development smoke. This separate
study compares the existing summary instructions with an explicitly structured
record at 256, 512 and 1024 completion tokens. It uses ordinary
`Compactor.optimize()` without verification. The structured treatment adds
organization and explicit entity/value/source/status bindings, so this is an
instruction-package comparison, not an isolated formatting experiment.

Every case runs one original-history baseline and six fixed variants: at most
13 model calls per case, without retries. Variants share the downstream model,
answer settings, threshold, recent-message count and token counter. Treatment
order rotates by case; the exact order is recorded. Fallbacks and errors stay
in the results and their calls remain charged. Do not interpret high accuracy
from rejected compactions as evidence of information-preserving summaries.

Real requests require explicit commands and case limits:

```bash
python -m benchmarks.study dev-live --config benchmarks/config.minimax.example.json --limit 4 --output artifacts/study-dev
python -m benchmarks.study eval-live --config benchmarks/config.minimax.example.json --calibration artifacts/study-dev/calibration.json --limit 28 --output artifacts/study-eval
```

Set `MINIMAX_API_KEY` in the process environment; never put it in a configuration
file or command argument. The example requires the optional `tiktoken` extra.
Its `o200k_base` counts are a consistent local size proxy, **not MiniMax's
tokenizer or billable usage**. Summary budgets cap requested completions and
can include thinking; they do not promise a visible summary length. Diagnostics
record the returned model, finish reason and reasoning-token usage when present.
`reasoning_effort` is an optional OpenAI-compatible model setting; check the
chosen provider's supported levels before using it.

The MiniMax example uses `reasoning_effort: low`. MiniMax-M3.1-Flash always
thinks and currently ignores temperature (effective temperature 1), so repeated
runs can differ. See the official [Chat Completions reference](https://platform.minimax.io/docs/api-reference/text-chat-openai).
Example rates are the [published PAYG rates](https://platform.minimax.io/docs/pricing/overview)
checked on 2026-11-24: USD 0.15 input / 0.60 output / 0.015 cached input per
million tokens. With a subscription key, reported USD is a **PAYG-equivalent
estimate**, not the subscription invoice or incremental cash spend. Replace
rates for your account; absent usage/rates remain unknown.

Development selects an eligible variant; evaluation requires the exact matching
code/data/settings manifest and carries that selection unchanged. All six
predeclared variants run on evaluation to describe the held-out curve; do not
select a new winner from evaluation. Four development cases and 28 synthetic
evaluation cases support descriptive results only, not statistical significance
or production-quality guarantees.

Outputs include the existing JSONL/CSV, metadata and report plus a development
`calibration.json` or evaluation `study.json`. `checkpoint.jsonl` is flushed
after each completed scenario. Use a new output directory for every run;
existing study checkpoints are rejected to avoid accidental duplicate calls.
