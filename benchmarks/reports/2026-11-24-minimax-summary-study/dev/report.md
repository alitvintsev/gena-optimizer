# Conversation compression benchmark

Real model results. USD is a published PAYG-equivalent estimate, not subscription cash spend.

Dataset: `synthetic-ru-en-v1` / `f2708051f1dab34d4f38a5dfb70b5e8f50d057897b6c9ccc82b8e553672e5542`.
Commit: `1b7b7ba46c1e23007ae5ad6ca3e5e0ac809c95e1`; dirty: `False`.
Code/scoring SHA256: `50549dcb34c8445da113828192410cf8851ba815e1ea3d5668caf4b309acf116`.
Compaction: `{"trigger_tokens": 200, "keep_recent": 2, "summary_max_tokens": 256}`.
Local counter: `{"kind": "tiktoken", "encoding": "o200k_base"}`.
Models and response settings: `{"answer_max_tokens": 512, "token_counter": {"kind": "tiktoken", "encoding": "o200k_base"}, "summarizer": {"provider": "openai", "model": "MiniMax-M3.1-Flash", "base_url": "https://api.minimax.io/v1", "api_key_env": "MINIMAX_API_KEY", "temperature": 1, "reasoning_effort": "low", "timeout": 60, "prices": {"input_per_million": 0.15, "output_per_million": 0.6, "cached_input_per_million": 0.015}}, "downstream": {"provider": "openai", "model": "MiniMax-M3.1-Flash", "base_url": "https://api.minimax.io/v1", "api_key_env": "MINIMAX_API_KEY", "temperature": 1, "reasoning_effort": "low", "timeout": 60, "prices": {"input_per_million": 0.15, "output_per_million": 0.6, "cached_input_per_million": 0.015}}}`.
Settings, models, prices and counter: [metadata.json](metadata.json).

Unknown usage/cost is not zero. All calls, including rejected compaction, are charged.
Context fact recall uses lexical patterns and can miss valid paraphrases.

A/R/N = compaction accepted / rejected / not triggered.
Forbidden shows detected claims / answers that could not be checked.
| Mode/category | n | Correct | Facts | Forbidden/? | Errors | A/R/N | Context | Calls | USD |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| original | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 2639 → 2639 | 4 | 0.000308 |
| prose-1024 | 4 | 100.0% | 100.0% | 0/0 | 0 | 2/2/0 | 2639 → 2541 | 8 | 0.002253 |
| prose-256 | 4 | 50.0% | 62.5% | 0/0 | 0 | 4/0/0 | 2639 → 1629 | 8 | 0.001643 |
| prose-512 | 4 | 50.0% | 83.3% | 0/0 | 0 | 2/2/0 | 2639 → 2405 | 8 | 0.001837 |
| structured-1024 | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2639 → 2639 | 8 | 0.002484 |
| structured-256 | 4 | 100.0% | 100.0% | 0/0 | 0 | 4/0/0 | 2639 → 1590 | 8 | 0.001593 |
| structured-512 | 4 | 75.0% | 91.7% | 0/0 | 0 | 2/2/0 | 2639 → 2450 | 8 | 0.001955 |
| original/negations_exceptions | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 592 → 592 | 1 | 0.000046 |
| original/numbers_units | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 732 → 732 | 1 | 0.000183 |
| original/task_status | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 601 → 601 | 1 | 0.000026 |
| original/updated_facts | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 714 → 714 | 1 | 0.000053 |
| prose-1024/negations_exceptions | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 592 → 592 | 2 | 0.000354 |
| prose-1024/numbers_units | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 732 → 677 | 2 | 0.000665 |
| prose-1024/task_status | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 601 → 601 | 2 | 0.000533 |
| prose-1024/updated_facts | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 714 → 671 | 2 | 0.000700 |
| prose-256/negations_exceptions | 1 | 0.0% | 50.0% | 0/0 | 0 | 1/0/0 | 592 → 428 | 2 | 0.000427 |
| prose-256/numbers_units | 1 | 0.0% | 0.0% | 0/0 | 0 | 1/0/0 | 732 → 407 | 2 | 0.000689 |
| prose-256/task_status | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 601 → 413 | 2 | 0.000265 |
| prose-256/updated_facts | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 714 → 381 | 2 | 0.000262 |
| prose-512/negations_exceptions | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 592 → 592 | 2 | 0.000452 |
| prose-512/numbers_units | 1 | 0.0% | 66.7% | 0/0 | 0 | 1/0/0 | 732 → 629 | 2 | 0.000509 |
| prose-512/task_status | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 601 → 601 | 2 | 0.000313 |
| prose-512/updated_facts | 1 | 0.0% | 66.7% | 0/0 | 0 | 1/0/0 | 714 → 583 | 2 | 0.000562 |
| structured-1024/negations_exceptions | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 592 → 592 | 2 | 0.000535 |
| structured-1024/numbers_units | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 732 → 732 | 2 | 0.000641 |
| structured-1024/task_status | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 601 → 601 | 2 | 0.000547 |
| structured-1024/updated_facts | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 714 → 714 | 2 | 0.000760 |
| structured-256/negations_exceptions | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 592 → 393 | 2 | 0.000365 |
| structured-256/numbers_units | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 732 → 420 | 2 | 0.000436 |
| structured-256/task_status | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 601 → 385 | 2 | 0.000379 |
| structured-256/updated_facts | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 714 → 392 | 2 | 0.000412 |
| structured-512/negations_exceptions | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 592 → 592 | 2 | 0.000376 |
| structured-512/numbers_units | 1 | 100.0% | 100.0% | 0/0 | 0 | 1/0/0 | 732 → 652 | 2 | 0.000502 |
| structured-512/task_status | 1 | 100.0% | 100.0% | 0/0 | 0 | 0/1/0 | 601 → 601 | 2 | 0.000483 |
| structured-512/updated_facts | 1 | 0.0% | 66.7% | 0/0 | 0 | 1/0/0 | 714 → 605 | 2 | 0.000595 |

## Quality regressions

- `dev-numbers_units-ru-01` / `prose-256`: expected `{"length": "18,40 см", "mass": "3,05 г", "conditioning_temperature": "22 °C"}`; observed `"{\n  \"length\": \"\",\n  \"mass\": \"\",\n  \"conditioning_temperature\": \"\",\n  \"note\": \"В материалах беседы нет параметров для образца «Ольха» (ни с суффиксом, ни без него) — сведения о длине, массе и температуре кондиционирования не приводились, поэтому значения не заполнены.\"\n}"`.
- `dev-numbers_units-ru-01` / `prose-512`: expected `{"length": "18,40 см", "mass": "3,05 г", "conditioning_temperature": "22 °C"}`; observed `"{\"length\":\"18,40 см\",\"mass\":\"3,05 г\",\"conditioning_temperature\":\"\"}"`.
- `dev-negations_exceptions-en-01` / `prose-256`: expected `{"sharing_status": "forbidden", "exception": "redacted excerpt approved by the archive steward"}`; observed `"{\"sharing_status\":\"forbidden\",\"exception\":\"a redacted excerpt approved by the archive steward\"}"`.
- `dev-updated_facts-ru-01` / `prose-512`: expected `{"owner": "Марина", "room": "Вяз", "date": "12 января"}`; observed `"{\"owner\":\"Марина\",\"room\":\"«Вяз»\",\"date\":\"12 января\"}"`.
- `dev-updated_facts-ru-01` / `structured-512`: expected `{"owner": "Марина", "room": "Вяз", "date": "12 января"}`; observed `"{\"owner\":\"Марина\",\"room\":\"«Вяз»\",\"date\":\"12 января\"}"`.

All incorrect answers, including original-history errors, remain in results.jsonl.

Development-selected variant: `structured-256`. Selection uses development only.

Fixed budgets: 256 / 512 / 1024 completion tokens, including provider reasoning when applicable; prose / structured formats. Unverified optimize() only.

At most 13 calls per case, no retries; one shared original baseline per case. Execution order rotates by case and is recorded in metadata.json and checkpoint.jsonl.
