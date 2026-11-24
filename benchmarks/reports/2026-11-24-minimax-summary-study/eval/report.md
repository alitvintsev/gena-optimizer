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
| original | 28 | 96.4% | 96.4% | 0/1 | 1 | 0/0/0 | 18890 → 18890 | 28 | unknown |
| prose-1024 | 28 | 89.3% | 89.3% | 0/2 | 2 | 9/17/0 | 18890 → 17266 + 2 unknown | 54 | unknown |
| prose-256 | 28 | 75.0% | 79.2% | 0/0 | 0 | 28/0/0 | 18890 → 11493 | 56 | 0.010994 |
| prose-512 | 28 | 82.1% | 86.3% | 0/0 | 0 | 19/9/0 | 18890 → 16815 | 56 | 0.013278 |
| structured-1024 | 28 | 100.0% | 100.0% | 0/0 | 0 | 0/28/0 | 18890 → 18890 | 56 | 0.018790 |
| structured-256 | 28 | 85.7% | 89.9% | 0/1 | 1 | 27/0/0 | 18890 → 10885 + 1 unknown | 55 | unknown |
| structured-512 | 28 | 82.1% | 83.9% | 0/1 | 0 | 14/14/0 | 18890 → 17053 | 56 | 0.013984 |
| original/code_json | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 2790 → 2790 | 4 | 0.000441 |
| original/entity_ids | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 2734 → 2734 | 4 | 0.000249 |
| original/negations_exceptions | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 2661 → 2661 | 4 | 0.000291 |
| original/numbers_units | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 2696 → 2696 | 4 | 0.000314 |
| original/sources | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 2687 → 2687 | 4 | 0.000173 |
| original/task_status | 4 | 75.0% | 75.0% | 0/1 | 1 | 0/0/0 | 2659 → 2659 | 4 | unknown |
| original/updated_facts | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/0/0 | 2663 → 2663 | 4 | 0.000181 |
| prose-1024/code_json | 4 | 100.0% | 100.0% | 0/0 | 0 | 1/3/0 | 2790 → 2733 | 8 | 0.001846 |
| prose-1024/entity_ids | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2734 → 2734 | 8 | 0.001850 |
| prose-1024/negations_exceptions | 4 | 100.0% | 100.0% | 0/0 | 0 | 2/2/0 | 2661 → 2595 | 8 | 0.001671 |
| prose-1024/numbers_units | 4 | 75.0% | 75.0% | 0/1 | 1 | 1/2/0 | 2696 → 1907 + 1 unknown | 7 | unknown |
| prose-1024/sources | 4 | 75.0% | 75.0% | 0/1 | 1 | 1/2/0 | 2687 → 2063 + 1 unknown | 7 | unknown |
| prose-1024/task_status | 4 | 75.0% | 75.0% | 0/0 | 0 | 2/2/0 | 2659 → 2626 | 8 | 0.002017 |
| prose-1024/updated_facts | 4 | 100.0% | 100.0% | 0/0 | 0 | 2/2/0 | 2663 → 2608 | 8 | 0.002177 |
| prose-256/code_json | 4 | 75.0% | 75.0% | 0/0 | 0 | 4/0/0 | 2790 → 1643 | 8 | 0.001821 |
| prose-256/entity_ids | 4 | 75.0% | 75.0% | 0/0 | 0 | 4/0/0 | 2734 → 1669 | 8 | 0.001509 |
| prose-256/negations_exceptions | 4 | 75.0% | 87.5% | 0/0 | 0 | 4/0/0 | 2661 → 1672 | 8 | 0.001660 |
| prose-256/numbers_units | 4 | 75.0% | 75.0% | 0/0 | 0 | 4/0/0 | 2696 → 1580 | 8 | 0.001562 |
| prose-256/sources | 4 | 100.0% | 100.0% | 0/0 | 0 | 4/0/0 | 2687 → 1630 | 8 | 0.001419 |
| prose-256/task_status | 4 | 75.0% | 75.0% | 0/0 | 0 | 4/0/0 | 2659 → 1657 | 8 | 0.001482 |
| prose-256/updated_facts | 4 | 50.0% | 66.7% | 0/0 | 0 | 4/0/0 | 2663 → 1642 | 8 | 0.001540 |
| prose-512/code_json | 4 | 75.0% | 75.0% | 0/0 | 0 | 3/1/0 | 2790 → 2447 | 8 | 0.002208 |
| prose-512/entity_ids | 4 | 100.0% | 100.0% | 0/0 | 0 | 3/1/0 | 2734 → 2382 | 8 | 0.001813 |
| prose-512/negations_exceptions | 4 | 50.0% | 62.5% | 0/0 | 0 | 3/1/0 | 2661 → 2342 | 8 | 0.001880 |
| prose-512/numbers_units | 4 | 75.0% | 75.0% | 0/0 | 0 | 3/1/0 | 2696 → 2425 | 8 | 0.002057 |
| prose-512/sources | 4 | 100.0% | 100.0% | 0/0 | 0 | 2/2/0 | 2687 → 2404 | 8 | 0.001850 |
| prose-512/task_status | 4 | 100.0% | 100.0% | 0/0 | 0 | 3/1/0 | 2659 → 2404 | 8 | 0.001645 |
| prose-512/updated_facts | 4 | 75.0% | 91.7% | 0/0 | 0 | 2/2/0 | 2663 → 2411 | 8 | 0.001824 |
| structured-1024/code_json | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2790 → 2790 | 8 | 0.002789 |
| structured-1024/entity_ids | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2734 → 2734 | 8 | 0.002695 |
| structured-1024/negations_exceptions | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2661 → 2661 | 8 | 0.002860 |
| structured-1024/numbers_units | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2696 → 2696 | 8 | 0.002521 |
| structured-1024/sources | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2687 → 2687 | 8 | 0.002506 |
| structured-1024/task_status | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2659 → 2659 | 8 | 0.002888 |
| structured-1024/updated_facts | 4 | 100.0% | 100.0% | 0/0 | 0 | 0/4/0 | 2663 → 2663 | 8 | 0.002530 |
| structured-256/code_json | 4 | 100.0% | 100.0% | 0/0 | 0 | 4/0/0 | 2790 → 1686 | 8 | 0.001495 |
| structured-256/entity_ids | 4 | 100.0% | 100.0% | 0/0 | 0 | 4/0/0 | 2734 → 1481 | 8 | 0.001182 |
| structured-256/negations_exceptions | 4 | 75.0% | 87.5% | 0/0 | 0 | 4/0/0 | 2661 → 1637 | 8 | 0.001345 |
| structured-256/numbers_units | 4 | 75.0% | 75.0% | 0/0 | 0 | 4/0/0 | 2696 → 1655 | 8 | 0.001641 |
| structured-256/sources | 4 | 75.0% | 75.0% | 0/1 | 1 | 3/0/0 | 2687 → 1224 + 1 unknown | 7 | unknown |
| structured-256/task_status | 4 | 100.0% | 100.0% | 0/0 | 0 | 4/0/0 | 2659 → 1576 | 8 | 0.001490 |
| structured-256/updated_facts | 4 | 75.0% | 91.7% | 0/0 | 0 | 4/0/0 | 2663 → 1626 | 8 | 0.001596 |
| structured-512/code_json | 4 | 75.0% | 87.5% | 0/0 | 0 | 2/2/0 | 2790 → 2510 | 8 | 0.001952 |
| structured-512/entity_ids | 4 | 100.0% | 100.0% | 0/0 | 0 | 2/2/0 | 2734 → 2401 | 8 | 0.001977 |
| structured-512/negations_exceptions | 4 | 50.0% | 50.0% | 0/1 | 0 | 2/2/0 | 2661 → 2439 | 8 | 0.001944 |
| structured-512/numbers_units | 4 | 75.0% | 75.0% | 0/0 | 0 | 2/2/0 | 2696 → 2444 | 8 | 0.002194 |
| structured-512/sources | 4 | 75.0% | 75.0% | 0/0 | 0 | 2/2/0 | 2687 → 2443 | 8 | 0.002143 |
| structured-512/task_status | 4 | 100.0% | 100.0% | 0/0 | 0 | 2/2/0 | 2659 → 2373 | 8 | 0.001908 |
| structured-512/updated_facts | 4 | 100.0% | 100.0% | 0/0 | 0 | 2/2/0 | 2663 → 2443 | 8 | 0.001867 |

## Quality regressions

- `numbers_units-ru-01` / `prose-256`: expected `{"thickness": "0,75 мм", "roll_width": "1,20 м", "box_mass": "12,50 кг"}`; observed `"{\"thickness\":\"не указано\",\"roll_width\":\"не указано\",\"box_mass\":\"не указано\"}"`.
- `numbers_units-ru-01` / `prose-512`: expected `{"thickness": "0,75 мм", "roll_width": "1,20 м", "box_mass": "12,50 кг"}`; observed `"{\"thickness\":\"не указано\",\"roll_width\":\"не указано\",\"box_mass\":\"не указано\"}"`.
- `numbers_units-ru-01` / `structured-256`: expected `{"thickness": "0,75 мм", "roll_width": "1,20 м", "box_mass": "12,50 кг"}`; observed `"{\"thickness\":\"\",\"roll_width\":\"\",\"box_mass\":\"\"}"`.
- `numbers_units-ru-02` / `prose-1024`: expected `{"voltage": "24 В", "warmup": "90 с", "flow_limit": "2,50 л/мин"}`; observed `""`.
- `numbers_units-ru-02` / `structured-512`: expected `{"voltage": "24 В", "warmup": "90 с", "flow_limit": "2,50 л/мин"}`; observed `"{\"voltage\":\"не указаны в доступном контексте\",\"warmup\":\"не указан в доступном контексте\",\"flow_limit\":\"не указан в доступном контексте\"}"`.
- `negations_exceptions-ru-02` / `structured-512`: expected `{"cover_open_status": "запрещено", "exception": "диагностика без подачи питания в присутствии инженера"}`; observed `"{\"cover_open_status\":\"Работа с открытой защитной крышкой — запрещено.\",\"exception\":\"Единственное исключение — диагностика без подачи питания в присутствии инженера.\"}"`.
- `negations_exceptions-ru-02` / `prose-512`: expected `{"cover_open_status": "запрещено", "exception": "диагностика без подачи питания в присутствии инженера"}`; observed `"{\"cover_open_status\":\"Работа при открытой защитной крышке в мастерской «Берег» запрещена.\",\"exception\":\"Разрешён только специальный случай: диагностика без подачи питания в присутствии инженера.\"}"`.
- `negations_exceptions-en-01` / `prose-512`: expected `{"copy_status": "forbidden", "exception": "encrypted research laptop with written custodian approval"}`; observed `"{\"copy_status\":\"forbidden\",\"exception\":\"an encrypted research laptop with written custodian approval\"}"`.
- `negations_exceptions-en-01` / `structured-256`: expected `{"copy_status": "forbidden", "exception": "encrypted research laptop with written custodian approval"}`; observed `"{\"copy_status\":\"forbidden\",\"exception\":\"an encrypted research laptop with written custodian approval\"}"`.
- `negations_exceptions-en-02` / `prose-256`: expected `{"deletion_status": "forbidden", "exception": "duplicate test tag in the sandbox repository"}`; observed `"{\"deletion_status\":\"forbidden\",\"exception\":\"a duplicate test tag in the sandbox repository\"}"`.

All incorrect answers, including original-history errors, remain in results.jsonl.

Development-selected variant: `structured-256`. Selection uses development only.

Fixed budgets: 256 / 512 / 1024 completion tokens, including provider reasoning when applicable; prose / structured formats. Unverified optimize() only.

At most 13 calls per case, no retries; one shared original baseline per case. Execution order rotates by case and is recorded in metadata.json and checkpoint.jsonl.
