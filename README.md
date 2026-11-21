# gena-optimizer

Small, composable tools for shrinking LLM prompts and context.

- **Prompt hygiene**: model-free clean-ups such as JSON minification, whitespace
  collapsing, base64 and HTML stripping, and log de-duplication.
- **Compaction**: replaces older conversation turns with a recall-first summary
  written by a model of your choice, keeping recent turns verbatim.
- **Any model**: a one-method `ChatModel` interface with built-in clients for
  OpenAI-compatible APIs and the Anthropic API.

No runtime dependencies. Python 3.10+. Fully typed.

## Install

```bash
pip install gena-optimizer
```

## Quick start

```python
from gena_optimizer import AnthropicModel, Compactor, Pipeline, PromptHygiene

pipeline = Pipeline(
    PromptHygiene(),
    Compactor(AnthropicModel("claude-haiku-5-5"), trigger_tokens=50_000),
)

result = pipeline.run(messages)  # messages: list of {"role": ..., "content": ...}
print(result.tokens_before, "->", result.tokens_after)
send_to_your_model([m.to_dict() for m in result.messages])
```

Every optimizer has one method, `optimize(messages) -> list[Message]`. It accepts
`Message` objects or plain role/content dicts and never mutates its input.
`Pipeline.run` does the same and also reports token counts.

## Prompt hygiene

`PromptHygiene` applies text transforms to each message. By default it uses only
meaning-preserving ones:

| Transform | What it does | Lossless | In defaults |
| --- | --- | --- | --- |
| `minify_json` | Removes whitespace from JSON text and fenced `json` blocks | Yes | Yes |
| `collapse_whitespace` | Drops trailing spaces and extra blank lines; keeps indentation | Yes | Yes |
| `strip_base64` | Replaces data URIs and long base64 runs with a placeholder | No | No |
| `dedupe_lines` | Collapses consecutive identical lines into `line [repeated Nx]` | No | No |
| `strip_html` | Reduces an HTML document to its visible text | No | No |
| `json_rows_to_csv` | Rewrites a JSON array of flat, same-shaped objects as CSV | Nearly (`null` becomes empty) | No |

Choose transforms explicitly, and limit them to certain roles if you wish:

```python
from gena_optimizer import (
    PromptHygiene,
    collapse_whitespace,
    dedupe_lines,
    json_rows_to_csv,
    strip_base64,
)

hygiene = PromptHygiene(
    [strip_base64, json_rows_to_csv, dedupe_lines, collapse_whitespace],
    roles={"user", "tool"},
)
clean = hygiene.optimize(messages)
```

A transform is any `Callable[[str], str]`, so your own are one function away:

```python
def drop_signature(text: str) -> str:
    return text.split("\n-- \n")[0]


PromptHygiene([drop_signature, collapse_whitespace])
```

Transforms are also plain functions you can call directly: `minify_json(text)`.

## Compaction

`Compactor` does nothing until the conversation exceeds `trigger_tokens`. Then it
summarizes every non-system message except the last `keep_recent`, and returns
`[system messages..., summary, recent messages...]`.

```python
from gena_optimizer import Compactor, OpenAICompatibleModel

compactor = Compactor(
    OpenAICompatibleModel("your-model-name"),
    trigger_tokens=100_000,
    keep_recent=6,
    verify=True,
)
messages = compactor.optimize(messages)
```

**On quality.** No summary is guaranteed lossless. `Compactor` is designed to lose
as little as a summary can:

- The default instructions favour completeness and list what must survive: goals,
  decisions, exact values, relevant code, current state, and open questions.
- Recent turns and system messages are kept word for word.
- `verify=True` adds a second call that lists what the summary missed and appends
  it. It costs one more pass over the transcript.
- If the result is not smaller than the input, the original is returned.

Pass `instructions=...` to say what matters in your domain. Measure quality on
your own evaluation set before relying on it.

Limits to know: the older turns are summarized in one call, so they must fit the
summarizer's context window. Message content must be text.

## Models

`ChatModel` is a protocol with one method:

```python
def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str: ...
```

Built-in clients use only the standard library:

```python
from gena_optimizer import AnthropicModel, OpenAICompatibleModel

AnthropicModel("claude-haiku-5-5")  # reads ANTHROPIC_API_KEY
OpenAICompatibleModel("your-model-name")  # reads OPENAI_API_KEY
OpenAICompatibleModel(  # any compatible server
    "local-model",
    base_url="http://localhost:11434/v1",
    max_tokens_field="max_tokens",
)
```

`OpenAICompatibleModel` works with any endpoint that speaks the Chat Completions
format. To use another SDK, wrap a function or write a small class:

```python
from gena_optimizer import CallableModel

model = CallableModel(lambda messages, max_tokens: my_client.chat(messages, max_tokens))
```

API failures raise `ModelError`. The clients do not retry; pass your own
`transport` function to add retries or custom HTTP behaviour.

## Token counting

Counts default to a rough estimate of four characters per token. For exact counts
pass any `Callable[[str], int]`:

```python
from gena_optimizer import Compactor, tiktoken_counter  # pip install "gena-optimizer[tiktoken]"

Compactor(model, token_counter=tiktoken_counter())
```

## Writing an optimizer

Subclass `Optimizer` and implement `_optimize`. It then works anywhere, including
inside a `Pipeline`:

```python
from gena_optimizer import Message, Optimizer


class SlidingWindow(Optimizer):
    def __init__(self, keep: int) -> None:
        self.keep = keep

    def _optimize(self, messages: list[Message]) -> list[Message]:
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        return [*system, *rest[-self.keep :]]
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
mypy
python -m build && twine check dist/*
```

## License

MIT
