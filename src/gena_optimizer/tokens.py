"""Token counting.

A token counter is any ``Callable[[str], int]``. The default is a dependency-free
estimate; pass a tokenizer-backed counter when you need exact numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from gena_optimizer.types import Message

TokenCounter = Callable[[str], int]

_CHARS_PER_TOKEN = 4
_TOKENS_PER_MESSAGE = 4  # rough cost of role and message framing


def approximate_token_count(text: str) -> int:
    """Estimate tokens as one per four characters (a common rule of thumb for English)."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def tiktoken_counter(encoding: str = "o200k_base") -> TokenCounter:
    """Return an exact counter backed by ``tiktoken`` (``pip install gena-optimizer[tiktoken]``)."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "tiktoken is not installed; run: pip install 'gena-optimizer[tiktoken]'"
        ) from exc
    encoder = tiktoken.get_encoding(encoding)
    return lambda text: len(encoder.encode(text))


def count_message_tokens(
    messages: Iterable[Message], counter: TokenCounter = approximate_token_count
) -> int:
    """Count the tokens in a conversation, including a small per-message overhead."""
    return sum(counter(message.content) + _TOKENS_PER_MESSAGE for message in messages)
