"""Chain optimizers and measure what they saved."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gena_optimizer.base import Optimizer
from gena_optimizer.tokens import TokenCounter, approximate_token_count, count_message_tokens
from gena_optimizer.types import Message, MessageLike, as_messages


@dataclass(frozen=True, slots=True)
class Result:
    """The outcome of :meth:`Pipeline.run`."""

    messages: list[Message]
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def ratio(self) -> float:
        """Size after optimization as a fraction of the original (1.0 = unchanged)."""
        return self.tokens_after / self.tokens_before if self.tokens_before else 1.0


class Pipeline(Optimizer):
    """Runs optimizers in order. A pipeline is itself an optimizer, so pipelines nest.

    Put cheap, lossless steps first (hygiene) and model-based steps last
    (compaction), so the model summarizes text that is already clean.
    """

    def __init__(
        self, *optimizers: Optimizer, token_counter: TokenCounter = approximate_token_count
    ) -> None:
        self.optimizers = optimizers
        self.token_counter = token_counter

    def _optimize(self, messages: list[Message]) -> list[Message]:
        for optimizer in self.optimizers:
            messages = optimizer.optimize(messages)
        return messages

    def run(self, messages: Iterable[MessageLike]) -> Result:
        """Optimize ``messages`` and report token counts before and after."""
        original = as_messages(messages)
        optimized = self._optimize(original)
        return Result(
            messages=optimized,
            tokens_before=count_message_tokens(original, self.token_counter),
            tokens_after=count_message_tokens(optimized, self.token_counter),
        )
