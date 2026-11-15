"""The interface every optimizer implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from gena_optimizer.types import Message, MessageLike, as_messages


class Optimizer(ABC):
    """Transforms a conversation into a smaller, equivalent one.

    Subclasses implement :meth:`_optimize`. Callers use :meth:`optimize`, which
    accepts :class:`Message` objects or plain role/content dicts and never
    mutates its input.
    """

    def optimize(self, messages: Iterable[MessageLike]) -> list[Message]:
        """Return an optimized copy of ``messages``."""
        return self._optimize(as_messages(messages))

    @abstractmethod
    def _optimize(self, messages: list[Message]) -> list[Message]:
        """Return the optimized conversation. Must not mutate ``messages``."""
