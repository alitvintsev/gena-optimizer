"""Core data types."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message with a role (``system``, ``user``, ``assistant``) and text."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Return the ``{"role": ..., "content": ...}`` form most chat APIs accept."""
        return {"role": self.role, "content": self.content}


MessageLike: TypeAlias = Message | Mapping[str, str]
"""A :class:`Message` or a ``{"role": ..., "content": ...}`` mapping."""


def as_messages(messages: Iterable[MessageLike]) -> list[Message]:
    """Normalize messages or role/content mappings into a list of :class:`Message`."""
    return [_as_message(item) for item in messages]


def _as_message(item: MessageLike) -> Message:
    if isinstance(item, Message):
        return item
    try:
        role, content = item["role"], item["content"]
    except (KeyError, TypeError) as exc:
        raise TypeError(
            "each message must be a Message or a mapping with 'role' and 'content'"
        ) from exc
    if not isinstance(role, str) or not isinstance(content, str):
        raise TypeError("message 'role' and 'content' must be strings")
    return Message(role, content)
