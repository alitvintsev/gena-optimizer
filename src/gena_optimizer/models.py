"""A minimal, provider-neutral interface to chat models.

Anything with a ``complete(messages, *, max_tokens) -> str`` method is a
:class:`ChatModel`. Two HTTP adapters are included and need no extra
dependencies:

* :class:`OpenAICompatibleModel` for the Chat Completions format, which OpenAI
  and many other providers and local servers speak.
* :class:`AnthropicModel` for the Anthropic Messages API.

Wrap any other client with :class:`CallableModel`, or write a class with a
``complete`` method.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gena_optimizer.errors import ModelError
from gena_optimizer.types import Message

Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], dict[str, Any]]
"""``(url, headers, json_body, timeout_seconds) -> parsed JSON response``."""


@runtime_checkable
class ChatModel(Protocol):
    """A language model that turns a conversation into a text reply."""

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        """Return the model's reply to ``messages``."""
        ...


class CallableModel:
    """Adapts a function ``(messages, max_tokens) -> str`` to :class:`ChatModel`."""

    def __init__(self, function: Callable[[Sequence[Message], int], str]) -> None:
        self._function = function

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        return self._function(messages, max_tokens)


def post_json(
    url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float
) -> dict[str, Any]:
    """POST a JSON body and return the parsed JSON response (standard library only)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ModelError(f"model API returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ModelError(f"could not reach model API at {url}: {exc}") from exc
    except ValueError as exc:
        raise ModelError("model API returned a response that is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ModelError("model API returned an unexpected response shape")
    return parsed


class OpenAICompatibleModel:
    """Chat Completions client for OpenAI and OpenAI-compatible endpoints.

    Args:
        model: Model identifier sent to the API.
        api_key: Defaults to the ``OPENAI_API_KEY`` environment variable. May be
            omitted for local servers that need no authentication.
        base_url: API root, without the ``/chat/completions`` suffix.
        max_tokens_field: Request field for the output limit. OpenAI uses
            ``max_completion_tokens``; some compatible servers expect ``max_tokens``.
        timeout: Seconds to wait for a response.
        extra_body: Additional request fields, such as ``{"temperature": 0}``.
        transport: HTTP function; replace it in tests or to add retries.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        max_tokens_field: str = "max_completion_tokens",
        timeout: float = 60.0,
        extra_body: Mapping[str, Any] | None = None,
        transport: Transport = post_json,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._max_tokens_field = max_tokens_field
        self._timeout = timeout
        self._extra_body = dict(extra_body or {})
        self._transport = transport

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {
            **self._extra_body,
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            self._max_tokens_field: max_tokens,
        }
        data = self._transport(self._url, headers, body, self._timeout)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("unexpected Chat Completions response shape") from exc
        if not isinstance(content, str):
            raise ModelError("Chat Completions response contained no text")
        return content


class AnthropicModel:
    """Client for the Anthropic Messages API.

    Args:
        model: Model identifier sent to the API.
        api_key: Defaults to the ``ANTHROPIC_API_KEY`` environment variable.
        base_url: API root, without the ``/v1/messages`` suffix.
        api_version: Value of the ``anthropic-version`` header.
        timeout: Seconds to wait for a response.
        extra_body: Additional request fields.
        transport: HTTP function; replace it in tests or to add retries.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        api_version: str = "2023-06-01",
        timeout: float = 60.0,
        extra_body: Mapping[str, Any] | None = None,
        transport: Transport = post_json,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ModelError("no API key: pass api_key or set ANTHROPIC_API_KEY")
        self.model = model
        self._headers = {"x-api-key": key, "anthropic-version": api_version}
        self._url = f"{base_url.rstrip('/')}/v1/messages"
        self._timeout = timeout
        self._extra_body = dict(extra_body or {})
        self._transport = transport

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        body: dict[str, Any] = {
            **self._extra_body,
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [m.to_dict() for m in messages if m.role != "system"],
        }
        if system:
            body["system"] = system
        data = self._transport(self._url, self._headers, body, self._timeout)
        try:
            text = "".join(
                block["text"] for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ModelError("unexpected Messages API response shape") from exc
        return text
