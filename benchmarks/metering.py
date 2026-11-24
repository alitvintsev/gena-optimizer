"""Dependency-free measurement for benchmark model calls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from gena_optimizer.models import ChatModel, Transport, post_json
from gena_optimizer.tokens import TokenCounter, count_message_tokens
from gena_optimizer.types import Message


class UsageTransport:
    """Capture normalized usage without retaining request or response content."""

    def __init__(self, provider: str, transport: Transport = post_json) -> None:
        if provider not in {"openai", "anthropic"}:
            raise ValueError("provider must be openai or anthropic")
        self.provider = provider
        self._transport = transport
        self.usage: dict[str, int] | None = None
        self.response_metadata: dict[str, Any] = {}

    def __call__(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float
    ) -> dict[str, Any]:
        self.usage = None
        self.response_metadata = {}
        response = self._transport(url, headers, body, timeout)
        self.usage = _normalize_usage(self.provider, response)
        # Only diagnostics, never raw responses, headers or reasoning text.
        self.response_metadata = {"actual_model": response.get("model")}
        if self.provider == "openai":
            choices = response.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                self.response_metadata["finish_reason"] = choices[0].get("finish_reason")
            usage = response.get("usage", {})
            if isinstance(usage, Mapping):
                details = usage.get("completion_tokens_details", {})
                if isinstance(details, Mapping):
                    self.response_metadata["reasoning_tokens"] = _token_count(
                        details.get("reasoning_tokens")
                    )
        else:
            self.response_metadata["finish_reason"] = response.get("stop_reason")
        return response


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _normalize_usage(provider: str, response: dict[str, Any]) -> dict[str, int] | None:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    if provider == "openai":
        input_tokens = _token_count(usage.get("prompt_tokens"))
        output_tokens = _token_count(usage.get("completion_tokens"))
        details = usage.get("prompt_tokens_details", {})
        if not isinstance(details, Mapping):
            return None
        cached = _token_count(details.get("cached_tokens", 0))
        cache_write = _token_count(details.get("cache_write_tokens", 0))
    else:
        input_tokens = _token_count(usage.get("input_tokens"))
        output_tokens = _token_count(usage.get("output_tokens"))
        cached = _token_count(usage.get("cache_read_input_tokens", 0))
        cache_write = _token_count(usage.get("cache_creation_input_tokens", 0))
    if input_tokens is None or output_tokens is None or cached is None or cache_write is None:
        return None
    if provider == "anthropic":
        input_tokens += cached + cache_write
    if cached + cache_write > input_tokens:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
    }


class MeteredModel:
    """Measure calls to an existing ChatModel and retain safe metadata only."""

    def __init__(
        self,
        model: ChatModel,
        *,
        name: str,
        counter: TokenCounter,
        prices: dict[str, float] | None = None,
        transport: UsageTransport | None = None,
        stages: tuple[str, ...] = ("downstream",),
    ) -> None:
        if not stages:
            raise ValueError("stages must not be empty")
        _validate_prices(prices)
        self.model = model
        self.name = name
        self.counter = counter
        self.prices = dict(prices) if prices is not None else None
        self.transport = transport
        self.stages = stages
        self.records: list[dict[str, Any]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        stage = self.stages[min(len(self.records), len(self.stages) - 1)]
        estimated_input = count_message_tokens(messages, self.counter)
        if self.transport is not None:
            self.transport.usage = None
            self.transport.response_metadata = {}
        reply = None
        error = None
        started = perf_counter()
        try:
            reply = self.model.complete(messages, max_tokens=max_tokens)
            return reply
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            latency = perf_counter() - started
            usage = (
                dict(self.transport.usage)
                if self.transport is not None and self.transport.usage is not None
                else None
            )
            self.records.append(
                {
                    "model": self.name,
                    "stage": stage,
                    "latency_seconds": latency,
                    "estimated_input_tokens": estimated_input,
                    "estimated_output_tokens": self.counter(reply) if reply is not None else None,
                    "usage": usage,
                    "response_metadata": dict(self.transport.response_metadata)
                    if self.transport is not None
                    else {},
                    "cost_usd": cost_for_usage(usage, self.prices),
                    "error": error,
                }
            )

    def reset(self) -> None:
        self.records.clear()
        if self.transport is not None:
            self.transport.usage = None
            self.transport.response_metadata = {}


def _validate_prices(prices: dict[str, float] | None) -> None:
    if prices is None:
        return
    for rate in prices.values():
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate < 0
        ):
            raise ValueError("prices must contain finite nonnegative numeric rates")


def cost_for_usage(usage: dict[str, int] | None, prices: dict[str, float] | None) -> float | None:
    """Price normalized usage using USD per million tokens; unknown stays None."""
    _validate_prices(prices)
    if usage is None or prices is None:
        return None
    input_tokens = _token_count(usage.get("input_tokens"))
    output_tokens = _token_count(usage.get("output_tokens"))
    cached = _token_count(usage.get("cached_input_tokens", 0))
    cache_write = _token_count(usage.get("cache_write_input_tokens", 0))
    if (
        input_tokens is None
        or output_tokens is None
        or cached is None
        or cache_write is None
        or cached + cache_write > input_tokens
    ):
        return None
    categories = [
        (input_tokens - cached - cache_write, "input_per_million"),
        (output_tokens, "output_per_million"),
    ]
    if cached:
        categories.append((cached, "cached_input_per_million"))
    if cache_write:
        categories.append((cache_write, "cache_write_input_per_million"))
    if any(key not in prices for _, key in categories):
        return None
    return sum(tokens * prices[key] / 1_000_000 for tokens, key in categories)


_USAGE_KEYS = ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens")


def _totals(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unknown_usage = sum(record.get("usage") is None for record in records)
    known_costs = [record["cost_usd"] for record in records if record.get("cost_usd") is not None]
    return {
        "calls": len(records),
        "latency_seconds": sum(record["latency_seconds"] for record in records),
        "usage": (
            {key: sum(record["usage"].get(key, 0) for record in records) for key in _USAGE_KEYS}
            if not unknown_usage
            else None
        ),
        "unknown_usage_calls": unknown_usage,
        "cost_usd": sum(known_costs) if len(known_costs) == len(records) else None,
        "known_cost_usd": sum(known_costs),
    }


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Total calls and group stages without treating missing usage as zero."""
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_stage.setdefault(record["stage"], []).append(record)
    return {
        **_totals(records),
        "by_stage": {stage: _totals(stage_records) for stage, stage_records in by_stage.items()},
    }
