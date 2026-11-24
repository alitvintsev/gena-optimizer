import json

import pytest
from benchmarks.metering import MeteredModel, UsageTransport, aggregate_records, cost_for_usage

from gena_optimizer import CallableModel, Message, ModelError, OpenAICompatibleModel


def test_usage_transport_normalizes_openai_and_keeps_response_unchanged():
    response = {
        "choices": [{"message": {"content": "OK"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }
    requests = []

    def transport(url, headers, body, timeout):
        requests.append((url, headers, body, timeout))
        return response

    metered = UsageTransport("openai", transport=transport)
    assert metered("https://example.test", {"Authorization": "secret"}, {"x": 1}, 3) is response
    assert requests == [("https://example.test", {"Authorization": "secret"}, {"x": 1}, 3)]
    assert metered.usage == {
        "input_tokens": 100,
        "output_tokens": 12,
        "cached_input_tokens": 40,
        "cache_write_input_tokens": 0,
    }


def test_usage_transport_preserves_openai_cache_writes_and_requires_their_tariff():
    metered = UsageTransport(
        "openai",
        transport=lambda *_: {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 20},
            }
        },
    )
    metered("unused", {}, {}, 3)
    assert metered.usage == {
        "input_tokens": 100,
        "output_tokens": 12,
        "cached_input_tokens": 40,
        "cache_write_input_tokens": 20,
    }
    prices = {
        "input_per_million": 2,
        "output_per_million": 8,
        "cached_input_per_million": 0.5,
    }
    assert cost_for_usage(metered.usage, prices) is None
    assert cost_for_usage(
        metered.usage, {**prices, "cache_write_input_per_million": 3}
    ) == pytest.approx(0.000256)


@pytest.mark.parametrize(
    "details",
    [
        {"cache_write_tokens": -1},
        {"cache_write_tokens": None},
        {"cache_write_tokens": True},
        {"cache_write_tokens": "20"},
        {"cache_write_tokens": 101},
        {"cached_tokens": 70, "cache_write_tokens": 40},
    ],
)
def test_usage_transport_does_not_price_invalid_or_overlapping_openai_cache_counts(details):
    metered = UsageTransport(
        "openai",
        transport=lambda *_: {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "prompt_tokens_details": details,
            }
        },
    )
    metered("unused", {}, {}, 3)
    assert metered.usage is None
    assert (
        cost_for_usage(
            metered.usage,
            {
                "input_per_million": 2,
                "output_per_million": 8,
                "cached_input_per_million": 0.5,
                "cache_write_input_per_million": 3,
            },
        )
        is None
    )


def test_usage_transport_normalizes_anthropic_total_input():
    metered = UsageTransport(
        "anthropic",
        transport=lambda *_: {
            "usage": {
                "input_tokens": 20,
                "output_tokens": 12,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 30,
            }
        },
    )
    metered("unused", {}, {}, 3)
    assert metered.usage == {
        "input_tokens": 100,
        "output_tokens": 12,
        "cached_input_tokens": 50,
        "cache_write_input_tokens": 30,
    }


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_usage_transport_defaults_only_optional_cache_counts(provider):
    response = (
        {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        if provider == "openai"
        else {"usage": {"input_tokens": 0, "output_tokens": 0}}
    )
    metered = UsageTransport(provider, transport=lambda *_: response)
    metered("unused", {}, {}, 3)
    assert metered.usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
    }


@pytest.mark.parametrize(
    ("provider", "usage"),
    [
        ("openai", None),
        ("openai", {"prompt_tokens": 10}),
        ("openai", {"completion_tokens": 2}),
        ("openai", {"prompt_tokens": -1, "completion_tokens": 2}),
        ("openai", {"prompt_tokens": 10, "completion_tokens": True}),
        ("anthropic", {"input_tokens": 10}),
        ("anthropic", {"output_tokens": 2}),
    ],
)
def test_usage_transport_does_not_invent_missing_or_invalid_usage(provider, usage):
    metered = UsageTransport(provider, transport=lambda *_: {"usage": usage})
    metered("unused", {}, {}, 3)
    assert metered.usage is None


def test_usage_transport_drops_stale_usage_on_missing_response_and_failure():
    replies = [
        {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {},
        {"usage": {"prompt_tokens": 20, "completion_tokens": 3}},
        TimeoutError("secret failure details"),
    ]

    def transport(*_):
        response = replies.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    metered = UsageTransport("openai", transport=transport)
    metered("unused", {}, {}, 3)
    assert metered.usage["input_tokens"] == 10
    metered("unused", {}, {}, 3)
    assert metered.usage is None
    metered("unused", {}, {}, 3)
    assert metered.usage["input_tokens"] == 20
    with pytest.raises(TimeoutError):
        metered("unused", {}, {}, 3)
    assert metered.usage is None


def test_cost_for_usage_separates_uncached_cached_and_cache_write_tokens():
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_input_tokens": 50,
        "cache_write_input_tokens": 30,
    }
    prices = {
        "input_per_million": 2,
        "output_per_million": 8,
        "cached_input_per_million": 0.5,
        "cache_write_input_per_million": 3,
    }
    assert cost_for_usage(usage, prices) == pytest.approx(0.000315)


@pytest.mark.parametrize(
    ("usage", "prices"),
    [
        (None, {"input_per_million": 1, "output_per_million": 2}),
        ({"input_tokens": 10, "output_tokens": 2}, None),
        ({"input_tokens": 10, "output_tokens": 2}, {"input_per_million": 1}),
        (
            {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 5},
            {"input_per_million": 1, "output_per_million": 2},
        ),
        (
            {"input_tokens": 10, "output_tokens": 2, "cache_write_input_tokens": 5},
            {"input_per_million": 1, "output_per_million": 2},
        ),
    ],
)
def test_cost_for_usage_is_unknown_when_usage_or_required_prices_are_missing(usage, prices):
    assert cost_for_usage(usage, prices) is None


def test_cost_for_usage_allows_omitting_unused_cache_prices():
    assert cost_for_usage(
        {"input_tokens": 10, "output_tokens": 2},
        {"input_per_million": 1, "output_per_million": 2},
    ) == pytest.approx(0.000014)


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf"), "2", True])
def test_invalid_prices_are_rejected_without_calling_model(rate):
    with pytest.raises(ValueError, match="price"):
        MeteredModel(
            CallableModel(lambda *_: pytest.fail("model must not run")),
            name="test",
            counter=len,
            prices={"input_per_million": rate, "output_per_million": 2},
        )
    with pytest.raises(ValueError, match="price"):
        cost_for_usage(
            {"input_tokens": 10, "output_tokens": 2},
            {"input_per_million": rate, "output_per_million": 2},
        )


def test_metered_model_records_reply_usage_cost_and_monotonic_latency(monkeypatch):
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr("benchmarks.metering.perf_counter", lambda: next(ticks))
    transport = UsageTransport(
        "openai",
        transport=lambda *_: {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 12},
        },
    )
    model = MeteredModel(
        OpenAICompatibleModel("m", api_key="secret-key", transport=transport),
        name="measured-model",
        counter=len,
        prices={"input_per_million": 2, "output_per_million": 8},
        transport=transport,
        stages=("summary", "verification"),
    )
    assert model.complete([Message("user", "private request")], max_tokens=20) == "OK"
    assert model.records == [
        {
            "model": "measured-model",
            "stage": "summary",
            "latency_seconds": 0.25,
            "estimated_input_tokens": 19,
            "estimated_output_tokens": 2,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 12,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
            "cost_usd": pytest.approx(0.000296),
            "response_metadata": {
                "actual_model": None,
                "finish_reason": None,
                "reasoning_tokens": None,
            },
            "error": None,
        }
    ]
    assert "private request" not in json.dumps(model.records)
    assert "secret-key" not in json.dumps(model.records)


def test_metered_model_preserves_failure_and_records_only_sanitized_metadata(monkeypatch):
    ticks = iter([1.0, 1.5])
    monkeypatch.setattr("benchmarks.metering.perf_counter", lambda: next(ticks))
    failure = ModelError("secret-header secret-body secret-error")

    def fail(*_):
        raise failure

    transport = UsageTransport("openai", transport=fail)
    model = MeteredModel(
        OpenAICompatibleModel("m", api_key="secret-header", transport=transport),
        name="safe-model",
        counter=len,
        transport=transport,
    )
    with pytest.raises(ModelError) as caught:
        model.complete([Message("user", "secret-body")])
    assert caught.value is failure
    assert model.records == [
        {
            "model": "safe-model",
            "stage": "downstream",
            "latency_seconds": 0.5,
            "estimated_input_tokens": 15,
            "estimated_output_tokens": None,
            "usage": None,
            "response_metadata": {},
            "cost_usd": None,
            "error": "ModelError",
        }
    ]
    assert "secret" not in json.dumps(model.records)


def test_metered_model_stage_mapping_repeats_last_and_reset_restarts():
    transport = UsageTransport("openai", transport=lambda *_: {})
    model = MeteredModel(
        CallableModel(lambda *_: "OK"),
        name="m",
        counter=len,
        transport=transport,
        stages=("summary", "verification"),
    )
    for _ in range(3):
        model.complete([Message("user", "question")])
    assert [record["stage"] for record in model.records] == [
        "summary",
        "verification",
        "verification",
    ]
    assert all(record["usage"] is None and record["cost_usd"] is None for record in model.records)
    transport.usage = {"input_tokens": 100, "output_tokens": 10}
    model.reset()
    assert model.records == []
    assert transport.usage is None
    model.complete([])
    assert model.records[0]["stage"] == "summary"


def test_metered_model_rejects_empty_stages():
    with pytest.raises(ValueError, match="stages"):
        MeteredModel(CallableModel(lambda *_: "OK"), name="m", counter=len, stages=())


def test_aggregate_records_totals_known_usage_and_separates_stages():
    records = [
        {
            "stage": "summary",
            "latency_seconds": 0.25,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cached_input_tokens": 30,
                "cache_write_input_tokens": 20,
            },
            "cost_usd": 0.001,
        },
        {
            "stage": "downstream",
            "latency_seconds": 0.75,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 20,
                "cached_input_tokens": 50,
                "cache_write_input_tokens": 0,
            },
            "cost_usd": 0.002,
        },
    ]
    result = aggregate_records(records)
    assert result["calls"] == 2
    assert result["latency_seconds"] == 1
    assert result["usage"] == {
        "input_tokens": 300,
        "output_tokens": 30,
        "cached_input_tokens": 80,
        "cache_write_input_tokens": 20,
    }
    assert result["unknown_usage_calls"] == 0
    assert result["cost_usd"] == pytest.approx(0.003)
    assert result["known_cost_usd"] == pytest.approx(0.003)
    assert result["by_stage"]["summary"]["calls"] == 1
    assert result["by_stage"]["downstream"]["usage"]["input_tokens"] == 200
    assert "by_stage" not in result["by_stage"]["summary"]


def test_aggregate_records_does_not_turn_unknown_usage_or_cost_into_zero():
    result = aggregate_records(
        [
            {
                "stage": "downstream",
                "latency_seconds": 0.1,
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "cost_usd": 0.001,
            },
            {"stage": "downstream", "latency_seconds": 0.2, "usage": None, "cost_usd": None},
        ]
    )
    assert result["calls"] == 2
    assert result["usage"] is None
    assert result["unknown_usage_calls"] == 1
    assert result["cost_usd"] is None
    assert result["known_cost_usd"] == 0.001
    assert result["by_stage"]["downstream"]["cost_usd"] is None


def test_aggregate_records_empty_input_has_zero_calls_and_totals():
    assert aggregate_records([]) == {
        "calls": 0,
        "latency_seconds": 0,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
        },
        "unknown_usage_calls": 0,
        "cost_usd": 0,
        "known_cost_usd": 0,
        "by_stage": {},
    }
