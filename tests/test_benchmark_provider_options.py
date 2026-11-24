import json

import pytest
from benchmarks.metering import MeteredModel, UsageTransport
from benchmarks.run import configuration, models

from gena_optimizer import Message, OpenAICompatibleModel, approximate_token_count


def test_reasoning_effort_is_forwarded_without_changing_answer_limit(tmp_path, monkeypatch):
    spec = {
        "provider": "openai",
        "model": "test-model",
        "api_key_env": "TEST_STUDY_KEY",
        "reasoning_effort": "low",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"summarizer": spec, "downstream": spec}))
    config = configuration(path)
    monkeypatch.setenv("TEST_STUDY_KEY", "test-only-key")
    seen = []

    def transport(url, headers, body, timeout):
        seen.append(body)
        return {"choices": [{"message": {"content": "answer"}}]}

    clients = models(config, approximate_token_count, live=True)
    for client in clients:
        client.transport._transport = transport
        assert client.complete([Message("user", "Question")], max_tokens=73) == "answer"
    assert [body["reasoning_effort"] for body in seen] == ["low", "low"]
    assert [body["max_completion_tokens"] for body in seen] == [73, 73]


@pytest.mark.parametrize("effort", ["none", "", 0, {"secret": "not-allowed"}])
def test_invalid_reasoning_effort_is_rejected_before_requests(tmp_path, effort):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "summarizer": {
                    "provider": "openai",
                    "model": "test",
                    "reasoning_effort": effort,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="reasoning_effort"):
        configuration(path)


def test_response_diagnostics_are_recorded_and_never_reused_after_failure():
    responses = iter(
        [
            {
                "model": "actual-test-model",
                "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
            RuntimeError("private provider response"),
        ]
    )

    def transport(url, headers, body, timeout):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    usage = UsageTransport("openai", transport)
    client = MeteredModel(
        OpenAICompatibleModel("requested", transport=usage),
        name="requested",
        counter=approximate_token_count,
        transport=usage,
    )
    assert client.complete([Message("user", "Question")]) == "partial"
    assert client.records[0]["response_metadata"] == {
        "actual_model": "actual-test-model",
        "finish_reason": "length",
        "reasoning_tokens": 2,
    }
    with pytest.raises(RuntimeError):
        client.complete([Message("user", "Next")])
    assert client.records[1]["response_metadata"] == {}
    assert client.records[1]["usage"] is None
    assert "private provider response" not in json.dumps(client.records)
