import pytest

from gena_optimizer import (
    AnthropicModel,
    CallableModel,
    ChatModel,
    Message,
    ModelError,
    OpenAICompatibleModel,
    approximate_token_count,
    count_message_tokens,
)

MESSAGES = [Message("system", "Be brief."), Message("user", "Hi")]


class Recorder:
    def __init__(self, response):
        self.response = response
        self.request = None

    def __call__(self, url, headers, body, timeout):
        self.request = {
            "url": url,
            "headers": dict(headers),
            "body": dict(body),
            "timeout": timeout,
        }
        return self.response


def test_openai_compatible_request_and_response():
    transport = Recorder({"choices": [{"message": {"content": "Hello"}}]})
    model = OpenAICompatibleModel(
        "m1",
        api_key="k",
        base_url="https://example.test/v1/",
        extra_body={"temperature": 0},
        transport=transport,
    )
    assert model.complete(MESSAGES, max_tokens=50) == "Hello"
    assert transport.request["url"] == "https://example.test/v1/chat/completions"
    assert transport.request["headers"] == {"Authorization": "Bearer k"}
    assert transport.request["body"] == {
        "temperature": 0,
        "model": "m1",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "max_completion_tokens": 50,
    }


def test_openai_compatible_without_key_and_custom_field(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    transport = Recorder({"choices": [{"message": {"content": "ok"}}]})
    model = OpenAICompatibleModel("local", max_tokens_field="max_tokens", transport=transport)
    model.complete(MESSAGES)
    assert transport.request["headers"] == {}
    assert transport.request["body"]["max_tokens"] == 1024


@pytest.mark.parametrize("response", [{}, {"choices": []}, {"choices": [{"message": {}}]}])
def test_openai_compatible_bad_response(response):
    model = OpenAICompatibleModel("m", api_key="k", transport=Recorder(response))
    with pytest.raises(ModelError):
        model.complete(MESSAGES)


def test_anthropic_request_and_response():
    transport = Recorder(
        {"content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Hello"}]}
    )
    model = AnthropicModel("m2", api_key="k", transport=transport)
    assert model.complete(MESSAGES, max_tokens=50) == "Hello"
    assert transport.request["url"] == "https://api.anthropic.com/v1/messages"
    assert transport.request["headers"] == {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    assert transport.request["body"] == {
        "model": "m2",
        "max_tokens": 50,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Hi"}],
    }


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ModelError):
        AnthropicModel("m")


def test_anthropic_bad_response():
    model = AnthropicModel("m", api_key="k", transport=Recorder({"error": "x"}))
    with pytest.raises(ModelError):
        model.complete(MESSAGES)


def test_callable_model_and_protocol():
    model = CallableModel(lambda messages, max_tokens: f"{len(messages)}:{max_tokens}")
    assert model.complete(MESSAGES, max_tokens=7) == "2:7"
    assert isinstance(model, ChatModel)
    assert isinstance(OpenAICompatibleModel("m"), ChatModel)


def test_token_counting():
    assert approximate_token_count("") == 0
    assert approximate_token_count("abcde") == 2
    assert count_message_tokens([Message("user", "abcd")]) == 5
    assert count_message_tokens([Message("user", "abcd")], lambda text: 10) == 14
