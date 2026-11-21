import pytest

from gena_optimizer import CompactionError, Compactor, Message, Pipeline, PromptHygiene
from gena_optimizer.compaction import SUMMARY_HEADER


class FakeModel:
    """Returns queued replies and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, *, max_tokens=1024):
        self.calls.append((list(messages), max_tokens))
        return self.replies.pop(0)


def conversation(turns=10, size=400):
    messages = [Message("system", "You are helpful.")]
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(Message(role, f"turn {i} " + "x" * size))
    return messages


def test_below_threshold_is_untouched_and_model_not_called():
    model = FakeModel()
    messages = conversation()
    assert Compactor(model, trigger_tokens=10_000).optimize(messages) == messages
    assert model.calls == []


def test_compacts_older_turns_and_keeps_system_and_recent():
    model = FakeModel("  The summary.  ")
    messages = conversation()
    result = Compactor(model, trigger_tokens=500, keep_recent=2, summary_max_tokens=777).optimize(
        messages
    )

    assert result == [
        messages[0],
        Message("user", SUMMARY_HEADER + "The summary."),
        *messages[-2:],
    ]
    ((prompt, max_tokens),) = model.calls
    assert max_tokens == 777
    assert prompt[0].role == "system"
    assert "<user>\nturn 0 " in prompt[1].content
    assert "turn 7 " in prompt[1].content
    assert "turn 8 " not in prompt[1].content  # recent turns are not summarized
    assert "You are helpful." not in prompt[1].content  # nor are system messages


@pytest.mark.parametrize(
    "omissions",
    ["- budget is $4,200", "NONE of the requirements were preserved", "NONE\nAdditional issue"],
)
def test_verify_appends_omissions(omissions):
    model = FakeModel("Summary.", omissions)
    result = Compactor(model, trigger_tokens=500, keep_recent=0, verify=True).optimize(
        conversation()
    )
    assert result[1].content.endswith(f"Summary.\n\nAdditional details:\n{omissions}")
    assert "<summary>\nSummary.\n</summary>" in model.calls[1][0][1].content


@pytest.mark.parametrize("reply", ["NONE", " none "])
def test_verify_with_nothing_missing(reply):
    model = FakeModel("Summary.", reply)
    result = Compactor(model, trigger_tokens=500, verify=True).optimize(conversation())
    assert result[1].content == SUMMARY_HEADER + "Summary."


def test_keeps_original_when_summary_is_not_smaller():
    messages = conversation()
    model = FakeModel("y" * 100_000)
    assert Compactor(model, trigger_tokens=500).optimize(messages) == messages


def test_keeps_original_when_verification_makes_summary_too_large():
    messages = conversation()
    model = FakeModel("Summary.", "NONE of the details survived: " + "y" * 100_000)
    assert Compactor(model, trigger_tokens=500, verify=True).optimize(messages) == messages
    assert len(model.calls) == 2


def test_custom_summary_limit_applies_to_verification():
    model = FakeModel("Summary.", "- budget is $4,200")
    result = Compactor(
        model, trigger_tokens=500, keep_recent=0, verify=True, summary_max_tokens=777
    ).optimize(conversation())
    assert result[1].content.endswith("Additional details:\n- budget is $4,200")
    assert [max_tokens for _, max_tokens in model.calls] == [777, 777]


def test_nothing_to_summarize_when_all_turns_are_recent():
    model = FakeModel()
    messages = conversation(turns=4)
    assert Compactor(model, trigger_tokens=10, keep_recent=4).optimize(messages) == messages
    assert model.calls == []


def test_empty_summary_raises():
    with pytest.raises(CompactionError):
        Compactor(FakeModel("   "), trigger_tokens=500).optimize(conversation())


@pytest.mark.parametrize("kwargs", [{"trigger_tokens": 0}, {"keep_recent": -1}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        Compactor(FakeModel(), **kwargs)


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_summary_limit_raises_before_model_is_called(limit):
    model = FakeModel()
    with pytest.raises(ValueError, match="summary_max_tokens must be positive"):
        Compactor(model, summary_max_tokens=limit)
    assert model.calls == []


def test_custom_token_counter_controls_trigger():
    model = FakeModel("s")
    compactor = Compactor(
        model, trigger_tokens=50, keep_recent=2, token_counter=lambda text: len(text)
    )
    assert len(compactor.optimize(conversation(turns=8, size=10))) < 9


def test_pipeline_runs_in_order_and_reports_savings():
    model = FakeModel("Short summary.")
    pipeline = Pipeline(PromptHygiene(), Compactor(model, trigger_tokens=500, keep_recent=2))
    result = pipeline.run([m.to_dict() for m in conversation()])

    assert result.tokens_after < result.tokens_before
    assert result.tokens_saved == result.tokens_before - result.tokens_after
    assert 0 < result.ratio < 1
    assert pipeline.optimize(conversation(turns=1, size=1)) == conversation(turns=1, size=1)


def test_empty_pipeline_ratio():
    assert Pipeline().run([]).ratio == 1.0
