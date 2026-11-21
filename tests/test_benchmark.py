import json

import pytest

from gena_optimizer import CallableModel, Message, approximate_token_count


def sample():
    return {
        "id": "sample",
        "category": "numbers_units",
        "language": "en",
        "history": [
            {"role": "user", "content": "The amount is 19 kg. " + "background " * 100},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "Use the existing delivery schedule."},
            {"role": "assistant", "content": "Ready for the question."},
        ],
        "question": 'What is the amount? Reply with the JSON string field "amount".',
    }


def label():
    return {
        "id": "sample",
        "reference_answer": {"amount": "19 kg"},
        "required_facts": [
            {"field": "amount", "value": "19 kg", "context_pattern": r"amount is 19 kg"}
        ],
        "forbidden_claims": [{"field": "amount", "value": "91 kg"}],
    }


def test_structured_scoring_and_invalid_answer():
    from benchmarks.dataset import evaluate

    context = [Message("user", "The amount is 19 kg.")]
    good = evaluate('{"amount":"19 kg"}', label(), context)
    assert good["correct"] is True
    assert good["required_fact_recall"] == 1
    assert good["context_fact_recall"] == 1
    bad = evaluate('{"amount":"91 kg"}', label(), context)
    assert bad["correct"] is False
    assert bad["forbidden_claims"] == 1
    invalid = evaluate("not json", label(), context)
    assert invalid["valid_answer"] is False
    assert invalid["required_fact_recall"] == 0
    assert invalid["forbidden_claims"] is None


def test_dataset_validation_rejects_recent_only_facts():
    from benchmarks.dataset import validate

    case = sample()
    case["history"][0]["content"] = "No answer here."
    case["history"][-1]["content"] = "The amount is 19 kg."
    with pytest.raises(ValueError, match="old history"):
        validate([case], {"sample": label()})


@pytest.mark.parametrize(
    ("reply", "trigger", "expected"),
    [("short", 1, "accepted"), ("x" * 20000, 1, "rejected"), ("short", 100000, "not_triggered")],
)
def test_compaction_status_and_question_preservation(reply, trigger, expected):
    from benchmarks.metering import MeteredModel
    from benchmarks.run import run_case

    summarizer = MeteredModel(
        CallableModel(lambda messages, max_tokens: reply),
        name="fake-summary",
        counter=approximate_token_count,
        stages=("summary", "verification"),
    )
    seen = []

    def answer(messages, max_tokens):
        seen.append((list(messages), max_tokens))
        return '{"amount":"19 kg"}'

    downstream = MeteredModel(
        CallableModel(answer), name="fake-answer", counter=approximate_token_count
    )
    config = {
        "compaction": {"trigger_tokens": trigger, "keep_recent": 2, "summary_max_tokens": 80},
        "answer_max_tokens": 100,
    }
    result = run_case(
        sample(), label(), "compact", config, summarizer, downstream, approximate_token_count
    )
    assert result["compaction_status"] == expected
    assert seen[0][0][-1] == Message("user", sample()["question"])
    assert seen[0][1] == 100
    assert "reference_answer" not in str(seen[0][0])
    assert result["summary_calls"] == (0 if expected == "not_triggered" else 1)
    assert result["cost_usd"] is None


def test_pipeline_rejection_is_relative_to_hygiene_input():
    from benchmarks.metering import MeteredModel
    from benchmarks.run import run_case

    case = sample()
    case["history"][0]["content"] += "\n\n\n\nend\t"
    summarizer = MeteredModel(
        CallableModel(lambda m, n: "x" * 20000), name="fake", counter=approximate_token_count
    )
    downstream = MeteredModel(
        CallableModel(lambda m, n: "{}"), name="fake", counter=approximate_token_count
    )
    config = {
        "compaction": {"trigger_tokens": 1, "keep_recent": 2, "summary_max_tokens": 80},
        "answer_max_tokens": 100,
    }
    result = run_case(
        case, label(), "pipeline", config, summarizer, downstream, approximate_token_count
    )
    assert result["compaction_status"] == "rejected"
    assert result["context_after"] == result["compactor_input_tokens"]
    assert result["context_after"] < result["context_before"]
    assert result["summary_calls"] == 1


def test_error_is_retained_without_exception_contents():
    from benchmarks.metering import MeteredModel
    from benchmarks.run import run_case

    def fail(messages, max_tokens):
        raise RuntimeError("secret-do-not-log")

    summarizer = MeteredModel(CallableModel(fail), name="fake", counter=approximate_token_count)
    downstream = MeteredModel(
        CallableModel(lambda m, n: "{}"), name="fake", counter=approximate_token_count
    )
    config = {
        "compaction": {"trigger_tokens": 1, "keep_recent": 2, "summary_max_tokens": 80},
        "answer_max_tokens": 100,
    }
    result = run_case(
        sample(), label(), "compact", config, summarizer, downstream, approximate_token_count
    )
    assert result["compaction_status"] == "error"
    assert result["error"] == "RuntimeError"
    assert result["calls"] == 1
    assert "secret-do-not-log" not in json.dumps(result)


def test_dataset_balance_and_reference_rules():
    from benchmarks.dataset import evaluate, load, validate

    cases, labels = load()
    dev, dev_labels = load("dev")
    stats = validate(cases, labels)
    assert stats["cases"] == 28
    assert set(stats["categories"].values()) == {4}
    assert stats["languages"] == {"ru": 14, "en": 14}
    assert len(dev) == 4 and not set(labels) & set(dev_labels)
    for case in cases:
        context = [Message(m["role"], m["content"]) for m in case["history"]]
        scored = evaluate(
            json.dumps(labels[case["id"]]["reference_answer"]), labels[case["id"]], context
        )
        assert scored["correct"] and scored["context_fact_recall"] == 1


def test_default_offline_smoke_writes_all_five_modes(tmp_path):
    from benchmarks.run import main

    assert main(["--limit", "1", "--output", str(tmp_path)]) == 0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert {row["mode"] for row in rows} == {
        "original",
        "hygiene",
        "compact",
        "verified",
        "pipeline",
    }
    assert all(row["cost_usd"] is None for row in rows)
    assert all(row["compaction_status"] == "accepted" for row in rows[2:])
    assert all(row["answer"] == "{}" for row in rows)
    assert "FAKE SMOKE" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "results.csv").is_file()


def test_calibration_can_limit_development_subset(tmp_path):
    from benchmarks.dataset import load
    from benchmarks.run import calibrate, configuration

    cases, labels = load("dev")
    result = calibrate(configuration(), cases[:1], labels, tmp_path)
    assert result["development_ids"] == [cases[0]["id"]]
    assert result["compaction"]["keep_recent"] >= 1


def test_calibration_pins_response_settings_and_scoring_code(tmp_path):
    from benchmarks.dataset import load
    from benchmarks.run import calibrate, calibration_matches, configuration

    cases, labels = load("dev")
    config = configuration()
    result = calibrate(config, cases[:1], labels, tmp_path)
    result["live"] = True  # Only exercise compatibility validation; no provider calls.
    assert calibration_matches(result, config)
    changed = {**config, "answer_max_tokens": 1}
    assert not calibration_matches(result, changed)
    result["code_sha256"] = "different scoring implementation"
    assert not calibration_matches(result, config)


@pytest.mark.parametrize("args", [["live"], ["calibrate-live"], ["live", "--limit", "0"]])
def test_live_calls_require_explicit_bounds(args):
    from benchmarks.run import main

    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("case_id", "wrong_context"),
    [
        (
            "entity_ids-ru-01",
            "Спектрометр «Лира» имеет инвентарный номер SP-0401 и "
            "серийный номер LY-1080. Спектрометр «Лира-М» имеет инвентарный номер SP-04O1 "  # noqa: RUF001 -- intentional Cyrillic entity
            "и серийный номер LY-1008.",
        ),
        (
            "entity_ids-ru-02",
            "Для проекта «Маяк» заявка о сетевом доступе имеет ID NET-0791 "  # noqa: RUF001 -- Russian text
            "и владельца Анна Сокол. Для проекта «Маяк-2» заявка о сетевом доступе имеет ID "  # noqa: RUF001 -- Russian text
            "NET-0719 и владельца Анна Соколова.",
        ),
        (
            "entity_ids-en-01",
            "Pump Kestrel has asset ID PU-1108 and serial KS-3060. "
            "Pump Kestrel-S has asset ID PU-1I08 and serial KS-3006.",
        ),
        (
            "entity_ids-en-02",
            "Project Harbor has access ticket AC-2078, owned by Elina Woods. "
            "Project Harbour has access ticket AC-2087, owned by Elena Wood.",
        ),
    ],
)
def test_context_scoring_rejects_facts_bound_to_the_wrong_entity(case_id, wrong_context):
    from benchmarks.dataset import evaluate, load

    _, labels = load()
    result = evaluate("{}", labels[case_id], [Message("user", wrong_context)])
    assert result["context_fact_recall"] == 0


def test_rejected_verified_compaction_still_charges_all_calls():
    from benchmarks.metering import MeteredModel, UsageTransport
    from benchmarks.run import run_case

    from gena_optimizer import OpenAICompatibleModel

    replies = iter(["Short summary.", "correction " * 2000, '{"amount":"19 kg"}'])

    def transport(url, headers, body, timeout):
        return {
            "choices": [{"message": {"content": next(replies)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    usage = UsageTransport("openai", transport)
    client = OpenAICompatibleModel("fake", api_key="fake", transport=usage)
    prices = {"input_per_million": 1, "output_per_million": 2}
    summarizer = MeteredModel(
        client,
        name="fake",
        counter=approximate_token_count,
        prices=prices,
        transport=usage,
        stages=("summary", "verification"),
    )
    downstream = MeteredModel(
        client, name="fake", counter=approximate_token_count, prices=prices, transport=usage
    )
    config = {
        "compaction": {"trigger_tokens": 1, "keep_recent": 2, "summary_max_tokens": 80},
        "answer_max_tokens": 100,
    }
    result = run_case(
        sample(), label(), "verified", config, summarizer, downstream, approximate_token_count
    )
    assert result["compaction_status"] == "rejected"
    assert result["calls"] == 3 and result["verification_calls"] == 1
    assert result["cost_usd"] == pytest.approx(0.00015)
    assert result["by_stage"]["verification"]["calls"] == 1
