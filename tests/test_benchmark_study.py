import copy
import json

import pytest

from gena_optimizer import CallableModel, Message, approximate_token_count


def test_run_case_forwards_summary_instructions_and_budget():
    from benchmarks.dataset import load
    from benchmarks.metering import MeteredModel
    from benchmarks.run import configuration, run_case

    cases, labels = load("dev")
    seen = []

    def summarize(messages, max_tokens):
        seen.append((messages, max_tokens))
        return "Short summary."

    summary = MeteredModel(CallableModel(summarize), name="fake", counter=approximate_token_count)
    answer = MeteredModel(
        CallableModel(lambda m, n: "{}"), name="fake", counter=approximate_token_count
    )
    config = configuration()
    config["summary_instructions"] = "Preserve exact entity-value bindings."
    config["compaction"]["summary_max_tokens"] = 512
    row = run_case(
        cases[0],
        labels[cases[0]["id"]],
        "compact",
        config,
        summary,
        answer,
        approximate_token_count,
    )
    assert seen[0][0][0] == Message("system", "Preserve exact entity-value bindings.")
    assert seen[0][1] == 512
    assert row["context"][-1] == {"role": "user", "content": cases[0]["question"]}
    assert "reference_answer" not in str(seen[0][0])


def test_study_runs_single_baseline_and_rotates_six_fixed_variants(tmp_path):
    from benchmarks.run import configuration
    from benchmarks.study import run_study

    result = run_study(configuration(), limit=2, output=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 14
    assert sum(row["mode"] == "original" for row in rows) == 2
    assert sum(row["calls"] for row in rows) == 26
    assert {row["mode"] for row in rows} == {
        "original",
        "prose-256",
        "prose-512",
        "prose-1024",
        "structured-256",
        "structured-512",
        "structured-1024",
    }
    variants = [row for row in rows if row["mode"] != "original"]
    assert all(row["verification_calls"] == 0 for row in variants)
    assert all(row["summary_budget"] in (256, 512, 1024) for row in variants)
    assert rows[0]["mode"] == "original" and rows[7]["mode"] == "prose-256"
    assert result["execution_order"][0] == {"id": rows[0]["id"], "mode": "original"}
    assert len((tmp_path / "checkpoint.jsonl").read_text().splitlines()) == 14
    assert result["summary"]["overall"]["calls"] == 26
    assert "FAKE SMOKE" in (tmp_path / "report.md").read_text()


def test_fixed_variants_propagate_budgets_and_prompts_to_models(tmp_path, monkeypatch):
    from benchmarks import run, study
    from benchmarks.metering import MeteredModel

    from gena_optimizer.compaction import DEFAULT_INSTRUCTIONS

    seen = []

    def summarize(messages, max_tokens):
        seen.append((messages[0].content, max_tokens))
        return "summary"

    def models(config, counter, *, live):
        return (
            MeteredModel(CallableModel(summarize), name="fake", counter=counter),
            MeteredModel(CallableModel(lambda m, n: "{}"), name="fake", counter=counter),
        )

    monkeypatch.setattr(run, "models", models)
    study.run_study(run.configuration(), limit=1, output=tmp_path)
    assert [budget for _, budget in seen] == [256, 512, 1024, 256, 512, 1024]
    assert [prompt for prompt, _ in seen[:3]] == [DEFAULT_INSTRUCTIONS] * 3
    assert all("entity" in prompt and "superseded" in prompt for prompt, _ in seen[3:])
    assert all("19 kg" not in prompt for prompt, _ in seen)


def test_checkpoint_retains_completed_scenarios_after_interruption(tmp_path, monkeypatch):
    from benchmarks import run, study
    from benchmarks.metering import MeteredModel

    def interrupt(messages, max_tokens):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        run,
        "models",
        lambda config, counter, live: (
            MeteredModel(CallableModel(interrupt), name="fake", counter=counter),
            MeteredModel(CallableModel(lambda m, n: "{}"), name="fake", counter=counter),
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        study.run_study(run.configuration(), limit=1, output=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "checkpoint.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["mode"] == "original"
    assert rows[1]["error"] == "KeyboardInterrupt"
    assert rows[1]["calls"] == 1 and rows[1]["call_records"][0]["error"] == "KeyboardInterrupt"
    assert (tmp_path / "metadata.json").is_file()
    with pytest.raises(ValueError, match="already"):
        study.run_study(run.configuration(), limit=1, output=tmp_path)


def test_rejected_summaries_keep_cost_and_are_ineligible(tmp_path, monkeypatch):
    from benchmarks import run, study
    from benchmarks.metering import MeteredModel, UsageTransport

    from gena_optimizer import OpenAICompatibleModel

    def models(config, counter, *, live):
        metered = []
        for role in ("summary", "downstream"):

            def transport(url, headers, body, timeout, role=role):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "x" * 100000 if role == "summary" else "{}",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }

            usage = UsageTransport("openai", transport)
            metered.append(
                MeteredModel(
                    OpenAICompatibleModel("fake", api_key="fake", transport=usage),
                    name=role,
                    counter=counter,
                    transport=usage,
                    prices={"input_per_million": 1, "output_per_million": 2},
                )
            )
        return metered

    monkeypatch.setattr(run, "models", models)
    result = study.run_study(run.configuration(), limit=1, output=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert all(row["compaction_status"] == "rejected" for row in rows[1:])
    assert all(row["cost_usd"] == pytest.approx(0.0001) for row in rows[1:])
    assert result["selected_variant"] is None
    assert result["summary"]["overall"]["cost_usd"] == pytest.approx(0.00065)


def test_eval_requires_matching_dev_manifest_and_keeps_dev_selection(tmp_path, monkeypatch):
    from benchmarks import study
    from benchmarks.run import configuration

    config = configuration()
    dev = study.run_study(config, limit=1, output=tmp_path / "dev")
    dev["selected_variant"] = "structured-1024"
    monkeypatch.setattr(study, "select_variant", lambda rows: pytest.fail("eval must not select"))
    result = study.run_study(
        config,
        split="eval",
        limit=1,
        output=tmp_path / "eval",
        calibration=dev,
    )
    assert result["selected_variant"] == "structured-1024"
    assert not set(result["case_ids"]) & set(dev["case_ids"])
    assert result["summary"]["overall"]["scenarios"] == 7
    for field in ("config", "code_sha256", "dataset_sha256", "grid"):
        mismatched = copy.deepcopy(dev)
        mismatched["manifest"][field] = None
        with pytest.raises(ValueError, match="match"):
            study.run_study(
                config, split="eval", limit=1, output=tmp_path / field, calibration=mismatched
            )
    with pytest.raises(ValueError, match="real"):
        study.run_study(
            config, split="eval", limit=1, live=True, output=tmp_path / "live", calibration=dev
        )
    overlap = copy.deepcopy(dev)
    overlap["case_ids"] = result["case_ids"]
    with pytest.raises(ValueError, match="disjoint"):
        study.run_study(
            config, split="eval", limit=1, output=tmp_path / "overlap", calibration=overlap
        )


def test_selection_prioritizes_quality_and_excludes_errors_and_no_compaction():
    from benchmarks.study import select_variant

    def row(
        mode, status="accepted", correct=True, recall=1, forbidden=0, after=50, cost=0.1, error=None
    ):
        return {
            "mode": mode,
            "compaction_status": status,
            "error": error,
            "correct": correct,
            "required_fact_recall": recall,
            "forbidden_claims": forbidden,
            "context_before": 100,
            "context_after": after,
            "cost_usd": cost,
        }

    assert (
        select_variant(
            [
                row("original"),
                row("prose-256", correct=False, after=1),
                row("prose-512", status="rejected", after=1),
                row("prose-1024", error="TimeoutError", after=1),
                row("structured-256", forbidden=1, after=1),
                row("structured-512", after=40),
                row("structured-1024", after=30),
            ]
        )
        == "structured-1024"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["dev-live"],
        ["eval-live"],
        ["dev-live", "--limit", "0"],
        ["dev-live", "--config", "unused.json"],
        ["eval-live", "--config", "unused.json", "--limit", "1"],
        ["smoke", "--limit", "-1"],
    ],
)
def test_live_cli_requires_config_limits_and_dev_calibration(args):
    from benchmarks.study import main

    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_default_cli_is_offline_smoke(tmp_path):
    from benchmarks.study import main

    assert main(["--limit", "1", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "calibration.json").is_file()
    assert json.loads((tmp_path / "metadata.json").read_text())["live"] is False
