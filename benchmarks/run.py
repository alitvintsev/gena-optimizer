"""Run with no arguments for offline validation, development calibration and smoke."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from gena_optimizer import (
    AnthropicModel,
    CallableModel,
    Compactor,
    Message,
    OpenAICompatibleModel,
    Pipeline,
    PromptHygiene,
    approximate_token_count,
    count_message_tokens,
    tiktoken_counter,
)
from gena_optimizer.compaction import DEFAULT_INSTRUCTIONS

from .dataset import VERSION, evaluate, fingerprint, load, validate
from .metering import MeteredModel, UsageTransport, aggregate_records

ROOT = Path(__file__).resolve().parents[1]
MODES = ("original", "hygiene", "compact", "verified", "pipeline")
ANSWER_INSTRUCTIONS = (
    "Answer the last user question using the conversation. Return only a JSON object "
    "with the requested field names and string values. Copy exact numbers, units, IDs, "
    "URLs and code when asked. Use the latest corrections; do not invent missing facts."
)
DEFAULT = {
    "token_counter": {"kind": "approximate"},
    "compaction": {"trigger_tokens": 200, "keep_recent": 2, "summary_max_tokens": 256},
    "answer_max_tokens": 512,
}
CANDIDATES = [
    {"trigger_tokens": 100000, "keep_recent": 2, "summary_max_tokens": 256},
    {"trigger_tokens": 200, "keep_recent": 2, "summary_max_tokens": 256},
    {"trigger_tokens": 400, "keep_recent": 2, "summary_max_tokens": 384},
]


def run_case(case, label, mode, config, summarizer, downstream, counter):
    """Use normal optimize/run, observing fallback against Compactor's own input."""
    if mode not in MODES:
        raise ValueError("unknown comparison mode")
    summarizer.reset()
    downstream.reset()
    messages = [Message("system", ANSWER_INSTRUCTIONS)]
    messages += [Message(m["role"], m["content"]) for m in case["history"]]
    messages += [Message("user", case["question"])]
    compactor_input = PromptHygiene().optimize(messages) if mode == "pipeline" else messages
    before = count_message_tokens(messages, counter)
    status, error, answer = "not_applicable", None, ""
    optimized = None
    phase = "optimization"
    started = time.perf_counter()
    try:
        compactor = Compactor(
            summarizer,
            verify=mode == "verified",
            token_counter=counter,
            instructions=config.get("summary_instructions", DEFAULT_INSTRUCTIONS),
            **config["compaction"],
        )
        if mode == "original":
            optimized = messages
        elif mode == "hygiene":
            optimized = PromptHygiene().optimize(messages)
        elif mode == "pipeline":
            optimized = (
                Pipeline(PromptHygiene(), compactor, token_counter=counter).run(messages).messages
            )
        else:
            optimized = compactor.optimize(messages)
        if mode in {"compact", "verified", "pipeline"}:
            status = (
                "not_triggered"
                if not summarizer.records
                else "rejected"
                if optimized == compactor_input
                else "accepted"
            )
        if optimized[-1] != compactor_input[-1]:
            raise ValueError("current question did not survive optimization")
        phase = "downstream"
        answer = downstream.complete(optimized, max_tokens=config["answer_max_tokens"])
    except Exception as exc:
        # Provider exceptions may contain headers or response text. Log only the type.
        error = type(exc).__name__
        if phase == "optimization":
            status = "error"
    records = [*summarizer.records, *downstream.records]
    result = {
        "id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "mode": mode,
        "compaction_status": status,
        "error": error,
        "context_before": before,
        "compactor_input_tokens": count_message_tokens(compactor_input, counter),
        "context_after": count_message_tokens(optimized, counter) if optimized else None,
        "summary_calls": min(1, len(summarizer.records)),
        "verification_calls": max(0, len(summarizer.records) - 1),
        "answer": answer,
        "reference_answer": label["reference_answer"],
        "context": [m.to_dict() for m in optimized] if optimized else None,
        "scenario_latency_seconds": time.perf_counter() - started,
        "call_records": records,
        **aggregate_records(records),
        **evaluate(answer, label, optimized or []),
    }
    if error:
        result["correct"] = False
    return result


def token_counter(config):
    spec = config["token_counter"]
    if spec["kind"] == "approximate":
        return approximate_token_count
    if spec["kind"] == "tiktoken":
        return tiktoken_counter(spec.get("encoding", "o200k_base"))
    raise ValueError("token_counter.kind must be approximate or tiktoken")


def configuration(path=None):
    config = copy.deepcopy(DEFAULT)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or set(supplied) - {*DEFAULT, "summarizer", "downstream"}:
            raise ValueError("unknown configuration fields")
        config.update(supplied)
    params = config["compaction"]
    if set(params) != {"trigger_tokens", "keep_recent", "summary_max_tokens"}:
        raise ValueError("compaction must specify trigger_tokens, keep_recent, summary_max_tokens")
    if any(type(value) is not int for value in params.values()):
        raise ValueError("compaction parameters must be integers")
    if min(params.values()) <= 0 or type(config["answer_max_tokens"]) is not int:
        raise ValueError("positive limits and keep_recent >= 1 are required")
    if config["answer_max_tokens"] <= 0:
        raise ValueError("answer_max_tokens must be positive")
    for role in ("summarizer", "downstream"):
        if role not in config:
            continue
        model = config[role]
        if set(model) - {
            "provider",
            "model",
            "base_url",
            "api_key_env",
            "temperature",
            "reasoning_effort",
            "max_tokens_field",
            "timeout",
            "prices",
        }:
            raise ValueError(f"{role}: unknown model fields; use api_key_env for credentials")
        if model.get("provider") not in {"openai", "anthropic"} or not model.get("model"):
            raise ValueError(f"{role}: provider and model are required")
        if "reasoning_effort" in model and (
            model["provider"] != "openai"
            or model["reasoning_effort"] not in ("low", "medium", "high", "xhigh", "max")
        ):
            raise ValueError(
                f"{role}: reasoning_effort requires an OpenAI-compatible model "
                "and low, medium, high, xhigh or max"
            )
        if "base_url" in model:
            url = urlsplit(model["base_url"])
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    f"{role}: base_url must not contain credentials or query parameters"
                )
    return config


def models(config, counter, *, live=False):
    result = []
    for role in ("summarizer", "downstream"):
        stages = ("summary", "verification") if role == "summarizer" else ("downstream",)
        if not live:

            def fake(messages, max_tokens, stage=role):
                if stage == "downstream":
                    return "{}"
                return (
                    "NONE"
                    if "List the omissions now." in messages[-1].content
                    else ("Offline synthetic summary; no quality claims.")
                )

            result.append(
                MeteredModel(
                    CallableModel(fake), name=f"fake-{role}", counter=counter, stages=stages
                )
            )
            continue
        spec = config.get(role)
        if not spec:
            raise ValueError(f"{role}: live mode requires a model configuration")
        env_name = spec.get("api_key_env")
        if not isinstance(env_name, str) or not os.environ.get(env_name):
            raise ValueError(f"{role}: configured API key environment variable is missing")
        transport = UsageTransport(spec["provider"])
        options = {
            "api_key": os.environ[env_name],
            "transport": transport,
            "extra_body": {"temperature": spec.get("temperature", 0)},
            "timeout": spec.get("timeout", 60),
        }
        if "reasoning_effort" in spec:
            options["extra_body"]["reasoning_effort"] = spec["reasoning_effort"]
        if "base_url" in spec:
            options["base_url"] = spec["base_url"]
        if spec["provider"] == "openai":
            options["max_tokens_field"] = spec.get("max_tokens_field", "max_completion_tokens")
            client = OpenAICompatibleModel(spec["model"], **options)
        else:
            client = AnthropicModel(spec["model"], **options)
        result.append(
            MeteredModel(
                client,
                name=f"{spec['provider']}:{spec['model']}",
                counter=counter,
                prices=spec.get("prices"),
                transport=transport,
                stages=stages,
            )
        )
    return result


def code_fingerprint():
    code = hashlib.sha256()
    paths = sorted((ROOT / "src/gena_optimizer").glob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py"))
    for path in paths:
        code.update(str(path.relative_to(ROOT)).encode())
        code.update(path.read_bytes())
    return code.hexdigest()


def experiment_settings(config):
    return {
        key: config.get(key)
        for key in ("answer_max_tokens", "token_counter", "summarizer", "downstream")
    }


def calibration_matches(calibration, config):
    return (
        calibration.get("live") is True
        and calibration.get("dataset_sha256") == fingerprint()
        and calibration.get("code_sha256") == code_fingerprint()
        and calibration.get("settings") == experiment_settings(config)
    )


def provenance(config, *, live, split, cases, calibration=None):

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    try:
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "dataset_version": VERSION,
        "dataset_sha256": fingerprint(),
        "scoring_and_code_sha256": code_fingerprint(),
        "commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "live": live,
        "split": split,
        "case_ids": [case["id"] for case in cases],
        "config": config,
        "calibration": calibration,
        "local_counter_note": (
            "Context sizes include estimated message framing; provider usage is separate."
        ),
        "quality_note": "Real model results"
        if live
        else "FAKE SMOKE: no model quality conclusions",
    }


def compare(cases, labels, config, *, live=False, modes=MODES):
    counter = token_counter(config)
    summarizer, downstream = models(config, counter, live=live)
    return [
        run_case(case, labels[case["id"]], mode, config, summarizer, downstream, counter)
        for case in cases
        for mode in modes
    ]


def summarize(rows):
    calls = [call for row in rows for call in row["call_records"]]
    return {
        "scenarios": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "correct_rate": sum(row["correct"] for row in rows) / len(rows),
        "required_fact_recall": sum(r["required_fact_recall"] for r in rows) / len(rows),
        "context_fact_recall": sum(r["context_fact_recall"] for r in rows) / len(rows),
        "invalid_answers": sum(not r["valid_answer"] for r in rows),
        "forbidden_claims": sum(r["forbidden_claims"] or 0 for r in rows),
        "unknown_forbidden": sum(r["forbidden_claims"] is None for r in rows),
        "compaction_status": dict(Counter(r["compaction_status"] for r in rows)),
        "context_before": sum(r["context_before"] for r in rows),
        "context_after_known": sum(r["context_after"] or 0 for r in rows),
        "unknown_contexts": sum(r["context_after"] is None for r in rows),
        **aggregate_records(calls),
    }


def save(output, rows, metadata):
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    grouped = {
        f"{mode}/{category}": summarize(
            [r for r in rows if r["mode"] == mode and r["category"] == category]
        )
        for mode in sorted({r["mode"] for r in rows})
        for category in sorted({r["category"] for r in rows})
        if any(r["mode"] == mode and r["category"] == category for r in rows)
    }
    summary = {
        "overall": summarize(rows),
        "by_mode": {
            mode: summarize([r for r in rows if r["mode"] == mode])
            for mode in sorted({r["mode"] for r in rows})
        },
        "by_mode_category": grouped,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Conversation compression benchmark",
        "",
        metadata["quality_note"],
        "",
        f"Dataset: `{VERSION}` / `{metadata['dataset_sha256']}`.",
        f"Commit: `{metadata['commit']}`; dirty: `{metadata['dirty']}`.",
        f"Code/scoring SHA256: `{metadata['scoring_and_code_sha256']}`.",
        f"Compaction: `{json.dumps(metadata['config']['compaction'])}`.",
        f"Local counter: `{json.dumps(metadata['config']['token_counter'])}`.",
        f"Models and response settings: `{json.dumps(experiment_settings(metadata['config']))}`.",
        "Settings, models, prices and counter: [metadata.json](metadata.json).",
        "",
        "Unknown usage/cost is not zero. All calls, including rejected compaction, are charged.",
        "Context fact recall uses lexical patterns and can miss valid paraphrases.",
        "",
        "A/R/N = compaction accepted / rejected / not triggered.",
        "Forbidden shows detected claims / answers that could not be checked.",
        "| Mode/category | n | Correct | Facts | Forbidden/? | Errors | A/R/N | Context | "
        "Calls | USD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for name, group in {**summary["by_mode"], **grouped}.items():
        cost = "unknown" if group["cost_usd"] is None else f"{group['cost_usd']:.6f}"
        states = group["compaction_status"]
        counts = "/".join(
            str(states.get(state, 0)) for state in ("accepted", "rejected", "not_triggered")
        )
        context = f"{group['context_before']} → {group['context_after_known']}"
        if group["unknown_contexts"]:
            context += f" + {group['unknown_contexts']} unknown"
        lines.append(
            f"| {name} | {group['scenarios']} | {group['correct_rate']:.1%} | "
            f"{group['required_fact_recall']:.1%} | "
            f"{group['forbidden_claims']}/{group['unknown_forbidden']} | "
            f"{group['errors']} | {counts} | {context} | {group['calls']} | {cost} |"
        )
    lines += ["", "## Quality regressions", ""]
    baselines = {r["id"]: r for r in rows if r["mode"] == "original"}
    regressions = [
        r
        for r in rows
        if r["mode"] != "original"
        and not r["correct"]
        and baselines.get(r["id"], {}).get("correct")
    ]
    if not metadata["live"]:
        lines.append("Not evaluated: fake outputs test plumbing only.")
    elif not baselines:
        lines.append("No original-history baseline in this development calibration run.")
    elif not regressions:
        lines.append("No regressions relative to correct original-history answers in this run.")
    for row in regressions[:10] if metadata["live"] else []:
        lines += [
            f"- `{row['id']}` / `{row['mode']}`: expected "
            f"`{json.dumps(row['reference_answer'], ensure_ascii=False)}`; observed "
            f"`{json.dumps(row['answer'], ensure_ascii=False)}`."
        ]
    lines += [
        "",
        "All incorrect answers, including original-history errors, remain in results.jsonl.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def calibrate(config, cases, labels, output, *, live=False):
    labels = {case["id"]: labels[case["id"]] for case in cases}
    trials = []
    for index, params in enumerate(CANDIDATES):
        candidate = {**config, "compaction": params}
        validate(cases, labels, keep_recent=params["keep_recent"])
        rows = compare(cases, labels, candidate, live=live, modes=MODES[2:])
        summary = save(
            output / f"candidate-{index}",
            rows,
            provenance(candidate, live=live, split="dev", cases=cases),
        )
        accepted = sum(row["compaction_status"] == "accepted" for row in rows)
        trials.append(
            {
                "parameters": params,
                "accepted": accepted,
                "quality": summary["overall"]["correct_rate"],
                "context_recall": summary["overall"]["context_fact_recall"],
                "errors": summary["overall"]["errors"],
            }
        )
    eligible = [trial for trial in trials if trial["accepted"] and not trial["errors"]]
    if not eligible:
        raise ValueError(
            "no development candidate accepted compaction; inspect calibration results"
        )
    chosen = max(
        eligible, key=lambda trial: (trial["quality"], trial["context_recall"], trial["accepted"])
    )
    result = {
        "live": live,
        "dataset_version": VERSION,
        "dataset_sha256": fingerprint(),
        "code_sha256": code_fingerprint(),
        "settings": experiment_settings(config),
        "compaction": chosen["parameters"],
        "trials": trials,
        "development_ids": [case["id"] for case in cases],
        "models": {key: config.get(key) for key in ("summarizer", "downstream")},
        "token_counter": config["token_counter"],
    }
    (output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="smoke",
        choices=("validate", "smoke", "calibrate-live", "live"),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--limit", type=int, help="maximum cases, not calls; live <=9 calls/case, calibration <=21"
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    args = parser.parse_args(argv)
    live = args.command in {"live", "calibrate-live"}
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if live and (not args.config or args.limit is None):
        parser.error("live commands require --config and an explicit positive --limit")
    try:
        config = configuration(args.config)
        cases, labels = load(keep_recent=config["compaction"]["keep_recent"])
        dev, dev_labels = load("dev", keep_recent=config["compaction"]["keep_recent"])
        if not 24 <= len(cases) <= 30 or set(labels) & set(dev_labels):
            raise ValueError("expected 24-30 evaluation cases disjoint from development")
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "eval": validate(cases, labels),
                        "dev": validate(dev, dev_labels),
                        "dataset_sha256": fingerprint(),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "calibrate-live":
            calibrate(config, dev[: args.limit], dev_labels, args.output, live=True)
            return 0
        if live:
            if not args.calibration:
                raise ValueError("live evaluation requires --calibration from calibrate-live")
            calibration = json.loads(args.calibration.read_text())
            if not calibration_matches(calibration, config):
                raise ValueError("calibration must match real models, settings, code and dataset")
        else:
            calibration = calibrate(config, dev, dev_labels, args.output / "development")
        config["compaction"] = calibration["compaction"]
        selected = cases[: args.limit] if args.limit else cases
        validate(
            selected,
            {case["id"]: labels[case["id"]] for case in selected},
            keep_recent=config["compaction"]["keep_recent"],
        )
        rows = compare(selected, labels, config, live=live)
        summary = save(
            args.output,
            rows,
            provenance(config, live=live, split="eval", cases=selected, calibration=calibration),
        )
        print(json.dumps({"output": str(args.output), **summary["overall"]}, ensure_ascii=False))
        return int(bool(summary["overall"]["errors"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
