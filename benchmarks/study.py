"""Fixed summary budget/format study; offline smoke unless explicitly run live.

Each case has one original baseline and six ordinary (unverified) compactions:
maximum 13 provider calls, without retries. Baseline usage is never duplicated.
A summary budget limits provider completion tokens, including reasoning when the
provider counts reasoning in its completion limit; it is not a length guarantee.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from gena_optimizer.compaction import DEFAULT_INSTRUCTIONS

from . import run
from .dataset import fingerprint, load

STRUCTURED_INSTRUCTIONS = (
    DEFAULT_INSTRUCTIONS
    + """

Use a compact structured record with these named sections:
- entity: bind each fact to the exact person, object, or task it describes;
- value / unit: retain exact values, numbers, dates, IDs and units together;
- source: retain source-to-claim bindings, file paths and exact URLs;
- status: distinguish done, in progress, blocked, next and open questions;
- superseded: mark obsolete facts explicitly and give the current correction;
- code: retain relevant code, commands, JSON and errors exactly, verbatim.
Preserve negations, exceptions, constraints, goals, decisions and their reasons.
Do not move a value, identifier, status or source to a different entity. Omit empty
sections. Keep the original language; reply only with the structured summary.
"""
)
FORMATS = {"prose": DEFAULT_INSTRUCTIONS, "structured": STRUCTURED_INSTRUCTIONS}
GRID = tuple(
    {
        "name": f"{format_name}-{budget}",
        "summary_format": format_name,
        "summary_budget": budget,
        "instructions": instructions,
    }
    for format_name, instructions in FORMATS.items()
    for budget in (256, 512, 1024)
)


def manifest(config):
    """Freeze every treatment and shared setting before looking at evaluation."""
    return {
        "study_version": 1,
        "dataset_sha256": fingerprint(),
        "code_sha256": run.code_fingerprint(),
        "config": copy.deepcopy(config),
        "grid": list(GRID),
        "verify": False,
        "max_calls_per_case": 13,
        "ordering": "rotate original + prose ascending + structured ascending by case index",
    }


def select_variant(rows):
    """Choose only on development: eligibility, quality, safety, then savings/cost."""
    candidates = []
    for mode in sorted({row["mode"] for row in rows} - {"original"}):
        group = [row for row in rows if row["mode"] == mode]
        if any(row["error"] for row in group) or not any(
            row["compaction_status"] == "accepted" for row in group
        ):
            continue
        score = (
            sum(row["correct"] for row in group) / len(group),
            sum(row["required_fact_recall"] for row in group) / len(group),
            -sum(row["forbidden_claims"] or 0 for row in group),
            -sum(row["forbidden_claims"] is None for row in group),
            sum(
                row["context_before"] - (row["context_after"] or row["context_before"])
                for row in group
            ),
            -sum(row["cost_usd"] if row["cost_usd"] is not None else float("inf") for row in group),
        )
        candidates.append((score, mode))
    return max(candidates)[1] if candidates else None


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _checkpoint(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_study(
    config,
    *,
    split="dev",
    limit=None,
    live=False,
    output=Path("artifacts/summary-study"),
    calibration=None,
):
    """Persist each completed scenario before the next potentially billed call.

    ``calibration`` is the complete development artifact (calibration.json).
    Evaluation reports every fixed variant and carries its preselected winner;
    evaluation labels never select a new winner.
    """
    if split not in {"dev", "eval"}:
        raise ValueError("split must be dev or eval")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if live and limit is None:
        raise ValueError("live study requires an explicit positive limit")
    frozen = manifest(config)
    cases, labels = load(split, keep_recent=config["compaction"]["keep_recent"])
    other, _ = load(
        "eval" if split == "dev" else "dev", keep_recent=config["compaction"]["keep_recent"]
    )
    if {case["id"] for case in cases} & {case["id"] for case in other}:
        raise ValueError("development and evaluation must be disjoint")
    cases = cases[:limit]
    if split == "eval":
        if not calibration or calibration.get("split") != "dev":
            raise ValueError("evaluation requires development calibration")
        if live and calibration.get("live") is not True:
            raise ValueError("live evaluation requires real development calibration")
        if calibration.get("manifest") != frozen:
            raise ValueError("development manifest must match config, grid, code and data exactly")
        if {case["id"] for case in cases} & set(calibration.get("case_ids", [])):
            raise ValueError("development and evaluation IDs must be disjoint")
        if calibration.get("selected_variant") not in {item["name"] for item in GRID}:
            raise ValueError("development must select an eligible variant before evaluation")
    output = Path(output)
    checkpoint = output / "checkpoint.jsonl"
    if checkpoint.exists():
        raise ValueError("output already contains a study checkpoint; use a new output directory")
    counter = run.token_counter(config)
    summarizer, downstream = run.models(config, counter, live=live)
    metadata = run.provenance(config, live=live, split=split, cases=cases)
    metadata.update({"manifest": frozen, "execution_order": [], "selected_variant": None})
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metadata.json", metadata)
    treatments = (None, *GRID)
    rows = []
    with checkpoint.open("x", encoding="utf-8") as stream:
        for index, case in enumerate(cases):
            shift = index % len(treatments)
            order = treatments[shift:] + treatments[:shift]
            for treatment in order:
                candidate = copy.deepcopy(config)
                if treatment:
                    candidate["compaction"]["summary_max_tokens"] = treatment["summary_budget"]
                    candidate["summary_instructions"] = treatment["instructions"]
                identity = {
                    "id": case["id"],
                    "mode": treatment["name"] if treatment else "original",
                    "summary_budget": treatment["summary_budget"] if treatment else None,
                    "summary_format": treatment["summary_format"] if treatment else None,
                    "execution_index": len(rows),
                }
                try:
                    row = run.run_case(
                        case,
                        labels[case["id"]],
                        "compact" if treatment else "original",
                        candidate,
                        summarizer,
                        downstream,
                        counter,
                    )
                except BaseException as exc:
                    # MeteredModel records interrupted attempts in its finally block.
                    records = [*summarizer.records, *downstream.records]
                    _checkpoint(
                        stream,
                        {
                            **identity,
                            "compaction_status": "interrupted",
                            "error": type(exc).__name__,
                            "call_records": records,
                            **run.aggregate_records(records),
                        },
                    )
                    raise
                row.update(identity)
                _checkpoint(stream, row)
                rows.append(row)
                metadata["execution_order"].append({"id": case["id"], "mode": row["mode"]})
    selected = select_variant(rows) if split == "dev" else calibration["selected_variant"]
    metadata["selected_variant"] = selected
    summary = run.save(output, rows, metadata)
    result = {
        "live": live,
        "split": split,
        "manifest": frozen,
        "case_ids": [case["id"] for case in cases],
        "selected_variant": selected,
        "execution_order": metadata["execution_order"],
        "summary": summary,
    }
    if split == "eval":
        result["development_ids"] = calibration["case_ids"]
    _write_json(output / ("calibration.json" if split == "dev" else "study.json"), result)
    with (output / "report.md").open("a", encoding="utf-8") as report:
        report.write(
            f"\nDevelopment-selected variant: `{selected}`. Selection uses development only.\n"
            "\nFixed budgets: 256 / 512 / 1024 completion tokens, including provider reasoning "
            "when applicable; prose / structured formats. Unverified optimize() only.\n"
            "\nAt most 13 calls per case, no retries; one shared original baseline per case. "
            "Execution order rotates by case and is recorded in metadata.json "
            "and checkpoint.jsonl.\n"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="smoke", choices=("smoke", "dev-live", "eval-live")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, help="maximum cases; at most 13 calls per case")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/summary-study"))
    args = parser.parse_args(argv)
    live = args.command != "smoke"
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if live and (not args.config or args.limit is None):
        parser.error("live commands require --config and an explicit positive --limit")
    if args.command == "eval-live" and not args.calibration:
        parser.error("eval-live requires --calibration from dev-live")
    try:
        calibration = json.loads(args.calibration.read_text()) if args.calibration else None
        result = run_study(
            run.configuration(args.config),
            split="eval" if args.command == "eval-live" else "dev",
            live=live,
            limit=args.limit,
            output=args.output,
            calibration=calibration,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "selected_variant": result["selected_variant"],
                    **result["summary"]["overall"],
                },
                ensure_ascii=False,
            )
        )
        return int(bool(result["summary"]["overall"]["errors"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
