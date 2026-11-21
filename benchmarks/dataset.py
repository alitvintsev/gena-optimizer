"""Separate experiment inputs and labels, validation, and deterministic scoring."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from gena_optimizer import Message

DATA = Path(__file__).with_name("data")
VERSION = "synthetic-ru-en-v1"
CATEGORIES = {
    "numbers_units",
    "negations_exceptions",
    "updated_facts",
    "entity_ids",
    "sources",
    "code_json",
    "task_status",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_number}: expected object")
        rows.append(row)
    return rows


def validate(cases: list[dict], labels: dict[str, dict], *, keep_recent: int = 2) -> dict:
    """Validate all rows, including facts in the part that will be summarized."""
    ids = [case.get("id") for case in cases]
    if not cases or len(set(ids)) != len(ids) or set(ids) != set(labels):
        raise ValueError("dataset: empty, duplicate IDs, or input/label ID mismatch")
    for case in cases:
        name = case["id"]
        if set(case) != {"id", "category", "language", "history", "question"}:
            raise ValueError(f"{name}: unexpected input fields (labels must be separate)")
        if not isinstance(name, str) or not name:
            raise ValueError("dataset: nonempty string IDs required")
        if case["category"] not in CATEGORIES or case["language"] not in {"ru", "en"}:
            raise ValueError(f"{name}: unknown category or language")
        history = case["history"]
        if not isinstance(history, list) or len(history) < 4:
            raise ValueError(f"{name}: history needs at least four messages")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"{name}: missing question")
        for message in history:
            if (
                set(message) != {"role", "content"}
                or message["role"] not in {"user", "assistant"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()
            ):
                raise ValueError(f"{name}: invalid history message")
        if keep_recent < 1 or keep_recent >= len(history):
            raise ValueError(f"{name}: keep_recent must retain question and leave old history")
        old = "\n".join(m["content"] for m in history[: len(history) + 1 - keep_recent])
        label = labels[name]
        reference = label.get("reference_answer")
        facts = label.get("required_facts")
        forbidden = label.get("forbidden_claims")
        if (
            label.get("id") != name
            or not isinstance(reference, dict)
            or not reference
            or not isinstance(facts, list)
            or not facts
            or not isinstance(forbidden, list)
            or not forbidden
        ):
            raise ValueError(f"{name}: incomplete annotation")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in reference.items()):
            raise ValueError(f"{name}: reference must map strings to strings")
        if {fact.get("field"): fact.get("value") for fact in facts} != reference or len(
            facts
        ) != len(reference):
            raise ValueError(f"{name}: reference/required facts mismatch")
        for fact in facts:
            try:
                matches = re.search(fact["context_pattern"], old, re.IGNORECASE | re.DOTALL)
            except (KeyError, TypeError, re.error) as exc:
                raise ValueError(f"{name}: invalid context_pattern") from exc
            if not fact["value"] or fact["value"] not in old or not matches:
                raise ValueError(f"{name}: required fact missing from old history")
        for claim in forbidden:
            if (
                not isinstance(claim.get("field"), str)
                or not isinstance(claim.get("value"), str)
                or reference.get(claim["field"]) == claim["value"]
            ):
                raise ValueError(f"{name}: invalid or contradictory forbidden claim")
    return {
        "cases": len(cases),
        "categories": dict(Counter(c["category"] for c in cases)),
        "languages": dict(Counter(c["language"] for c in cases)),
    }


def load(split: str = "eval", *, keep_recent: int = 2) -> tuple[list[dict], dict[str, dict]]:
    prefix = "dev_" if split == "dev" else ""
    cases = read_jsonl(DATA / f"{prefix}cases.jsonl")
    rows = read_jsonl(DATA / f"{prefix}labels.jsonl")
    labels = {row["id"]: row for row in rows}
    if len(labels) != len(rows):
        raise ValueError(f"{split}: duplicate label IDs")
    validate(cases, labels, keep_recent=keep_recent)
    return cases, labels


def fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(DATA.glob("*.jsonl")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _unique_object(pairs: list[tuple]) -> dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate answer field")
    return obj


def evaluate(answer: str, label: dict, context: list[Message]) -> dict:
    """Exact structured answer checks; context regex recall is a lexical proxy."""
    try:
        parsed = json.loads(answer, object_pairs_hook=_unique_object)
        valid = isinstance(parsed, dict) and all(
            isinstance(value, str) for value in parsed.values()
        )
    except ValueError:
        parsed, valid = {}, False
    facts = label["required_facts"]
    text = "\n".join(message.content for message in context)
    return {
        "valid_answer": valid,
        "correct": valid and parsed == label["reference_answer"],
        "required_fact_recall": sum(
            valid and parsed.get(fact["field"]) == fact["value"] for fact in facts
        )
        / len(facts),
        "context_fact_recall": sum(
            bool(re.search(fact["context_pattern"], text, re.IGNORECASE | re.DOTALL))
            for fact in facts
        )
        / len(facts),
        "forbidden_claims": sum(
            parsed.get(claim["field"]) == claim["value"] for claim in label["forbidden_claims"]
        )
        if valid
        else None,
    }
