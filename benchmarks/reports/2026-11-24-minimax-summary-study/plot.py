"""Plot published results offline: uv run --no-project --with matplotlib python plot.py."""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
rows = [json.loads(line) for line in (directory / "eval/results.jsonl").read_text().splitlines()]
budgets = [256, 512, 1024]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
for package, color in [("prose", "#255fba"), ("structured", "#b34b15")]:
    accuracy, saved, cost = [], [], []
    for budget in budgets:
        group = [row for row in rows if row["mode"] == f"{package}-{budget}"]
        assert len(group) == 28
        accuracy.append(100 * sum(row["correct"] for row in group) / len(group))
        known = [row for row in group if row["context_after"] is not None]
        saved.append(
            100
            * (
                1
                - sum(row["context_after"] for row in known)
                / sum(row["context_before"] for row in known)
            )
            if known
            else math.nan
        )
        cost.append(
            sum(row["cost_usd"] for row in group)
            if all(row["cost_usd"] is not None for row in group)
            else math.nan
        )
    for ax, values in zip(axes, (accuracy, saved, cost), strict=True):
        ax.plot(budgets, values, marker="o", color=color, label=package)
baseline = [row for row in rows if row["mode"] == "original"]
assert len(baseline) == 28
axes[0].axhline(
    100 * sum(row["correct"] for row in baseline) / len(baseline),
    color="#555555",
    linestyle="--",
    label="original",
)
if all(row["cost_usd"] is not None for row in baseline):
    axes[2].axhline(
        sum(row["cost_usd"] for row in baseline), color="#555555", linestyle="--", label="original"
    )
for ax, title, ylabel in zip(
    axes,
    (
        "Answer accuracy (all 28 cases)",
        "Local context savings (known pairs)",
        "Full scenario PAYG-equivalent cost",
    ),
    ("Exact annotated answer, %", "Saved local tokens, %", "USD / 28 one-answer scenarios"),
    strict=True,
):
    ax.set(title=title, ylabel=ylabel, xlabel="Requested completion-token cap")
    ax.set_xticks(budgets)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
axes[0].set_ylim(0, 105)
axes[0].annotate(
    "all 28 fallbacks",
    xy=(1024, 100),
    xytext=(650, 65),
    fontsize=9,
    color="#b34b15",
    arrowprops={"arrowstyle": "->", "color": "#b34b15"},
)
axes[1].set_ylim(bottom=0)
axes[2].set_ylim(bottom=0)
fig.suptitle("MiniMax-M3.1-Flash · one held-out synthetic run · 2026-11-24", fontsize=14)
fig.supxlabel(
    "Caps include thinking. Gaps in cost mean unknown usage; USD is not subscription cash spend.",
    fontsize=9,
)
fig.savefig(directory / "curves.png", dpi=160)
