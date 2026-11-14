"""A scheduled-limit run over training: accuracy, length against the token limit, and cut-offs, by swap count.

uv run --no-project --with matplotlib python schedule_plot.py [RUN]   (default swaps-7b-tags-sched128)
Reads runs/<RUN>/rollouts and runs/<RUN>/limits.jsonl; writes figures/<RUN>-schedule.png.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from plots import accuracy, load, smoothed, tokens

LEVELS = [2, 4, 8, 16]
COLORS = dict(zip(LEVELS, plt.cm.viridis([0.0, 0.3, 0.6, 0.85])))


def limit_at(changes: list[dict], step: int) -> int:
    return [c["limit"] for c in changes if c["step"] <= step][-1]


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "swaps-7b-tags-sched128"
    rows = load(run)["train"]
    changes = [json.loads(line) for line in Path("runs", run, "limits.jsonl").open()]
    last = max(max(rows[level]) for level in LEVELS)
    cut_off = lambda row: float(row["tokens"] >= limit_at(changes, row["step"]) - 2)
    for level in LEVELS:
        for step, at in rows[level].items():
            for row in at:
                row["step"] = step

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained", height_ratios=[1, 1.2, 0.8])
    panels = [(axes[0], accuracy, "Accuracy"), (axes[1], tokens, "Completion tokens"), (axes[2], cut_off, "Cut off at the limit")]
    for ax, metric, title in panels:
        for level in LEVELS:
            ax.plot(*smoothed(rows[level], metric), color=COLORS[level], linewidth=2, label=f"{level} swaps")
        for change in changes[1:]:
            ax.axvline(change["step"], color="0.6", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    steps = [c["step"] for c in changes] + [last]
    axes[1].step(steps, [c["limit"] for c in changes] + [changes[-1]["limit"]], where="post", color="black", linewidth=1.5,
                 label="Token limit")
    for ax in (axes[0], axes[2]):
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter("{x:.0%}")
    axes[1].set_ylim(0)
    axes[2].set_xlabel("Training step")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-schedule.png", dpi=150)
