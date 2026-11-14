"""Compare the runs that differ only in the token limit: figures/limits-*.png and summary numbers.

uv run --no-project --with matplotlib python limits.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from plots import accuracy, load, newline, smoothed, tokens

RUNS = {256: "swaps-7b-tags-256", 512: "swaps-7b-tags-512", 768: "swaps-7b-tags-v2", 1280: "swaps-7b-tags-long"}
LEVELS = [2, 4, 8, 16]
COLORS = {limit: plt.get_cmap("magma")(x) for limit, x in zip(RUNS, [0.82, 0.62, 0.42, 0.15])}
LATE = range(150, 200)


def cut_off(limit: int):
    return lambda row: float(row["tokens"] >= limit - 2)


def by_level(data, metric_for):
    """For each limit and level, the smoothed training curve of a metric (metric_for(limit) gives the metric)."""
    return {limit: {level: smoothed(data[limit]["train"][level], metric_for(limit)) for level in LEVELS} for limit in RUNS}


def grid(name: str, curves, ylabel: str, percent: bool, limit_lines: bool = False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, level in zip(axes.flat, LEVELS):
        for limit in RUNS:
            ax.plot(*curves[limit][level], color=COLORS[limit], linewidth=2, label=f"{limit:,} tokens")
        if limit_lines:
            top = ax.get_ylim()[1]
            for limit in RUNS:
                if limit < top:
                    ax.axhline(limit, color=COLORS[limit], linewidth=1, alpha=0.6)
            ax.set_ylim(0, top)
        if percent:
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_title(f"{level} swaps")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    axes[0, 0].legend(title="Token limit", loc="lower right" if percent else "upper right")
    fig.suptitle(f"OLMo 3 7B, by token limit: {ylabel.lower()}")
    fig.savefig(f"figures/limits-{name}.png", dpi=150)
    plt.close(fig)


def late_mean(data, limit, level, metric):
    rows = [r for s in LATE for r in data[limit]["train"][level].get(s, [])]
    return sum(map(metric, rows)) / len(rows)


def first_step(curve, threshold=0.9):
    return next((s for s, v in zip(*curve) if v >= threshold), None)


def summary(data, acc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    for limit in RUNS:
        ax.plot(LEVELS, [late_mean(data, limit, level, tokens) for level in LEVELS], "o-", color=COLORS[limit], linewidth=2,
                markersize=7, label=f"{limit:,} tokens")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LEVELS, [str(level) for level in LEVELS])
    ax.set_xlabel("Swaps")
    ax.set_ylabel("Completion tokens, steps 150 to 199")
    ax.set_ylim(0)
    ax.set_title("Length once the task is learned")
    ax.legend(title="Token limit")
    ax = axes[1]
    width = 0.2
    for i, limit in enumerate(RUNS):
        steps = [first_step(acc[limit][level]) for level in LEVELS]
        ax.bar([j + (i - 1.5) * width for j in range(len(LEVELS))], [s or 0 for s in steps], width * 0.9, color=COLORS[limit],
               label=f"{limit:,} tokens")
    ax.set_xticks(range(len(LEVELS)), [str(level) for level in LEVELS])
    ax.set_xlabel("Swaps")
    ax.set_ylabel("Training step")
    ax.set_title("First step with 90% training accuracy")
    for ax in axes:
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("OLMo 3 7B, by token limit")
    fig.savefig("figures/limits-summary.png", dpi=150)
    plt.close(fig)


def example_lines(data):
    """A correct 16-swap response near the end of each run, for quoting."""
    for limit in RUNS:
        rows = [r for s in (195, 196, 197, 198, 199) for r in data[limit]["train"][16].get(s, []) if r["reward"]]
        print(f"\n--- {limit} ({rows[0]['tokens']} tokens)\n{rows[0]['completion'][:900]}")


if __name__ == "__main__":
    data = {limit: load(run) for limit, run in RUNS.items()}
    acc = by_level(data, lambda limit: accuracy)
    grid("accuracy", acc, "Accuracy", percent=True)
    grid("length", by_level(data, lambda limit: tokens), "Completion tokens", percent=False, limit_lines=True)
    grid("newlines", by_level(data, lambda limit: newline), "Completions with a newline", percent=True)
    grid("cutoffs", by_level(data, cut_off), "Completions cut off at the limit", percent=True)
    summary(data, acc)
    stats = {}
    for limit in RUNS:
        stats[limit] = {level: {"step 90%": first_step(acc[limit][level]), "late tokens": round(late_mean(data, limit, level, tokens)),
                                "late newline": round(late_mean(data, limit, level, newline), 2),
                                "late accuracy": round(late_mean(data, limit, level, accuracy), 3),
                                "early cut-off": round(sum(map(cut_off(limit), (r for s in range(0, 40) for r in data[limit]["train"][level].get(s, []))))
                                                       / sum(len(data[limit]["train"][level].get(s, [])) for s in range(0, 40)), 2)} for level in LEVELS}
    print(json.dumps(stats, indent=1))
    example_lines(data)
