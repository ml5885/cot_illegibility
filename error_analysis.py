"""Figures for error_analysis.md: how OLMo 3 7B's wrong answers break down across the token-limit runs, once the task
is learned.

uv run --no-project --with matplotlib --with datasets python error_analysis.py
Writes figures/error-analysis-{causes,lists,positions,tracking}.png.
"""

from collections import Counter

import matplotlib.pyplot as plt

from cot_errors import STEPS, first_wrong_step
from errors import cause, limit_at, list_detail
from plots import load
from traces import list_steps

# run, label, split, first step counted (once the task is learned; for the gradual run, once its limit reached 128)
RUNS = [
    ("swaps-7b-tags-long", "1,280 tokens", "train", 50),
    ("swaps-7b-tags-v2", "768 tokens", "train", 50),
    ("swaps-7b-tags-512", "512 tokens", "train", 50),
    ("swaps-7b-tags-256", "256 tokens", "train", 50),
    ("swaps-7b-tags-sched128", "128 tokens, lowered gradually", "train", 151),
    ("swaps-7b-tags-2and4", "Trained on 2 and 4 swaps", "eval", 40),
]
COLORS = dict(zip([r[0] for r in RUNS], [plt.get_cmap("magma")(x) for x in (0.12, 0.3, 0.45, 0.6, 0.73)] + ["#2a78d6"]))
LEVEL = 16
CAUSES = [
    ("Wrong list", "Wrong list of swaps", "#2a78d6"),
    ("Tracking error", "Tracking error", "#eb6834"),
    ("Cut off", "Cut off", "#eda100"),
    ("Answer differs", "Answer differs from last stated box", "#1baf7a"),
]


def rows_for(run: str, split: str, first: int) -> list[dict]:
    """Rows at LEVEL swaps from step `first` on, each with its step."""
    data = load(run)[split][LEVEL]
    return [r | {"step": step} for step, rs in data.items() if step >= first for r in rs]


def legend_below(ax, ncol: int):
    """A legend under the axes that starts at the axes' left edge, not under the tick labels."""
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=ncol, fontsize=13, borderaxespad=0)


def bars(ax, shares: dict[str, list[tuple[float, str, str]]], title: str, ncol: int):
    for i, parts in enumerate(shares.values()):
        left = 0.0
        for width, label, color in parts:
            ax.barh(i, width, left=left, color=color, edgecolor="white", label=label if i == 0 else None)
            left += width
    ax.set_yticks(range(len(shares)), list(shares))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.set_title(title)
    legend_below(ax, ncol)


if __name__ == "__main__":
    rows = {run: rows_for(run, split, first) for run, _, split, first in RUNS}
    labels = {run: label for run, label, _, _ in RUNS}

    causes, kinds, skips, positions, tracking = {}, {}, {}, {}, {}
    for run, label in labels.items():
        limits = limit_at(run)
        wrong = [r for r in rows[run] if not r["reward"]]
        counts = Counter(cause(r, limits(r["step"])) for r in wrong)
        n = len(wrong)
        causes[f"{label} ({n})"] = [(counts[k] / n, name, color) for k, name, color in CAUSES] + [
            ((n - sum(counts[k] for k, _, _ in CAUSES)) / n, "Other", "0.8")]
        detail = list_detail(rows[run])
        k = detail["kinds"]
        total = sum(k.values())
        kinds[f"{label} ({total})"] = [(k["left out only"] / total, "Left out swaps", "#2a78d6"),
                                      (k["added only"] / total, "Added extra swaps", "#e87ba4"),
                                      (k["both"] / total, "Left out and added", "#7a5195")]
        skips[label] = (detail["later_copy"] / detail["skips"], detail["chance"] / detail["skips"], int(detail["skips"]))
        left = [[m is None for m in list_steps(r)[1]] for r in rows[run]]
        positions[run] = [sum(flags[i] for flags in left) / len(left) for i in range(LEVEL)]
        firsts = Counter(first_wrong_step(r) for r in wrong if cause(r, limits(r["step"])) == "Tracking error")
        m = sum(firsts.values())
        tracking[f"{label} ({m})"] = [(firsts[key] / m, name, color) for key, name, color in STEPS]
        print(f"{label}: {n} wrong of {len(rows[run])}; causes {dict(counts)}; kinds {dict(k)}; "
              f"skips {tuple(round(x, 2) for x in skips[label][:2])} of {skips[label][2]}; first wrong steps {dict(firsts)}")

    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    bars(ax, causes, f"Wrong {LEVEL}-swap answers", ncol=2)
    fig.savefig("figures/error-analysis-causes.png", dpi=150)

    fig = plt.figure(figsize=(16, 4.8), layout="constrained")
    left_fig, right_fig = fig.subfigures(1, 2, width_ratios=[1.15, 1])
    bars(left_fig.subplots(), kinds, "Wrong lists of swaps, by kind", ncol=3)
    ax = right_fig.subplots()
    for i, label in enumerate(skips):
        observed, chance, _ = skips[label]
        ax.plot([chance, observed], [i, i], color="0.75", linewidth=2, zorder=1)
        ax.scatter([chance], [i], s=60, facecolor="white", edgecolor="0.4", linewidth=1.5, zorder=2,
                   label="If the landing point were random" if i == 0 else None)
        ax.scatter([observed], [i], s=60, color="#1f4e8c", zorder=2, label="Observed" if i == 0 else None)
    ax.set_yticks(range(len(skips)), [f"{label} ({skips[label][2]})" for label in skips])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.set_title("Skips that jump to a repeated pair")
    legend_below(ax, ncol=1)
    fig.savefig("figures/error-analysis-lists.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for run, shares in positions.items():
        ax.plot(range(1, LEVEL + 1), shares, "o-", color=COLORS[run], linewidth=2, markersize=4, label=labels[run])
    ax.set_xticks(range(1, LEVEL + 1))
    ax.set_xlabel("Position of the swap in the puzzle")
    ax.set_ylabel("Responses that leave it out")
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.set_ylim(0)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_title(f"Left-out swaps by position ({LEVEL} swaps)")
    ax.legend(loc="upper left", fontsize=12)
    fig.savefig("figures/error-analysis-positions.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    bars(ax, tracking, "First wrong step in tracking errors", ncol=1)
    fig.savefig("figures/error-analysis-tracking.png", dpi=150)
