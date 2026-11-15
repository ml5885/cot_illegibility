"""Figures for error_analysis.md: how OLMo 3 7B's wrong answers break down across the token-limit runs, once the task
is learned.

uv run --no-project --with matplotlib python error_analysis.py
Writes figures/error-analysis-causes.png and figures/error-analysis-lists.png.
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from errors import cause, list_detail
from plots import load

# run, label, split, first step counted, token limit there
RUNS = [
    ("swaps-7b-tags-long", "1,280 tokens", "train", 50, 1280),
    ("swaps-7b-tags-v2", "768 tokens", "train", 50, 768),
    ("swaps-7b-tags-512", "512 tokens", "train", 50, 512),
    ("swaps-7b-tags-256", "256 tokens", "train", 50, 256),
    ("swaps-7b-tags-sched128", "128 tokens, lowered gradually", "train", 151, 128),
    ("swaps-7b-tags-2and4", "Trained on 2 and 4 swaps", "eval", 40, 768),
]
LEVEL = 16
CAUSES = [
    ("Wrong list", "Wrong list of swaps", "#2a78d6"),
    ("Tracking error", "Tracking error", "#eb6834"),
    ("Cut off", "Cut off", "#eda100"),
    ("Answer differs", "Answer differs from last stated box", "#1baf7a"),
]


def rows_for(run: str, split: str, first: int) -> list[dict]:
    """The gradual 128-token run writes "Swap C,F moves to C"; the swap-list parser reads pairs as "C and F"."""
    data = load(run)[split][LEVEL]
    rows = [r for step, rs in data.items() if step >= first for r in rs]
    return [r | {"completion": re.sub(r"\b([A-F]),([A-F])\b", r"\1 and \2", r["completion"])} for r in rows]


def bars(ax, shares: dict[str, list[tuple[float, str, str]]], title: str):
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


if __name__ == "__main__":
    causes, kinds, skips, merged = {}, {}, {}, {}
    for run, label, split, first, limit in RUNS:
        rows = rows_for(run, split, first)
        wrong = [r for r in rows if not r["reward"]]
        counts = Counter(cause(r, limit) for r in wrong)
        n = len(wrong)
        causes[f"{label} ({n})"] = [(counts[k] / n, name, color) for k, name, color in CAUSES] + [
            ((n - sum(counts[k] for k, _, _ in CAUSES)) / n, "Other", "0.8")]
        detail = list_detail(rows)
        k = detail["kinds"]
        total = sum(k.values())
        kinds[f"{label} ({total})"] = [(k["left out only"] / total, "Left out swaps", "#2a78d6"),
                                      (k["added only"] / total, "Added extra swaps", "#e87ba4"),
                                      (k["both"] / total, "Left out and added", "#7a5195")]
        # In the gradual run's terse lists, in-order alignment pins a jump to an earlier copy of the landing pair, so the
        # skip check misses these jumps; that run is left out of the skips panel.
        if run != "swaps-7b-tags-sched128":
            skips[label] = (detail["later_copy"] / detail["skips"], detail["chance"] / detail["skips"], int(detail["skips"]))
        merged[label] = detail["merged"]
        print(f"{label}: {n} wrong of {len(rows)}; causes {dict(counts)}; kinds {dict(k)}; skips {skips.get(label)}; merged {merged[label]:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    bars(ax, causes, f"Wrong {LEVEL}-swap answers once the task is learned")
    fig.legend(loc="outside lower center", ncol=3, fontsize=13)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/error-analysis-causes.png", dpi=150)

    fig = plt.figure(figsize=(16, 4.6), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.15, 1])
    ax = left.subplots()
    bars(ax, kinds, "Wrong lists of swaps, by kind")
    left.legend(loc="outside lower center", ncol=3, fontsize=13)
    ax = right.subplots()
    labels = list(skips)
    for i, label in enumerate(labels):
        observed, chance, n = skips[label]
        ax.plot([chance, observed], [i, i], color="0.75", linewidth=2, zorder=1)
        ax.scatter([chance], [i], s=60, facecolor="white", edgecolor="0.4", linewidth=1.5, zorder=2,
                   label="If the landing point were random" if i == 0 else None)
        ax.scatter([observed], [i], s=60, color="#1f4e8c", zorder=2, label="Observed" if i == 0 else None)
    ax.set_yticks(range(len(labels)), [f"{label} ({skips[label][2]})" for label in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.set_title("Skips that jump to a repeated pair")
    right.legend(loc="outside lower center", ncol=2, fontsize=13)
    fig.savefig("figures/error-analysis-lists.png", dpi=150)
