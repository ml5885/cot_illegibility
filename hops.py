"""The 64-token run answers without reasoning; compare its answers with rules that follow the key for a few moves.

uv run --no-project --with matplotlib python hops.py [RUN] [FIRST_STEP]

Rule k follows the key through its first k moves only: find the first swap that involves the key's box, move the key
to the other box, and repeat from the next swap, k times. With every move followed, it is the correct answer. Uses
training samples from FIRST_STEP on (default 150) and writes figures/<RUN>-hops.png.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import plots  # noqa: F401  (shared figure style: serif fonts)
from traces import SWAP, START

HOPS = [1, 2, 3]
COLORS = {1: "#f5b08c", 2: "#eb6834", 3: "#b8441a"}


def follow(start: str, swaps: list[tuple[str, str]], hops: int) -> str:
    key = start
    for a, b in swaps:
        if hops and key in (a, b):
            key, hops = b if key == a else a, hops - 1
    return key


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "swaps-7b-tags-64"
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    stats = defaultdict(lambda: defaultdict(float))
    for path in sorted(Path("runs", run, "rollouts").glob("train_*.jsonl")):
        if int(path.stem[-5:]) < first:
            continue
        for line in path.open():
            row = json.loads(line)
            question = row["prompt"].split("User:")[-1]
            start, swaps = START.search(question).group(1), [tuple(s) for s in SWAP.findall(question)]
            s = stats[row["level"]]
            s["n"] += 1
            s["model"] += row["reward"]
            for k in HOPS:
                rule = follow(start, swaps, k)
                s[f"rule {k}"] += rule == row["answer"]
                s[f"agree {k}"] += rule == row["parsed"]
    levels = sorted(stats)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    axes[0].plot(levels, [stats[l]["model"] / stats[l]["n"] for l in levels], "o-", color="#2a78d6", linewidth=2.5, label="64-token model")
    for k in HOPS:
        label = "Follow the key's first move" if k == 1 else f"Follow the key's first {k} moves"
        axes[0].plot(levels, [stats[l][f"rule {k}"] / stats[l]["n"] for l in levels], "o--", color=COLORS[k], linewidth=1.5, label=label)
        axes[1].plot(levels, [stats[l][f"agree {k}"] / stats[l]["n"] for l in levels], "o-", color=COLORS[k], linewidth=2, label=label)
    axes[0].axhline(1 / 6, color="gray", linestyle=":", linewidth=1, label="Chance (1/6)")
    axes[0].set_title("Accuracy")
    axes[1].set_title("Agreement with rule")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, [str(l) for l in levels])
        ax.set_xlabel("Swaps")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    axes[0].legend(loc="lower left", fontsize=13)
    fig.suptitle("OLMo 3 7B, 64-token limit")
    fig.savefig(f"figures/{run}-hops.png", dpi=150)
    for l in levels:
        print(l, {k: round(v / stats[l]["n"], 2) for k, v in stats[l].items() if k != "n"})
