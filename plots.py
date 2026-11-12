"""Per-run training curves from saved rollouts: uv run --with matplotlib plots.py RUN [RUN ...]"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

WINDOW = 5


def load(run: str) -> dict[str, dict[int, dict[int, list[dict]]]]:
    """rows[split][level][step]"""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(Path("runs", run, "rollouts").glob("*.jsonl")):
        split, step = path.stem.rsplit("_", 1)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[split][row["level"]][int(step)].append(row)
    return rows


def accuracy(row: dict) -> float:
    return row["reward"]


def length(row: dict) -> float:
    return len(row["completion"])


def smoothed(by_step: dict[int, list[dict]], metric) -> tuple[list[int], list[float]]:
    steps = sorted(by_step)
    means = [sum(map(metric, by_step[s])) / len(by_step[s]) for s in steps]
    return steps, [sum(w := means[max(0, i - WINDOW + 1) : i + 1]) / len(w) for i in range(len(means))]


def plot(run: str):
    rows = load(run)
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (metric, title) in zip(axes, [(accuracy, "Accuracy"), (length, "Completion length (characters)")]):
        for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(rows["train"])):
            ax.plot(*smoothed(rows["train"][level], metric), color=color, label=f"{level} swaps")
            held_out = rows["eval"][level]
            ax.plot(
                sorted(held_out),
                [sum(map(metric, held_out[s])) / len(held_out[s]) for s in sorted(held_out)],
                "o",
                color=color,
                markeredgecolor="black",
            )
        ax.set_title(title)
        ax.set_xlabel("Training step")
    axes[0].axhline(1 / 6, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].legend()
    fig.suptitle(f"{model} on the box-swap task. Lines: training samples ({WINDOW}-step mean). Dots: held-out puzzles.")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}.png", dpi=150)


if __name__ == "__main__":
    for run in sys.argv[1:]:
        plot(run)
