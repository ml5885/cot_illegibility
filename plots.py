"""Figures from saved rollouts.

uv run --with matplotlib plots.py [--start-box] RUN [RUN ...]   training curves per run
uv run --with matplotlib plots.py --formats MODEL                 base model, think tags vs plain format
uv run --with matplotlib plots.py --lengths RUN                   completion length by outcome, per level
uv run --with matplotlib plots.py --length-signal RUN             what the reward says about length
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

WINDOW = 5
BIN, MIN_SAMPLES = 20, 5
OUTCOMES = [("Correct", "#2a78d6"), ("Wrong answer", "#eb6834"), ("No valid answer", "#1baf7a")]
START = re.compile(r"A key is in box (\w)\.")


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


def names_start(row: dict) -> float:
    return float(row["parsed"] is not None and row["parsed"].strip() == START.search(row["prompt"]).group(1))


def smoothed(by_step: dict[int, list[dict]], metric) -> tuple[list[int], list[float]]:
    steps = sorted(by_step)
    means = [sum(map(metric, by_step[s])) / len(by_step[s]) for s in steps]
    return steps, [sum(w := means[max(0, i - WINDOW + 1) : i + 1]) / len(w) for i in range(len(means))]


def plot(run: str, start_box: bool):
    rows = load(run)
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    panels = [(accuracy, "Accuracy"), (length, "Completion length (characters)")]
    if start_box:
        panels.insert(1, (names_start, "Answers naming the starting box"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4))
    for ax, (metric, title) in zip(axes, panels):
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
    for ax in axes[:-1]:
        ax.axhline(1 / 6, color="gray", linestyle=":", linewidth=1)
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(loc="lower right", ncol=2)
    fig.suptitle(f"{model}: training samples (lines) and held-out puzzles (dots)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}.png", dpi=150)


def outcome(row: dict) -> str:
    return "Correct" if row["reward"] else "Wrong answer" if row["parsed"] is not None else "No valid answer"


def plot_lengths(run: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    fig, axes = plt.subplots(2, len(rows), figsize=(4 * len(rows), 5), sharex="col", height_ratios=[1, 3])
    for (top, ax), level in zip(axes.T, sorted(rows)):
        starts = range(0, max(rows[level]) + 1, BIN)
        centers = [s + BIN / 2 for s in starts]
        bins = [[r for step in range(start, start + BIN) for r in rows[level].get(step, [])] for start in starts]
        floor = [0.0] * len(bins)
        for name, color in OUTCOMES:
            lengths = [[length(r) for r in b if outcome(r) == name] for b in bins]
            share = [len(v) / len(b) for v, b in zip(lengths, bins)]
            top.bar(centers, share, BIN - 2, bottom=floor, color=color)
            floor = [f + s for f, s in zip(floor, share)]
            means = [sum(v) / len(v) if len(v) >= MIN_SAMPLES else float("nan") for v in lengths]
            ax.plot(centers, means, "o-", color=color, label=name)
        top.set_title(f"{level} swaps")
        top.set_ylim(0, 1)
        ax.set_xlabel("Training step")
        ax.set_ylim(0)
    axes[0, 0].set_ylabel("Share of samples")
    axes[1, 0].set_ylabel("Completion length (characters)")
    axes[1, 0].legend(loc="upper right")
    fig.suptitle(f"{model}: completion length by outcome")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-lengths.png", dpi=150)


def length_gaps(rows: list[dict]) -> list[float]:
    """Correct minus incorrect mean length, for each puzzle whose samples were not all scored alike."""
    puzzles = defaultdict(lambda: ([], []))
    for row in rows:
        puzzles[row["prompt"]][int(row["reward"])].append(length(row))
    return [sum(right) / len(right) - sum(wrong) / len(wrong) for wrong, right in puzzles.values() if wrong and right]


def plot_length_signal(run: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    shares = [
        (lambda row: float("\n" in row["completion"]), "Completions with a newline"),
        (lambda row: float("</answer>" not in row["completion"]), "Completions with no closed answer tag"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(rows)):
        starts = range(0, max(rows[level]) + 1, BIN)
        gaps = [length_gaps([r for step in range(start, start + BIN) for r in rows[level].get(step, [])]) for start in starts]
        means = [sum(g) / len(g) if len(g) >= MIN_SAMPLES else float("nan") for g in gaps]
        axes[0].plot([s + BIN / 2 for s in starts], means, "o-", color=color, label=f"{level} swaps")
        for ax, (metric, _) in zip(axes[1:], shares):
            ax.plot(*smoothed(rows[level], metric), color=color)
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=1)
    axes[0].set_title("Correct minus incorrect length (characters)")
    axes[0].legend()
    for ax, (_, title) in zip(axes[1:], shares):
        ax.set_title(title)
    for ax in axes:
        ax.set_xlabel("Training step")
    fig.suptitle(model)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-length-signal.png", dpi=150)


def reweighted(rows: list[dict], metric) -> float:
    """Baselines were sampled on puzzles where the key often stays put. The current puzzle is that
    distribution conditioned so the key moves 5 times in 6, so weighting the two cases gives its accuracy."""
    stayed = [row["answer"] == START.search(row["prompt"]).group(1) for row in rows]
    mean = lambda keep: sum(metric(r) for r, s in zip(rows, stayed) if s == keep) / stayed.count(keep)
    return 5 / 6 * mean(False) + 1 / 6 * mean(True)


def plot_formats(model: str):
    panels = [(accuracy, "Accuracy"), (lambda row: float(row["parsed"] is not None), "Completions with a parseable answer")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for fmt, label, color in [("tags", "Think tags", "tab:blue"), ("plain", "No tags", "tab:orange")]:
        levels = load(f"baseline-swaps-{fmt}-{model}")["baseline"]
        for ax, (metric, _) in zip(axes, panels):
            ax.plot(
                range(len(levels)),
                [reweighted(levels[level][0], metric) for level in sorted(levels)],
                "o-",
                color=color,
                label=label,
            )
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("Number of swaps")
        ax.set_xticks(range(len(levels)), sorted(levels))
        ax.set_ylim(-0.03, 1.03)
    axes[0].axhline(1 / 6, color="gray", linestyle=":", linewidth=1)
    axes[0].legend()
    fig.suptitle(f"{model} before training")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/baseline-formats-{model}.png", dpi=150)


if __name__ == "__main__":
    if sys.argv[1] == "--formats":
        plot_formats(sys.argv[2])
        sys.exit()
    if sys.argv[1] in ("--lengths", "--length-signal"):
        (plot_lengths if sys.argv[1] == "--lengths" else plot_length_signal)(sys.argv[2])
        sys.exit()
    runs = [arg for arg in sys.argv[1:] if arg != "--start-box"]
    for run in runs:
        plot(run, start_box=len(runs) < len(sys.argv) - 1)
