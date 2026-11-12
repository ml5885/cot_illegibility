"""Figures from saved rollouts.

uv run --with matplotlib plots.py [--start-box] RUN [RUN ...]   training curves per run
uv run --with matplotlib plots.py --formats MODEL                 base model, think tags vs plain format
uv run --with matplotlib plots.py --lengths RUN                   completion length by outcome, per level
uv run --with matplotlib plots.py --newlines RUN                  share of completions with a newline, by outcome
uv run --with matplotlib plots.py --length-signal RUN             what the reward says about length
uv run --with matplotlib plots.py --correlation RUN               correlation of length with correctness
uv run --with matplotlib plots.py --token-limit RUN               completion tokens against the token limit
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
    "legend.framealpha": 1,
})

WINDOW = 5
BIN, MIN_SAMPLES = 20, 5
OUTCOMES = [("Correct", "#2a78d6"), ("Wrong answer", "#eb6834"), ("Wrong format", "#1baf7a")]
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
    return "Correct" if row["reward"] else "Wrong answer" if row["parsed"] is not None else "Wrong format"


def newline(row: dict) -> float:
    return float("\n" in row["completion"])


BY_OUTCOME = {
    "--lengths": (lambda row: row["tokens"], "Length (tokens)", "completion length by outcome", "{x:,.0f}", "lengths", True),
    "--newlines": (newline, "With a newline", "completions with a newline, by outcome", "{x:.0%}", "newlines", False),
}


def plot_by_outcome(run: str, metric, label: str, title: str, tick_format: str, suffix: str, token_limit: bool):
    rows = load(run)["train"]
    args = json.loads(Path("runs", run, "args.json").read_text())
    model = args["model"].split("/")[-1]
    grid_rows = (len(rows) + 1) // 2
    fig, axes = plt.subplots(
        2 * grid_rows, 2, figsize=(10, 4.4 * grid_rows), sharex=True, height_ratios=[1, 3] * grid_rows, layout="constrained"
    )
    panels = [axes[2 * (i // 2) + 1, i % 2] for i in range(len(rows))]
    for i, (ax, level) in enumerate(zip(panels, sorted(rows))):
        top = axes[2 * (i // 2), i % 2]
        starts = range(0, max(rows[level]) + 1, BIN)
        centers = [s + BIN / 2 for s in starts]
        bins = [[r for step in range(start, start + BIN) for r in rows[level].get(step, [])] for start in starts]
        floor = [0.0] * len(bins)
        for name, color in OUTCOMES:
            lengths = [[metric(r) for r in b if outcome(r) == name] for b in bins]
            share = [len(v) / len(b) for v, b in zip(lengths, bins)]
            top.bar(centers, share, BIN - 3, bottom=floor, color=color)
            floor = [f + s for f, s in zip(floor, share)]
            means = [sum(v) / len(v) if len(v) >= MIN_SAMPLES else float("nan") for v in lengths]
            ax.plot(centers, means, "o-", color=color, label=name, linewidth=2, markersize=6)
        top.set_title(f"{level} swaps")
        top.set_ylim(0, 1)
        top.set_yticks([0, 1], ["0%", "100%"] if i % 2 == 0 else ["", ""])
        top.spines["left"].set_visible(False)
        top.tick_params(length=0)
        if token_limit:
            ax.axhline(args["max_tokens"], color="black", linestyle=":", linewidth=1.5, label="Token limit")
        ax.sharey(panels[0])
        ax.tick_params(labelbottom=True, labelleft=i % 2 == 0)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        if i % 2 == 0:
            top.set_ylabel("Proportion")
            ax.set_ylabel(label)
    for ax in panels[-2:]:
        ax.set_xlabel("Training step")
    panels[0].set_ylim(0, 1.05 * max([ax.dataLim.y1 for ax in panels] + [args["max_tokens"]] * token_limit))
    panels[0].yaxis.set_major_formatter(tick_format)
    fig.legend(*panels[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.suptitle(f"{model}: {title}")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-{suffix}.png", dpi=150)


def sides(rows: list[dict], metric) -> list[tuple[float, float]]:
    """(incorrect mean, correct mean) of the metric, for each puzzle whose samples were not all scored alike."""
    puzzles = defaultdict(lambda: ([], []))
    for row in rows:
        puzzles[row["prompt"]][int(row["reward"])].append(metric(row))
    return [(sum(wrong) / len(wrong), sum(right) / len(right)) for wrong, right in puzzles.values() if wrong and right]


def plot_length_signal(run: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    fig, (by_outcome, by_level) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    starts = range(0, max(max(by_step) for by_step in rows.values()) + 1, BIN)
    binned = lambda by_step: [[r for step in range(start, start + BIN) for r in by_step.get(step, [])] for start in starts]
    mean = lambda values: sum(values) / len(values) if len(values) >= MIN_SAMPLES else float("nan")
    pooled = [sum((sides(b, newline) for b in bins), []) for bins in zip(*(binned(rows[level]) for level in rows))]
    for side, (name, color) in [(1, OUTCOMES[0]), (0, ("Incorrect", OUTCOMES[1][1]))]:
        by_outcome.plot([s + BIN / 2 for s in starts], [mean([pair[side] for pair in b]) for b in pooled], "o-", color=color, label=name)
    for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(rows)):
        by_level.plot(*smoothed(rows[level], newline), color=color, label=f"{level} swaps")
    by_outcome.set_title("Completions with a newline (incorrect vs. correct)")
    by_level.set_title("Completions with a newline (by swap count)")
    by_outcome.yaxis.set_major_formatter("{x:.0%}")
    by_outcome.set_ylim(0, 0.9)
    for ax in (by_outcome, by_level):
        ax.legend(loc="upper right")
        ax.set_xlabel("Training step")
        ax.set_xlim(-5, 205)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-length-signal.png", dpi=150)


def correlation(rows: list[dict], within_puzzle: bool) -> tuple[float, float]:
    """Pearson r between completion length and reward, with its standard error."""
    puzzles = defaultdict(list)
    for row in rows:
        puzzles[row["prompt"] if within_puzzle else None].append((length(row), row["reward"]))
    pairs = [(x - sum(a for a, _ in g) / len(g), y - sum(b for _, b in g) / len(g)) for g in puzzles.values() for x, y in g]
    if sum(row["reward"] == 0 for row in rows) < MIN_SAMPLES or sum(row["reward"] for row in rows) < MIN_SAMPLES:
        return float("nan"), float("nan")
    r = sum(x * y for x, y in pairs) / (sum(x * x for x, _ in pairs) * sum(y * y for _, y in pairs)) ** 0.5
    return r, ((1 - r * r) / (len(pairs) - 2)) ** 0.5


def plot_correlation(run: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    panels = [
        ("All samples", False, lambda row: True),
        ("Within a puzzle", True, lambda row: True),
        ("Within a puzzle, wrong format excluded", True, lambda row: outcome(row) != "Wrong format"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True, layout="constrained")
    levels = sorted(rows)
    for offset, color, level in zip(range(len(levels)), plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), levels):
        starts = range(0, max(rows[level]) + 1, BIN)
        bins = [[r for step in range(start, start + BIN) for r in rows[level].get(step, [])] for start in starts]
        for ax, (title, within, keep) in zip(axes, panels):
            r, error = zip(*(correlation([row for row in b if keep(row)], within) for b in bins))
            xs = [s + BIN / 2 + 1.5 * (offset - (len(levels) - 1) / 2) for s in starts]
            ax.errorbar(xs, r, yerr=error, fmt="o-", color=color, label=f"{level} swaps", capsize=2, elinewidth=1)
            ax.set_title(title)
    for ax in axes:
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Training step")
        ax.set_xlim(-5, 205)
    axes[0].set_ylabel("Correlation of length with correctness")
    axes[0].legend(loc="lower left")
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-correlation.png", dpi=150)


def plot_token_limit(run: str):
    rows = load(run)["train"]
    args = json.loads(Path("runs", run, "args.json").read_text())
    limit, half = args["max_tokens"], BIN // 2
    fig, axes = plt.subplots(4, 2, figsize=(10, 8.8), sharex=True, height_ratios=[1, 3] * 2, layout="constrained")
    for i, level in enumerate(sorted(rows)):
        top, ax = axes[2 * (i // 2), i % 2], axes[2 * (i // 2) + 1, i % 2]
        starts = range(0, max(rows[level]) + 1, half)
        centers = [s + half / 2 for s in starts]
        bins = [sorted(r["tokens"] for step in range(start, start + half) for r in rows[level].get(step, [])) for start in starts]
        top.bar(centers, [sum(t >= limit - 2 for t in b) / len(b) for b in bins], half - 2, color="#eb6834")
        ax.fill_between(
            centers, [b[len(b) // 20] for b in bins], [b[-len(b) // 20] for b in bins], color="#2a78d6", alpha=0.25, linewidth=0,
            label="5th to 95th percentile",
        )
        ax.plot(centers, [b[len(b) // 2] for b in bins], color="#2a78d6", linewidth=2, label="Median")
        ax.plot(centers, [b[-1] for b in bins], ".", color="0.35", label="Longest")
        ax.axhline(limit, color="black", linestyle="--", linewidth=1, label="Token limit")
        top.set_title(f"{level} swaps")
        top.set_ylim(0, 0.15)
        top.set_yticks([0, 0.15], ["0%", "15%"] if i % 2 == 0 else ["", ""])
        top.spines["left"].set_visible(False)
        top.tick_params(length=0)
        ax.set_ylim(0, limit * 1.06)
        ax.tick_params(labelbottom=True, labelleft=i % 2 == 0)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        if i % 2 == 0:
            top.set_ylabel("Cut off")
            ax.set_ylabel("Completion tokens")
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.suptitle(f"{args['model'].split('/')[-1]}: completion length against the {limit}-token limit")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-token-limit.png", dpi=150)


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
    if sys.argv[1] in BY_OUTCOME:
        plot_by_outcome(sys.argv[2], *BY_OUTCOME[sys.argv[1]])
        sys.exit()
    OTHER = {"--length-signal": plot_length_signal, "--correlation": plot_correlation, "--token-limit": plot_token_limit}
    if sys.argv[1] in OTHER:
        OTHER[sys.argv[1]](sys.argv[2])
        sys.exit()
    runs = [arg for arg in sys.argv[1:] if arg != "--start-box"]
    for run in runs:
        plot(run, start_box=len(runs) < len(sys.argv) - 1)
