"""Figures from saved rollouts.

uv run --with matplotlib plots.py [--start-box] RUN [RUN ...]   training curves per run
uv run --with matplotlib plots.py --formats MODEL                 base model, think tags vs plain format
uv run --with matplotlib plots.py --lengths RUN                   completion length by outcome, per level
uv run --with matplotlib plots.py --newlines RUN                  share of completions with a newline, by outcome
uv run --with matplotlib plots.py --length-signal RUN             what the reward says about length
uv run --with matplotlib plots.py --correlation RUN               correlation of length with correctness
uv run --with matplotlib plots.py --advantages RUN                distribution of GRPO advantages over training
uv run --with matplotlib plots.py --formatting RUN                formatting features over training, and newline use with and without each
uv run --with matplotlib plots.py --lead RUN [RUN ...]            does the advantage gap for newlines predict the next change in their use
uv run --with matplotlib plots.py --push RUN                      advantage-weighted difference in newlines and length within puzzles
uv run --with matplotlib plots.py --newline-correlation RUN       correlation of using a newline with correctness
uv run --with matplotlib plots.py --token-limit RUN               completion tokens against the token limit
uv run --with matplotlib --with transformers plots.py --traces RUN   what correct held-out traces spend their tokens on
uv run --with matplotlib plots.py --tracking RUN                  per-swap tracking and standalone location statements, by swap count
uv run --with matplotlib plots.py --coverage RUN                  how many of a puzzle's swaps the reasoning writes out
uv run --with matplotlib plots.py --train-eval RUN                newline use on training and held-out puzzles
uv run --with matplotlib plots.py --templates RUN                 share of responses containing each common sentence template
uv run --with matplotlib plots.py --okay RUN                      responses that open with "Okay"
uv run --with matplotlib plots.py --errors RUN                    what went wrong in the responses that scored 0, as shares of them
"""

import json
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from traces import PAIR, START, puzzle, step_counts, swap_mentions, swap_steps

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "figure.titleweight": "normal",
    "axes.titleweight": "normal",
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
OUTCOMES = [("Correct", "#2a78d6"), ("Incorrect", "#eb6834")]


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


def held_out_curve(by_step: dict[int, list[dict]], metric) -> tuple[list[int], list[float]]:
    steps = sorted(by_step)
    return steps, [sum(map(metric, by_step[s])) / len(by_step[s]) for s in steps]


def plot(run: str, start_box: bool):
    rows = load(run)
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    panels = [(accuracy, "Accuracy"), (length, "Completion length (characters)")]
    if start_box:
        panels.insert(1, (names_start, "Answers naming the starting box"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.4), layout="constrained")
    for ax, (metric, title) in zip(axes, panels):
        for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(rows["train"] | rows["eval"])):
            if level in rows["train"]:
                ax.plot(*smoothed(rows["train"][level], metric), color=color, label=f"{level} swaps")
            else:
                ax.plot(*held_out_curve(rows["eval"][level], metric), color=color, label=f"{level} swaps, held-out only")
        ax.set_title(title)
        ax.set_xlabel("Training step")
    for ax in axes[:-1]:
        ax.axhline(1 / 6, color="gray", linestyle=":", linewidth=1)
        ax.set_ylim(-0.03, 1.03)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}.png", dpi=150)


def outcome(row: dict) -> str:
    return "Correct" if row["reward"] else "Incorrect"


def tokens(row: dict) -> float:
    return row["tokens"]


def newline(row: dict) -> float:
    return float("\n" in row["completion"])


BY_OUTCOME = {
    "--lengths": (tokens, "Length (tokens)", "completion length by outcome", "{x:,.0f}", "lengths"),
    "--newlines": (newline, "With a newline", "completions with a newline, by outcome", "{x:.0%}", "newlines"),
}


def plot_by_outcome(run: str, metric, label: str, title: str, tick_format: str, suffix: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
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
        ax.sharey(panels[0])
        ax.tick_params(labelbottom=True, labelleft=i % 2 == 0)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        if i % 2 == 0:
            top.set_ylabel("Proportion")
            ax.set_ylabel(label)
    for ax in panels[-2:]:
        ax.set_xlabel("Training step")
    panels[0].set_ylim(0, 1.03 if "%" in tick_format else 1.05 * max(ax.dataLim.y1 for ax in panels))
    panels[0].yaxis.set_major_formatter(tick_format)
    fig.legend(*panels[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle(f"{model}: {title}")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-{suffix}.png", dpi=150)


def plot_length_signal(run: str):
    held_out = load(run)["eval"]
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, layout="constrained")
    accuracy_ax, length_ax, kinds_ax, newline_ax = axes.flatten()
    by_level = [(accuracy_ax, accuracy, "Accuracy"), (length_ax, tokens, "Completion length in tokens"), (newline_ax, newline, "Completions with a newline")]
    for ax, metric, title in by_level:
        for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(rows | held_out)):
            if level in rows:
                ax.plot(*smoothed(rows[level], metric), color=color, label=f"{level} swaps")
            else:
                ax.plot(*held_out_curve(held_out[level], metric), color=color, label=f"{level} swaps, held-out only")
        ax.set_title(title)
    pooled = defaultdict(list)
    for by_step in rows.values():
        for step, step_rows in by_step.items():
            pooled[step] += step_rows
    kinds = [("A newline", "\n", "#eb6834"), ("A blank line", "\n\n", "#e87ba4"), ("Two spaces in a row", "  ", "0.45")]
    for name, text, color in kinds:
        kinds_ax.plot(*smoothed(pooled, lambda row: float(text in row["completion"])), color=color, label=name, linewidth=2)
    kinds_ax.set_title("Completions containing")
    for ax in axes.flatten():
        ax.set_ylim(0)
    for ax in (accuracy_ax, kinds_ax, newline_ax):
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 1.03)
    for ax in (kinds_ax, newline_ax):
        ax.set_xlabel("Training step")
    accuracy_ax.legend(loc="best")
    kinds_ax.legend(loc="best")
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-length-signal.png", dpi=150)


def correlation(rows: list[dict], metric, within_puzzle: bool) -> tuple[float, float]:
    """Pearson r between the metric and reward, with its standard error. Undefined when either barely varies."""
    puzzles = defaultdict(list)
    for row in rows:
        puzzles[row["prompt"] if within_puzzle else None].append((metric(row), row["reward"]))
    pairs = [(x - sum(a for a, _ in g) / len(g), y - sum(b for _, b in g) / len(g)) for g in puzzles.values() for x, y in g]
    rare = lambda values: min(Counter(values).most_common()[-1][1], len(values) - Counter(values).most_common(1)[0][1]) < MIN_SAMPLES
    spread_x, spread_y = sum(x * x for x, _ in pairs), sum(y * y for _, y in pairs)
    if rare([row["reward"] for row in rows]) or (metric is newline and rare([metric(row) for row in rows])) or not spread_x * spread_y:
        return float("nan"), float("nan")
    r = sum(x * y for x, y in pairs) / (spread_x * spread_y) ** 0.5
    return r, ((1 - r * r) / (len(pairs) - 2)) ** 0.5


CORRELATIONS = {"--correlation": (length, "completion length", "correlation"), "--newline-correlation": (newline, "using a newline", "newline-correlation")}


def plot_correlation(run: str, metric, name: str, suffix: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    panels = [("All samples", False), ("Within a puzzle", True)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    levels = sorted(rows)
    last = max(max(by_step) for by_step in rows.values())
    for offset, color, level in zip(range(len(levels)), plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), levels):
        starts = range(0, last + 1, BIN)
        bins = [[r for step in range(start, start + BIN) for r in rows[level].get(step, [])] for start in starts]
        for ax, (title, within) in zip(axes, panels):
            r, error = zip(*(correlation(b, metric, within) for b in bins))
            xs = [s + BIN / 2 + 1.5 * (offset - (len(levels) - 1) / 2) for s in starts]
            ax.errorbar(xs, r, yerr=error, fmt="o-", color=color, label=f"{level} swaps", capsize=2, elinewidth=1)
            ax.set_title(title)
    for ax in axes:
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Training step")
        ax.set_xlim(-5, last + 6)
    axes[0].set_ylabel("Correlation with correctness")
    axes[1].legend(loc="best")
    fig.suptitle(f"{model}: {name} and correctness")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-{suffix}.png", dpi=150)


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
        top.set_ylim(0, 1)
        top.set_yticks([0, 1], ["0%", "100%"] if i % 2 == 0 else ["", ""])
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
    fig.suptitle(f"{args['model'].split('/')[-1]}: completion length and the token limit")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-token-limit.png", dpi=150)


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CONCLUSION = re.compile(r"Therefore|Thus|So,? after|After all|After these|In conclusion|Finally,|Hence|At the end|After performing|After completing")
PARTS = ["Opening", "Swaps that move the key", "Swaps that leave it", "Conclusion", "Closing tag and answer"]


def tracking(row: dict) -> list[str]:
    return [outcome for outcome, _ in swap_steps(row)]


def right_share(rows: list[dict]) -> float:
    steps = [step for row in rows for step in swap_steps(row)]
    return sum(outcome == "right" for outcome, _ in steps) / len(steps)


def standalone_share(rows: list[dict]) -> float:
    steps = [step for row in rows for step in swap_steps(row)]
    return sum(standalone for _, standalone in steps) / len(steps)


def trace_parts(row: dict, count) -> dict:
    """Split a completion into the text spent on each swap of its puzzle."""
    swaps, _, moves = puzzle(row)
    reasoning, tag, rest = row["completion"].partition("</think>")
    starts = [m and m.start() for m in swap_mentions(reasoning, swaps)]
    found = [s for s in starts if s is not None]
    conclusion = CONCLUSION.search(reasoning, found[-1]) if found else None
    end = conclusion.start() if conclusion else len(reasoning)
    size = dict(zip(found, map(count, (reasoning[a:b] for a, b in zip(found, found[1:] + [end])))))
    per_swap = [(moved, size[s]) for s, moved in zip(starts, moves) if s is not None]
    return {
        "moving": [n for moved, n in per_swap if moved],
        "staying": [n for moved, n in per_swap if not moved],
        "Opening": count(reasoning[: found[0]] if found else reasoning),
        "Conclusion": count(reasoning[end:]),
        "Closing tag and answer": count(tag + rest),
    }


# Real sentences to show beside the standalone-statement panel: run -> (swap count, training step, text).
EXAMPLES = {
    "swaps-7b-tags-v2": [
        (4, 5, 'Step 5\n"3. Swap boxes E and B:\nThe key moves from box B to box E."'),
        (4, 100, 'Step 100\n"1. Swap the contents of boxes D and A:\nThe key, being in box D, will move to box A.\nSo now, the key is in box A."'),
        (4, 187, 'Step 187\n"5. Swap the contents of boxes A and E:\nThe key, initially in E, will move to box A."'),
    ],
    "swaps-7b-tags-2and4": [
        (4, 5, 'Step 5\n"4. Swap F and A:\nThe contents of F and A are exchanged.\nThe key moves from A to F."'),
        (4, 88, 'Step 88\n"4. Swap the contents of boxes A and F:\nThe key was in F before this swap,\nso it moves to A. Now, the key is in box A."'),
    ],
}


def plot_tracking(run: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width, examples = 5, EXAMPLES.get(run, [])
    panels = [
        ("Correctly tracks the key's location", right_share),
        ("Swaps followed by a standalone location statement", standalone_share),
    ]
    if examples:
        grid = plt.figure(figsize=(15, 8), layout="constrained").subplot_mosaic("\n".join(["x" * 4 + "a" * 5 + "y" * 5, "l" * 4 + "c" * 5 + "r" * 5]))
        fig, axes, margins = grid["a"].figure, [grid["a"], grid["c"]], [grid["l"], grid["r"]]
        for name in "xylr":
            grid[name].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    binned = {}
    for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(rows)):
        bins = defaultdict(list)
        for step, step_rows in rows[level].items():
            bins[step // width * width] += step_rows
        binned[level] = dict(sorted(bins.items()))
        for ax, (_, metric) in zip(axes, panels):
            ax.plot([s + width / 2 for s in binned[level]], [metric(rs) for rs in binned[level].values()], color=color, label=f"{level} swaps")
    for ax, (title, _) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 1.03)
    axes[0].legend(loc="lower right")
    for (level, step, text), (side, x, height) in zip(examples, [(0, 0.8, 0.3), (1, 0.05, 0.75), (1, 0.05, 0.3)]):
        start = step // width * width
        axes[1].annotate(
            text, xy=(start + width / 2, standalone_share(binned[level][start])), xytext=(x, height),
            textcoords=margins[side].transAxes, va="center", ha="left" if side else "right", multialignment="left", fontsize=10.5, bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.4"},
            arrowprops={"arrowstyle": "->", "color": "0.25"}, annotation_clip=False,
        ).set_in_layout(False)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-tracking.png", dpi=150)


def plot_traces(run: str):
    from transformers import AutoTokenizer

    args = json.loads(Path("runs", run, "args.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args["model"])
    count = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    held_out = load(run)["eval"]
    level = max(held_out)
    steps = [s for s in sorted(held_out[level]) if s >= 40]
    solved = set.intersection(*({r["prompt"] for r in held_out[level][s] if r["reward"]} for s in steps))
    traces = {s: [trace_parts(r, count) for r in held_out[level][s] if r["prompt"] in solved] for s in steps}
    mean = lambda values: sum(values) / len(values)
    fig, (per_swap, parts) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for side, key, color in [(0, "moving", PALETTE[1]), (1, "staying", PALETTE[2])]:
        label = PARTS[1 + side]
        per_swap.plot(steps, [mean([n for t in traces[s] for n in t[key]]) for s in steps], "o-", color=color, label=label)
    totals = {
        "Opening": [mean([t["Opening"] for t in traces[s]]) for s in steps],
        "Swaps that move the key": [mean([sum(t["moving"]) for t in traces[s]]) for s in steps],
        "Swaps that leave it": [mean([sum(t["staying"]) for t in traces[s]]) for s in steps],
        "Conclusion": [mean([t["Conclusion"] for t in traces[s]]) for s in steps],
        "Closing tag and answer": [mean([t["Closing tag and answer"] for t in traces[s]]) for s in steps],
    }
    floor = [0.0] * len(steps)
    for name, color in zip(PARTS, [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], "0.6"]):
        parts.bar(steps, totals[name], 14, bottom=floor, color=color, label=name)
        floor = [f + v for f, v in zip(floor, totals[name])]
    per_swap.set_title("Tokens per swap addressed")
    per_swap.set_ylim(0)
    per_swap.legend(loc="lower left")
    parts.set_title("Tokens by part of the response")
    parts.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    for ax in (per_swap, parts):
        ax.set_xlabel("Training step")
    fig.suptitle(f"{args['model'].split('/')[-1]}: correct {level}-swap responses")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-traces.png", dpi=150)


def plot_coverage(run: str):
    data = load(run)
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width = 5
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(data["train"] | data["eval"])):
        trained = level in data["train"]
        bins = defaultdict(list)
        for step, rows in data["train" if trained else "eval"][level].items():
            bins[step // width * width + width / 2 if trained else step] += rows
        bins = dict(sorted(bins.items()))
        ax.plot(
            list(bins), [sum("skipped" not in tracking(r) for r in rs) / len(rs) for rs in bins.values()], color=color,
            label=f"{level} swaps" if trained else f"{level} swaps, held-out only",
        )
    ax.set_title("Responses that write out every swap")
    ax.set_xlabel("Training step")
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.set_ylim(0, 1.03)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-coverage.png", dpi=150)


def plot_train_eval(run: str):
    data = load(run)
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    levels = sorted(data["train"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, layout="constrained")
    for ax, level in zip(axes.flatten(), levels):
        ax.plot(*smoothed(data["train"][level], newline), color=PALETTE[0], label="Training puzzles")
        ax.plot(*held_out_curve(data["eval"][level], newline), "o-", color=PALETTE[1], markersize=4, label="Held-out puzzles")
        ax.set_title(f"{level} swaps")
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    axes[0, 1].legend(loc="upper right")
    fig.suptitle(f"{model}: completions with a newline")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-train-eval.png", dpi=150)


def sentence_templates(row: dict) -> set[str]:
    """Sentences of the reasoning with whitespace collapsed, box letters as X and numbers as N. A sentence ends
    at a period followed by a space, except after a list marker such as "3."."""
    text = " ".join(row["completion"].partition("</think>")[0].split())
    text = re.sub(r"\d+", "N", re.sub(r"\b[A-F]\b", "X", text))
    return {sentence.strip() for sentence in re.split(r"(?<=[^N]\.)\s+", text) if sentence.strip()}


def common_templates(run: str, width: int, floor: float = 0.1) -> tuple[list[str], dict[int, dict[str, float]]]:
    """Templates in at least `floor` of the responses of some 20-step window, ordered by when they peak, and
    the share of responses containing each template in every `width`-step window."""
    counts, sizes = defaultdict(lambda: defaultdict(int)), defaultdict(int)
    for by_step in load(run)["train"].values():
        for step, rows in by_step.items():
            for row in rows:
                sizes[step] += 1
                for template in sentence_templates(row):
                    counts[step][template] += 1
    def shares(size: int) -> dict[int, dict[str, float]]:
        totals, n = defaultdict(lambda: defaultdict(int)), defaultdict(int)
        for step in counts:
            n[step // size * size] += sizes[step]
            for template, count in counts[step].items():
                totals[step // size * size][template] += count
        return {start: {t: c / n[start] for t, c in totals[start].items()} for start in sorted(totals)}
    coarse = shares(BIN)
    kept = {t for window in coarse.values() for t, share in window.items() if share >= floor}
    peak = lambda t: max(coarse, key=lambda start: coarse[start].get(t, 0))
    order = sorted(kept, key=lambda t: (peak(t), -coarse[peak(t)][t]))
    return order, shares(width)


def plot_templates(run: str):
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width = 10
    order, shares = common_templates(run, width)
    columns = 3
    rows = -(-len(order) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.1 * rows), sharex=True, sharey=True, layout="constrained")
    for ax, template in zip(axes.flatten(), order):
        ax.plot([start + width / 2 for start in shares], [shares[start].get(template, 0) for start in shares], color=PALETTE[0], linewidth=2)
        ax.set_title("\n".join(textwrap.wrap(template, 40)), fontsize=13)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 1.03)
    for ax in axes.flatten()[len(order) :]:
        ax.axis("off")
    fig.supxlabel("Training step", fontsize=13)
    fig.supylabel("Responses containing the sentence", fontsize=13)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-templates.png", dpi=130)


def first_word(row: dict) -> str:
    return (re.findall(r"[A-Za-z']+", row["completion"]) or [""])[0]


FORMATTING = [
    ("Numbered list", lambda row: bool(re.search(r"(?:^|\s)1[.)]\s.*\s2[.)]\s", " ".join(row["completion"].split())))),
    ("Dash bullets", lambda row: bool(re.search(r"(?:^|\s)- \S", " ".join(row["completion"].split())))),
    ('"Step N:" headers', lambda row: bool(re.search(r"Step \d+:", row["completion"]))),
    ('Opens with "Let\'s"', lambda row: first_word(row) == "Let's"),
    ('Opens with "Okay"', lambda row: first_word(row) == "Okay"),
    ('Opens with "To"', lambda row: first_word(row) == "To"),
    ("Longer than typical for its swap count", None),
    ("Correct", lambda row: bool(row["reward"])),
]


def plot_formatting(run: str):
    data = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width = 10
    bins = defaultdict(list)
    for by_step in data.values():
        for step, rows in by_step.items():
            bins[step // width * width] += rows
    bins = dict(sorted(bins.items()))
    centers = [start + width / 2 for start in bins]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), sharex=True, sharey=True, layout="constrained")
    for ax, (name, has) in zip(axes.flatten(), FORMATTING):
        frequency, with_it, without = [], [], []
        for rows in bins.values():
            typical = {level: sum(r["tokens"] for r in rows if r["level"] == level) / sum(r["level"] == level for r in rows) for level in data}
            flags = [has(r) if has else r["tokens"] > typical[r["level"]] for r in rows]
            sides = [[newline(r) for r, flag in zip(rows, flags) if flag == side] for side in (True, False)]
            frequency.append(sum(flags) / len(rows))
            with_it.append(sum(sides[0]) / len(sides[0]) if len(sides[0]) >= 20 else float("nan"))
            without.append(sum(sides[1]) / len(sides[1]) if len(sides[1]) >= 20 else float("nan"))
        ax.fill_between(centers, frequency, color="0.85", label="Responses with the feature")
        ax.plot(centers, with_it, "o-", color=PALETTE[1], markersize=4, label="Newline use, with the feature")
        ax.plot(centers, without, "o-", color=PALETTE[0], markersize=4, label="Newline use, without it")
        ax.set_title(name)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-formatting.png", dpi=150)


def advantages(rows: list[dict]) -> list[float]:
    """Each response's GRPO advantage: reward minus its puzzle group's mean, over the group's standard deviation."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["prompt"]].append(row["reward"])
    values = []
    for rewards in groups.values():
        mean = sum(rewards) / len(rewards)
        spread = (sum((x - mean) ** 2 for x in rewards) / (len(rewards) - 1)) ** 0.5
        values += [(x - mean) / (spread + 1e-4) for x in rewards]
    return values


def plot_advantages(run: str):
    data = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    steps = sorted({step for by_step in data.values() for step in by_step})
    by_step = {step: advantages([row for level in data.values() for row in level.get(step, [])]) for step in steps}
    windows = [(0, 20), (40, 60), (steps[-1] - 19, steps[-1] + 1)]
    fig, (share, *hists) = plt.subplots(1, 4, figsize=(17, 4.2), layout="constrained")
    nonzero = [sum(a != 0 for a in by_step[s]) / len(by_step[s]) for s in steps]
    share.plot(steps, [sum(w := nonzero[max(0, i - WINDOW + 1) : i + 1]) / len(w) for i in range(len(steps))], color="0.2", linewidth=2)
    share.set_title("Responses with a nonzero advantage")
    share.set_xlabel("Training step")
    share.yaxis.set_major_formatter("{x:.0%}")
    share.set_ylim(0, 1.03)
    edges = [-2.75 + 0.25 * i for i in range(23)]
    for ax, (lo, hi) in zip(hists, windows):
        values = [a for s in range(lo, hi) for a in by_step[s]]
        kept = [a for a in values if a != 0]
        ax.hist([a for a in kept if a < 0], bins=edges, weights=[1 / len(values)] * sum(a < 0 for a in kept), color=PALETTE[1], label="Incorrect")
        ax.hist([a for a in kept if a > 0], bins=edges, weights=[1 / len(values)] * sum(a > 0 for a in kept), color=PALETTE[0], label="Correct")
        ax.set_title(f"Steps {lo} to {hi - 1}")
        ax.set_xlabel("Advantage")
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 0.2)
    hists[0].set_ylabel("Share of all responses")
    hists[0].legend(loc="upper left")
    fig.suptitle(f"{model}: GRPO advantages")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-advantages.png", dpi=150)


def group_advantages(rows: list[dict]) -> list[tuple[dict, float, list[dict]]]:
    """(response, its GRPO advantage, its puzzle group) for every response in a group with mixed rewards."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["prompt"]].append(row)
    out = []
    for group in groups.values():
        rewards = [r["reward"] for r in group]
        mean = sum(rewards) / len(group)
        spread = (sum((x - mean) ** 2 for x in rewards) / (len(group) - 1)) ** 0.5
        if spread:
            out += [(row, (row["reward"] - mean) / (spread + 1e-4), group) for row in group]
    return out


def plot_lead(runs: list[str]):
    """One point per 5-step window in which newline use still varies (between 1% and 99% of responses)."""
    width = 5
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for run, color in zip(runs, PALETTE):
        data = load(run)["train"]
        steps = sorted({step for by_step in data.values() for step in by_step})
        pooled = [[row for by_step in data.values() for row in by_step.get(step, [])] for step in steps]
        rate = [sum(map(newline, rows)) / len(rows) for rows in pooled]
        gap = [
            sum(a * (newline(row) - sum(map(newline, group)) / len(group)) for row, a, group in group_advantages(rows)) / len(rows)
            for rows in pooled
        ]
        window = lambda values: [sum(values[i : i + width]) / width for i in range(0, len(values) - width + 1, width)]
        rates, gaps = window(rate), window(gap)
        points = [(gaps[i], rates[i + 1] - rates[i]) for i in range(len(rates) - 1) if 0.01 < rates[i] < 0.99]
        xs, ys = zip(*points)
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        r = sum((x - mean_x) * (y - mean_y) for x, y in points) / (sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)) ** 0.5
        trained = ", ".join(map(str, sorted(data)))
        label = f"r = {r:+.2f}, {len(points)} windows" + (f" (trained on {trained} swaps)" if len(runs) > 1 else "")
        ax.scatter(xs, ys, color=color, s=36, label=label)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Advantage gap for newline responses in a 5-step window")
    ax.set_ylabel("Change in newline use into the next window")
    ax.yaxis.set_major_formatter("{x:+.0%}")
    ax.legend(loc="lower right", fontsize=10)
    fig.suptitle(json.loads(Path("runs", runs[0], "args.json").read_text())["model"].split("/")[-1])
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/newline-lead.png", dpi=150)


def plot_cutoffs(runs: list[str]):
    width = 10
    fig, axes = plt.subplots(2, len(runs), figsize=(5.5 * len(runs), 7), sharex=True, sharey=True, layout="constrained", squeeze=False)
    for (top, bottom), run in zip(axes.T, runs):
        data = load(run)["train"]
        limit = json.loads(Path("runs", run, "args.json").read_text())["max_tokens"]
        pooled = defaultdict(list)
        for by_step in data.values():
            for step, rows in by_step.items():
                pooled[step] += rows
        top.plot(*smoothed(pooled, newline), color="#eb6834", linewidth=2)
        top.set_title(f"{limit:,}-token limit")
        wrong = defaultdict(list)
        for step, rows in pooled.items():
            wrong[step // width * width] += [row["tokens"] >= limit - 2 for row in rows if not row["reward"]]
        starts = sorted(wrong)
        shares = [sum(wrong[s]) / len(wrong[s]) for s in starts]
        bottom.bar([s + width / 2 for s in starts], shares, width - 2, color="#eda100")
        for s, share in zip(starts, shares):
            n = len(wrong[s])
            bottom.text(s + width / 2, share + 0.01, f"{n / 1000:.1f}k" if n >= 1000 else n, ha="center", va="bottom", fontsize=7, color="0.3")
        bottom.set_xlabel("Training step")
    for ax in axes.flat:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter("{x:.0%}")
    axes[0, 0].set_ylabel("Completions with a newline")
    axes[1, 0].set_ylabel("Wrong answers cut off")
    fig.suptitle(json.loads(Path("runs", runs[0], "args.json").read_text())["model"].split("/")[-1])
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/cutoffs-newlines.png", dpi=150)


def plot_length_push(runs: list[str]):
    """Per step, the sum over responses of advantage × (length − group mean length), sign flipped so up means shorter."""
    width = 10
    sources = [("Groups with a cut-off answer", "#eda100"), ("Groups with other wrong answers only", "#2a78d6")]
    fig, axes = plt.subplots(2, len(runs), figsize=(5.5 * len(runs), 7), sharex=True, layout="constrained", squeeze=False)
    for (top, bottom), run in zip(axes.T, runs):
        data = load(run)["train"]
        limit = json.loads(Path("runs", run, "args.json").read_text())["max_tokens"]
        pooled = defaultdict(list)
        for by_step in data.values():
            for step, rows in by_step.items():
                pooled[step] += rows
        top.plot(*smoothed(pooled, newline), color="#eb6834", linewidth=2)
        top.set_ylim(0, 1)
        top.yaxis.set_major_formatter("{x:.0%}")
        top.set_title(f"{limit:,}-token limit")
        push = defaultdict(lambda: [0.0, 0.0])
        for step, rows in pooled.items():
            for row, advantage, group in group_advantages(rows):
                cut = any(r["tokens"] >= limit - 2 for r in group)
                mean = sum(r["tokens"] for r in group) / len(group)
                push[step // width * width][0 if cut else 1] -= advantage * (row["tokens"] - mean) / width
        starts = sorted(push)
        centers = [s + width / 2 for s in starts]
        below = [0.0] * len(starts)
        for i, (label, color) in enumerate(sources):
            values = [push[s][i] for s in starts]
            base = [b if v >= 0 else 0 for b, v in zip(below, values)]
            bottom.bar(centers, values, width - 2, bottom=base, color=color, label=label)
            below = [b + max(v, 0) for b, v in zip(below, values)]
        bottom.axhline(0, color="black", linewidth=0.8)
        bottom.grid(axis="y", color="0.9", linewidth=0.8)
        bottom.set_axisbelow(True)
        bottom.set_xlabel("Training step")
        bottom.set_ylim(-400, 1000)
        for center, total in zip(centers, below):
            if total > 1000:
                bottom.text(center, 985, f"{total / 1000:.1f}k", ha="center", va="top", fontsize=7, color="white")
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("Completions with a newline")
    axes[1, 0].set_ylabel("Push toward shorter")
    fig.align_ylabels(axes[:, 0])
    axes[1, 0].legend(loc="upper right", fontsize=10)
    fig.suptitle(json.loads(Path("runs", runs[0], "args.json").read_text())["model"].split("/")[-1])
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/length-push.png", dpi=150)


def plot_push(run: str):
    data = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width = 10
    typical = {level: sum(r["tokens"] for rows in by_step.values() for r in rows) / sum(map(len, by_step.values())) for level, by_step in data.items()}
    longer = lambda row, group: row["tokens"] > sum(r["tokens"] for r in group) / len(group)
    features = [
        (newline, "Completions with a newline", "{x:.0%}", lambda row, group: bool(newline(row)), "With a newline", "Without"),
        (lambda row: row["tokens"] / typical[row["level"]], "Relative completion length", "{x:.2f}", longer, "Longer than its group's mean", "Shorter"),
    ]
    steps = sorted({step for by_step in data.values() for step in by_step})
    pooled = {step: [row for by_step in data.values() for row in by_step.get(step, [])] for step in steps}
    scored = {step: group_advantages(rows) for step, rows in pooled.items()}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, layout="constrained")
    starts = range(0, steps[-1] + 1, width)
    for (top, bottom), (feature, title, tick_format, has, yes, no) in zip(axes.T, features):
        top.plot(*smoothed(pooled, feature), color="0.2", linewidth=2)
        top.set_title(title)
        top.yaxis.set_major_formatter(tick_format)
        for side, label, color in [(True, yes, PALETTE[1]), (False, no, PALETTE[0])]:
            bins = [[a for s in range(start, start + width) for row, a, group in scored.get(s, []) if has(row, group) == side] for start in starts]
            kept = [(start + width / 2, b) for start, b in zip(starts, bins) if len(b) >= MIN_SAMPLES]
            means = [sum(b) / len(b) for _, b in kept]
            errors = [(sum((a - m) ** 2 for a in b) / (len(b) - 1) / len(b)) ** 0.5 for (_, b), m in zip(kept, means)]
            bottom.errorbar([x for x, _ in kept], means, yerr=errors, fmt="o-", color=color, label=label, capsize=2, markersize=4)
        bottom.axhline(0, color="gray", linestyle=":", linewidth=1)
        bottom.set_title("Mean advantage")
        bottom.set_xlabel("Training step")
        bottom.set_ylim(-1.5, 1.5)
        bottom.legend(loc="upper left")
    axes[0, 0].set_ylim(0, 1.03)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-push.png", dpi=150)


def opens_with_okay(row: dict) -> float:
    return float(row["completion"].lstrip().startswith("Okay"))


def plot_okay(run: str):
    data = load(run)
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    fig, (by_level, by_outcome, accuracies) = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True, layout="constrained")
    for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(data["train"] | data["eval"])):
        if level in data["train"]:
            by_level.plot(*smoothed(data["train"][level], opens_with_okay), color=color, label=f"{level} swaps")
        else:
            by_level.plot(*held_out_curve(data["eval"][level], opens_with_okay), color=color, label=f"{level} swaps, held-out only")
    bins = defaultdict(list)
    for by_step in data["train"].values():
        for step, rows in by_step.items():
            bins[step // BIN * BIN] += rows
    bins = dict(sorted(bins.items()))
    centers = [start + BIN / 2 for start in bins]
    mean = lambda values: sum(values) / len(values) if len(values) >= MIN_SAMPLES else float("nan")
    for name, color, reward in [("Correct responses", OUTCOMES[0][1], 1), ("Incorrect responses", OUTCOMES[1][1], 0)]:
        by_outcome.plot(centers, [mean([opens_with_okay(r) for r in rs if r["reward"] == reward]) for rs in bins.values()], "o-", color=color, label=name)
    for name, color, okay in [('Opens with "Okay"', PALETTE[2], 1), ("Opens with anything else", "0.45", 0)]:
        accuracies.plot(centers, [mean([r["reward"] for r in rs if opens_with_okay(r) == okay]) for rs in bins.values()], "o-", color=color, label=name)
    by_level.set_title('Responses that open with "Okay"')
    by_outcome.set_title('Open with "Okay", by outcome')
    accuracies.set_title("Accuracy, by opening")
    for ax in (by_level, by_outcome, accuracies):
        ax.set_xlabel("Training step")
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 1.03)
        ax.legend(loc="best")
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-okay.png", dpi=150)


ERRORS = [
    ("Cut off at the token limit", "#eda100"),
    ("Answer malformed", "0.65"),
    ("Left out or added a swap", "#2a78d6"),
    ("Right swaps, tracked wrongly", "#eb6834"),
    ("Other", "0.3"),
]


def error_cause(row: dict, max_tokens: int) -> str:
    """The first of these that applies to a response that scored 0."""
    if row["parsed"] is None:
        return ERRORS[0][0] if row["tokens"] >= max_tokens - 2 else ERRORS[1][0]
    if any(step_counts(row)):
        return ERRORS[2][0]
    return ERRORS[3][0] if "wrong" in tracking(row) else ERRORS[4][0]


def plot_errors(run: str):
    data = load(run)
    args = json.loads(Path("runs", run, "args.json").read_text())
    levels = sorted(data["train"] | data["eval"])
    evaluated = sorted(next(iter(data["eval"].values())))
    width = evaluated[1] - evaluated[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, layout="constrained")
    for ax, level in zip(axes.flatten(), levels):
        trained = level in data["train"]
        bins = defaultdict(list)
        for step, rows in data["train" if trained else "eval"][level].items():
            bins[(step // width + 1) * width if trained else step] += rows
        bins = dict(sorted(bins.items()))
        causes = [[error_cause(r, args["max_tokens"]) for r in rs if not r["reward"]] for rs in bins.values()]
        floor = [0.0] * len(bins)
        shown = [len(c) >= MIN_SAMPLES for c in causes]
        for name, color in ERRORS:
            share = [c.count(name) / len(c) if ok else 0 for c, ok in zip(causes, shown)]
            ax.bar(list(bins), share, 0.8 * width, bottom=floor, color=color, label=name)
            floor = [f + v for f, v in zip(floor, share)]
        for x, c, ok in zip(bins, causes, shown):
            if ok:
                ax.annotate(len(c), (x, 1), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(f"{level} swaps" if trained else f"{level} swaps, held-out only", pad=14)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of wrong responses")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle(f"{args['model'].split('/')[-1]}: error analysis")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-error-shares.png", dpi=150)


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
    if sys.argv[1] == "--lead":
        plot_lead(sys.argv[2:])
        sys.exit()
    if sys.argv[1] == "--cutoffs":
        plot_cutoffs(sys.argv[2:])
        sys.exit()
    if sys.argv[1] == "--length-push":
        plot_length_push(sys.argv[2:])
        sys.exit()
    if sys.argv[1] in CORRELATIONS:
        plot_correlation(sys.argv[2], *CORRELATIONS[sys.argv[1]])
        sys.exit()
    if sys.argv[1] in BY_OUTCOME:
        plot_by_outcome(sys.argv[2], *BY_OUTCOME[sys.argv[1]])
        sys.exit()
    OTHER = {
        "--length-signal": plot_length_signal, "--token-limit": plot_token_limit,
        "--traces": plot_traces, "--tracking": plot_tracking, "--coverage": plot_coverage,
        "--okay": plot_okay, "--push": plot_push, "--formatting": plot_formatting, "--advantages": plot_advantages, "--templates": plot_templates, "--train-eval": plot_train_eval, "--errors": plot_errors,
    }
    if sys.argv[1] in OTHER:
        OTHER[sys.argv[1]](sys.argv[2])
        sys.exit()
    runs = [arg for arg in sys.argv[1:] if arg != "--start-box"]
    for run in runs:
        plot(run, start_box=len(runs) < len(sys.argv) - 1)
