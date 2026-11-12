"""Figures from saved rollouts.

uv run --with matplotlib plots.py [--start-box] RUN [RUN ...]   training curves per run
uv run --with matplotlib plots.py --formats MODEL                 base model, think tags vs plain format
uv run --with matplotlib plots.py --lengths RUN                   completion length by outcome, per level
uv run --with matplotlib plots.py --newlines RUN                  share of completions with a newline, by outcome
uv run --with matplotlib plots.py --length-signal RUN             what the reward says about length
uv run --with matplotlib plots.py --correlation RUN               correlation of length with correctness
uv run --with matplotlib plots.py --token-limit RUN               completion tokens against the token limit
uv run --with matplotlib plots.py --blank-lines RUN               where the first blank line falls; reward vs a lenient parser
uv run --with matplotlib --with transformers plots.py --traces RUN   what correct held-out traces spend their tokens on
uv run --with matplotlib plots.py --tracking RUN                  per-swap tracking and standalone location statements, by swap count
uv run --with matplotlib plots.py --coverage RUN                  how many of a puzzle's swaps the reasoning writes out
uv run --with matplotlib plots.py --okay RUN                      responses that open with "Okay"
uv run --with matplotlib plots.py --errors RUN                    what went wrong in the responses that scored 0
uv run --with matplotlib plots.py --error-shares RUN              the same, as a share of the wrong responses
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from traces import PAIR, START, newline_after_swaps, puzzle, swap_mentions, swap_steps

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
    panels[0].set_ylim(0, 1.05 * max(ax.dataLim.y1 for ax in panels))
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
    for ax in (kinds_ax, newline_ax):
        ax.set_xlabel("Training step")
    accuracy_ax.legend(loc="best")
    kinds_ax.legend(loc="best")
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
    panels = [("All samples", False), ("Within a puzzle", True)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    levels = sorted(rows)
    for offset, color, level in zip(range(len(levels)), plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), levels):
        starts = range(0, max(rows[level]) + 1, BIN)
        bins = [[r for step in range(start, start + BIN) for r in rows[level].get(step, [])] for start in starts]
        for ax, (title, within) in zip(axes, panels):
            r, error = zip(*(correlation(b, within) for b in bins))
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
    fig.suptitle(f"{args['model'].split('/')[-1]}: completion length and the token limit")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-token-limit.png", dpi=150)


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LOCATIONS = [
    "Right after <think>", "After the opening sentence", "Between swap steps",
    "Just before </think>", "Between the tags", "Inside the answer",
]


def blank_line_location(row: dict) -> str | None:
    text = row["completion"]
    at, think, answer = text.find("\n\n"), text.find("</think>"), text.find("<answer>")
    if at < 0:
        return None
    if not text[:at].strip():
        return LOCATIONS[0]
    if think < 0 or at < think:
        if think >= 0 and not text[at:think].strip():
            return LOCATIONS[3]
        return LOCATIONS[2] if PAIR.search(text[:at]) else LOCATIONS[1]
    return LOCATIONS[4] if answer < 0 or at < answer else LOCATIONS[5]


def lenient_correct(row: dict) -> float:
    """Right letter in the last answer tag, whatever the whitespace, closing tags or wording around it."""
    answer = row["completion"].rpartition("<answer>")
    letters = set(re.findall(r"\b[A-F]\b", answer[2].split("</answer>")[0]))
    return float(bool(answer[1]) and letters == {row["answer"]})


def plot_blank_lines(run: str):
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    pooled = defaultdict(list)
    for by_step in rows.values():
        for step, step_rows in by_step.items():
            pooled[step] += step_rows
    fig, (where, scores) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    shares = [smoothed(pooled, lambda row, name=name: float(blank_line_location(row) == name)) for name in LOCATIONS]
    where.stackplot(shares[0][0], [share for _, share in shares], labels=LOCATIONS, colors=PALETTE)
    where.set_title("Where the first blank line falls")
    where.yaxis.set_major_formatter("{x:.0%}")
    where.legend(loc="upper right", fontsize=10)
    scores.plot(*smoothed(pooled, accuracy), color=PALETTE[0], linewidth=2, label="Rewarded")
    scores.plot(*smoothed(pooled, lenient_correct), color=PALETTE[1], linewidth=2, linestyle="--", label="Correct under a lenient parser")
    scores.set_title("Reward against a lenient parser")
    scores.yaxis.set_major_formatter("{x:.0%}")
    scores.legend(loc="lower right")
    for ax in (where, scores):
        ax.set_xlabel("Training step")
        ax.set_ylim(0)
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-blank-lines.png", dpi=150)


CONCLUSION = re.compile(r"Therefore|Thus|So,? after|After all|After these|In conclusion|Finally,|Hence|At the end|After performing|After completing")
PARTS = ["Opening", "Swaps that move the key", "Swaps that leave it", "Conclusion", "Closing tag and answer"]


def tracking(row: dict) -> list[str]:
    return [outcome for outcome, _ in swap_steps(row)]


def right_share(rows: list[dict]) -> float:
    steps = [step for row in rows for step in swap_steps(row)]
    return sum(outcome == "right" for outcome, _ in steps) / len(steps)


def newline_step_share(rows: list[dict]) -> float:
    steps = [step for row in rows for step in newline_after_swaps(row)]
    return sum(steps) / len(steps)


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
        ("Swaps followed by a newline", newline_step_share),
        ("Swaps followed by a standalone location statement", standalone_share),
    ]
    if examples:
        grid = plt.figure(figsize=(15, 8), layout="constrained").subplot_mosaic("\n".join(["a" * 7 + "b" * 7, "l" * 4 + "c" * 5 + "r" * 5]))
        fig, axes, margins = grid["a"].figure, [grid["a"], grid["b"], grid["c"]], [grid["l"], grid["r"]]
        for margin in margins:
            margin.axis("off")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
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
        axes[2].annotate(
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
    ("Answer malformed", "0.6"),
    ("Left out a swap", "#2a78d6"),
    ("Tracked a swap wrongly", "#eb6834"),
    ("Reasoning right, answer differs", "#e87ba4"),
    ("Key location not stated", "#1baf7a"),
]


def error_cause(row: dict, max_tokens: int) -> str:
    """The first of these that applies to a response that scored 0."""
    outcomes = tracking(row)
    if row["parsed"] is None:
        return ERRORS[0][0] if row["tokens"] >= max_tokens - 2 else ERRORS[1][0]
    for outcome, cause in [("skipped", ERRORS[2][0]), ("wrong", ERRORS[3][0])]:
        if outcome in outcomes:
            return cause
    return ERRORS[4][0] if set(outcomes) == {"right"} else ERRORS[5][0]


def plot_errors(run: str, of_wrong: bool = False):
    data = load(run)
    args = json.loads(Path("runs", run, "args.json").read_text())
    levels = sorted(data["train"] | data["eval"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, layout="constrained")
    for ax, level in zip(axes.flatten(), levels):
        trained = level in data["train"]
        bins = defaultdict(list)
        for step, rows in data["train" if trained else "eval"][level].items():
            bins[step // BIN * BIN + BIN / 2 if trained else step] += rows
        bins = dict(sorted(bins.items()))
        causes = [[error_cause(r, args["max_tokens"]) for r in rs if not r["reward"]] for rs in bins.values()]
        floor = [0.0] * len(bins)
        shown = [len(c) >= MIN_SAMPLES or not of_wrong for c in causes]
        for name, color in ERRORS:
            share = [c.count(name) / (len(c) if of_wrong else len(rs)) if ok else 0 for c, rs, ok in zip(causes, bins.values(), shown)]
            ax.bar(list(bins), share, (BIN if trained else 10) - 3, bottom=floor, color=color, label=name)
            floor = [f + v for f, v in zip(floor, share)]
        if of_wrong:
            for x, c, ok in zip(bins, causes, shown):
                if ok:
                    ax.annotate(len(c), (x, 1), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(f"{level} swaps" if trained else f"{level} swaps, held-out only", pad=14 if of_wrong else None)
        ax.yaxis.set_major_formatter("{x:.0%}")
        if of_wrong:
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of wrong responses" if of_wrong else "Share of responses")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle(f"{args['model'].split('/')[-1]}: error analysis")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-{'error-shares' if of_wrong else 'errors'}.png", dpi=150)


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
    OTHER = {
        "--length-signal": plot_length_signal, "--correlation": plot_correlation, "--token-limit": plot_token_limit,
        "--blank-lines": plot_blank_lines, "--traces": plot_traces, "--tracking": plot_tracking, "--coverage": plot_coverage,
        "--okay": plot_okay, "--errors": plot_errors, "--error-shares": lambda run: plot_errors(run, of_wrong=True),
    }
    if sys.argv[1] in OTHER:
        OTHER[sys.argv[1]](sys.argv[2])
        sys.exit()
    runs = [arg for arg in sys.argv[1:] if arg != "--start-box"]
    for run in runs:
        plot(run, start_box=len(runs) < len(sys.argv) - 1)
