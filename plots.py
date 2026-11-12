"""Figures from saved rollouts.

uv run --with matplotlib plots.py [--start-box] RUN [RUN ...]   training curves per run
uv run --with matplotlib plots.py --formats MODEL                 base model, think tags vs plain format
uv run --with matplotlib plots.py --lengths RUN                   completion length by outcome, per level
uv run --with matplotlib plots.py --newlines RUN                  share of completions with a newline, by outcome
uv run --with matplotlib plots.py --length-signal RUN             what the reward says about length
uv run --with matplotlib plots.py --correlation RUN               correlation of length with correctness
uv run --with matplotlib plots.py --token-limit RUN               completion tokens against the token limit
uv run --with matplotlib plots.py --blank-lines RUN               where the first blank line falls; reward vs a lenient parser
uv run --with matplotlib plots.py --groups RUN                    rewarded vs unrewarded samples of the same puzzle
uv run --with matplotlib --with transformers plots.py --traces RUN   what correct held-out traces spend their tokens on
uv run --with matplotlib plots.py --tracking RUN                  blank lines against per-swap tracking, by opening sentence
uv run --with matplotlib plots.py --openers RUN                   the same, per opening sentence and swap count
uv run --with matplotlib plots.py --styles RUN                    blank-line vs plain responses to the same puzzle
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
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, layout="constrained")
    axes = axes.flatten()
    pooled = defaultdict(list)
    for by_step in rows.values():
        for step, step_rows in by_step.items():
            pooled[step] += step_rows
    kinds = [("A newline", "\n", "#eb6834"), ("A blank line", "\n\n", "#e87ba4"), ("Two spaces in a row", "  ", "0.45")]
    for name, text, color in kinds:
        axes[0].plot(*smoothed(pooled, lambda row: float(text in row["completion"])), color=color, label=name, linewidth=2)
    axes[0].set_title("Completions containing")
    by_level = [(newline, "Completions with a newline"), (tokens, "Completion length in tokens"), (accuracy, "Accuracy")]
    for ax, (metric, title) in zip(axes[1:], by_level):
        for color, level in zip(plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), sorted(rows)):
            ax.plot(*smoothed(rows[level], metric), color=color, label=f"{level} swaps")
        ax.set_title(title)
    for ax in axes:
        ax.set_ylim(0)
    for ax in axes[2:]:
        ax.set_xlabel("Training step")
    for ax in axes[[0, 1, 3]]:
        ax.yaxis.set_major_formatter("{x:.0%}")
    for ax in axes[:2]:
        ax.legend(loc="upper right")
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
    fig.suptitle(f"{args['model'].split('/')[-1]}: completion length against the {limit}-token limit")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-token-limit.png", dpi=150)


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PAIR = re.compile(r"\b([A-F]) and ([A-F])\b")
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


def plot_groups(run: str):
    """One observation per GRPO group: the samples drawn for one puzzle at one step."""
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    subsets = [("All groups", lambda group: True), ("Groups with every answer well formed", lambda group: all(r["parsed"] is not None for r in group))]
    metrics = [(newline, "Newline use, rewarded − unrewarded", "{x:+.0%}"), (tokens, "Tokens, rewarded − unrewarded", "{x:+.0f}")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, sharey="row", layout="constrained")
    levels = sorted(rows)
    for offset, color, level in zip(range(len(levels)), plt.cm.viridis([0.0, 0.3, 0.6, 0.85]), levels):
        starts = range(0, max(rows[level]) + 1, BIN)
        xs = [s + BIN / 2 + 2 * (offset - (len(levels) - 1) / 2) for s in starts]
        for column, (title, keep) in zip(axes.T, subsets):
            mixed, share = [], []
            for start in starts:
                groups = defaultdict(list)
                for step in range(start, start + BIN):
                    for row in rows[level].get(step, []):
                        groups[step, row["prompt"]].append(row)
                kept = [g for g in groups.values() if 0 < sum(r["reward"] for r in g) < len(g) and keep(g)]
                mixed.append(kept)
                share.append(len(kept) / len(groups))
            column[0].plot(xs, share, "o-", color=color, label=f"{level} swaps", markersize=4)
            column[0].set_title(f"{title}\nShare with mixed rewards")
            for ax, (metric, name, tick_format) in zip(column[1:], metrics):
                side = lambda group, reward: [metric(r) for r in group if r["reward"] == reward]
                gaps = [[sum(side(g, 1)) / len(side(g, 1)) - sum(side(g, 0)) / len(side(g, 0)) for g in b] for b in mixed]
                enough = [len(g) >= MIN_SAMPLES for g in gaps]
                means = [sum(g) / len(g) if ok else float("nan") for g, ok in zip(gaps, enough)]
                errors = [
                    (sum((x - m) ** 2 for x in g) / (len(g) - 1) / len(g)) ** 0.5 if ok else float("nan")
                    for g, m, ok in zip(gaps, means, enough)
                ]
                ax.errorbar(xs, means, yerr=errors, fmt="o-", color=color, capsize=2, elinewidth=1, markersize=4)
                for x, m, e, g, ok in zip(xs, means, errors, gaps, enough):
                    if ok:
                        ax.annotate(len(g), (x, m + e), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=color)
                ax.axhline(0, color="gray", linestyle=":", linewidth=1)
                ax.set_title(name)
                ax.yaxis.set_major_formatter(tick_format)
    axes[0, 0].yaxis.set_major_formatter("{x:.0%}")
    axes[0, 0].set_ylim(0)
    axes[0, 1].legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-groups.png", dpi=150)


SWAP = re.compile(r"Swap the contents of boxes (\w) and (\w)\.")
CONCLUSION = re.compile(r"Therefore|Thus|So,? after|After all|After these|In conclusion|Finally,|Hence|At the end|After performing|After completing")
PARTS = ["Opening", "Swaps that move the key", "Swaps that leave it", "Conclusion", "Closing tag and answer"]


def puzzle(row: dict) -> tuple[list[tuple[str, str]], list[str], list[bool]]:
    """The swaps of a row's puzzle, the key's box after each, and whether each moved the key."""
    question = row["prompt"].split("User:")[-1]
    key, after, moves = START.search(question).group(1), [], []
    swaps = SWAP.findall(question)
    for a, b in swaps:
        moves.append(key in (a, b))
        key = b if key == a else a if key == b else key
        after.append(key)
    return swaps, after, moves


def swap_mentions(reasoning: str, swaps: list[tuple[str, str]]) -> list[re.Match | None]:
    """Where the reasoning takes up each swap, matched in order by the pair of boxes it names.
    "boxes X and Y" is preferred, since the model often repeats the bare pair inside a step."""
    found, at = [], 0
    for a, b in swaps:
        named = [m for m in PAIR.finditer(reasoning, at) if set(m.groups()) == {a, b}]
        mention = next((m for m in named if reasoning[: m.start()].endswith("boxes ")), named[0] if named else None)
        found.append(mention)
        at = mention.end() if mention else at
    return found


def swap_steps(row: dict) -> list[tuple[str, bool]]:
    """For each swap: whether the last box named in its step is where the key really is, and whether the step
    states that location in more than one sentence. Whitespace is normalized first."""
    swaps, after, _ = puzzle(row)
    reasoning = " ".join(row["completion"].partition("</think>")[0].split())
    mentions = swap_mentions(reasoning, swaps)
    starts = [m.start() for m in mentions if m] + [len(reasoning)]
    steps = []
    for mention, box in zip(mentions, after):
        if not mention:
            steps.append(("skipped", False))
            continue
        sentences = reasoning[mention.end() : next(s for s in starts if s > mention.start())].split(". ")
        named = [re.findall(r"\b[A-F]\b", sentence) for sentence in sentences]
        last = [letters[-1] for letters in named if letters]
        outcome = "unstated" if not last else "right" if last[-1] == box else "wrong"
        restated = sum("key" in sentence and letters[-1:] == last[-1:] for sentence, letters in zip(sentences, named)) > 1
        steps.append((outcome, bool(last) and restated))
    return steps


def tracking(row: dict) -> list[str]:
    return [outcome for outcome, _ in swap_steps(row)]


def right_share(rows: list[dict]) -> float:
    steps = [step for row in rows for step in swap_steps(row)]
    return sum(outcome == "right" for outcome, _ in steps) / len(steps)


def restated_share(rows: list[dict]) -> float:
    steps = [step for row in rows for step in swap_steps(row)]
    return sum(restated for _, restated in steps) / len(steps)


def blank_after_opening(row: dict) -> float:
    rest = row["completion"].strip()[len(opening(row)) :]
    return float("\n\n" in rest[: len(rest) - len(rest.lstrip())])


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
        "possible": (sum(moves), len(moves) - sum(moves)),
        "Opening": count(reasoning[: found[0]] if found else reasoning),
        "Conclusion": count(reasoning[end:]),
        "Closing tag and answer": count(tag + rest),
    }


def opening(row: dict) -> str:
    return re.split(r"(?<=[.:!?])\s", row["completion"].strip(), maxsplit=1)[0]


def plot_tracking(run: str):
    """Observational: the opening is whichever sentence the model chose, not one we prefilled."""
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width, enough = 5, 30
    bins = defaultdict(list)
    for by_step in rows.values():
        for step, step_rows in by_step.items():
            bins[step // width * width] += step_rows
    counts = defaultdict(int)
    for row in sum(bins.values(), []):
        counts[opening(row)] += 1
    groups = [("All responses", None)] + [(name, name) for name in sorted(counts, key=counts.get, reverse=True)[:3]]
    panels = [
        ("Responses with a blank line", lambda rs: sum("\n\n" in r["completion"] for r in rs) / len(rs)),
        ("Swaps followed by the right key location", lambda rs: share(rs, "right")),
        ("Swaps skipped or left without a location", lambda rs: share(rs, "skipped") + share(rs, "unstated")),
    ]

    def share(rs, outcome):
        outcomes = [o for r in rs for o in tracking(r)]
        return outcomes.count(outcome) / len(outcomes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True, layout="constrained")
    for (label, name), color in zip(groups, ["black"] + PALETTE):
        kept = {start: [r for r in rs if name in (None, opening(r))] for start, rs in sorted(bins.items())}
        kept = {start: rs for start, rs in kept.items() if len(rs) >= enough}
        for ax, (_, metric) in zip(axes, panels):
            ax.plot([s + width / 2 for s in kept], [metric(rs) for rs in kept.values()], "o-", color=color, label=label, markersize=4,
                    linewidth=2.5 if name is None else 1.5)
    for ax, (title, _) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.set_ylim(0, 1.03)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=10, title="Opening sentence")
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
    fig, (covered, per_swap, parts) = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    for side, key, color in [(0, "moving", PALETTE[1]), (1, "staying", PALETTE[2])]:
        label = PARTS[1 + side]
        covered.plot(steps, [sum(len(t[key]) for t in traces[s]) / sum(t["possible"][side] for t in traces[s]) for s in steps], "o-", color=color, label=label)
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
    covered.set_title("Swaps addressed in the reasoning")
    covered.yaxis.set_major_formatter("{x:.0%}")
    covered.set_ylim(0, 1.05)
    covered.legend(loc="lower right")
    per_swap.set_title("Tokens per swap addressed")
    per_swap.set_ylim(0)
    parts.set_title("Tokens by part of the response")
    parts.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    for ax in (covered, per_swap, parts):
        ax.set_xlabel("Training step")
    fig.suptitle(f"{args['model'].split('/')[-1]}: correct {level}-swap traces on {len(solved)} held-out puzzles")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-traces.png", dpi=150)


def plot_openers(run: str):
    """Observational: each line follows the responses that happened to start with that sentence."""
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width, enough = 10, 10
    counts = defaultdict(int)
    for by_step in rows.values():
        for step, step_rows in by_step.items():
            if 10 <= step < 70:
                for row in step_rows:
                    counts[opening(row)] += 1
    openers = sorted(counts, key=counts.get, reverse=True)[:4]
    panels = [
        ("Opening followed\nby a blank line", lambda rs: sum(map(blank_after_opening, rs)) / len(rs), "{x:.0%}"),
        ("Swaps followed by the\nright key location", right_share, "{x:.0%}"),
        ("Swaps whose step\nrestates the location", restated_share, "{x:.0%}"),
        ("Responses\nper 10 steps", len, "{x:.0f}"),
    ]
    levels = sorted(rows)
    fig, axes = plt.subplots(len(panels), len(levels), figsize=(4 * len(levels), 10), sharex=True, sharey="row", layout="constrained")
    for column, level in zip(axes.T, levels):
        for name, color in zip(openers, PALETTE):
            bins = defaultdict(list)
            for step, step_rows in rows[level].items():
                bins[step // width * width] += [r for r in step_rows if opening(r) == name]
            bins = dict(sorted(bins.items()))
            for ax, (_, metric, _) in zip(column, panels):
                kept = {start: rs for start, rs in bins.items() if metric is len or len(rs) >= enough}
                ax.plot([s + width / 2 for s in kept], [metric(rs) for rs in kept.values()], "o-", color=color, label=name, markersize=3)
        column[0].set_title(f"{level} swaps")
        column[-1].set_xlabel("Training step")
    for ax, (title, _, tick_format) in zip(axes[:, 0], panels):
        ax.set_ylabel(title)
        ax.yaxis.set_major_formatter(tick_format)
        ax.set_ylim(0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=10, title="Opening sentence")
    fig.suptitle(model)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-openers.png", dpi=150)


def plot_styles(run: str):
    """Blank-line minus plain responses within a rollout group that has both, one observation per group."""
    rows = load(run)["train"]
    model = json.loads(Path("runs", run, "args.json").read_text())["model"].split("/")[-1]
    width = 10
    groups = defaultdict(list)
    for by_step in rows.values():
        for step, step_rows in by_step.items():
            for row in step_rows:
                groups[step, row["prompt"]].append(row)
    subsets = [("All responses", lambda r: True), ("Rewarded responses only", lambda r: r["reward"] == 1)]
    metrics = [("Swaps followed by the right key location", right_share), ("Swaps whose step restates the location", restated_share)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey="row", layout="constrained")
    for column, (subset, keep) in zip(axes.T, subsets):
        for ax, (title, metric) in zip(column, metrics):
            gaps = defaultdict(list)
            for (step, _), group in groups.items():
                blank = [r for r in group if keep(r) and "\n\n" in r["completion"]]
                plain = [r for r in group if keep(r) and "\n\n" not in r["completion"]]
                if blank and plain:
                    gaps[step // width * width].append(metric(blank) - metric(plain))
            gaps = {start: g for start, g in sorted(gaps.items()) if len(g) >= MIN_SAMPLES}
            xs = [start + width / 2 for start in gaps]
            means = [sum(g) / len(g) for g in gaps.values()]
            errors = [(sum((x - m) ** 2 for x in g) / (len(g) - 1) / len(g)) ** 0.5 for g, m in zip(gaps.values(), means)]
            ax.errorbar(xs, means, yerr=errors, fmt="o-", color=PALETTE[0], capsize=3)
            for x, m, e, g in zip(xs, means, errors, gaps.values()):
                ax.annotate(len(g), (x, m + e), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"{subset}\n{title}" if ax is column[0] else title)
            ax.yaxis.set_major_formatter("{x:+.0%}")
        column[-1].set_xlabel("Training step")
    fig.suptitle(f"{model}: blank-line minus plain responses to the same puzzle")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-styles.png", dpi=150)


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
        "--blank-lines": plot_blank_lines, "--groups": plot_groups, "--traces": plot_traces, "--tracking": plot_tracking,
        "--openers": plot_openers, "--styles": plot_styles,
    }
    if sys.argv[1] in OTHER:
        OTHER[sys.argv[1]](sys.argv[2])
        sys.exit()
    runs = [arg for arg in sys.argv[1:] if arg != "--start-box"]
    for run in runs:
        plot(run, start_box=len(runs) < len(sys.argv) - 1)
