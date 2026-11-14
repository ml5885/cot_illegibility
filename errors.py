"""Error statistics for each run: markdown tables (printed) and a figure (figures/<run>-errors.png).

Each wrong answer gets the first cause that applies: cut off at the token limit; no answer that parses; a wrong
list of swaps (one left out, added, or both); the right list but a wrong stated key location; the right list and
locations but a final answer that differs from the last stated location; otherwise unclear (a location not stated).

uv run --no-project --with matplotlib python errors.py [RUN ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from plots import load, tracking
from traces import list_steps

CAUSES = ["Wrong list", "Tracking error", "No answer", "Cut off", "Answer differs", "Unclear"]
KINDS = [("left out only", "Swaps left out only"), ("added only", "Steps added only"), ("both", "Both")]
EXTRAS = [
    ("after the last swap", "After the last swap"),
    ("a pair from the problem", "Repeats a pair from the problem"),
    ("a pair not in the problem", "A pair not in the problem"),
]
PHASES = {
    "swaps-7b-tags-v2": [(0, 49), (50, 99), (100, 199)],
    "swaps-7b-tags-long": [(0, 49), (50, 99), (100, 199)],
    "swaps-7b-tags-512": [(0, 49), (50, 99), (100, 199)],
    "swaps-7b-tags-256": [(0, 49), (50, 99), (100, 199)],
    "swaps-7b-tags-2and4": [(0, 39), (40, 100)],
}
MIN_BAR = 10


def cause(row: dict, limit: int) -> str:
    if row["parsed"] is None:
        return "Cut off" if row["tokens"] >= limit - 2 else "No answer"
    _, mentions, extra = list_steps(row)
    if extra or any(m is None for m in mentions):
        return "Wrong list"
    outcomes = tracking(row)
    if "wrong" in outcomes:
        return "Tracking error"
    return "Answer differs" if outcomes and outcomes[-1] == "right" else "Unclear"


def list_detail(rows: list[dict]) -> dict:
    """Among responses with a wrong list: kinds, how many swaps are left out and where, and what the extra steps are.
    Over all responses: how often the second of two back-to-back identical swaps is left out."""
    kinds, left_counts, quarters = Counter(), Counter(), Counter()
    merged = repeats = later_copy = skips = chance = 0.0
    extra_kinds = Counter()
    for row in rows:
        swaps, mentions, extra = list_steps(row)
        for i in range(1, len(swaps)):
            if set(swaps[i]) == set(swaps[i - 1]) and mentions[i - 1] is not None:
                repeats += 1
                merged += mentions[i] is None
        left = [i for i, m in enumerate(mentions) if m is None]
        if not left and not extra:
            continue
        kinds["left out only" if left and not extra else "added only" if extra and not left else "both"] += 1
        if left:
            left_counts[min(len(left), 3)] += 1
        for i in left:
            quarters[min(3, i * 4 // len(swaps))] += 1
        written = [i for i, m in enumerate(mentions) if m is not None]
        for a, b in zip(written, written[1:]):
            if b > a + 2:
                skips += 1
                later_copy += set(swaps[b - 1]) == set(swaps[a])
                chance += sum(set(swaps[k]) == set(swaps[a]) for k in range(a + 1, len(swaps))) / (len(swaps) - a - 1)
        last = max((m.start() for m in mentions if m), default=0)
        pairs = {frozenset(s) for s in swaps}
        for m in extra:
            extra_kinds["after the last swap" if m.start() > last else "a pair from the problem" if frozenset(m.groups()) in pairs else "a pair not in the problem"] += 1
    return {
        "n": sum(kinds.values()), "kinds": kinds, "left_counts": left_counts, "quarters": quarters, "extra": extra_kinds,
        "merged": merged / repeats if repeats else None, "skips": skips, "later_copy": later_copy, "chance": chance,
    }


def stats(run: str) -> list[dict]:
    """One entry per swap count and training phase."""
    data = load(run)
    limit = json.loads(Path("runs", run, "args.json").read_text())["max_tokens"]
    held_out = {level for level in data["eval"] if level not in data["train"]}
    out = []
    for level in sorted(set(data["train"]) | held_out):
        split = "eval" if level in held_out else "train"
        for lo, hi in PHASES[run]:
            rows = [r for s, rs in data[split][level].items() if lo <= s <= hi for r in rs]
            if not rows:
                continue
            wrong = [r for r in rows if not r["reward"]]
            right = [r for r in rows if r["reward"]]
            out.append({
                "level": level, "held_out": split == "eval", "phase": (lo, hi), "responses": len(rows), "right": len(right),
                "causes": Counter(cause(r, limit) for r in wrong),
                "lucky": sum(cause(r | {"reward": 0}, limit) not in ("Answer differs", "Unclear") for r in right),
                **list_detail(rows),
            })
    return out


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def tables(groups: list[dict]) -> str:
    cause_rows, list_rows, extra_rows = [], [], []
    for g in groups:
        lo, hi = g["phase"]
        label = f"{g['level']}{' (held-out)' if g['held_out'] else ''} | {lo}–{hi}"
        wrong = g["responses"] - g["right"]
        cause_rows.append(
            f"| {label} | {g['responses']} | {pct(g['right'] / g['responses'])} | {wrong} | "
            + " | ".join(pct(g["causes"][c] / wrong) if wrong else "–" for c in CAUSES)
            + f" | {pct(g['lucky'] / g['right']) if g['right'] else '–'} |"
        )
        n = g["n"]
        if n >= MIN_BAR:
            k, lc, q, e = g["kinds"], g["left_counts"], g["quarters"], g["extra"]
            ql = sum(q.values())
            skips = g["skips"]
            list_rows.append(
                f"| {label} | {n} ({pct(n / g['responses'])}) | {pct(k['left out only'] / n)} | {pct(k['added only'] / n)} | {pct(k['both'] / n)} | "
                f"{lc[1]} / {lc[2]} / {lc[3]} | "
                + (" / ".join(pct(q[i] / ql) for i in range(4)) if ql else "–")
                + f" | {pct(g['merged'])} | " + (f"{pct(g['later_copy'] / skips)} of {skips:.0f} ({pct(g['chance'] / skips)})" if skips else "–") + " |"
            )
            ne = sum(e.values())
            if ne:
                extra_rows.append(f"| {label} | {ne} | " + " | ".join(pct(e[key] / ne) for key, _ in EXTRAS) + " |")
    out = ["Wrong answers by cause (share of wrong answers), and correct answers reached with a wrong list or a wrong stated location:\n"]
    out.append("| Swaps \\| steps | Responses | Accuracy | Wrong | " + " | ".join(CAUSES) + " | Correct with wrong reasoning |")
    out.append("|---|" + "---|" * (len(CAUSES) + 4))
    out += cause_rows
    if list_rows:
        out.append("\nResponses with a wrong list of swaps:\n")
        out.append("| Swaps \\| steps | Responses (share) | Left out only | Added only | Both | Left out 1 / 2 / 3+ | Left-out swaps by quarter of the list | Back-to-back repeats merged (all responses) | Skips of 2+ swaps landing after a later copy (chance) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        out += list_rows
    if extra_rows:
        out.append("\nExtra steps in those responses:\n")
        out.append("| Swaps \\| steps | Extra steps | " + " | ".join(label for _, label in EXTRAS) + " |")
        out.append("|---|---|" + "---|" * len(EXTRAS))
        out += extra_rows
    return "\n".join(out)


def phase_label(phase: tuple[int, int]) -> str:
    lo, hi = phase
    return f"{lo}+" if hi >= 100 and lo >= 100 else f"{lo}–{hi}"


def model_label(run: str) -> str:
    args = json.loads(Path("runs", run, "args.json").read_text())
    extra = ", trained on 2 and 4 swaps" if args["swaps"] == [2, 4] else ""
    return f"{args['model'].split('/')[-1]}, {args['max_tokens']:,}-token limit{extra}"


def figure(run: str, groups: list[dict]):
    """Heatmaps per swap count and training phase (each column sums to 100% within its block): why wrong answers are
    wrong, what kind of wrong list, what the extra steps are; then how often long skips land right after a later copy
    of the last pair written, against chance."""
    from matplotlib.patches import Rectangle

    phases = PHASES[run]
    levels = sorted({(g["level"], g["held_out"]) for g in groups})
    spacing = len(phases) + 0.35
    col = {(g["level"], g["phase"]): li * spacing + phases.index(g["phase"]) for g in groups for li, (level, _) in enumerate(levels) if level == g["level"]}
    blocks = [
        ("Wrong answers, by cause", [(c, c) for c in CAUSES], lambda g: g["causes"]),
        ("Responses with a wrong list of swaps, by kind", KINDS, lambda g: g["kinds"]),
        ("Extra steps in those responses, by kind", EXTRAS, lambda g: g["extra"]),
    ]
    skips = []
    for level, held_out in levels:
        own = [g for g in groups if g["level"] == level]
        n = sum(g["skips"] for g in own)
        if n >= 20:
            skips.append((f"{level} swaps, {n:.0f} skips", sum(g["later_copy"] for g in own) / n, sum(g["chance"] for g in own) / n))

    # Fixed layout in inches, so every title starts at the same left margin above the row labels.
    width, label_margin, margin, row = 13, 3.3, 0.25, 0.42
    header, title_gap, block_gap = 0.45, 0.4, 0.3
    heights = [(len(rows) + 0.6) * row for _, rows, _ in blocks] + [0.55 * (len(skips) + 0.2)]
    total = 0.7 + header + sum(heights) + len(heights) * (title_gap + block_gap) + 0.3
    fig = plt.figure(figsize=(width, total))
    fig.text(margin / width, 1 - 0.25 / total, f"{model_label(run)}: errors by training step", fontsize=16, va="top")
    titles = [f"{title} (%)" for title, _, _ in blocks] + ["Skips of 2+ swaps that land right after a later copy of the last pair written (all steps)"]
    axes, top = [], 0.7
    for i, (title, height) in enumerate(zip(titles, heights)):
        fig.text(margin / width, 1 - top / total, title, fontsize=14, va="top")
        top += title_gap + header * (i == 0)
        axes.append(fig.add_axes([label_margin / width, 1 - (top + height) / total, (width - label_margin - margin) / width, height / total]))
        top += height + block_gap

    blues = plt.get_cmap("Blues")
    for i, (ax, (_, rows, counts)) in enumerate(zip(axes, blocks)):
        for g in groups:
            c, x = counts(g), col[g["level"], g["phase"]]
            n = sum(c.values())
            for y, (key, _) in enumerate(rows):
                if n < MIN_BAR:
                    ax.add_patch(Rectangle((x - 0.46, y - 0.44), 0.92, 0.88, facecolor="none", edgecolor="0.85", linewidth=0.8))
                    ax.text(x, y, "–", ha="center", va="center", fontsize=11, color="0.6")
                    continue
                v = c[key] / n
                ax.add_patch(Rectangle((x - 0.46, y - 0.44), 0.92, 0.88, facecolor=blues(0.05 + 0.9 * v), edgecolor="none"))
                ax.text(x, y, f"{100 * v:.0f}", ha="center", va="center", fontsize=11, color="white" if v > 0.5 else "0.15")
            ax.text(x, len(rows) - 0.2, f"{n:,}", ha="center", va="center", fontsize=10, color="0.45")
        ax.set_yticks([*range(len(rows)), len(rows) - 0.2], [label for _, label in rows] + ["Count"])
        ax.get_yticklabels()[-1].set_color("0.45")
        ax.set_ylim(len(rows) + 0.1, -0.5)
        ax.set_xlim(-0.55, max(col.values()) + 0.55)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 0:
            ax.xaxis.tick_top()
            ax.set_xticks(list(col.values()), [phase_label(phase) for _, phase in col], fontsize=10, color="0.35")
            swaps = ax.secondary_xaxis("top")
            swaps.set_xticks([li * spacing + (len(phases) - 1) / 2 for li in range(len(levels))],
                             [f"{level} swaps{' (held-out)' if held_out else ''}" for level, held_out in levels], fontsize=13)
            swaps.tick_params(length=0, pad=22)
            swaps.spines["top"].set_visible(False)
        else:
            ax.set_xticks([])

    ax = axes[-1]
    for y, (_, observed, chance) in enumerate(skips):
        ax.plot([chance, observed], [y, y], color="0.8", linewidth=2.5, zorder=1)
        ax.scatter(chance, y, s=80, facecolor="white", edgecolor="0.4", linewidth=1.5, zorder=2, label="If the landing point were random" if y == 0 else None)
        ax.scatter(observed, y, s=80, color="#1d4f91", zorder=3, label="Observed" if y == 0 else None)
        low, high = sorted([(chance, "0.4"), (observed, "#1d4f91")])
        ax.text(low[0] - 0.012, y, f"{low[0]:.0%}", ha="right", va="center", fontsize=11, color=low[1])
        ax.text(high[0] + 0.012, y, f"{high[0]:.0%}", ha="left", va="center", fontsize=11, color=high[1])
    ax.set_yticks(range(len(skips)), [label for label, _, _ in skips])
    ax.set_ylim(len(skips) - 0.4, -0.6)
    ax.set_xlim(0, 0.5)
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=10)

    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-errors.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for run in sys.argv[1:] or list(PHASES):
        if Path("runs", run, "args.json").exists():
            groups = stats(run)
            figure(run, groups)
            print(f"\n## {run}\n\n{tables(groups)}")
