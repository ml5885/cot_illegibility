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

CAUSES = ["Cut off", "No answer", "Wrong list", "Tracking error", "Answer differs", "Unclear"]
CAUSE_COLORS = ["#eda100", "0.65", "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
KINDS = [("left out only", "Swaps left out only", "#1d4f91"), ("added only", "Steps added only", "#5b93d9"), ("both", "Both", "#b9d3f2")]
EXTRAS = [
    ("after the last swap", "After the last swap", "#4f3596"),
    ("a pair from the problem", "Repeats a pair from the problem", "#9580d6"),
    ("a pair not in the problem", "A pair not in the problem", "#d6ccf1"),
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
                extra_rows.append(f"| {label} | {ne} | " + " | ".join(pct(e[key] / ne) for key, _, _ in EXTRAS) + " |")
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
        out.append("| Swaps \\| steps | Extra steps | " + " | ".join(label for _, label, _ in EXTRAS) + " |")
        out.append("|---|---|" + "---|" * len(EXTRAS))
        out += extra_rows
    return "\n".join(out)


def phase_label(phase: tuple[int, int]) -> str:
    lo, hi = phase
    return f"{lo}+" if hi >= 100 and lo >= 100 else f"{lo}–{hi}"


def figure(run: str, groups: list[dict]):
    """Stacked bars per swap count and training phase: why wrong answers are wrong, what kind of wrong list, what
    the extra steps are; and how often long skips land right after a later copy of the last pair written."""
    phases = PHASES[run]
    levels = sorted({(g["level"], g["held_out"]) for g in groups})
    slot = {(g["level"], g["phase"]): li * (len(phases) + 1) + phases.index(g["phase"]) for g in groups for li, lv in enumerate(levels) if lv[0] == g["level"]}

    def stacked(ax, title, parts, counts):
        for g in groups:
            c = counts(g)
            n = sum(c.values())
            x, floor = slot[g["level"], g["phase"]], 0.0
            if n < MIN_BAR:
                ax.text(x, 0.01, f"{n}", ha="center", va="bottom", fontsize=8, color="0.55")
                continue
            for key, _, color in parts:
                ax.bar(x, c[key] / n, 0.8, bottom=floor, color=color, edgecolor="white", linewidth=1)
                floor += c[key] / n
            ax.text(x, 1.01, f"{n:,}", ha="center", va="bottom", fontsize=8, color="0.3")
        for _, label, color in parts:
            ax.bar(0, 0, color=color, label=label)
        ax.set_title(title, pad=16)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), layout="constrained")
    stacked(axes[0, 0], "Wrong answers, by cause", [(c, c, color) for c, color in zip(CAUSES, CAUSE_COLORS)], lambda g: g["causes"])
    stacked(axes[0, 1], "Responses with a wrong list of swaps, by kind", KINDS, lambda g: g["kinds"])
    stacked(axes[1, 0], "Extra steps in those responses", EXTRAS, lambda g: g["extra"])

    ax = axes[1, 1]
    for li, (level, held_out) in enumerate(levels):
        own = [g for g in groups if g["level"] == level]
        skips = sum(g["skips"] for g in own)
        if skips < 20:
            continue
        x = li * (len(phases) + 1) + (len(phases) - 1) / 2
        ax.bar(x - 0.45, sum(g["later_copy"] for g in own) / skips, 0.8, color="0.25", label="Observed" if not ax.patches else None)
        ax.bar(x + 0.45, sum(g["chance"] for g in own) / skips, 0.8, color="0.75", label="If the landing point were random" if len(ax.patches) == 1 else None)
        ax.text(x, 1.01, f"{skips:.0f} skips", ha="center", va="bottom", fontsize=8, color="0.3")
    ax.set_title("Skips of 2+ swaps that land right after a later copy\nof the last pair written (all steps)", pad=16)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)

    for i, ax in enumerate(axes.flat):
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        if i < 3:
            ax.set_xticks([slot[g["level"], g["phase"]] for g in groups], [phase_label(g["phase"]) for g in groups], fontsize=8)
        else:
            ax.set_xticks([])
        for li, (level, held_out) in enumerate(levels):
            ax.text(li * (len(phases) + 1) + (len(phases) - 1) / 2, -0.07 if i < 3 else -0.03, f"{level} swaps{' (held-out)' if held_out else ''}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top")
        ax.set_xlim(-0.7, len(levels) * (len(phases) + 1) - 1.3)

    args = json.loads(Path("runs", run, "args.json").read_text())
    extra = ", trained on 2 and 4 swaps" if args["swaps"] == [2, 4] else ""
    fig.suptitle(f"{args['model'].split('/')[-1]}, {args['max_tokens']:,}-token limit{extra}: error statistics by training step")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{run}-errors.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for run in sys.argv[1:] or list(PHASES):
        if Path("runs", run, "args.json").exists():
            groups = stats(run)
            figure(run, groups)
            print(f"\n## {run}\n\n{tables(groups)}")
