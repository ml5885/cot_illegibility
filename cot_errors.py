"""What base models get wrong when they reason (runs/baseline-swaps-cot-*), and the first wrong step of tracking errors.

uv run --no-project --with matplotlib --with datasets python cot_errors.py

Each wrong answer gets the first cause that applies: cut off at the token limit; no single box in the answer tag;
fewer than half of the swaps taken up in the reasoning; a wrong list of swaps (one left out or added); a step that
puts the key in the wrong box; otherwise other. For tracking errors, the first wrong step is classed by what the
model did to the key there. Writes figures/cot-errors.png.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from cot_baselines import ANSWER, BOX, correct
from plots import PALETTE
from traces import START, list_steps, puzzle, stated_boxes, step_texts, swap_mentions, swap_steps
from zero_shot import COLORS

LIMIT = 1280
CAUSES = [
    ("Cut off", "Cut off", PALETTE[3]),
    ("No answer", "No single box in the answer tag", PALETTE[4]),
    ("Skips", "Takes up fewer than half the swaps", PALETTE[2]),
    ("Wrong list", "Wrong list of swaps", PALETTE[0]),
    ("Tracking error", "Tracking error", PALETTE[1]),
    ("Other", "Other", "0.75"),
]
# Shades of the tracking-error colour, since these split that cause.
STEPS = [
    ("unrelated", "Moves the key on a swap that doesn't involve its box", "#b8441a"),
    ("missed", "Leaves the key in place when its box is swapped", PALETTE[1]),
    ("elsewhere", "Moves the key to the wrong box", "#f5b08c"),
    ("other", "Other", "0.75"),
]


def reasoning_only(row: dict) -> dict:
    """The row with its completion cut at the answer tag and closed with </think>, which Qwen3 loses in decoding."""
    completion = row["completion"]
    end = completion.find("<answer>")
    return row | {"completion": (completion if end < 0 else completion[:end]) + "</think>"}


def cause(row: dict) -> str:
    tags = ANSWER.findall(row["completion"])
    if len(set(BOX.findall(tags[-1])) if tags else ()) != 1:
        return "Cut off" if row["tokens"] >= LIMIT - 2 else "No answer"
    row = reasoning_only(row)
    swaps, mentions, extra = list_steps(row)
    taken = sum(m is not None for m in mentions)
    if taken < len(swaps) / 2:
        return "Skips"
    if extra or taken < len(swaps):
        return "Wrong list"
    return "Tracking error" if any(outcome == "wrong" for outcome, _ in swap_steps(row)) else "Other"


def first_wrong_step(row: dict) -> str:
    row = reasoning_only(row)
    swaps, after, moves = puzzle(row)
    reasoning = " ".join(row["completion"].partition("</think>")[0].split())
    before = START.search(row["prompt"].split("User:")[-1]).group(1)
    for swap, text, box, moved in zip(swaps, step_texts(reasoning, swap_mentions(reasoning, swaps)), after, moves):
        if text is not None and (stated := stated_boxes(text)) and stated[-1] != box:
            if moved:
                return "missed" if stated[-1] == before else "elsewhere"
            return "unrelated" if stated[-1] in swap else "other"
        before = box
    return "other"


def shares(ax, rows: dict[str, list[str]], parts: list[tuple[str, str, str]], title: str):
    for i, (model, values) in enumerate(rows.items()):
        counts, left = Counter(values), 0.0
        for key, label, color in parts:
            width = counts[key] / len(values)
            ax.barh(i, width, left=left, color=color, edgecolor="white", label=label if i == 0 else None)
            left += width
    ax.set_yticks(range(len(rows)), list(rows))
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10)


if __name__ == "__main__":
    causes, steps = {}, {}
    for model in COLORS:
        rows = [json.loads(line) for line in Path("runs", f"baseline-swaps-cot-{model}", "rollouts", "baseline_00000.jsonl").open()]
        wrong = [row for row in rows if not correct(row["completion"], row["answer"])]
        causes[model] = [cause(row) for row in wrong]
        steps[model] = [first_wrong_step(row) for row, c in zip(wrong, causes[model]) if c == "Tracking error"]
        start = [row for row in wrong if (tags := ANSWER.findall(row["completion"])) and len(set(BOX.findall(tags[-1]))) == 1]
        print(f"{model:16s} wrong {len(wrong):4d}  " + "  ".join(f"{k} {v / len(wrong):.2f}" for k, v in Counter(causes[model]).most_common())
              + f"\n{'':16s} first wrong step: " + "  ".join(f"{k} {v / len(steps[model]):.2f}" for k, v in Counter(steps[model]).most_common())
              + f"\n{'':16s} wrong answers naming the starting box: "
              + f"{sum(BOX.findall(ANSWER.findall(r['completion'])[-1])[0] == START.search(r['prompt']).group(1) for r in start) / len(start):.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), layout="constrained", sharey=True)
    shares(axes[0], causes, CAUSES, "Why wrong answers are wrong (2 to 16 swaps)")
    shares(axes[1], steps, STEPS, "Tracking errors: what the first wrong step does")
    axes[0].invert_yaxis()  # shared with the right panel
    fig.suptitle("Base models with reasoning: errors")
    fig.savefig("figures/cot-errors.png", dpi=150)
