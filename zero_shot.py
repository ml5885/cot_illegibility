"""Answering with no reasoning: base models against simple rules that read the puzzle without tracking the key.

uv run --no-project --with matplotlib --with datasets python zero_shot.py
Reads runs/baseline-swaps-noreason-*/ (from baseline.py --no-reasoning) and writes figures/zero-shot.png.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import plots  # noqa: F401  (shared figure style: serif fonts)
from rollouts import by_level
from tasks import swap_problems

LEVELS = [1, 2, 4, 8, 16, 32]
# One hue per family: Llama blue, OLMo pink, Qwen in shades of purple (lighter for smaller, Qwen3.5 darker than Qwen3).
COLORS = {
    "Llama-3.1-8B": "#2a78d6",
    "Olmo-3-1025-7B": "#e0569b",
    "Qwen3-4B-Base": "#c3a6e8",
    "Qwen3-8B-Base": "#9a72d6",
    "Qwen3.5-4B-Base": "#7243b8",
    "Qwen3.5-9B-Base": "#3f1d7a",
}
START = re.compile(r"A key is in box ([A-F])")
SWAP = re.compile(r"Swap the contents of boxes ([A-F]) and ([A-F])")


def rules(question: str) -> dict[str, dict[str, float]]:
    """Each rule as a distribution over boxes."""
    start = START.search(question).group(1)
    swaps = SWAP.findall(question)
    counts = Counter(b for pair in swaps for b in pair)
    most = [b for b in counts if counts[b] == max(counts.values())]
    last = swaps[-1]
    return {
        "The starting box": {start: 1.0},
        "A box in the last swap": {b: 0.5 for b in last},
        "The second box of the last swap": {last[1]: 1.0},
        "The most-swapped box": {b: 1 / len(most) for b in most},
    }


def rule_accuracy(n: int = 2000) -> dict[str, dict[int, float]]:
    out = defaultdict(dict)
    for i, level in enumerate(LEVELS):
        problems = swap_problems(i, n, level)
        for name in rules(problems[0].question):
            out[name][level] = sum(rules(p.question)[name].get(p.answer, 0) for p in problems) / n
    return out


def model_results() -> dict[str, dict]:
    results = {}
    for run in sorted(Path("runs").glob("baseline-swaps-noreason-*")):
        if not (run / "letters.jsonl").exists():
            continue
        rows = [json.loads(line) for line in (run / "rollouts" / "baseline_00000.jsonl").open()]
        letters = [json.loads(line) for line in (run / "letters.jsonl").open()]
        sampled = by_level(rows)
        per = defaultdict(lambda: defaultdict(list))
        for row in letters:
            probs, rule = row["probs"], rules(row["question"])
            stats = per[row["level"]]
            mass = sum(probs.values())
            stats["P(correct)"].append(probs.get(row["answer"], 0))
            stats["P(correct), among box letters"].append(probs.get(row["answer"], 0) / mass if mass else 1 / 6)
            stats["Most likely box correct"].append(max("ABCDEF", key=lambda b: probs.get(b, 0)) == row["answer"])
            stats["P(any box letter)"].append(mass)
            stats["P(starting box)"].append(probs.get(START.search(row["question"]).group(1), 0))
            stats["P(a box in the last swap)"].append(sum(probs.get(b, 0) for b in rule["A box in the last swap"]))
        results[run.name.removeprefix("baseline-swaps-noreason-")] = {
            level: {"Sampled accuracy": sampled[f"accuracy/level_{level}"]} | {k: sum(v) / len(v) for k, v in stats.items()}
            for level, stats in sorted(per.items())
        }
    return results


def table(rows: dict[str, dict[int, float]], title: str) -> str:
    levels = sorted({level for r in rows.values() for level in r})
    lines = [f"| {title} | " + " | ".join(f"{level}" for level in levels) + " |", "|---|" + "---|" * len(levels)]
    lines += [f"| {name} | " + " | ".join(f"{r[level]:.2f}" if level in r else "–" for level in levels) + " |" for name, r in rows.items()]
    return "\n".join(lines)


def figure(results: dict[str, dict], baselines: dict[str, dict[int, float]]):
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for model, by_lvl in results.items():
        color = COLORS[model]
        levels = sorted(by_lvl)
        ax.plot(levels, [by_lvl[level]["P(correct), among box letters"] for level in levels], "o-", color=color, label=model, linewidth=2)
    best = {level: max(r[level] for r in baselines.values()) for level in LEVELS}
    ax.plot(LEVELS, [best[level] for level in LEVELS], color="0.4", linestyle="--", linewidth=1, label="Best simple rule")
    ax.axhline(1 / 6, color="gray", linestyle=":", linewidth=1, label="Chance (1/6)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LEVELS, [str(level) for level in LEVELS])
    ax.set_xlabel("Swaps")
    ax.set_ylabel("Probability on the correct box\n(among the six box letters)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Base models answering with no reasoning")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/zero-shot.png", dpi=150)


if __name__ == "__main__":
    random.seed(0)
    baselines = rule_accuracy()
    print("Simple rules (accuracy):\n\n" + table(baselines, "Rule \\| swaps"))
    results = model_results()
    for model, by_lvl in results.items():
        measures = {m: {level: by_lvl[level][m] for level in by_lvl} for m in next(iter(by_lvl.values()))}
        print(f"\n{model}:\n\n" + table(measures, "Measure \\| swaps"))
    if results:
        figure(results, baselines)
