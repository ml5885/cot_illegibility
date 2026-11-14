"""Base models answering with reasoning, with and without "Think step by step." (runs/baseline-swaps-cot-*, -cotstep-*).

uv run --no-project --with matplotlib --with datasets python cot_baselines.py

A sample counts as correct when its last answer tag names exactly one box and it is the right one. The strict reward is
not used here: Qwen3's tokenizer treats </think> as a special token, which vLLM drops from the text, so every Qwen3
sample fails the tag check.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from generalization import puzzle_se
from zero_shot import COLORS

PANELS = {"baseline-swaps-cot-": "Question as is", "baseline-swaps-cotstep-": 'With "Think step by step."'}
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BOX = re.compile(r"\b([A-F])\b")


def correct(completion: str, answer: str) -> bool:
    tags = ANSWER.findall(completion)
    return bool(tags) and set(BOX.findall(tags[-1])) == {answer}


def results(run: Path) -> dict[int, list[dict]]:
    by = defaultdict(list)
    for line in (run / "rollouts" / "baseline_00000.jsonl").open():
        r = json.loads(line)
        by[r["level"]].append({"prompt": r["prompt"], "reward": correct(r["completion"], r["answer"])})
    return dict(sorted(by.items()))


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained", sharey=True)
    for ax, (prefix, title) in zip(axes, PANELS.items()):
        for model in COLORS:
            by = results(Path("runs", prefix + model))
            levels = list(by)
            acc = [sum(r["reward"] for r in by[l]) / len(by[l]) for l in levels]
            ax.errorbar(levels, acc, yerr=[puzzle_se(by[l]) for l in levels], fmt="o-", color=COLORS[model], linewidth=2,
                        markersize=5, capsize=2, label=model)
            print(f"{title:28s} {model:16s} " + "  ".join(f"{l}: {a:.2f}" for l, a in zip(levels, acc)))
        ax.axhline(1 / 6, color="gray", linestyle=":", linewidth=1, label="Chance (1/6)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, [str(l) for l in levels])
        ax.set_xlabel("Swaps")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_title(title)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="outside right center", fontsize=10)
    fig.suptitle("Performance with reasoning (bars: one standard error over puzzles)")
    fig.savefig("figures/cot-baselines.png", dpi=150)
