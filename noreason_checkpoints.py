"""Answering without reasoning across RL checkpoints (runs/noreason-<run>-<step>, from baseline.py --no-reasoning --adapter).

uv run --no-project --with matplotlib --with datasets python noreason_checkpoints.py

Step 0 is the base model (runs/baseline-swaps-noreason-Olmo-3-1025-7B). Writes figures/noreason-checkpoints.png.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import plots  # noqa: F401  (shared figure style: serif fonts)

RUNS = {
    "swaps-7b-tags-long": "1,280-token limit",
    "swaps-7b-tags-512": "512-token limit",
    "swaps-7b-tags-256": "256-token limit",
    "swaps-7b-tags-2and4": "Trained on 2 and 4 swaps",
}
BASE = Path("runs", "baseline-swaps-noreason-Olmo-3-1025-7B")


def p_correct(run: Path) -> dict[int, float]:
    """Mean probability on the correct box, among the six box letters, per swap count."""
    per = defaultdict(list)
    for line in (run / "letters.jsonl").open():
        row = json.loads(line)
        mass = sum(row["probs"].values())
        per[row["level"]].append(row["probs"].get(row["answer"], 0) / mass if mass else 1 / 6)
    return {level: sum(v) / len(v) for level, v in sorted(per.items())}


if __name__ == "__main__":
    base = p_correct(BASE)
    levels = list(base)
    colors = dict(zip(levels, plt.cm.viridis([0.0, 0.2, 0.4, 0.6, 0.8, 0.95])))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharey=True)
    for ax, (run, title) in zip(axes.flat, RUNS.items()):
        steps = sorted(int(d.name.rsplit("-", 1)[1]) for d in Path("runs").glob(f"noreason-{run}-*"))
        by_step = {0: base} | {s: p_correct(Path("runs", f"noreason-{run}-{s}")) for s in steps}
        for level in levels:
            ax.plot(list(by_step), [r[level] for r in by_step.values()], "o-", color=colors[level], markersize=4, linewidth=2,
                    label=f"{level} swap" + "s" * (level > 1))
        ax.axhline(1 / 6, color="gray", linestyle=":", linewidth=1, label="Chance (1/6)")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability on the correct box\n(among the six box letters)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    fig.suptitle("OLMo 3 7B checkpoints answering without reasoning")
    fig.savefig("figures/noreason-checkpoints.png", dpi=150)
    for run in RUNS:
        final = max(int(d.name.rsplit("-", 1)[1]) for d in Path("runs").glob(f"noreason-{run}-*"))
        print(run, final, {level: round(p, 2) for level, p in p_correct(Path("runs", f"noreason-{run}-{final}")).items()})
