"""Length generalization: final checkpoints evaluated on 16 to 32 swaps (runs/gen-*, from baseline.py with --adapter).

uv run --no-project --with matplotlib python generalization.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from errors import cause

RUNS = {
    "swaps-7b-tags-256": "256-token limit",
    "swaps-7b-tags-512": "512-token limit",
    "swaps-7b-tags-v2": "768-token limit",
    "swaps-7b-tags-long": "1,280-token limit",
    "swaps-7b-tags-2and4": "Trained on 2 and 4 swaps",
}
COLORS = dict(zip(RUNS, [plt.get_cmap("magma")(x) for x in (0.82, 0.62, 0.42, 0.15)] + ["#2a78d6"]))
LIMIT = 3072


def rows(run: str) -> dict[int, list[dict]]:
    by = defaultdict(list)
    for line in Path("runs", f"gen-{run}", "rollouts", "baseline_00000.jsonl").open():
        r = json.loads(line)
        by[r["level"]].append(r)
    return dict(sorted(by.items()))


def puzzle_se(rs: list[dict]) -> float:
    """Standard error of accuracy over puzzles (samples of one puzzle are averaged first)."""
    by = defaultdict(list)
    for r in rs:
        by[r["prompt"]].append(r["reward"])
    means = [sum(v) / len(v) for v in by.values()]
    m = sum(means) / len(means)
    return (sum((x - m) ** 2 for x in means) / (len(means) - 1) / len(means)) ** 0.5


if __name__ == "__main__":
    data = {run: rows(run) for run in RUNS if Path("runs", f"gen-{run}").exists()}
    # Two subfigures so each legend sits under the panels it belongs to.
    fig = plt.figure(figsize=(19, 6.2), layout="constrained")
    runs_fig, errors_fig = fig.subfigures(1, 2, width_ratios=[2, 0.85])
    axes = list(runs_fig.subplots(1, 2)) + [errors_fig.subplots()]
    for run, by in data.items():
        levels = list(by)
        acc = [sum(r["reward"] for r in by[l]) / len(by[l]) for l in levels]
        se = [puzzle_se(by[l]) for l in levels]
        axes[0].errorbar(levels, acc, yerr=se, fmt="o-", color=COLORS[run], linewidth=2, markersize=5, capsize=2, label=RUNS[run])
        axes[1].plot(levels, [sum(r["tokens"] for r in by[l]) / len(by[l]) for l in levels], "o-", color=COLORS[run], linewidth=2,
                     markersize=5, label=RUNS[run])
    axes[0].axhline(1 / 6, color="gray", linestyle=":", linewidth=1, label="Chance (1/6)")
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter("{x:.0%}")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[1].set_ylabel("Completion tokens")
    axes[1].set_ylim(0)
    axes[1].set_title("Length")
    for ax in axes[:2]:
        ax.set_xlabel("Swaps")
        ax.set_xticks([16, 18, 20, 22, 24, 28, 32])
        ax.grid(axis="y", color="0.9", linewidth=0.8)

    # Why wrong answers are wrong, pooled over 17 to 32 swaps.
    ax = axes[2]
    # Greens and greys, so no colour here is also a run's colour in the other panels.
    parts = [("Wrong list", "Wrong list of swaps", "#2f8f6b"), ("Tracking error", "Tracking error", "#9bc53d"),
             ("Answer differs", "Answer differs from last stated box", "#4d4d4d"), ("Cut off", "Cut off", "#999999")]
    for i, (run, by) in enumerate(data.items()):
        wrong = [r for level, rs in by.items() if level > 16 for r in rs if not r["reward"]]
        counts = defaultdict(int)
        for r in wrong:
            counts[cause(r, LIMIT)] += 1
        floor = 0.0
        for key, label, color in parts + [("other", "Other", "0.86")]:
            n = counts[key] if key != "other" else len(wrong) - sum(counts[k] for k, _, _ in parts)
            ax.barh(i, n / len(wrong), left=floor, color=color, edgecolor="white", label=label if i == 0 else None)
            floor += n / len(wrong)
    ax.set_yticks(range(len(data)), [RUNS[run] for run in data])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.set_title("Errors")
    errors_fig.legend(loc="outside lower center", ncol=2, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    runs_fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="outside lower center", ncol=3, fontsize=11)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/generalization.png", dpi=150)

    for run, by in data.items():
        print(f"\n{RUNS[run]}")
        for level, rs in by.items():
            wrong = [r for r in rs if not r["reward"]]
            causes = defaultdict(int)
            for r in wrong:
                causes[cause(r, LIMIT)] += 1
            print(f"  {level:2d} swaps: acc {sum(r['reward'] for r in rs) / len(rs):.2f} ± {puzzle_se(rs):.2f}, "
                  f"tokens {sum(r['tokens'] for r in rs) / len(rs):.0f}, cut off {sum(r['tokens'] >= LIMIT - 2 for r in rs)}, "
                  f"wrong causes {dict(causes)}")
