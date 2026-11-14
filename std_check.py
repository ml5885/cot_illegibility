"""Does GRPO's division of advantages by the group's reward std change what training pushes on?

For every training step, the advantage-weighted sum of a response feature (a line break in the reasoning; completion
tokens), with advantages as trained (divided by the group std) and without the division. Points on a line through
the origin mean the std only rescales the push.

uv run --no-project --with matplotlib python std_check.py [RUN]   (default swaps-7b-tags-v2); writes figures/<RUN>-std-check.png
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import plots  # noqa: F401  (shared figure style: serif fonts)

FEATURES = {
    "Line break in the reasoning": lambda r: float("\n" in r["completion"].partition("</think>")[0]),
    "Completion tokens": lambda r: r["tokens"],
}

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "swaps-7b-tags-v2"
    pushes = defaultdict(list)
    for path in sorted(Path("runs", run, "rollouts").glob("train_*.jsonl")):
        groups = defaultdict(list)
        for line in path.open():
            row = json.loads(line)
            groups[row["prompt"]].append(row)
        raw, scaled = defaultdict(float), defaultdict(float)
        for group in groups.values():
            rewards = [r["reward"] for r in group]
            mean, std = statistics.mean(rewards), statistics.stdev(rewards)
            if not std:
                continue
            for r in group:
                for name, feature in FEATURES.items():
                    raw[name] += (r["reward"] - mean) * feature(r)
                    scaled[name] += (r["reward"] - mean) / (std + 1e-4) * feature(r)
        for name in FEATURES:
            pushes[name].append((raw[name], scaled[name]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, (name, points) in zip(axes, pushes.items()):
        ax.scatter(*zip(*points), s=12, color="#2a78d6", alpha=0.6)
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.axvline(0, color="0.7", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Push without the std division")
        ax.set_ylabel("Push as trained")
    fig.savefig(f"figures/{run}-std-check.png", dpi=150)
