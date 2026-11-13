"""Search for correlations between features of the reasoning that hold up against training trends.

Scan A: within-group association. Among responses to the same problem at the same step, do responses with feature X
differ in Y? Conditioning on the group removes trends over training.
Scan B: lagged changes. In non-overlapping 10-step bins, does a change in X go with a change in Y in the same bin
or the next one? Compared against a null where X is circularly shifted in time, and required to hold in both runs.
"""

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from interactive import FEATURES
from plots import group_advantages, load, swap_steps
from traces import step_counts

RUNS = ["swaps-7b-tags-v2", "swaps-7b-tags-long"]
WIDTH = 10
BINARY = {
    "newline": FEATURES["a newline"],
    "blank line": FEATURES["a blank line"],
    "standalone statement": FEATURES["a standalone location statement"],
    "every swap written": FEATURES["every swap written out"],
    "numbered list": FEATURES["a numbered list"],
    "Step N: headers": FEATURES['"Step N:" headers'],
    "opens Let's": FEATURES['the opening "Let\'s"'],
    "opens To": FEATURES['the opening "To"'],
}
# Pairs linked by definition rather than by anything the model learned.
TAUTOLOGIES = {
    frozenset(p)
    for p in [
        ("newline", "blank line"), ("cut off", "correct"), ("cut off", "length"), ("wrong swap list", "correct"),
        ("every swap written", "wrong swap list"), ("location right", "correct"), ("cut off", "every swap written"),
    ]
}


def features(row: dict, limit: int) -> dict:
    steps = swap_steps(row)
    left_out, added = step_counts(row)
    out = {name: float(bool(f(row))) for name, f in BINARY.items()}
    return out | {
        "length": float(row["tokens"]),
        "correct": float(row["reward"]),
        "cut off": float(row["tokens"] >= limit - 2),
        "location right": sum(o == "right" for o, _ in steps) / len(steps) if steps else None,
        "standalone share": sum(s for _, s in steps) / len(steps) if steps else None,
        "wrong swap list": float(bool(left_out or added)),
    }


def table(run: str) -> list[dict]:
    """One row per training response with its features, cached in runs/<run>/features.jsonl."""
    cache = Path("runs", run, "features.jsonl")
    if cache.exists():
        return [json.loads(line) for line in cache.read_text().splitlines()]
    limit = json.loads(Path("runs", run, "args.json").read_text())["max_tokens"]
    rows = []
    for level, by_step in load(run)["train"].items():
        for step, step_rows in by_step.items():
            for row, advantage, _ in group_advantages(step_rows):
                row["_advantage"] = advantage
            for row in step_rows:
                rows.append({"step": step, "level": level, "prompt": row["prompt"], "advantage": row.get("_advantage")} | features(row, limit))
    cache.write_text("\n".join(map(json.dumps, rows)))
    return rows


NAMES = list(BINARY) + ["cut off", "wrong swap list", "correct", "length", "location right", "standalone share"]
BINARY_NAMES = list(BINARY) + ["cut off", "wrong swap list", "correct"]


def scan_within(rows: list[dict]) -> dict:
    """For binary X and any Y: mean over groups (same prompt and step, X varies) of mean Y with X minus without.
    Length is divided by the group's mean length, so it is a relative difference."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["step"], r["prompt"])].append(r)
    out = {}
    for x in BINARY_NAMES:
        for y in NAMES:
            if x == y:
                continue
            diffs = []
            for g in groups.values():
                yes = [r[y] for r in g if r[x] and r[y] is not None]
                no = [r[y] for r in g if not r[x] and r[y] is not None]
                if yes and no:
                    scale = st.mean(r["length"] for r in g) if y == "length" else 1
                    diffs.append((st.mean(yes) - st.mean(no)) / scale)
            if len(diffs) >= 30:
                mean, se = st.mean(diffs), st.stdev(diffs) / len(diffs) ** 0.5
                out[(x, y)] = {"effect": mean, "z": mean / se if se else 0.0, "groups": len(diffs)}
    return out


def binned(rows: list[dict]) -> dict:
    """(level -> bin start -> feature -> mean), plus pooled advantage gaps per bin (level 0)."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        b = r["step"] // WIDTH * WIDTH
        for name in NAMES:
            if r[name] is not None:
                cells[r["level"]][b][name].append(r[name])
        cells[r["level"]][b]["mixed groups"].append(float(r["advantage"] is not None))
    series = {level: {b: {k: st.mean(v) for k, v in feats.items()} for b, feats in sorted(bins.items())} for level, bins in cells.items()}
    gaps = defaultdict(dict)
    for x in BINARY_NAMES:
        if x == "correct":
            continue
        by_bin = defaultdict(lambda: ([], []))
        for r in rows:
            if r["advantage"] is not None:
                by_bin[r["step"] // WIDTH * WIDTH][0 if r[x] else 1].append(r["advantage"])
        for b, (yes, no) in by_bin.items():
            gaps[b][f"advantage gap: {x}"] = st.mean(yes) - st.mean(no) if len(yes) >= 5 and len(no) >= 5 else None
    longer = defaultdict(lambda: ([], []))
    groups = defaultdict(list)
    for r in rows:
        if r["advantage"] is not None:
            groups[(r["step"], r["prompt"])].append(r)
    for (step, _), g in groups.items():
        mean = st.mean(r["length"] for r in g)
        for r in g:
            longer[step // WIDTH * WIDTH][0 if r["length"] > mean else 1].append(r["advantage"])
    for b, (yes, no) in longer.items():
        gaps[b]["advantage gap: length"] = st.mean(yes) - st.mean(no) if len(yes) >= 5 and len(no) >= 5 else None
    return {"levels": series, "gaps": dict(sorted(gaps.items()))}


def standardized_deltas(values: list[float | None]) -> list[float | None]:
    deltas = [b - a if a is not None and b is not None else None for a, b in zip(values, values[1:])]
    known = [d for d in deltas if d is not None]
    if len(known) < 4 or st.pstdev(known) == 0:
        return [None] * len(deltas)
    mean, sd = st.mean(known), st.pstdev(known)
    return [(d - mean) / sd if d is not None else None for d in deltas]


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 8:
        return None
    xs, ys = zip(*pairs)
    if st.pstdev(xs) == 0 or st.pstdev(ys) == 0:
        return None
    return st.correlation(xs, ys)


def lagged_pairs(cells: list[tuple[list, list, list]], lag: int, shifts: list[int] | None = None) -> list[tuple[float, float]]:
    """(ΔX at bin t, ΔY at bin t + lag) over all cells; Y bins where the share is saturated are skipped."""
    pairs = []
    for i, (dx, dy, usable) in enumerate(cells):
        if shifts:
            dx = dx[shifts[i] :] + dx[: shifts[i]]
        for t in range(len(dx) - lag):
            if dx[t] is not None and dy[t + lag] is not None and usable[t + lag]:
                pairs.append((dx[t], dy[t + lag]))
    return pairs


def scan_changes(series: dict, names: list[str], targets: list[str], rng: random.Random, trials: int = 300) -> dict:
    """series: cell -> list of per-bin dicts. For each ordered (X, Y) and lag 0 or 1: pooled r of standardized changes,
    a circular-shift null p-value, and the smallest r after leaving out any one observation."""
    out = {}
    cells = list(series.values())
    for x in names:
        for y in targets:
            if x == y:
                continue
            prepared = []
            for bins in cells:
                xs, ys = [b.get(x) for b in bins], [b.get(y) for b in bins]
                shares = y not in ("length",) and not y.startswith("advantage gap")
                usable = [ys[t + 1] is not None and (not shares or 0.02 < ys[t] < 0.98) for t in range(len(ys) - 1)]
                prepared.append((standardized_deltas(xs), standardized_deltas(ys), usable))
            for lag in (0, 1):
                pairs = lagged_pairs(prepared, lag)
                r = pearson(pairs)
                if r is None:
                    continue
                null = []
                for _ in range(trials):
                    shifts = [rng.randrange(1, max(2, len(p[0]))) for p in prepared]
                    rn = pearson(lagged_pairs(prepared, lag, shifts))
                    if rn is not None:
                        null.append(abs(rn))
                p = (1 + sum(v >= abs(r) for v in null)) / (1 + len(null))
                loo = [pearson(pairs[:i] + pairs[i + 1 :]) for i in range(len(pairs))]
                weakest = min(loo, key=lambda v: abs(v) if v is not None and (v > 0) == (r > 0) else -1) if loo else None
                spearman = pearson(list(zip(rank([a for a, _ in pairs]), rank([b for _, b in pairs]))))
                out[(x, y, lag)] = {"r": r, "p": p, "weakest": weakest, "spearman": spearman, "n": len(pairs)}
    return out


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for position, i in enumerate(order):
        ranks[i] = float(position)
    return ranks


def drop_timing(series_by_run: dict, names: list[str]) -> dict:
    """For each run x level cell where newline use falls by over 30 points, find the 10-step bin of steepest fall and
    score each feature's change in that bin (and the one before) against its typical change in that cell."""
    scores = defaultdict(list)
    for run, levels in series_by_run.items():
        for level, bins in levels.items():
            newline = [b.get("newline") for b in bins]
            if max(newline) - newline[-1] < 0.3:
                continue
            deltas = [b - a for a, b in zip(newline, newline[1:])]
            drop = min(range(len(deltas)), key=deltas.__getitem__)
            for x in names:
                if x in ("newline", "blank line"):
                    continue
                d = standardized_deltas([b.get(x) for b in bins])
                for offset, label in [(0, "same bin"), (-1, "bin before")]:
                    t = drop + offset
                    if 0 <= t < len(d) and d[t] is not None:
                        scores[(x, label)].append({"run": run, "level": level, "bin": t * WIDTH, "z": d[t]})
    return scores


if __name__ == "__main__" and "--plots" not in sys.argv:
    rng = random.Random(0)
    runs = sys.argv[1:] or RUNS
    tables = {run: table(run) for run in runs}
    within = {run: scan_within(rows) for run, rows in tables.items()}
    bins = {run: binned(rows) for run, rows in tables.items()}
    level_series = {run: {lvl: list(b["levels"][lvl].values()) for lvl in b["levels"]} for run, b in bins.items()}
    pooled_series = {}
    for run, b in bins.items():
        levels = b["levels"]
        starts = sorted({s for l in levels for s in levels[l]})
        pooled = []
        for s in starts:
            shares = {k: st.mean(levels[l][s][k] for l in levels if s in levels[l] and levels[l][s].get(k) is not None) for k in NAMES + ["mixed groups"]}
            pooled.append(shares | b["gaps"].get(s, {}))
        pooled_series[run] = pooled
    level_names = NAMES + ["mixed groups"]
    changes = {run: scan_changes(level_series[run], level_names, level_names, rng) for run in runs}
    gap_names = sorted({k for b in bins.values() for g in b["gaps"].values() for k in g})
    gap_changes = {run: scan_changes({0: pooled_series[run]}, gap_names, NAMES, rng) for run in runs}
    timing = drop_timing(level_series, level_names)
    out = Path("figures/correlations.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "within": {run: {f"{x}|{y}": v for (x, y), v in w.items()} for run, w in within.items()},
        "changes": {run: {f"{x}|{y}|{lag}": v for (x, y, lag), v in c.items()} for run, c in changes.items()},
        "gap_changes": {run: {f"{x}|{y}|{lag}": v for (x, y, lag), v in c.items()} for run, c in gap_changes.items()},
        "timing": {f"{x}|{label}": v for (x, label), v in timing.items()},
        "series": level_series, "pooled": pooled_series,
        "tautologies": [sorted(p) for p in TAUTOLOGIES],
    }))


RUN_LABELS = {"swaps-7b-tags-v2": "768-token limit", "swaps-7b-tags-long": "1,280-token limit", "swaps-7b-tags-512": "512-token limit (first 60 steps)"}


def plot_report(path: str = "figures/correlations.json"):
    import matplotlib.pyplot as plt

    from plots import PALETTE

    d = json.loads(Path(path).read_text())
    held_out = {"swaps-7b-tags-512": {f"{x}|{y}": v for (x, y), v in scan_within(table("swaps-7b-tags-512")).items()}}
    within = d["within"] | held_out
    runs = list(within)

    # Within-group differences in length and correctness.
    xs = ["newline", "numbered list", "opens Let's", "standalone statement", "every swap written"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, layout="constrained")
    for ax, (y, title, scale) in zip(axes, [("length", "Length relative to the group's mean", 1), ("correct", "Share correct", 1)]):
        for k, run in enumerate(runs):
            values = [within[run].get(f"{x}|{y}") for x in xs]
            pos = [i + (k - 1) * 0.26 for i in range(len(xs))]
            ax.barh(pos, [v["effect"] * scale for v in values], 0.24, xerr=[2 * abs(v["effect"] / v["z"]) * scale if v["z"] else 0 for v in values],
                    color=PALETTE[k], label=RUN_LABELS[run], error_kw={"elinewidth": 1, "capsize": 2})
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.xaxis.set_major_formatter("{x:+.0%}")
    axes[0].set_yticks(range(len(xs)), [x[0].upper() + x[1:] for x in xs])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=10)
    fig.savefig("figures/correlations-within.png", dpi=150)

    # Replication of correlations between changes, across the two finished runs.
    taut = {frozenset(p) for p in d["tautologies"]}
    fig, ax = plt.subplots(figsize=(6.4, 6), layout="constrained")
    for source, color, label in [("changes", "0.6", "Feature changes"), ("gap_changes", PALETTE[1], "Advantage gap, then feature change")]:
        a, b = d[source][RUNS[0]], d[source][RUNS[1]]
        keys = [k for k in a if k in b and frozenset(k.split("|")[:2]) not in taut]
        ax.scatter([a[k]["r"] for k in keys], [b[k]["r"] for k in keys], s=12, color=color, alpha=0.7, label=label)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.plot([-1, 1], [-1, 1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("r, 768-token limit")
    ax.set_ylabel("r, 1,280-token limit")
    ax.set_title("Correlations between changes, both runs")
    ax.legend(loc="upper left", fontsize=10)
    fig.savefig("figures/correlations-changes.png", dpi=150)

    # Openings and newlines over training.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, run in zip(axes, RUNS):
        pooled = d["pooled"][run]
        steps = [i * WIDTH + WIDTH / 2 for i in range(len(pooled))]
        for name, color in [("newline", "#eb6834"), ("opens Let's", PALETTE[0]), ("opens To", PALETTE[2])]:
            ax.plot(steps, [p[name] for p in pooled], color=color, linewidth=2, label={"newline": "Completions with a newline"}.get(name, name.replace("opens", "Opens with")))
        ax.set_title(RUN_LABELS[run])
        ax.set_xlabel("Training step")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter("{x:.0%}")
    axes[0].legend(loc="center right", fontsize=10)
    fig.savefig("figures/correlations-openings.png", dpi=150)


if __name__ == "__main__" and "--plots" in sys.argv:
    plot_report()
