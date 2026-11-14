"""Error statistics for each run, as markdown tables: wrong answers by cause, and a breakdown of wrong lists of swaps.

Each wrong answer gets the first cause that applies: cut off at the token limit; no answer that parses; a wrong
list of swaps (one left out, added, or both); the right list but a wrong stated key location; the right list and
locations but a final answer that differs from the last stated location; otherwise unclear (a location not stated).
"""

import json
import sys
from collections import Counter
from pathlib import Path

from plots import load, tracking
from traces import list_steps, puzzle

CAUSES = ["Cut off", "No answer", "Wrong list", "Tracking error", "Answer differs", "Unclear"]
PHASES = {
    "swaps-7b-tags-v2": [(0, 49), (50, 99), (100, 199)],
    "swaps-7b-tags-long": [(0, 49), (50, 99), (100, 199)],
    "swaps-7b-tags-512": [(0, 49), (50, 99), (100, 199)],
    "swaps-7b-tags-2and4": [(0, 39), (40, 100)],
}


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
    merged = repeats = jumps = later_copy = skips = chance = 0.0
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
    n = sum(kinds.values())
    return {
        "n": n, "kinds": kinds, "left_counts": left_counts, "quarters": quarters, "extra": extra_kinds,
        "merged": merged / repeats if repeats else None, "skips": skips,
        "later_copy": later_copy / skips if skips else None, "chance": chance / skips if skips else None,
    }


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def tables(run: str) -> str:
    data = load(run)
    limit = json.loads(Path("runs", run, "args.json").read_text())["max_tokens"]
    held_out = {level for level in data["eval"] if level not in data["train"]}
    out = []
    cause_rows, list_rows, extra_rows = [], [], []
    for level in sorted(set(data["train"]) | held_out):
        split = "eval" if level in held_out else "train"
        for lo, hi in PHASES[run]:
            rows = [r for s, rs in data[split][level].items() if lo <= s <= hi for r in rs]
            if not rows:
                continue
            label = f"{level}{' (held-out)' if split == 'eval' else ''} | {lo}–{hi}"
            wrong = [r for r in rows if not r["reward"]]
            right = [r for r in rows if r["reward"]]
            causes = Counter(cause(r, limit) for r in wrong)
            lucky = sum(1 for r in right if cause(r | {"reward": 0}, limit) not in ("Answer differs", "Unclear")) if right else 0
            cause_rows.append(
                f"| {label} | {len(rows)} | {pct(len(right) / len(rows))} | {len(wrong)} | "
                + " | ".join(pct(causes[c] / len(wrong)) if wrong else "–" for c in CAUSES)
                + f" | {pct(lucky / len(right)) if right else '–'} |"
            )
            d = list_detail(rows)
            if d["n"] >= 10:
                k, lc, q, e = d["kinds"], d["left_counts"], d["quarters"], d["extra"]
                ql = sum(q.values())
                list_rows.append(
                    f"| {label} | {d['n']} ({pct(d['n'] / len(rows))}) | {pct(k['left out only'] / d['n'])} | {pct(k['added only'] / d['n'])} | {pct(k['both'] / d['n'])} | "
                    f"{lc[1]} / {lc[2]} / {lc[3]} | "
                    + (" / ".join(pct(q[i] / ql) for i in range(4)) if ql else "–")
                    + f" | {pct(d['merged'])} | " + (f"{pct(d['later_copy'])} of {d['skips']:.0f} ({pct(d['chance'])})" if d["skips"] else "–") + " |"
                )
                ne = sum(e.values())
                if ne:
                    extra_rows.append(
                        f"| {label} | {ne} | {pct(e['after the last swap'] / ne)} | {pct(e['a pair from the problem'] / ne)} | {pct(e['a pair not in the problem'] / ne)} |"
                    )
    out.append("Wrong answers by cause (share of wrong answers), and correct answers reached with a wrong list or a wrong stated location:\n")
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
        out.append("| Swaps \\| steps | Extra steps | After the last swap | Repeats a pair from the problem | Pair not in the problem |")
        out.append("|---|---|---|---|---|")
        out += extra_rows
    return "\n".join(out)


if __name__ == "__main__":
    for run in sys.argv[1:] or list(PHASES):
        if Path("runs", run, "args.json").exists():
            print(f"\n## {run}\n\n{tables(run)}")
