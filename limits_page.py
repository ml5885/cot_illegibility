"""token_limits.html: the text and figures of token_limits.md, with example responses side by side by swap count.

uv run --no-project --with markdown python limits_page.py

The examples at each chosen step are the median-length training response at each swap count. A step with no puzzle
at some swap count borrows the nearest step that has one, and says so.
"""

import html
import json
import re
from pathlib import Path

import markdown

from tasks import TAGS
from traces import puzzle

LEVELS = [2, 4, 8, 16]
RUNS = [
    ("64 tokens", "swaps-7b-tags-64", 64, [
        (0, "The base model."),
        (6, "Starts to reason, then stops and answers."),
        (9, "One sentence."),
        (40, "An empty think block, as in every response from step 30 to 170."),
        (190, "The answer letter inside the think block too. Training accuracy at 4 and 8 swaps jumps around the switch "
              "(steps 170 to 178), from 63% to 78% and from 26% to 45%."),
    ]),
    ("128 tokens", "swaps-7b-tags-128", 128, [
        (20, "Describes tracking without doing it on long puzzles."),
        (42, "Short guesses at 16 swaps fit the limit."),
        (56, "One clause per swap; every 16-swap response is cut off from here on."),
        (190, "The final style."),
    ]),
    ("256 tokens", "swaps-7b-tags-256", 256, [
        (10, "Numbered steps run together."),
        (40, "Newlines and full sentences."),
        (190, "One short line per swap."),
    ]),
    ("512 tokens", "swaps-7b-tags-512", 512, [
        (15, "No newlines yet."),
        (50, "Near peak length, bold headers and bullets."),
        (190, "The same layout, shorter."),
    ]),
    ("768 tokens", "swaps-7b-tags-v2", 768, [
        (30, "Newlines at their peak."),
        (100, "The same numbered steps on one line."),
        (190, "Shorter, still on one line."),
    ]),
    ("1,280 tokens", "swaps-7b-tags-long", 1280, [
        (30, "\"Step n:\" paragraphs."),
        (120, "Numbered swaps with indented explanations."),
        (190, "Newlines dropped on short puzzles, and a recap added."),
    ]),
]

CSS = """
body { margin: 0; background: #fff; color: #222; font: 17px/1.6 "Iowan Old Style", Charter, Georgia, serif; }
main { max-width: 1240px; margin: 0 auto; padding: 40px 24px 80px; }
h1 { font-size: 2.2em; margin: 0.4em 0 0.6em; }
h2 { font-size: 1.6em; margin: 1.6em 0 0.5em; }
h3 { font-size: 1.25em; margin: 1.6em 0 0.4em; }
p, ul { max-width: 900px; }
img { display: block; max-width: min(100%, 1000px); margin: 1em 0; }
code { font: 0.85em ui-monospace, Menlo, monospace; background: #f2f2f2; padding: 1px 5px; border-radius: 3px; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border-bottom: 1px solid #ddd; padding: 4px 14px 4px 0; text-align: left; }
.step { margin: 1.4em 0 0.4em; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.cell { border: 1px solid #e3e3e3; border-radius: 4px; padding: 8px 10px; font-size: 13px; line-height: 1.45; }
.head { font-weight: bold; }
.right { color: #1a7f4b; }
.wrong { color: #c0392b; }
.puzzle { color: #777; margin: 2px 0 6px; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 420px; overflow-y: auto; }
@media (max-width: 800px) { .grid { grid-template-columns: minmax(0, 1fr); } }
"""


def rows_at(run: str, step: int, level: int) -> tuple[int, list[dict]]:
    """Training rows at the step, or the nearest step with that swap count."""
    for offset in range(0, 10):
        for s in (step + offset, step - offset):
            path = Path("runs", run, "rollouts", f"train_{s:05d}.jsonl")
            if path.exists() and (rows := [r for r in map(json.loads, path.open()) if r["level"] == level]):
                return s, rows
    raise LookupError(run, step, level)


def cell(run: str, limit: int, step: int, level: int) -> str:
    found, rows = rows_at(run, step, level)
    row = sorted(rows, key=lambda r: r["tokens"])[len(rows) // 2]
    swaps, _, _ = puzzle(row)
    start = re.search(r"A key is in box (\w)", row["prompt"]).group(1)
    answer = TAGS.parse_answer(row["completion"])
    if row["reward"]:
        verdict = f'<span class="right">right</span>, answer {row["answer"]}'
    else:
        said = "cut off" if row["tokens"] >= limit - 2 else f"said {answer}" if answer else "no answer"
        verdict = f'<span class="wrong">wrong</span> ({said}), answer {row["answer"]}'
    note = f", from step {found}" if found != step else ""
    return (
        f'<div class="cell"><div class="head">{level} swaps: {verdict}</div>'
        f'<div class="puzzle">Key in {start}; swaps {", ".join(a + b for a, b in swaps)}. {row["tokens"]} tokens{note}.</div>'
        f'<div class="text">{html.escape("<think>" + row["completion"].rstrip())}</div></div>'
    )


def examples() -> str:
    parts = [
        "<h2>Over training</h2>",
        "<p>For each run, a few points in training. At each, the median-length training response at each swap count, in full, "
        "with the puzzle (where the key starts and the swaps, in order) and whether the answer is right.</p>",
    ]
    for title, run, limit, steps in RUNS:
        parts.append(f"<h3>{title}</h3>")
        for step, note in steps:
            parts.append(f'<p class="step"><b>Step {step}.</b> {html.escape(note)}</p>')
            parts.append('<div class="grid">' + "".join(cell(run, limit, step, level) for level in LEVELS) + "</div>")
    return "\n".join(parts)


if __name__ == "__main__":
    text = Path("token_limits.md").read_text()
    # The examples go just before the caveats.
    before, caveats = text[: text.index("## Caveats")], text[text.index("## Caveats") :]
    render = lambda md: markdown.markdown(md, extensions=["tables"])
    body = render(before) + examples() + render(caveats)
    Path("token_limits.html").write_text(
        f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Token limits</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n"
    )
