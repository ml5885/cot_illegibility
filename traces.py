"""Read a swap-puzzle response step by step: which swaps it takes up and where it says the key is after each."""

import re

START = re.compile(r"A key is in box (\w)\.")
SWAP = re.compile(r"Swap the contents of boxes (\w) and (\w)\.")
PAIR = re.compile(r"\b([A-F])\s*(?:and|with|&)\s*(?:box\s+)?([A-F])\b")
# A list marker ("3.", "Step 3:") with no sentence break between it and the pair: the pair heads a step.
HEADER = re.compile(r"\d+\s*[.:)]\s*[^.:\n]*$")
# Boxes named only to say the key is not there, or as a group, are not statements of where the key is.
NOT_THERE = re.compile(
    r"(?:\bnot|n't|\bnever|\bno key is)\s+(?:\w+\s+){0,2}?(?:in|to|into)\s+(?:either\s+)?(?:box(?:es)?\s+)?[A-F]\b"
    r"(?:\s*(?:,|or|and|nor)\s*(?:box\s+)?[A-F]\b)*"
)
GROUP = re.compile(r"\b[A-F]\s*(?:and|or|nor|with|&)\s*(?:box\s+)?[A-F]\b|\((?:\s*[A-F]\s*,)+[^)]*\)")
PLACED = re.compile(r"\b(?:in|to|into|at)\s+(?:box\s+)?([A-F])\b")
LETTER = re.compile(r"\b[A-F]\b")


def puzzle(row: dict) -> tuple[list[tuple[str, str]], list[str], list[bool]]:
    """The swaps of a row's puzzle, the key's box after each, and whether each moved the key."""
    question = row["prompt"].split("User:")[-1]
    key, after, moves = START.search(question).group(1), [], []
    swaps = SWAP.findall(question)
    for a, b in swaps:
        moves.append(key in (a, b))
        key = b if key == a else a if key == b else key
        after.append(key)
    return swaps, after, moves


def align(swaps: list[tuple[str, str]], pool: list[re.Match]) -> list[re.Match | None]:
    """The longest in-order matching of swaps to named pairs; among equally long ones, the latest."""
    length = [[0] * (len(pool) + 1) for _ in range(len(swaps) + 1)]
    for i, swap in enumerate(swaps, 1):
        for j, mention in enumerate(pool, 1):
            same = set(mention.groups()) == set(swap)
            length[i][j] = max(length[i - 1][j], length[i][j - 1], length[i - 1][j - 1] + same)
    found, i, j = [None] * len(swaps), len(swaps), len(pool)
    while i and j:
        if set(pool[j - 1].groups()) == set(swaps[i - 1]) and length[i][j] == length[i - 1][j - 1] + 1:
            found[i - 1] = pool[j - 1]
            i, j = i - 1, j - 1
        elif length[i][j] == length[i][j - 1]:
            j -= 1
        else:
            i -= 1
    return found


def step_texts(reasoning: str, mentions: list[re.Match | None]) -> list[str | None]:
    starts = [m.start() for m in mentions if m] + [len(reasoning)]
    return [m and reasoning[m.end() : next(s for s in starts if s > m.start())] for m in mentions]


def step_pairs(reasoning: str, expected: int) -> list[re.Match]:
    """The pairs of boxes that look like the start of a step: those heading a numbered item if there are enough
    of them, else those worded "boxes X and Y", else every pair named."""
    named = list(PAIR.finditer(reasoning))
    heads = [m for m in named if HEADER.search(reasoning[max(0, m.start() - 80) : m.start()])]
    worded = [m for m in named if reasoning[: m.start()].endswith("boxes ")]
    return next((ms for ms in (heads, worded) if 2 * len(ms) >= expected), named)


def swap_mentions(reasoning: str, swaps: list[tuple[str, str]]) -> list[re.Match | None]:
    """Where the reasoning takes up each swap, matched in order by the pair of boxes it names.
    The model repeats pairs inside a step and sometimes lists the swaps before working through them. So pairs
    that head a numbered step are used when there are enough of them, then pairs worded "boxes X and Y", then
    any pair; and of the earliest and the latest matching, the one whose steps more often place the key wins."""
    pool = step_pairs(reasoning, len(swaps))
    earliest = align(swaps[::-1], pool[::-1])[::-1]
    latest = align(swaps, pool)
    placed = lambda mentions: sum(bool(text and stated_boxes(text)) for text in step_texts(reasoning, mentions))
    return latest if placed(latest) > placed(earliest) else earliest


def stated_boxes(step: str) -> list[str]:
    """The box each sentence of a step puts the key in, in order. Sentences about the key count first, and
    "in/to box X" counts before a bare letter."""
    sentences = [GROUP.sub("", NOT_THERE.sub("", sentence)) for sentence in step.split(". ")]
    about_key = [s for s in sentences if "key" in s.lower()]
    for group in (about_key, sentences):
        for pattern in (PLACED, LETTER):
            boxes = [found[-1][-1] for s in group if (found := pattern.findall(s))]
            if boxes:
                return boxes
    return []


def swap_steps(row: dict) -> list[tuple[str, bool]]:
    """For each swap: whether its step ends with the key in the right box ("right", "wrong"), names no box for
    the key ("unstated"), or the swap is never taken up as given ("skipped"); and whether, after working out
    where the key goes, the step states the result in a sentence of its own ("So now, the key is in box A.").
    Whitespace is normalized first."""
    swaps, after, _ = puzzle(row)
    reasoning = " ".join(row["completion"].partition("</think>")[0].split())
    steps = []
    for text, box in zip(step_texts(reasoning, swap_mentions(reasoning, swaps)), after):
        if text is None:
            steps.append(("skipped", False))
            continue
        boxes = stated_boxes(text)
        outcome = "unstated" if not boxes else "right" if boxes[-1] == box else "wrong"
        steps.append((outcome, boxes.count(boxes[-1]) > 1 if boxes else False))
    return steps


def newline_after_swaps(row: dict) -> list[bool]:
    """For each swap the reasoning writes out: whether its step, as written, contains a line break."""
    reasoning = row["completion"].partition("</think>")[0]
    texts = step_texts(reasoning, swap_mentions(reasoning, puzzle(row)[0]))
    return ["\n" in text for text in texts if text is not None]


def step_counts(row: dict) -> tuple[int, int]:
    """How many of the puzzle's swaps the reasoning leaves out, and how many steps it writes that match no swap
    of the puzzle at that point (a swap repeated, invented or miscopied). Only numbered or "boxes X and Y" steps
    are counted as extra, from the first matched step on, so a pair repeated inside a step is not one."""
    swaps = puzzle(row)[0]
    reasoning = " ".join(row["completion"].partition("</think>")[0].split())
    matched = [m for m in swap_mentions(reasoning, swaps) if m]
    pool = step_pairs(reasoning, len(swaps))
    if not matched or len(pool) == len(PAIR.findall(reasoning)) and len(pool) > 2 * len(swaps):
        return len(swaps) - len(matched), 0
    spans = {m.span() for m in matched}
    numbered = all(HEADER.search(reasoning[max(0, m.start() - 80) : m.start()]) for m in pool)
    extra = [
        m for before, m in zip(pool, pool[1:])
        if m.start() > matched[0].start() and m.span() not in spans and (numbered or set(m.groups()) != set(before.groups()))
    ]
    return len(swaps) - len(matched), len(extra)
