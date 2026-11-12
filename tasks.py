import random
import re
from collections import defaultdict
from dataclasses import dataclass

from datasets import Dataset, load_dataset

ANSWER = re.compile(r"Answer:\s*([^\n]+)")
BOXED = re.compile(r"\\boxed\{(.*)\}")
STOP = "\nProblem:"
MATH_EVAL = 500
PROMPT = (
    'Solve the following problem. Reason step by step, then end with a line of the form "Answer: X".\n\n'
    "Problem: {}\nReasoning:"
)


@dataclass
class Problem:
    prompt: str
    answer: str
    level: int = 0


def swap_problem(rng: random.Random, n_swaps: int, n_boxes: int = 6) -> Problem:
    boxes = [chr(ord("A") + i) for i in range(n_boxes)]
    key = rng.choice(boxes)
    lines = [f"There are {n_boxes} boxes labeled {boxes[0]} to {boxes[-1]}. A key is in box {key}."]
    for _ in range(n_swaps):
        a, b = rng.sample(boxes, 2)
        lines.append(f"Swap the contents of boxes {a} and {b}.")
        key = b if key == a else a if key == b else key
    lines.append("Which box contains the key?")
    return Problem(PROMPT.format(" ".join(lines)), key, n_swaps)


def swap_problems(seed: int, n: int, n_swaps: int) -> list[Problem]:
    rng = random.Random(seed)
    return [swap_problem(rng, n_swaps) for _ in range(n)]


def math_rows() -> list[tuple[str, str]]:
    """Unique (problem, answer) pairs; problems that appear with conflicting answers are dropped."""
    answers = defaultdict(set)
    for row in load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train"):
        answers[row["prompt"].strip()].add(row["ground_truth"])
    return [(prompt, a.pop()) for prompt, a in answers.items() if len(a) == 1]


def math_problems(split: str, seed: int, n: int) -> list[Problem]:
    """MATH_EVAL problems from a fixed shuffle are held out for eval; train draws from the rest."""
    rows = math_rows()
    random.Random(0).shuffle(rows)
    if split == "eval":
        rows = rows[:MATH_EVAL]
    else:
        rows = rows[MATH_EVAL:]
        random.Random(seed).shuffle(rows)
    return [Problem(PROMPT.format(prompt), answer) for prompt, answer in rows[:n]]


def normalize(answer: str) -> str:
    answer = answer.strip().strip("$.* ")
    if boxed := BOXED.search(answer):
        answer = boxed.group(1)
    return answer.replace(",", "").replace(" ", "")


def parse_answer(completion: str) -> str | None:
    """The answer counts only if it is on the final line, so nothing after it can be rewarded."""
    last_line = completion.removesuffix(STOP).rstrip().rsplit("\n", 1)[-1]
    match = ANSWER.search(last_line)
    return normalize(match.group(1)) if match else None


def reward(completion: str, answer: str) -> float:
    return float(parse_answer(completion) == normalize(answer))


def dataset(task: str, split: str, seed: int, n: int, levels: list[int]) -> Dataset:
    if task == "swaps":
        seed += 10_000 * (split == "eval")
        problems = [p for i, k in enumerate(levels) for p in swap_problems(seed + i, n // len(levels), k)]
    else:
        problems = math_problems(split, seed, n)
    return Dataset.from_list([vars(p) for p in problems])
