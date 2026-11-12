import random
import re
from dataclasses import dataclass

from datasets import Dataset, load_dataset

ANSWER = re.compile(r"Answer:\s*([^\n]+)")
BOXED = re.compile(r"\\boxed\{(.*)\}")
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


def math_problems(seed: int, n: int) -> list[Problem]:
    rows = load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train").shuffle(seed=seed)
    return [Problem(PROMPT.format(row["prompt"]), row["ground_truth"]) for row in rows.select(range(n))]


def normalize(answer: str) -> str:
    answer = answer.strip().strip("$.* ")
    if boxed := BOXED.search(answer):
        answer = boxed.group(1)
    return answer.replace(",", "").replace(" ", "")


def parse_answer(completion: str) -> str | None:
    match = ANSWER.search(completion)
    return normalize(match.group(1)) if match else None


def reward(completion: str, answer: str) -> float:
    return float(parse_answer(completion) == normalize(answer))


def dataset(task: str, seed: int, n: int, levels: list[int]) -> Dataset:
    if task == "swaps":
        problems = [p for i, k in enumerate(levels) for p in swap_problems(seed + i, n // len(levels), k)]
    else:
        problems = math_problems(seed, n)
    return Dataset.from_list([vars(p) for p in problems])
