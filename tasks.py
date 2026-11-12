import random
import re
from collections import defaultdict
from dataclasses import dataclass

from datasets import Dataset, load_dataset

ANSWER = re.compile(r"</think>\s*<answer>(.*?)</answer>", re.DOTALL)
BOXED = re.compile(r"\\boxed\{(.*)\}")
STOP = "</answer>"
MATH_EVAL = 500
# Modeled on the DeepSeek R1-Zero template, with the opening tag prefilled.
PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: {} Assistant: <think>"
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
    """The completion continues an open <think>, so it must close it and then give a closed <answer>."""
    match = ANSWER.search(completion)
    return normalize(match.group(1)) if match else None


def reasoning(completion: str) -> str:
    return completion.split("</think>")[0]


def reward(completion: str, answer: str) -> float:
    return float(parse_answer(completion) == normalize(answer))


def dataset(task: str, split: str, seed: int, n: int, levels: list[int]) -> Dataset:
    if task == "swaps":
        seed += 10_000 * (split == "eval")
        problems = [p for i, k in enumerate(levels) for p in swap_problems(seed + i, n // len(levels), k)]
    else:
        problems = math_problems(split, seed, n)
    return Dataset.from_list([vars(p) for p in problems])
