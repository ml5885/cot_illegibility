import random
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from datasets import Dataset, load_dataset

BOXED = re.compile(r"\\boxed\{(.*)\}")
TAGGED_ANSWER = re.compile(r"</think>\s*<answer>(.*?)</answer>", re.DOTALL)
PLAIN_ANSWER = re.compile(r"Answer:\s*(.+)")
MATH_EVAL = 500


def split_tags(completion: str) -> tuple[str, str | None]:
    """The completion continues an open <think>, so it must close it and then give a closed <answer>."""
    match = TAGGED_ANSWER.search(completion)
    return (completion[: match.start()], match.group(1)) if match else (completion, None)


def split_plain(completion: str) -> tuple[str, str | None]:
    """The answer counts only if it is on the final line, so nothing after it can be rewarded."""
    *reasoning, last_line = completion.removesuffix(PLAIN.stop).rstrip().rsplit("\n", 1)
    match = PLAIN_ANSWER.search(last_line)
    return ("".join(reasoning), match.group(1)) if match else (completion, None)


@dataclass(frozen=True)
class Format:
    template: str
    stop: str
    skip: str  # appended to the prompt, it forces an answer with no reasoning
    split: Callable[[str], tuple[str, str | None]]  # completion -> (reasoning, raw answer)

    def reasoning(self, completion: str) -> str:
        return self.split(completion)[0].strip()

    def parse_answer(self, completion: str) -> str | None:
        answer = self.split(completion)[1]
        return None if answer is None else normalize(answer)

    def reward(self, completion: str, answer: str) -> float:
        return float(self.parse_answer(completion) == normalize(answer))


# Modeled on the DeepSeek R1-Zero template, with the opening tag prefilled.
TAGS = Format(
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "User: {} Assistant: <think>",
    "</answer>",
    " </think> <answer>",
    split_tags,
)
PLAIN = Format(
    'Solve the following problem. Reason step by step, then end with a line of the form "Answer: X".\n\n'
    "Problem: {}\nReasoning:",
    "\nProblem:",
    "\nAnswer:",
    split_plain,
)
FORMATS = {"tags": TAGS, "plain": PLAIN}


@dataclass
class Problem:
    question: str
    answer: str
    level: int = 0


def swap_problem(rng: random.Random, n_swaps: int, n_boxes: int = 6) -> Problem:
    """The key ends in a uniformly random box, so naming the starting box is no better than guessing."""
    boxes = [chr(ord("A") + i) for i in range(n_boxes)]
    start = rng.choice(boxes)
    moves = n_swaps > 0 and rng.random() < 1 - 1 / n_boxes
    while True:
        key, swaps = start, [rng.sample(boxes, 2) for _ in range(n_swaps)]
        for a, b in swaps:
            key = b if key == a else a if key == b else key
        if (key != start) == moves:
            break
    lines = [f"There are {n_boxes} boxes labeled {boxes[0]} to {boxes[-1]}. A key is in box {start}."]
    lines += [f"Swap the contents of boxes {a} and {b}." for a, b in swaps]
    lines.append("Which box contains the key? Answer with the letter of the box.")
    return Problem(" ".join(lines), key, n_swaps)


def swap_problems(seed: int, n: int, n_swaps: int) -> list[Problem]:
    rng = random.Random(seed)
    return [swap_problem(rng, n_swaps) for _ in range(n)]


def math_rows() -> list[tuple[str, str]]:
    """Unique (problem, answer) pairs; problems that appear with conflicting answers are dropped."""
    answers = defaultdict(set)
    for row in load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train"):
        answers[row["prompt"].strip()].add(row["ground_truth"])
    return [(question, a.pop()) for question, a in answers.items() if len(a) == 1]


def math_problems(split: str, seed: int, n: int) -> list[Problem]:
    """MATH_EVAL problems from a fixed shuffle are held out for eval; train draws from the rest."""
    rows = math_rows()
    random.Random(0).shuffle(rows)
    if split == "eval":
        rows = rows[:MATH_EVAL]
    else:
        rows = rows[MATH_EVAL:]
        random.Random(seed).shuffle(rows)
    return [Problem(question, answer) for question, answer in rows[:n]]


def normalize(answer: str) -> str:
    answer = answer.strip().strip("$.* ")
    if boxed := BOXED.search(answer):
        answer = boxed.group(1)
    return answer.replace(",", "").replace(" ", "")


def dataset(task: str, split: str, seed: int, n: int, levels: list[int], fmt: Format) -> Dataset:
    """Columns: prompt, answer, level."""
    if task == "swaps":
        seed += 10_000 * (split == "eval")
        problems = [p for i, k in enumerate(levels) for p in swap_problems(seed + i, n // len(levels), k)]
    else:
        problems = math_problems(split, seed, n)
    return Dataset.from_list(
        [{"prompt": fmt.template.format(p.question), "answer": p.answer, "level": p.level} for p in problems]
    )
