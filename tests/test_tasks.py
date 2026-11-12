import random
import re

import pytest

from tasks import dataset, normalize, parse_answer, reward, swap_problem, swap_problems

SWAP = re.compile(r"Swap the contents of boxes (\w) and (\w)\.")


def simulate(prompt: str) -> str:
    """Independent solver: track every box's contents, then find the key."""
    contents = {box: box == re.search(r"A key is in box (\w)\.", prompt).group(1) for box in "ABCDEF"}
    for a, b in SWAP.findall(prompt):
        contents[a], contents[b] = contents[b], contents[a]
    return next(box for box, has_key in contents.items() if has_key)


@pytest.mark.parametrize("n_swaps", [0, 1, 2, 8, 40])
def test_swap_answer_matches_independent_simulation(n_swaps):
    for problem in swap_problems(seed=n_swaps, n=200, n_swaps=n_swaps):
        assert len(SWAP.findall(problem.prompt)) == n_swaps
        assert problem.answer == simulate(problem.prompt)
        assert problem.level == n_swaps


def test_swap_prompt_format():
    prompt = swap_problem(random.Random(0), 3).prompt
    assert prompt.count("Problem:") == 1 and prompt.count("Answer:") == 1
    assert "\n\nProblem: There are 6 boxes labeled A to F. A key is in box " in prompt
    assert prompt.endswith(" Which box contains the key?\nReasoning:")
    assert all(a != b for a, b in SWAP.findall(prompt))


def test_swap_problems_deterministic_and_seed_dependent():
    assert swap_problems(1, 20, 8) == swap_problems(1, 20, 8)
    assert swap_problems(1, 20, 8) != swap_problems(2, 20, 8)


def test_swap_answers_roughly_uniform():
    answers = [p.answer for p in swap_problems(0, 6000, 30)]
    assert all(800 < answers.count(box) < 1200 for box in "ABCDEF")


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("A to C.\nAnswer: C", "C"),
        ("Answer: C.", "C"),
        ("Answer:C\nmore text", "C"),
        ("so it is 34.\nAnswer: $34$", "34"),
        ("Answer: \\boxed{-3}", "-3"),
        ("Answer: $\\boxed{-3}$.", "-3"),
        ("Answer: 1,234", "1234"),
        ("Answer: **7**", "7"),
        ("Answer: 5\nAnswer: 6", "5"),
        ("the answer is C", None),
        ("", None),
    ],
)
def test_parse_answer(completion, expected):
    assert parse_answer(completion) == expected


def test_normalize_is_idempotent():
    for raw in [" $\\boxed{12}$. ", "1,000", "C.", "-3"]:
        assert normalize(normalize(raw)) == normalize(raw)


@pytest.mark.parametrize(
    "completion, answer, expected",
    [
        ("Answer: C", "C", 1.0),
        ("Answer: D", "C", 0.0),
        ("Answer: box C", "C", 0.0),
        ("C", "C", 0.0),
        ("Answer: 34", "34", 1.0),
        ("Answer: 340", "34", 0.0),
        ("Answer: -3", "3", 0.0),
    ],
)
def test_reward(completion, answer, expected):
    assert reward(completion, answer) == expected


def test_swap_dataset_splits_levels_evenly_and_differs_by_seed():
    data = dataset("swaps", seed=0, n=40, levels=[4, 8])
    assert data.column_names == ["prompt", "answer", "level"]
    assert data["level"] == [4] * 20 + [8] * 20
    assert all(answer == simulate(prompt) for prompt, answer in zip(data["prompt"], data["answer"]))
    assert not set(data["prompt"]) & set(dataset("swaps", seed=10_000, n=40, levels=[4, 8])["prompt"])


def test_swap_levels_use_different_problems():
    data = dataset("swaps", seed=0, n=40, levels=[8, 8])
    assert data["prompt"][:20] != data["prompt"][20:]
