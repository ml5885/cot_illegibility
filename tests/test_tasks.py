import random
import re

import pytest

from tasks import MATH_EVAL, PROMPT, dataset, math_problems, math_rows, normalize, parse_answer, reasoning, reward, swap_problem, swap_problems

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
    assert prompt.startswith("A conversation between User and Assistant.")
    assert "User: There are 6 boxes labeled A to F. A key is in box " in prompt
    assert prompt.endswith(" Which box contains the key? Assistant: <think>")
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
        (" A to C. </think> <answer> C </answer>", "C"),
        ("A to C.</think><answer>C</answer>", "C"),
        ("x</think>\n\n<answer>C.</answer>", "C"),
        ("so 34.</think> <answer>$34$</answer>", "34"),
        ("x</think> <answer>\\boxed{-3}</answer>", "-3"),
        ("x</think> <answer>1,234</answer>", "1234"),
        ("x</think> <answer>**7**</answer>", "7"),
        ("x</think> <answer>5</answer> <answer>6</answer>", "5"),
        ("x</think> <answer>3", None),
        ("x <answer>C</answer>", None),
        ("x</think> so the answer is <answer>C</answer>", None),
        ("x</think> C", None),
        ("the answer is C", None),
        ("", None),
    ],
)
def test_parse_answer(completion, expected):
    assert parse_answer(completion) == expected


def test_reasoning_is_the_text_before_the_closing_think_tag():
    assert reasoning(" A to C. </think> <answer>C</answer>") == " A to C. "
    assert reasoning("never closes") == "never closes"


def test_normalize_is_idempotent():
    for raw in [" $\\boxed{12}$. ", "1,000", "C.", "-3"]:
        assert normalize(normalize(raw)) == normalize(raw)


@pytest.mark.parametrize(
    "completion, answer, expected",
    [
        ("x</think> <answer>C</answer>", "C", 1.0),
        ("x</think> <answer>D</answer>", "C", 0.0),
        ("x</think> <answer>box C</answer>", "C", 0.0),
        ("C", "C", 0.0),
        ("the answer is C</think>", "C", 0.0),
        ("x</think> <answer>34</answer>", "34", 1.0),
        ("x</think> <answer>340</answer>", "34", 0.0),
        ("x</think> <answer>-3</answer>", "3", 0.0),
    ],
)
def test_reward(completion, answer, expected):
    assert reward(completion, answer) == expected


def test_swap_dataset_splits_levels_evenly_and_differs_by_seed():
    data = dataset("swaps", "train", seed=0, n=40, levels=[4, 8])
    assert data.column_names == ["prompt", "answer", "level"]
    assert data["level"] == [4] * 20 + [8] * 20
    assert all(answer == simulate(prompt) for prompt, answer in zip(data["prompt"], data["answer"]))
    assert not set(data["prompt"]) & set(dataset("swaps", "eval", seed=0, n=40, levels=[4, 8])["prompt"])


def test_swap_levels_use_different_problems():
    data = dataset("swaps", "train", seed=0, n=40, levels=[8, 8])
    assert data["prompt"][:20] != data["prompt"][20:]


def test_math_eval_is_fixed_and_disjoint_from_every_train_draw():
    eval_prompts = {p.prompt for p in math_problems("eval", seed=0, n=MATH_EVAL)}
    assert eval_prompts == {p.prompt for p in math_problems("eval", seed=7, n=MATH_EVAL)}
    train_a = [p.prompt for p in math_problems("train", seed=0, n=12_000)]
    train_b = [p.prompt for p in math_problems("train", seed=1, n=12_000)]
    assert train_a != train_b and set(train_a) == set(train_b)
    assert len(set(train_a)) == len(train_a) == len(math_rows()) - MATH_EVAL
    assert len(eval_prompts) == MATH_EVAL
    assert not eval_prompts & set(train_a) and not eval_prompts & set(train_b)


def test_math_dataset_format():
    data = dataset("math", "eval", seed=0, n=50, levels=[])
    assert data.column_names == ["prompt", "answer", "level"]
    head, tail = PROMPT.split("{}")
    for prompt, answer in zip(data["prompt"], data["answer"]):
        assert prompt.startswith(head) and prompt.endswith(tail)
        assert reward(f" so we get it. </think> <answer> {answer} </answer>", answer) == 1.0
        assert reward(f"x</think> <answer>\\boxed{{{answer}}}</answer>", answer) == 1.0
