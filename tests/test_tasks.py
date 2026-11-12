import random
import re
from collections import Counter

import pytest

from tasks import FORMATS, MATH_EVAL, PLAIN, TAGS, dataset, math_problems, math_rows, normalize, swap_problem, swap_problems

SWAP = re.compile(r"Swap the contents of boxes (\w) and (\w)\.")
START = re.compile(r"A key is in box (\w)\.")


def simulate(question: str) -> str:
    """Independent solver: track every box's contents, then find the key."""
    contents = {box: box == re.search(r"A key is in box (\w)\.", question).group(1) for box in "ABCDEF"}
    for a, b in SWAP.findall(question):
        contents[a], contents[b] = contents[b], contents[a]
    return next(box for box, has_key in contents.items() if has_key)


@pytest.mark.parametrize("n_swaps", [0, 1, 2, 8, 40])
def test_swap_answer_matches_independent_simulation(n_swaps):
    for problem in swap_problems(seed=n_swaps, n=200, n_swaps=n_swaps):
        assert len(SWAP.findall(problem.question)) == n_swaps
        assert problem.answer == simulate(problem.question)
        assert problem.level == n_swaps


def test_swap_question_format():
    question = swap_problem(random.Random(0), 3).question
    assert question.startswith("There are 6 boxes labeled A to F. A key is in box ")
    assert question.endswith(" Which box contains the key? Answer with the letter of the box.")
    assert all(a != b for a, b in SWAP.findall(question))


def test_swap_problems_deterministic_and_seed_dependent():
    assert swap_problems(1, 20, 8) == swap_problems(1, 20, 8)
    assert swap_problems(1, 20, 8) != swap_problems(2, 20, 8)


def test_swap_answers_roughly_uniform():
    answers = [p.answer for p in swap_problems(0, 6000, 30)]
    assert all(800 < answers.count(box) < 1200 for box in "ABCDEF")


def shortcut_accuracy(problems, guess):
    return sum(guess(p.question) == p.answer for p in problems) / len(problems)


@pytest.mark.parametrize("n_swaps", [1, 2, 3, 4, 8, 32])
def test_naming_the_starting_box_is_no_better_than_guessing(n_swaps):
    problems = swap_problems(0, 6000, n_swaps)
    assert abs(shortcut_accuracy(problems, lambda q: START.search(q).group(1)) - 1 / 6) < 0.02


@pytest.mark.parametrize("n_swaps", [2, 3, 4, 8, 32])
def test_positional_shortcuts_stay_well_below_solving(n_swaps):
    problems = swap_problems(0, 6000, n_swaps)
    most_mentioned = lambda q: Counter(box for pair in SWAP.findall(q) for box in pair).most_common(1)[0][0]
    for guess in (lambda q: SWAP.findall(q)[-1][0], lambda q: SWAP.findall(q)[-1][1], most_mentioned):
        assert shortcut_accuracy(problems, guess) < 0.45


def test_zero_swaps_leaves_the_key_in_place():
    assert all(START.search(p.question).group(1) == p.answer for p in swap_problems(0, 50, 0))


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
        ("x\nAnswer: C", None),
        ("", None),
    ],
)
def test_tags_parse_answer(completion, expected):
    assert TAGS.parse_answer(completion) == expected


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("A to C.\nAnswer: C", "C"),
        ("Answer: C.", "C"),
        ("Answer: C\n\n", "C"),
        ("Answer: C\nProblem:", "C"),
        ("Answer: C\nProblem: a new one\nAnswer: D", "D"),
        ("1) no. 2) Answer: D", "D"),
        ("so it is 34.\nAnswer: $34$", "34"),
        ("Answer: $\\boxed{-3}$.", "-3"),
        ("Answer: 5\nAnswer: 6", "6"),
        ("Answer:C\nmore text", None),
        ("x</think> <answer>C</answer>", None),
        ("the answer is C", None),
        ("", None),
    ],
)
def test_plain_parse_answer(completion, expected):
    assert PLAIN.parse_answer(completion) == expected


def test_normalize_is_idempotent():
    for raw in [" $\\boxed{12}$. ", "1,000", "C.", "-3"]:
        assert normalize(normalize(raw)) == normalize(raw)


@pytest.mark.parametrize(
    "fmt, completion, answer, expected",
    [
        (TAGS, "x</think> <answer>C</answer>", "C", 1.0),
        (TAGS, "x</think> <answer>D</answer>", "C", 0.0),
        (TAGS, "x</think> <answer>box C</answer>", "C", 0.0),
        (TAGS, "the answer is C</think>", "C", 0.0),
        (TAGS, "x</think> <answer>340</answer>", "34", 0.0),
        (TAGS, "x</think> <answer>-3</answer>", "3", 0.0),
        (PLAIN, "Answer: C", "C", 1.0),
        (PLAIN, "Answer: D", "C", 0.0),
        (PLAIN, "Answer: C\nBut wait, maybe D.", "C", 0.0),
        (PLAIN, "C", "C", 0.0),
        (PLAIN, "Answer: 340", "34", 0.0),
    ],
)
def test_reward(fmt, completion, answer, expected):
    assert fmt.reward(completion, answer) == expected


@pytest.mark.parametrize("fmt", FORMATS.values())
def test_template_has_one_slot_and_a_reachable_stop(fmt):
    assert fmt.template.count("{}") == 1
    assert fmt.template.format("Q?").count("Q?") == 1


def test_tags_prompt_ends_inside_an_open_think_tag():
    prompt = dataset("swaps", "train", 0, 2, [3], TAGS)["prompt"][0]
    assert prompt.startswith("A conversation between User and Assistant.")
    assert "User: There are 6 boxes" in prompt and prompt.endswith("Answer with the letter of the box. Assistant: <think>")


def test_plain_prompt_has_no_tags():
    prompt = dataset("swaps", "train", 0, 2, [3], PLAIN)["prompt"][0]
    assert "<think>" not in prompt and "<answer>" not in prompt
    assert "\n\nProblem: There are 6 boxes" in prompt and prompt.endswith("Answer with the letter of the box.\nReasoning:")


def test_formats_share_the_same_problems():
    tags, plain = (dataset("swaps", "train", 3, 40, [4, 8], fmt) for fmt in (TAGS, PLAIN))
    assert tags["answer"] == plain["answer"] and tags["level"] == plain["level"]
    assert [simulate(p) for p in tags["prompt"]] == [simulate(p) for p in plain["prompt"]] == tags["answer"]


def test_swap_dataset_splits_levels_evenly_and_eval_differs_from_train():
    data = dataset("swaps", "train", seed=0, n=40, levels=[4, 8], fmt=TAGS)
    assert data.column_names == ["prompt", "answer", "level"]
    assert data["level"] == [4] * 20 + [8] * 20
    assert not set(data["prompt"]) & set(dataset("swaps", "eval", seed=0, n=40, levels=[4, 8], fmt=TAGS)["prompt"])


def test_swap_levels_use_different_problems():
    data = dataset("swaps", "train", seed=0, n=40, levels=[8, 8], fmt=TAGS)
    assert data["prompt"][:20] != data["prompt"][20:]


def test_math_eval_is_fixed_and_disjoint_from_every_train_draw():
    eval_questions = {p.question for p in math_problems("eval", seed=0, n=MATH_EVAL)}
    assert eval_questions == {p.question for p in math_problems("eval", seed=7, n=MATH_EVAL)}
    train_a = [p.question for p in math_problems("train", seed=0, n=20_000)]
    train_b = [p.question for p in math_problems("train", seed=1, n=20_000)]
    assert train_a != train_b and set(train_a) == set(train_b)
    assert len(set(train_a)) == len(train_a) == len(math_rows()) - MATH_EVAL
    assert len(eval_questions) == MATH_EVAL
    assert not eval_questions & set(train_a)


@pytest.mark.parametrize("fmt, solve", [(TAGS, " so. </think> <answer> {} </answer>"), (PLAIN, " so.\nAnswer: {}")])
def test_math_dataset_format(fmt, solve):
    data = dataset("math", "eval", seed=0, n=50, levels=[], fmt=fmt)
    assert data.column_names == ["prompt", "answer", "level"]
    head, tail = fmt.template.split("{}")
    for prompt, answer in zip(data["prompt"], data["answer"]):
        assert prompt.startswith(head) and prompt.endswith(tail)
        assert fmt.reward(solve.format(answer), answer) == 1.0


@pytest.mark.parametrize(
    "fmt, completion, reasoning",
    [
        (TAGS, " The key moves to B. </think> <answer>B</answer>", "The key moves to B."),
        (TAGS, " first </think> <think> second </think> <answer> B </answer>", "first </think> <think> second"),
        (TAGS, " never closes", "never closes"),
        (PLAIN, " Step one.\nStep two.\nAnswer: B", "Step one.\nStep two."),
        (PLAIN, " no final line", "no final line"),
    ],
)
def test_reasoning_is_everything_before_the_answer(fmt, completion, reasoning):
    assert fmt.reasoning(completion) == reasoning


@pytest.mark.parametrize("fmt", [TAGS, PLAIN])
def test_skipping_the_reasoning_still_parses(fmt):
    completion = fmt.skip + " E " + (fmt.stop if fmt is TAGS else "")
    assert fmt.parse_answer(completion) == "E" and fmt.reasoning(completion) == ""
