import pytest

from traces import puzzle, swap_steps

TEMPLATE = "User: There are 6 boxes labeled A to F. A key is in box {}. {} Which box contains the key? Assistant: <think>"


def row(start, swaps, completion):
    question = " ".join(f"Swap the contents of boxes {a} and {b}." for a, b in swaps)
    return {"prompt": TEMPLATE.format(start, question), "completion": completion}


def outcomes(*args):
    return [outcome for outcome, _ in swap_steps(row(*args))]


def test_puzzle_tracks_the_key_and_which_swaps_move_it():
    swaps, after, moves = puzzle(row("A", ["AB", "CD", "BC"], ""))
    assert swaps == [("A", "B"), ("C", "D"), ("B", "C")]
    assert after == ["B", "B", "C"] and moves == [True, False, True]


@pytest.mark.parametrize(
    "completion",
    [
        " 1. Swap the contents of boxes A and B: The key in box A moves to box B. 2. Swap the contents of boxes C and D:"
        " The key is in box B, so it remains there. 3. Swap the contents of boxes B and C: The key, in B, will move to C."
        "</think> <answer> C </answer>",
        " Step 1: Swap A and B. The key was in box A, so it is now in box B. Step 2: Swap C and D. No change, as the key"
        " is in B. Step 3: Swap B and C. The key moves from B to C. So the key is in box C.</think> <answer>C</answer>",
        " Let's go.\n\n1. Swap A and B: key moves to B.\n\n2. Swap C and D: the key stays in B.\n\n3. Swap B and C: the key"
        " ends up in C.\n\nTherefore the key is in box C.</think> <answer> C </answer>",
        " First we swap box A with box B, so the key goes to box B. Then C & D are swapped, which leaves the key in box B."
        " Last, B and C are swapped and the key moves to box C.</think> <answer> C </answer>",
    ],
)
def test_correct_walkthroughs_in_several_styles_read_as_right(completion):
    assert outcomes("A", ["AB", "CD", "BC"], completion) == ["right", "right", "right"]


def test_a_wrong_location_is_caught_at_the_step_where_it_happens():
    completion = (
        " 1. Swap the contents of boxes A and B: The key moves to box B. 2. Swap the contents of boxes C and D: The key"
        " moves to box D. 3. Swap the contents of boxes B and C: The key is in box D, so it stays in box D.</think>"
    )
    assert outcomes("A", ["AB", "CD", "BC"], completion) == ["right", "wrong", "wrong"]


def test_boxes_named_only_as_the_swapped_pair_or_as_not_holding_the_key_are_ignored():
    completion = (
        " 1. Swap the contents of boxes C and B. The key is in box A, so it remains there. The key is not affected by"
        " swapping boxes C and B. 2. Swap the contents of boxes D and E: The key is in box A, so it doesn't move to box D."
        " 3. Swap the contents of boxes E and F: The key is in A. No key is moved to box E or F.</think>"
    )
    assert outcomes("A", ["CB", "DE", "EF"], completion) == ["right", "right", "right"]


def test_a_step_that_never_places_the_key_is_unstated():
    completion = (
        " 1. Swap the contents of boxes C and D: The key is not in boxes C or D, so its location does not change."
        " 2. Swap the contents of boxes A and B: The key moves to box B.</think>"
    )
    assert outcomes("A", ["CD", "AB"], completion) == ["unstated", "right"]


def test_the_same_pair_twice_in_a_row_with_the_pair_repeated_inside_each_step():
    completion = (
        " 1. Swap the contents of boxes F and C: The key is in box C, so when we swap boxes F and C, the key will move to"
        " box F. 2. Swap the contents of boxes F and C: The key is in box F, so when we swap boxes F and C, the key will"
        " move to box C.</think>"
    )
    assert outcomes("C", ["FC", "FC"], completion) == ["right", "right"]


def test_dropping_the_last_swap_is_skipped_not_wrong():
    completion = " 1. Swap A and B: The key moves to B. 2. Swap C and D: The key stays in B. So the key is in box B.</think>"
    assert outcomes("A", ["AB", "CD", "BC"], completion) == ["right", "right", "skipped"]


def test_dropping_a_swap_in_the_middle_does_not_misalign_the_rest():
    completion = (
        " 1. Swap A and B: The key moves to B. 2. Swap E and F: The key stays in B. 3. Swap B and C: The key moves to C."
        " 4. Swap C and D: The key moves to D.</think>"
    )
    assert outcomes("A", ["AB", "CD", "EF", "BC", "CD"], completion) == ["right", "skipped", "right", "right", "right"]


def test_a_list_of_the_swaps_before_the_walkthrough_is_not_taken_for_the_walkthrough():
    completion = (
        " The instructions are: 1. Swap the contents of boxes A and B. 2. Swap the contents of boxes B and C. Now step by"
        " step. 1. Swap the contents of boxes A and B: the key moves to box B. 2. Swap the contents of boxes B and C: the"
        " key moves to box C.</think>"
    )
    assert outcomes("A", ["AB", "BC"], completion) == ["right", "right"]


def test_restating_counts_sentences_that_place_the_key_in_the_final_box():
    once = " 1. Swap A and B: The key, in A, will move to B.</think>"
    twice = " 1. Swap A and B: The key is in box A, so it moves to box B. So now the key is in box B.</think>"
    assert swap_steps(row("A", ["AB"], once)) == [("right", False)]
    assert swap_steps(row("A", ["AB"], twice)) == [("right", True)]


def test_layout_does_not_change_the_reading():
    plain = " 1. Swap A and B: The key moves to B. 2. Swap B and C: The key moves to C.</think>"
    spaced = " 1. Swap A and B:\n\n   The key moves to B.\n\n2. Swap B and C:\n\n   The key moves to C.\n\n</think>"
    assert swap_steps(row("A", ["AB", "BC"], plain)) == swap_steps(row("A", ["AB", "BC"], spaced))
