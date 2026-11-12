import json

import pytest

from tasks import PLAIN, TAGS
from train import grpo_config, logged_reward, parse_args


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_logged_reward_scores_and_records_every_completion(tmp_path):
    state = [(3, "train")]
    logs = []
    correct = logged_reward(tmp_path, TAGS, lambda: state[0], logs.append)
    assert correct.__name__ == "correct"

    kwargs = dict(prompts=["p1", "p1", "p2"], answer=["C", "C", "34"], level=[4, 4, 0], completion_ids=[[1, 2], [2], [3]])
    assert correct(completions=["x</think><answer>C</answer>", "x</think><answer>D</answer>", "junk"], **kwargs) == [1.0, 0.0, 0.0]
    assert correct(completions=["junk", "junk", "junk"], **kwargs) == [0.0, 0.0, 0.0]

    state[0] = (4, "eval")
    assert correct(completions=["x", "x", "</think> <answer>34</answer>"], **kwargs) == [0.0, 0.0, 1.0]

    assert sorted(p.name for p in tmp_path.iterdir()) == ["eval_00004.jsonl", "train_00003.jsonl"]
    train, evaluation = read(tmp_path / "train_00003.jsonl"), read(tmp_path / "eval_00004.jsonl")
    assert len(train) == 6 and len(evaluation) == 3
    assert train[0] == {
        "prompt": "p1", "completion": "x</think><answer>C</answer>",
        "answer": "C", "parsed": "C", "level": 4, "reward": 1.0, "tokens": 2, "reasoning_chars": 1,
    }
    assert logs[0]["train_levels/accuracy/level_4"] == 0.5 and logs[0]["train_levels/tokens/level_4"] == 1.5
    assert logs[1]["train_levels/accuracy/level_4"] == 0.25  # running mean over both batches of step 3
    assert logs[2] == {
        "eval_levels/accuracy/level_4": 0.0, "eval_levels/parsed/level_4": 0.0,
        "eval_levels/tokens/level_4": 1.5, "eval_levels/reasoning_chars/level_4": 1.0,
        "eval_levels/accuracy/level_0": 1.0, "eval_levels/parsed/level_0": 1.0,
        "eval_levels/tokens/level_0": 1.0, "eval_levels/reasoning_chars/level_0": 0.0,
    }
    assert [r["parsed"] for r in train[:3]] == ["C", "D", None]
    assert evaluation[2]["reward"] == 1.0 and evaluation[2]["parsed"] == "34"


def config(*argv):
    return grpo_config(parse_args(["--task", "swaps", "--name", "t", *argv]), "runs/t")


def test_default_config_is_outcome_only_no_kl():
    c = config()
    assert c.beta == 0.0 and c.temperature == 1.0 and c.loss_type == "dapo"
    assert c.generation_kwargs == {"stop": ["</answer>"], "include_stop_str_in_output": True}
    assert c.use_vllm and c.vllm_mode == "colocate"


def test_plain_format_changes_only_the_stop_string():
    plain, tags = config("--format", "plain").to_dict(), config().to_dict()
    assert plain.pop("generation_kwargs") == {"stop": ["\nProblem:"], "include_stop_str_in_output": True}
    tags.pop("generation_kwargs")
    assert plain == tags


def test_logged_reward_uses_the_given_format(tmp_path):
    kwargs = dict(prompts=["p"], answer=["C"], level=[1], completion_ids=[[1]])
    completions = ["x</think><answer>C</answer>"]
    assert logged_reward(tmp_path, TAGS, lambda: (0, "train"), lambda stats: None)(completions=completions, **kwargs) == [1.0]
    assert logged_reward(tmp_path, PLAIN, lambda: (0, "train"), lambda stats: None)(completions=completions, **kwargs) == [0.0]
    assert logged_reward(tmp_path, PLAIN, lambda: (0, "train"), lambda stats: None)(completions=["x\nAnswer: C"], **kwargs) == [1.0]


@pytest.mark.parametrize("prompts, generations, batch", [(32, 8, 16), (4, 4, 8), (8, 16, 16), (32, 8, 256)])
def test_one_optimizer_step_uses_every_prompt_once(prompts, generations, batch):
    c = config("--prompts-per-step", str(prompts), "--generations", str(generations), "--batch-size", str(batch))
    assert c.per_device_train_batch_size * c.gradient_accumulation_steps == prompts * generations
    assert c.generation_batch_size == prompts * generations
    assert c.steps_per_generation == c.gradient_accumulation_steps
    assert c.per_device_eval_batch_size == batch
    assert c.model_init_kwargs == {"dtype": "bfloat16"}


@pytest.mark.parametrize("prompts, generations, batch", [(32, 8, 12), (3, 8, 16), (32, 8, 4)])
def test_incompatible_batch_shapes_are_rejected(prompts, generations, batch):
    with pytest.raises(AssertionError):
        config("--prompts-per-step", str(prompts), "--generations", str(generations), "--batch-size", str(batch))
