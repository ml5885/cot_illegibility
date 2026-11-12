import json

import pytest

from train import grpo_config, logged_reward, parse_args


def test_logged_reward_scores_and_records_every_completion(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    state = [(3, "train")]
    correct = logged_reward(path, lambda: state[0])
    assert correct.__name__ == "correct"

    kwargs = dict(prompts=["p1", "p1", "p2"], answer=["C", "C", "34"], level=[4, 4, 0], completion_ids=[[1], [2], [3]])
    assert correct(completions=["Answer: C", "Answer: D", "junk"], **kwargs) == [1.0, 0.0, 0.0]

    state[0] = (4, "eval")
    assert correct(completions=["x", "x", "Answer: 34"], **kwargs) == [0.0, 0.0, 1.0]

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 6
    assert rows[0] == {
        "step": 3, "split": "train", "prompt": "p1", "completion": "Answer: C",
        "answer": "C", "level": 4, "reward": 1.0,
    }
    assert [r["split"] for r in rows] == ["train"] * 3 + ["eval"] * 3
    assert rows[5]["reward"] == 1.0 and rows[5]["step"] == 4


def config(*argv):
    return grpo_config(parse_args(["--task", "swaps", "--name", "t", *argv]), "runs/t")


def test_default_config_is_outcome_only_no_kl():
    c = config()
    assert c.beta == 0.0 and c.temperature == 1.0 and c.loss_type == "dapo"
    assert c.generation_kwargs == {"stop": ["\nProblem:"]}
    assert c.use_vllm and c.vllm_mode == "colocate"


@pytest.mark.parametrize("prompts, generations, batch", [(32, 8, 16), (4, 4, 8), (8, 16, 16), (32, 8, 256)])
def test_one_optimizer_step_uses_every_prompt_once(prompts, generations, batch):
    c = config("--prompts-per-step", str(prompts), "--generations", str(generations), "--batch-size", str(batch))
    assert c.per_device_train_batch_size * c.gradient_accumulation_steps == prompts * generations
    assert c.generation_batch_size == prompts * generations
    assert c.steps_per_generation == c.gradient_accumulation_steps
    assert c.per_device_eval_batch_size == batch


@pytest.mark.parametrize("prompts, generations, batch", [(32, 8, 12), (3, 8, 16), (32, 8, 4)])
def test_incompatible_batch_shapes_are_rejected(prompts, generations, batch):
    with pytest.raises(AssertionError):
        config("--prompts-per-step", str(prompts), "--generations", str(generations), "--batch-size", str(batch))
