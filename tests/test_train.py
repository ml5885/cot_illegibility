import json

from train import logged_reward


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
