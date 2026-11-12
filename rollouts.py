"""Every sampled completion is saved, one file per split and step, in the layout view/index.html reads."""

import json
from pathlib import Path

from tasks import Format


def score_and_save(
    directory: Path, fmt: Format, split: str, step: int,
    prompts: list[str], completions: list[str], answers: list[str], levels: list[int],
) -> list[float]:
    directory.mkdir(parents=True, exist_ok=True)
    rewards = [fmt.reward(c, a) for c, a in zip(completions, answers)]
    with open(directory / f"{split}_{step:05d}.jsonl", "a") as f:
        for prompt, completion, answer, level, reward in zip(prompts, completions, answers, levels, rewards):
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "answer": answer,
                "parsed": fmt.parse_answer(completion),
                "level": level,
                "reward": reward,
            }) + "\n")
    return rewards
