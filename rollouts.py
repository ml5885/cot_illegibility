"""Every sampled completion is saved, one file per split and step, in the layout view/index.html reads."""

import json
from pathlib import Path

from tasks import Format


def score_and_save(
    directory: Path, fmt: Format, split: str, step: int,
    prompts: list[str], completions: list[str], answers: list[str], levels: list[int], tokens: list[int],
) -> list[dict]:
    """`tokens` is each completion's length in tokens. Returns the saved rows."""
    rows = [
        {
            "prompt": prompt,
            "completion": completion,
            "answer": answer,
            "parsed": fmt.parse_answer(completion),
            "level": level,
            "reward": fmt.reward(completion, answer),
            "tokens": n,
            "reasoning_chars": len(fmt.reasoning(completion)),
        }
        for prompt, completion, answer, level, n in zip(prompts, completions, answers, levels, tokens)
    ]
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{split}_{step:05d}.jsonl", "a") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return rows


def by_level(rows: list[dict]) -> dict[str, float]:
    """Mean accuracy, parse rate and lengths for each difficulty level."""
    stats = {}
    for level in sorted({row["level"] for row in rows}):
        at = [row for row in rows if row["level"] == level]
        stats[f"accuracy/level_{level}"] = sum(row["reward"] for row in at) / len(at)
        stats[f"parsed/level_{level}"] = sum(row["parsed"] is not None for row in at) / len(at)
        stats[f"tokens/level_{level}"] = sum(row["tokens"] for row in at) / len(at)
        stats[f"reasoning_chars/level_{level}"] = sum(row["reasoning_chars"] for row in at) / len(at)
    return stats
