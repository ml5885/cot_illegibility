"""Does each reasoning step read the key's box from the text before it? A behavioural test with prefilled responses.

For each checkpoint, take its own correct (greedy) response to a held-out puzzle and pick a step that states where the
key is. Cut the response right after that box letter and let the model continue (greedy) in three conditions:

- clean: nothing changed, a control (it should still answer right);
- text edit: the stated box replaced by another one, so the text and the prompt imply different final answers;
- prompt edit: that step's swap in the prompt changed so the key ends up elsewhere, the written text left as it was.

We record whether the final answer follows the written text or the prompt.

uv run python prefill.py --checkpoints swaps-7b-tags-256:200 swaps-7b-tags-long:60 ...   (GPU host)
Writes runs/prefill/<run>-<step>.jsonl and prints the shares per checkpoint, swap count and condition.
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from tasks import TAGS, dataset
from traces import SWAP, puzzle, swap_mentions, step_texts

BOXES = "ABCDEF"
PLACED = re.compile(r"\b(?:in|to|into|at)\s+(?:box\s+)?([A-F])\b")


def follow(key: str, swaps: list[tuple[str, str]]) -> str:
    for a, b in swaps:
        key = b if key == a else a if key == b else key
    return key


def cut_point(row: dict, rng: random.Random) -> tuple[int, int] | None:
    """A random step k (not the last) whose text puts the key in the right box: (k, index of that box letter in the
    completion). The letter is the last "in/to/into/at (box) X" in the step that names the right box."""
    swaps, after, _ = puzzle(row)
    reasoning = row["completion"].partition("</think>")[0]
    mentions = swap_mentions(reasoning, swaps)
    candidates = []
    for k, (mention, text) in enumerate(zip(mentions[:-1], step_texts(reasoning, mentions)[:-1])):
        if mention is None or text is None:
            continue
        hits = [m for m in PLACED.finditer(text) if m.group(1) == after[k]]
        if hits:
            candidates.append((k, mention.end() + hits[-1].start(1)))
    return rng.choice(candidates) if candidates else None


def edits(row: dict, k: int, rng: random.Random) -> dict | None:
    """The text edit and prompt edit for step k, with the final answer each source implies. None if no edit makes
    the two sources disagree."""
    swaps, after, moves = puzzle(row)
    start = re.search(r"A key is in box (\w)", row["prompt"]).group(1)
    rest = swaps[k + 1 :]
    # Text edit: another box for the key after step k; the prompt still implies the true answer.
    boxes = [b for b in BOXES if b != after[k] and follow(b, rest) != row["answer"]]
    if not boxes:
        return None
    box = rng.choice(boxes)
    # Prompt edit: swap k changed so that it moves the key if it did not, or leaves it if it did.
    before = after[k - 1] if k else start
    for _ in range(20):
        if moves[k]:
            pair = tuple(rng.sample([b for b in BOXES if b != before], 2))
        else:
            pair = (before, rng.choice([b for b in BOXES if b != before]))[:: rng.choice((1, -1))]
        edited = swaps[:k] + [pair] + rest
        if follow(start, edited) != row["answer"]:
            break
    else:
        return None
    match = list(SWAP.finditer(row["prompt"]))[k]
    prompt = row["prompt"][: match.start()] + f"Swap the contents of boxes {pair[0]} and {pair[1]}." + row["prompt"][match.end() :]
    return {
        "box": box, "text_answer": follow(box, rest),
        "pair": pair, "prompt": prompt, "prompt_answer": follow(start, edited),
    }


def outcome(answer: str | None, text: str, prompt: str) -> str:
    return "text" if answer == text else "prompt" if answer == prompt else "none" if answer is None else "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="run:step")
    parser.add_argument("--swaps", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--problems", type=int, default=100, help="per swap count")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The system nvcc is too old to JIT-compile FlashInfer's sampler.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    problems = dataset("swaps", "eval", args.seed, args.problems * len(args.swaps), args.swaps, TAGS)
    llm = LLM("allenai/Olmo-3-1025-7B", dtype="bfloat16", gpu_memory_utilization=0.9, seed=args.seed, max_model_len=4096,
              enable_lora=True, max_lora_rank=64)
    out = Path("runs", "prefill")
    out.mkdir(parents=True, exist_ok=True)

    for i, spec in enumerate(args.checkpoints, 1):
        run, step = spec.split(":")
        limit = json.loads(Path("runs", run, "args.json").read_text())["max_tokens"]
        lora = LoRARequest(spec, i, f"runs/{run}/checkpoint-{step}")
        greedy = SamplingParams(temperature=0, max_tokens=limit, stop=[TAGS.stop], include_stop_str_in_output=True)
        first = llm.generate(list(problems["prompt"]), greedy, lora_request=lora)
        rng = random.Random(args.seed)
        cases = []
        for problem, output in zip(problems, first):
            row = dict(problem) | {"completion": output.outputs[0].text}
            if TAGS.reward(row["completion"], row["answer"]) != 1 or not (cut := cut_point(row, rng)):
                continue
            k, index = cut
            edit = edits(row, k, rng)
            if edit is None:
                continue
            kept = row["completion"][:index]
            cases.append(row | {"step": k, "letter_index": index} | edit | {"inputs": {
                "clean": row["prompt"] + kept + row["completion"][index],
                "text": row["prompt"] + kept + edit["box"],
                "prompt": edit["prompt"] + kept + row["completion"][index],
            }})
        inputs = [(case, name, text) for case in cases for name, text in case["inputs"].items()]
        continued = llm.generate([text for _, _, text in inputs], greedy, lora_request=lora)
        counts = Counter()
        with open(out / f"{run}-{step}.jsonl", "w") as f:
            for (case, name, text), output in zip(inputs, continued):
                completion = text.split(" Assistant: <think>", 1)[1] + output.outputs[0].text
                answer = TAGS.parse_answer(completion)
                result = "right" if name == "clean" and answer == case["answer"] else (
                    "wrong" if name == "clean" else
                    outcome(answer, case["text_answer"], case["answer"]) if name == "text" else
                    outcome(answer, case["answer"], case["prompt_answer"]))
                counts[case["level"], name, result] += 1
                f.write(json.dumps({
                    "level": case["level"], "condition": name, "step": case["step"], "answer": case["answer"],
                    "box": case["box"], "text_answer": case["text_answer"], "pair": case["pair"],
                    "prompt_answer": case["prompt_answer"], "given": answer, "result": result,
                    "prompt": text.split(" Assistant: <think>", 1)[0],
                    "completion": completion,
                }) + "\n")
        print(f"\n{spec}: {len(cases)} cases from {len(problems)} puzzles")
        for level in args.swaps:
            for name in ("clean", "text", "prompt"):
                c = {r: n for (l, cond, r), n in counts.items() if l == level and cond == name}
                total = sum(c.values())
                if total:
                    print(f"  {level:2d} swaps, {name:6s} n={total:3d}  " + "  ".join(f"{r} {n / total:.2f}" for r, n in sorted(c.items())))


if __name__ == "__main__":
    main()
