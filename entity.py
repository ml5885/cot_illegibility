"""No-reasoning entity tracking: can a base model name the box that holds an object, with one or six objects and 0 to 2 swaps?
Closer to the entity-tracking setups of Kim & Schuster (2023) and Prakash et al. than our 16-swap task.

uv run python entity.py --model MODEL      (GPU host; writes runs/entity-<model>/letters.jsonl)
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

from tasks import FORMATS

# The system nvcc is too old to JIT-compile FlashInfer's sampler.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams  # noqa: E402

BOXES = "ABCDEF"
OBJECTS = ["key", "apple", "coin", "ring", "book", "cup"]
CONDITIONS = [(1, 0), (1, 1), (1, 2), (6, 0), (6, 1), (6, 2)]  # (objects, swaps)


def puzzle(rng: random.Random, n_objects: int, n_swaps: int) -> tuple[str, str]:
    """A question and the box holding the asked-about object."""
    objects = rng.sample(OBJECTS, n_objects)
    boxes = dict(zip(rng.sample(BOXES, n_objects), objects))
    if n_objects == 1:
        lines = [f"There are 6 boxes labeled A to F. A {objects[0]} is in box {next(iter(boxes))}."]
    else:
        lines = ["There are 6 boxes labeled A to F."] + [f"Box {b} contains the {o}." for b, o in sorted(boxes.items())]
    target = rng.choice(objects)
    start = next(b for b, o in boxes.items() if o == target)
    # As in the main task, the asked-about object ends somewhere else 5 times in 6, so naming its first box is no better than guessing.
    moves = n_swaps > 0 and rng.random() < 5 / 6
    while True:
        swaps, answer = [rng.sample(BOXES, 2) for _ in range(n_swaps)], start
        for a, b in swaps:
            answer = b if answer == a else a if answer == b else answer
        if (answer != start) == moves:
            break
    lines += [f"Swap the contents of boxes {a} and {b}." for a, b in swaps]
    lines.append(f"Which box contains the {target}? Answer with the letter of the box.")
    return " ".join(lines), answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--problems", type=int, default=200, help="per condition")
    args = parser.parse_args()

    fmt = FORMATS["tags"]
    rng = random.Random(0)
    items = [(objs, swaps, *puzzle(rng, objs, swaps)) for objs, swaps in CONDITIONS for _ in range(args.problems)]
    prompts = [fmt.template.format(question) + fmt.skip for _, _, question, _ in items]

    llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.9, seed=0, max_model_len=2048)
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0, logprobs=20))

    out = Path("runs") / f"entity-{args.model.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: defaultdict(float))
    with open(out / "letters.jsonl", "w") as f:
        for (objs, swaps, question, answer), output in zip(items, outputs):
            probs = defaultdict(float)
            for logprob in output.outputs[0].logprobs[0].values():
                if (letter := logprob.decoded_token.strip()) in set(BOXES):
                    probs[letter] += math.exp(logprob.logprob)
            f.write(json.dumps({"objects": objs, "swaps": swaps, "question": question, "answer": answer, "probs": probs}) + "\n")
            mass = sum(probs.values())
            s = stats[objs, swaps]
            s["n"] += 1
            s["p"] += probs[answer] / mass if mass else 1 / 6
            s["argmax"] += max(BOXES, key=lambda b: probs[b]) == answer
    for (objs, swaps), s in sorted(stats.items()):
        print(f"objects={objs} swaps={swaps}  P(correct)={s['p'] / s['n']:.3f}  argmax={s['argmax'] / s['n']:.3f}")


if __name__ == "__main__":
    main()
