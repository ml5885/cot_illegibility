"""Does the model open a think block on its own? Checkpoints given only "User: <question> Assistant:", with no
instruction about reasoning or tags and nothing prefilled.

uv run python unprompted.py --run swaps-7b-tags-long [--every 20]   (GPU host)
Writes runs/unprompted/<run>.jsonl (every sample) and prints, per checkpoint and swap count, the share that open
<think>, the share that write anything before naming a box, and accuracy with and without a think block.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tasks import TAGS, dataset

ANSWER = re.compile(r"<answer>\s*([A-F])\s*</answer>|\b(?:box|Box|answer is|Answer:)\s*([A-F])\b")


def given(completion: str) -> str | None:
    """The box in the answer tag, else the last box named as "box X" or "Answer: X"."""
    found = ANSWER.findall(completion)
    tagged = [a for a, _ in found if a]
    return tagged[-1] if tagged else (found[-1][1] if found else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--problems", type=int, default=40, help="per swap count")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    problems = dataset("swaps", "eval", 0, args.problems * 4, [2, 4, 8, 16], TAGS)
    questions = [p.split("User: ", 1)[1].rsplit(" Assistant:", 1)[0] for p in problems["prompt"]]
    prompts = [f"User: {q} Assistant:" for q in questions]
    limit = json.loads(Path("runs", args.run, "args.json").read_text())["max_tokens"]
    llm = LLM("allenai/Olmo-3-1025-7B", dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096, enable_lora=True,
              max_lora_rank=64)
    params = SamplingParams(n=args.k, temperature=1.0, max_tokens=limit, stop=["</answer>", "\nUser:"],
                            include_stop_str_in_output=True)
    steps = sorted(int(p.name.split("-")[1]) for p in Path("runs", args.run).glob("checkpoint-*"))
    out = Path("runs", "unprompted")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.run}.jsonl", "w") as f:
        for i, step in enumerate([0] + [s for s in steps if s % args.every == 0], 1):
            lora = LoRARequest(f"{args.run}-{step}", i, f"runs/{args.run}/checkpoint-{step}") if step else None
            stats = defaultdict(lambda: defaultdict(int))
            for problem, output in zip(problems, llm.generate(prompts, params, lora_request=lora)):
                for sample in output.outputs:
                    text = sample.text
                    think = text.lstrip().startswith("<think>")
                    box = given(text)
                    first_box = re.search(r"\b[A-F]\b", text)
                    reasons = bool(first_box) and len(text[: first_box.start()].split()) > 3
                    correct = box == problem["answer"]
                    s = stats[problem["level"]]
                    s["n"] += 1
                    s["think"] += think
                    s["reasons"] += reasons
                    s["correct"] += correct
                    s["think correct"] += think and correct
                    f.write(json.dumps({"step": step, "level": problem["level"], "answer": problem["answer"], "given": box,
                                        "think": think, "reasons": reasons, "completion": text}) + "\n")
            for level, s in sorted(stats.items()):
                print(f"{args.run} step {step:3d} {level:2d} swaps: think {s['think'] / s['n']:.2f}  reasons {s['reasons'] / s['n']:.2f}  "
                      f"accuracy {s['correct'] / s['n']:.2f}  accuracy with think {s['think correct'] / max(1, s['think']):.2f}")


if __name__ == "__main__":
    main()
