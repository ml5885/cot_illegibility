"""How likely is a line break at the first step boundaries, checkpoint by checkpoint, on fixed prefixes?

Each held-out puzzle gets two fixed openings ("Let's ..." and "To ..."), and we read the next-token distribution at two
points: right after the opening sentence, and right after a fixed first step. The probe is the probability of a token
that starts with a line break, against one that starts with a space, among the top 20 tokens.

uv run python newline_probe.py --runs swaps-7b-tags-long swaps-7b-tags-512 [--every 5]   (GPU host)
Writes runs/newline-probe/<run>.jsonl, one row per checkpoint, opening and point.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

from tasks import TAGS, dataset

OPENINGS = {
    "Let's": " Let's go through this step by step to determine where the key ends up. Initially, the key is in box {start}.",
    "To": " To determine which box contains the key, we need to follow each swap. Initially, the key is in box {start}.",
}


def first_step(question: str) -> str:
    start = re.search(r"A key is in box (\w)", question).group(1)
    a, b = re.search(r"Swap the contents of boxes (\w) and (\w)", question).groups()
    if start in (a, b):
        return f" 1. Swap {a} and {b}: the key moves to box {b if start == a else a}."
    return f" 1. Swap {a} and {b}: the key stays in box {start}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--problems", type=int, default=50, help="per swap count")
    args = parser.parse_args()

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    problems = dataset("swaps", "eval", 0, args.problems * 4, [2, 4, 8, 16], TAGS)
    prompts, keys = [], []
    for problem in problems:
        start = re.search(r"A key is in box (\w)", problem["prompt"]).group(1)
        for name, opening in OPENINGS.items():
            text = problem["prompt"] + opening.format(start=start)
            prompts += [text, text + first_step(problem["prompt"])]
            keys += [(name, "after the opening", problem["level"]), (name, "after step 1", problem["level"])]
    llm = LLM("allenai/Olmo-3-1025-7B", dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096, enable_lora=True,
              max_lora_rank=64)
    params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    out = Path("runs", "newline-probe")
    out.mkdir(parents=True, exist_ok=True)
    lora_id = 0
    for run in args.runs:
        steps = sorted(int(p.name.split("-")[1]) for p in Path("runs", run).glob("checkpoint-*"))
        with open(out / f"{run}.jsonl", "w") as f:
            for step in [0] + [s for s in steps if s % args.every == 0]:
                lora_id += 1
                lora = LoRARequest(f"{run}-{step}", lora_id, f"runs/{run}/checkpoint-{step}") if step else None
                sums = {}
                for key, output in zip(keys, llm.generate(prompts, params, lora_request=lora)):
                    probs = {"newline": 0.0, "space": 0.0}
                    for logprob in output.outputs[0].logprobs[0].values():
                        text = logprob.decoded_token
                        if text.startswith("\n"):
                            probs["newline"] += math.exp(logprob.logprob)
                        elif text.startswith(" "):
                            probs["space"] += math.exp(logprob.logprob)
                    s = sums.setdefault(key, [0.0, 0.0, 0])
                    s[0] += probs["newline"]
                    s[1] += probs["space"]
                    s[2] += 1
                for (opening, point, level), (newline, space, n) in sorted(sums.items()):
                    f.write(json.dumps({"step": step, "opening": opening, "point": point, "level": level,
                                        "newline": newline / n, "space": space / n}) + "\n")
                print(run, step, {k[:2]: round(v[0] / v[2], 3) for k, v in sums.items() if k[2] == 16})


if __name__ == "__main__":
    main()
