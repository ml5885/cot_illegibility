"""Base-model pass@1 / pass@k on a task, with per-problem results for difficulty filtering."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams

from tasks import Problem, dataset, parse_answer, reward

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["swaps", "math"], required=True)
parser.add_argument("--model", default="allenai/Olmo-3-1025-7B")
parser.add_argument("--swaps", type=int, nargs="+", default=[2, 4, 8, 16, 32, 48])
parser.add_argument("--problems", type=int, default=600, help="total, split evenly across swap levels")
parser.add_argument("--k", type=int, default=64)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("runs/baseline"))
args = parser.parse_args()

problems = [Problem(**row) for row in dataset(args.task, args.seed, args.problems, args.swaps)]

llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.9, seed=args.seed)
sampling = SamplingParams(
    n=args.k, temperature=args.temperature, max_tokens=args.max_tokens, stop=["\nProblem:"]
)
start = time.time()
outputs = llm.generate([p.prompt for p in problems], sampling)
elapsed = time.time() - start

levels = defaultdict(lambda: defaultdict(float))
args.out.mkdir(parents=True, exist_ok=True)
with open(args.out / f"{args.task}_{args.model.split('/')[-1]}.jsonl", "w") as f:
    for problem, output in zip(problems, outputs):
        completions = [c.text for c in output.outputs]
        n_correct = sum(reward(c, problem.answer) for c in completions)
        stats = levels[problem.level]
        stats["problems"] += 1
        stats["correct"] += n_correct
        stats["solved"] += n_correct > 0
        stats["parsed"] += sum(parse_answer(c) is not None for c in completions)
        stats["tokens"] += sum(len(c.token_ids) for c in output.outputs)
        f.write(json.dumps({
            "level": problem.level,
            "prompt": problem.prompt,
            "answer": problem.answer,
            "n_correct": n_correct,
            "completions": completions,
        }) + "\n")

for level, s in sorted(levels.items()):
    samples = s["problems"] * args.k
    print(
        f"level={level:3d}  pass@1={s['correct'] / samples:.3f}  "
        f"pass@{args.k}={s['solved'] / s['problems']:.3f}  "
        f"format={s['parsed'] / samples:.3f}  tok/sample={s['tokens'] / samples:.0f}"
    )
print(f"tok/s={sum(s['tokens'] for s in levels.values()) / elapsed:.0f}")
