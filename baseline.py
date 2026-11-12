"""Base-model pass@1 / pass@k on a task, per difficulty level."""

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from rollouts import score_and_save
from tasks import FORMATS, dataset

# The system nvcc is too old to JIT-compile FlashInfer's sampler.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["swaps", "math"], required=True)
parser.add_argument("--format", choices=FORMATS, default="tags")
parser.add_argument("--split", choices=["train", "eval"], default="train")
parser.add_argument("--model", default="allenai/Olmo-3-1025-7B")
parser.add_argument("--swaps", type=int, nargs="+", default=[2, 4, 8, 16, 32, 48])
parser.add_argument("--problems", type=int, default=600, help="total, split evenly across swap levels")
parser.add_argument("--k", type=int, default=64)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

fmt = FORMATS[args.format]
problems = dataset(args.task, args.split, args.seed, args.problems, args.swaps, fmt)
out = Path("runs") / f"baseline-{args.task}-{args.format}-{args.model.split('/')[-1]}"
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
(out / "args.json").write_text(json.dumps(vars(args), indent=2))

llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=0.9, seed=args.seed)
sampling = SamplingParams(
    n=args.k,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    stop=[fmt.stop],
    include_stop_str_in_output=True,
)
start = time.time()
outputs = llm.generate(problems["prompt"], sampling)
elapsed = time.time() - start

levels = defaultdict(lambda: defaultdict(float))
for problem, output in zip(problems, outputs):
    completions = [c.text for c in output.outputs]
    rewards = score_and_save(
        out / "rollouts", fmt, "baseline", 0,
        [problem["prompt"]] * args.k, completions, [problem["answer"]] * args.k, [problem["level"]] * args.k,
    )
    stats = levels[problem["level"]]
    stats["problems"] += 1
    stats["correct"] += sum(rewards)
    stats["solved"] += any(rewards)
    stats["parsed"] += sum(fmt.parse_answer(c) is not None for c in completions)
    stats["tokens"] += sum(len(c.token_ids) for c in output.outputs)

for level, s in sorted(levels.items()):
    samples = s["problems"] * args.k
    print(
        f"level={level:3d}  pass@1={s['correct'] / samples:.3f}  "
        f"pass@{args.k}={s['solved'] / s['problems']:.3f}  "
        f"format={s['parsed'] / samples:.3f}  tok/sample={s['tokens'] / samples:.0f}"
    )
print(f"tok/s={sum(s['tokens'] for s in levels.values()) / elapsed:.0f}")
