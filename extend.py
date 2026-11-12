"""Continue the rollouts of a run that were cut off at its token limit, to see how many would have been rewarded."""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tasks import FORMATS, normalize

# The system nvcc is too old to JIT-compile FlashInfer's sampler.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

# The reward rule in force before the tags format required exactly one think block and one answer.
ORIGINAL = re.compile(r"</think>\s*<answer>(.*?)</answer>", re.DOTALL)

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--steps", type=int, nargs=2, required=True, help="first and last training step to take rollouts from")
parser.add_argument("--adapter", help="LoRA checkpoint that sampled those steps; omit for the base model")
parser.add_argument("--max-tokens", type=int, default=2048, help="total budget, including the tokens already written")
parser.add_argument("--k", type=int, default=8, help="continuations per cut-off rollout")
args = parser.parse_args()

run = json.loads(Path("runs", args.run, "args.json").read_text())
fmt = FORMATS[run["format"]]
rows = [
    json.loads(line) | {"step": step}
    for step in range(args.steps[0], args.steps[1] + 1)
    for line in Path("runs", args.run, "rollouts", f"train_{step:05d}.jsonl").read_text().splitlines()
]
cut = [row for row in rows if row["tokens"] >= run["max_tokens"] - 2 and not row["reward"]]

llm = LLM(
    run["model"], dtype="bfloat16", gpu_memory_utilization=0.9, seed=0, max_model_len=4096,
    enable_lora=bool(args.adapter), max_lora_rank=64,
)
sampling = SamplingParams(
    n=args.k, temperature=run["temperature"], max_tokens=args.max_tokens - run["max_tokens"],
    stop=[fmt.stop], include_stop_str_in_output=True,
)
lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
outputs = llm.generate([row["prompt"] + row["completion"] for row in cut], sampling, lora_request=lora)

out = Path("runs", args.run, "extended", f"steps_{args.steps[0]:05d}_{args.steps[1]:05d}.jsonl")
out.parent.mkdir(exist_ok=True)
flips = defaultdict(float)
with out.open("w") as file:
    for row, output in zip(cut, outputs):
        completions = [row["completion"] + c.text for c in output.outputs]
        original = [
            float((m := ORIGINAL.search(c)) is not None and normalize(m.group(1)) == normalize(row["answer"])) for c in completions
        ]
        flips[row["level"]] += sum(original) / args.k
        record = row | {
            "continuations": [c.text for c in output.outputs],
            "tokens_added": [len(c.token_ids) for c in output.outputs],
            "reward_original_rule": original,
            "reward_strict_rule": [fmt.reward(c, row["answer"]) for c in completions],
        }
        file.write(json.dumps(record) + "\n")

for level in sorted({row["level"] for row in rows}):
    total, capped = sum(r["level"] == level for r in rows), sum(r["level"] == level for r in cut)
    print(
        f"level={level:3d}  rollouts={total}  cut_off={capped}  "
        f"would_be_rewarded={flips[level]:.1f}  share_of_all_rollouts={flips[level] / total:.4f}"
    )
