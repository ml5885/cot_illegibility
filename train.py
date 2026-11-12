"""Outcome-only GRPO with LoRA on the swap or math task."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from tasks import STOP, dataset, reward


def logged_reward(path: Path, state: Callable[[], tuple[int, str]]):
    """Reward on the final answer only; every sampled completion is appended to `path`."""

    def correct(prompts, completions, answer, level, **kwargs):
        rewards = [reward(c, a) for c, a in zip(completions, answer)]
        step, split = state()
        with open(path, "a") as f:
            for row in zip(prompts, completions, answer, level, rewards):
                f.write(json.dumps(dict(zip(
                    ["step", "split", "prompt", "completion", "answer", "level", "reward"],
                    [step, split, *row],
                ))) + "\n")
        return rewards

    return correct


def grpo_config(args: argparse.Namespace, out: Path):
    from trl import GRPOConfig

    completions_per_step = args.prompts_per_step * args.generations
    assert completions_per_step % args.batch_size == 0 and args.batch_size % args.generations == 0
    return GRPOConfig(
        output_dir=str(out),
        run_name=args.name,
        seed=args.seed,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=completions_per_step // args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_generations=args.generations,
        max_completion_length=args.max_tokens,
        temperature=args.temperature,
        generation_kwargs={"stop": [STOP]},
        beta=args.beta,
        loss_type="dapo",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=0,
        report_to="wandb",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["swaps", "math"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--swaps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--prompts-per-step", type=int, default=32)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16, help="completions per forward pass")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--eval-problems", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--vllm-memory", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main():
    args = parse_args()

    from peft import LoraConfig
    from trl import GRPOTrainer

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=logged_reward(
            out / "rollouts.jsonl",
            lambda: (trainer.state.global_step, "train" if trainer.model.training else "eval"),
        ),
        args=grpo_config(args, out),
        train_dataset=dataset(args.task, "train", args.seed, args.max_steps * args.prompts_per_step, args.swaps),
        eval_dataset=dataset(args.task, "eval", args.seed, args.eval_problems, args.swaps),
        peft_config=LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    trainer.train()
    trainer.save_model(str(out / "final"))


if __name__ == "__main__":
    import os

    os.environ.setdefault("WANDB_PROJECT", "cot-illegibility")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    main()
