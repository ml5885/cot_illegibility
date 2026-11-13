"""Outcome-only GRPO with LoRA on the swap or math task."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from rollouts import by_level, score_and_save
from tasks import FORMATS, Format, dataset


def logged_reward(directory: Path, fmt: Format, state: Callable[[], tuple[int, str]], log: Callable[[dict], None]):
    """Reward on the final answer only. `state` returns the current (step, split); `log` gets per-level stats."""

    seen = {"key": None, "rows": []}

    def correct(prompts, completions, completion_ids, answer, level, **kwargs):
        step, split = state()
        if seen["key"] != (step, split):
            seen.update(key=(step, split), rows=[])
        tokens = [len(ids) for ids in completion_ids]
        rows = score_and_save(directory, fmt, split, step, prompts, completions, answer, level, tokens)
        seen["rows"] += rows  # eval arrives in several batches; log the running mean over the whole step
        log({f"{split}_levels/{name}": value for name, value in by_level(seen["rows"]).items()})
        return [row["reward"] for row in rows]

    return correct


def grpo_config(args: argparse.Namespace, out: Path):
    from trl import GRPOConfig

    completions_per_step = args.prompts_per_step * args.generations
    assert completions_per_step % args.batch_size == 0 and args.batch_size % args.generations == 0
    assert args.batch_size % args.eval_generations == 0
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
        num_generations_eval=args.eval_generations,
        max_completion_length=args.max_tokens,
        temperature=args.temperature,
        generation_kwargs={"stop": [FORMATS[args.format].stop], "include_stop_str_in_output": True},
        beta=args.beta,
        loss_type="dapo",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_max_model_length=args.max_context,
        model_init_kwargs={"dtype": "bfloat16"},
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
    parser.add_argument("--format", choices=FORMATS, default="tags", help="think/answer tags or plain text")
    parser.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--swaps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--eval-swaps", type=int, nargs="+", help="held-out swap counts, default --swaps")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--prompts-per-step", type=int, default=32)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16, help="completions per forward pass")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--eval-problems", type=int, default=240)
    parser.add_argument("--eval-generations", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--vllm-memory", type=float, default=0.42)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", help="checkpoint directory to continue training from; raise --max-steps past its step")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    import wandb
    from peft import LoraConfig
    from trl import GRPOTrainer

    fmt = FORMATS[args.format]
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=logged_reward(
            out / "rollouts",
            fmt,
            lambda: (trainer.state.global_step, "train" if trainer.model.training else "eval"),
            lambda stats: wandb.log(stats, commit=False),
        ),
        args=grpo_config(args, out),
        train_dataset=dataset(args.task, "train", args.seed, args.max_steps * args.prompts_per_step, args.swaps, fmt),
        eval_dataset=dataset(args.task, "eval", args.seed, args.eval_problems, args.eval_swaps or args.swaps, fmt),
        peft_config=LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    # vLLM's transformers backend patches the backbone class's __call__ in place to consult this
    # flag, which only its own instances set; the trainer's copy must take the plain forward.
    type(trainer.model.get_base_model().base_model).do_not_compile = True
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out / "final"))


if __name__ == "__main__":
    import os

    os.environ.setdefault("WANDB_PROJECT", "cot-illegibility")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    main()
