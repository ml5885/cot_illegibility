"""Outcome-only GRPO with LoRA on the swap or math task."""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from transformers import TrainerCallback

from datasets import Dataset

from rollouts import by_level, score_and_save
from tasks import FORMATS, Format, dataset


def logged_reward(
    directory: Path, fmt: Format, state: Callable[[], tuple[int, str]], log: Callable[[dict], None],
    record: Callable[[str, int, list[dict]], None],
):
    """Reward on the final answer only. `state` returns the current (step, split); `log` gets per-level stats;
    `record` gets the scored rows."""

    seen = {"key": None, "rows": []}

    def correct(prompts, completions, completion_ids, answer, level, **kwargs):
        step, split = state()
        if seen["key"] != (step, split):
            seen.update(key=(step, split), rows=[])
        tokens = [len(ids) for ids in completion_ids]
        rows = score_and_save(directory, fmt, split, step, prompts, completions, answer, level, tokens)
        record(split, step, rows)
        seen["rows"] += rows  # eval arrives in several batches; log the running mean over the whole step
        log({f"{split}_levels/{name}": value for name, value in by_level(seen["rows"]).items()})
        return [row["reward"] for row in rows]

    return correct


def fixed_mix(data: Dataset, per_step: int, seed: int) -> Dataset:
    """Orders the puzzles so that every step's prompts hold the same number at each swap count, shuffled within the step.
    The trainer must then take the dataset in order (shuffle_dataset=False)."""
    rng = random.Random(seed)
    by_level = defaultdict(list)
    for i, level in enumerate(data["level"]):
        by_level[level].append(i)
    share, rest = divmod(per_step, len(by_level))
    assert not rest, "prompts per step must split evenly across swap counts"
    for indices in by_level.values():
        rng.shuffle(indices)
    order = []
    for step in range(len(data) // per_step):
        block = [i for indices in by_level.values() for i in indices[step * share : (step + 1) * share]]
        rng.shuffle(block)
        order += block
    return data.select(order)


class LimitSchedule(TrainerCallback):
    """Lowers the token limit by `factor`, down to `floor`, whenever training accuracy at every swap count has been at
    least `target` over the last `patience` steps at the current limit. Changes are appended to `path` as JSON lines."""

    def __init__(self, trainer, floor: int, factor: float, target: float, patience: int, path: Path, log: Callable[[dict], None]):
        self.trainer, self.floor, self.factor, self.target, self.patience, self.path = trainer, floor, factor, target, patience, path
        self.log = log
        self.rewards = defaultdict(lambda: defaultdict(list))  # step -> level -> rewards
        self.since = 0  # first step generated at the current limit
        self.write(0, trainer.max_completion_length)

    def record(self, split: str, step: int, rows: list[dict]):
        if split == "train":
            for row in rows:
                self.rewards[step][row["level"]].append(row["reward"])

    def write(self, step: int, limit: int):
        with self.path.open("a") as f:
            f.write(json.dumps({"step": step, "limit": limit}) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        step, limit = state.global_step, self.trainer.max_completion_length
        self.log({"token_limit": limit})
        window = range(step - self.patience, step)
        if limit <= self.floor or window.start < self.since:
            return
        pooled = defaultdict(list)
        for s in window:
            for level, rewards in self.rewards[s].items():
                pooled[level] += rewards
        if all(sum(r) / len(r) >= self.target for r in pooled.values()):
            limit = max(self.floor, round(limit * self.factor))
            self.trainer.max_completion_length = self.trainer.vllm_generation.max_completion_length = limit
            self.since = step
            self.write(step, limit)


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
        shuffle_dataset=args.mix == "random",
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
    parser.add_argument("--mix", choices=["fixed", "random"], default="fixed",
                        help="fixed: every step has the same number of puzzles at each swap count; random: shuffled")
    parser.add_argument("--min-tokens", type=int, help="lower the limit from --max-tokens toward this (see LimitSchedule)")
    parser.add_argument("--limit-factor", type=float, default=0.9)
    parser.add_argument("--limit-target", type=float, default=0.9, help="training accuracy needed at every swap count")
    parser.add_argument("--limit-patience", type=int, default=10, help="steps at that accuracy before each cut")
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

    train_data = dataset(args.task, "train", args.seed, args.max_steps * args.prompts_per_step, args.swaps, fmt)
    if args.mix == "fixed":
        train_data = fixed_mix(train_data, args.prompts_per_step, args.seed)
    schedule = None
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=logged_reward(
            out / "rollouts",
            fmt,
            lambda: (trainer.state.global_step, "train" if trainer.model.training else "eval"),
            lambda stats: wandb.log(stats, commit=False),
            lambda split, step, rows: schedule and schedule.record(split, step, rows),
        ),
        args=grpo_config(args, out),
        train_dataset=train_data,
        eval_dataset=dataset(args.task, "eval", args.seed, args.eval_problems, args.eval_swaps or args.swaps, fmt),
        peft_config=LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    # vLLM's transformers backend patches the backbone class's __call__ in place to consult this
    # flag, which only its own instances set; the trainer's copy must take the plain forward.
    type(trainer.model.get_base_model().base_model).do_not_compile = True
    if args.min_tokens:
        assert not args.resume, "the schedule's state is not saved in checkpoints"
        (out / "limits.jsonl").unlink(missing_ok=True)
        schedule = LimitSchedule(trainer, args.min_tokens, args.limit_factor, args.limit_target, args.limit_patience, out / "limits.jsonl",
                                  lambda stats: wandb.log(stats, commit=False))
        trainer.add_callback(schedule)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out / "final"))


if __name__ == "__main__":
    import os

    os.environ.setdefault("WANDB_PROJECT", "cot-illegibility")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    main()
