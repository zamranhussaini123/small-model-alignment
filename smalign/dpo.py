"""From-scratch Direct Preference Optimization loop.

Used unchanged for both preference datasets (HH-RLHF and UltraFeedback); only
the data paths and the run directory differ.

Device layout on Kaggle T4x2, as the manual suggests: the trainable policy sits
on ``cuda:0`` in fp32 (gradient scaling needs fp32 master weights) and the
frozen reference sits on ``cuda:1`` in fp16. With one GPU both share it and the
reference is still fp16 and still under ``no_grad``.

Checkpoint selection is by validation *reward accuracy*, not validation loss,
because that is the quantity the manual defines as the preference diagnostic.

Like SFT, the run is resumable: example order is deterministic in
``(seed, epoch)`` and a resume checkpoint is written every ``resume_every``
optimizer steps. See :mod:`smalign.checkpointing`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from smalign.batching import build_preference_batch
from smalign.checkpointing import (
    TimeBudget,
    clear_resume_state,
    directory_size_gib,
    epoch_order,
    format_eta,
    free_space_gib,
    load_resume_state,
    restore_rng,
    save_resume_state,
)
from smalign.objectives import dpo_logps_from_batches, dpo_loss_from_logps
from smalign.prompts import format_prompt, format_response
from smalign.sft import save_checkpoint
from smalign.utils import (
    DEFAULT_SEED,
    Stopwatch,
    append_jsonl,
    count_parameters,
    describe_environment,
    make_grad_scaler,
    load_causal_lm,
    read_json,
    release_cuda_memory,
    run_with_oom_backoff,
    set_seed,
    write_json,
)


@dataclass
class DPOConfig:
    run_name: str = "dpo"
    policy_path: str = "outputs/sft/best"
    reference_path: str = ""  # defaults to policy_path: a frozen copy of the SFT checkpoint
    train_path: str = "data/processed/hh_train.json"
    val_path: str = "data/processed/hh_val.json"
    output_dir: str = "outputs/dpo_hh"
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 256
    per_device_batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float = 1e-6
    optimizer: str = "rmsprop"  # "rmsprop" (DPO paper) or "adamw"
    lr_schedule: str = "constant"  # the manual's default is no decay for DPO
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    epochs: int = 1
    eval_every: int = 50  # optimizer steps
    log_every: int = 5
    eval_batch_size: int = 4
    eval_subset: int = 300  # periodic evals use a fixed slice; the final eval uses everything
    max_val_examples: int = 1000
    fp16: bool = True
    save_dtype: str = "float16"
    policy_device: str = "cuda:0"
    reference_device: str = "cuda:1"
    num_workers: int = 2
    seed: int = DEFAULT_SEED
    max_train_examples: int = 0
    resume_every: int = 100  # optimizer steps between resume checkpoints
    allow_resume: bool = True
    max_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.grad_accum_steps


class PreferenceDataset(Dataset):
    """Prompt / chosen / rejected rows under the shared chat template."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "prompt": format_prompt(row["instruction"]),
            "chosen": format_response(row["chosen"]),
            "rejected": format_response(row["rejected"]),
        }


def make_collator(tokenizer, max_length: int, max_prompt_length: int):
    def collate(samples: list[dict]) -> dict:
        return build_preference_batch(
            tokenizer,
            [sample["prompt"] for sample in samples],
            [sample["chosen"] for sample in samples],
            [sample["rejected"] for sample in samples],
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            pad_to="longest",
        )

    return collate


def resolve_devices(config: DPOConfig) -> tuple[torch.device, torch.device]:
    """Policy and reference devices, degrading gracefully below two GPUs."""

    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    count = torch.cuda.device_count()
    policy = torch.device(config.policy_device if count >= 1 else "cpu")
    if count >= 2:
        reference = torch.device(config.reference_device)
    else:
        reference = policy
        print("[dpo] only one GPU visible: the frozen reference shares it in fp16")
    return policy, reference


@torch.no_grad()
def evaluate(policy_model, reference_model, loader, beta: float, policy_device) -> dict:
    """Validation DPO loss, reward accuracy, and reward margin."""

    was_training = policy_model.training
    policy_model.eval()

    total_loss = 0.0
    correct = 0
    margin_sum = 0.0
    chosen_reward_sum = 0.0
    rejected_reward_sum = 0.0
    rows = 0

    # Scoring four sequences per pair against a 152k vocabulary is the biggest
    # allocation in a DPO run; clear the allocator before asking for it.
    release_cuda_memory()

    def score(part: dict) -> tuple[float, int, float, float, float, int]:
        logps = dpo_logps_from_batches(
            policy_model, reference_model, part, policy_device=policy_device
        )
        policy_chosen, policy_rejected, reference_chosen, reference_rejected = logps
        loss, _ = dpo_loss_from_logps(
            policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=beta
        )
        batch_rows = policy_chosen.shape[0]
        chosen_rewards = beta * (policy_chosen - reference_chosen)
        rejected_rewards = beta * (policy_rejected - reference_rejected)
        return (
            float(loss.item()) * batch_rows,
            int((chosen_rewards > rejected_rewards).sum().item()),
            float((chosen_rewards - rejected_rewards).sum().item()),
            float(chosen_rewards.sum().item()),
            float(rejected_rewards.sum().item()),
            batch_rows,
        )

    for batches in loader:
        for part in run_with_oom_backoff(score, batches):
            loss_sum, hits, margin, chosen_sum, rejected_sum, batch_rows = part
            total_loss += loss_sum
            correct += hits
            margin_sum += margin
            chosen_reward_sum += chosen_sum
            rejected_reward_sum += rejected_sum
            rows += batch_rows

    if was_training:
        policy_model.train()

    denominator = max(rows, 1)
    return {
        "val_loss": total_loss / denominator,
        "val_reward_accuracy": correct / denominator,
        "val_reward_margin": margin_sum / denominator,
        "val_chosen_reward": chosen_reward_sum / denominator,
        "val_rejected_reward": rejected_reward_sum / denominator,
        "val_pairs": rows,
    }


def train_dpo(config: DPOConfig) -> dict:
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    policy_device, reference_device = resolve_devices(config)
    reference_path = config.reference_path or config.policy_path
    budget = TimeBudget(config.max_seconds)

    train_rows = read_json(config.train_path)
    val_rows = read_json(config.val_path)
    if config.max_train_examples:
        train_rows = train_rows[: config.max_train_examples]
    val_rows = val_rows[: config.max_val_examples]

    tokenizer = AutoTokenizer.from_pretrained(config.policy_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = load_causal_lm(config.policy_path, torch.float32)
    policy_model.config.use_cache = False
    policy_model.to(policy_device)
    policy_model.train()

    # The reference matches the policy's precision whenever it has a GPU to
    # itself. That is what makes the manual's invariant hold exactly: with
    # identical weights *and* identical arithmetic, every initial reward pair
    # ties, so the first evaluation reports loss log 2, margin 0, and strict
    # reward accuracy 0.
    #
    # An fp16 reference is close but not equal - autocast keeps some ops in
    # fp32 for an fp32 model and in fp16 for an fp16 one - which showed up as a
    # spurious initial margin of ~5e-4 and an initial accuracy near 0.4.
    # fp16 is kept only for the single-GPU fallback, where the reference has to
    # share memory with the policy and its optimizer state.
    if reference_device == policy_device and policy_device.type == "cuda":
        reference_dtype = torch.float16
        print(f"[{config.run_name}] single GPU: reference in fp16 to save memory "
              "(expect a small non-zero initial reward margin)")
    else:
        reference_dtype = torch.float32
    reference_model = load_causal_lm(reference_path, reference_dtype)
    reference_model.config.use_cache = False
    reference_model.to(reference_device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    collate = make_collator(tokenizer, config.max_length, config.max_prompt_length)
    train_dataset = PreferenceDataset(train_rows)

    def make_val_loader(rows):
        return DataLoader(
            PreferenceDataset(rows),
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )

    quick_val_loader = make_val_loader(val_rows[: config.eval_subset or len(val_rows)])
    full_val_loader = make_val_loader(val_rows)

    batches_per_epoch = len(train_dataset) // config.per_device_batch_size
    steps_per_epoch = batches_per_epoch // config.grad_accum_steps
    total_steps = max(steps_per_epoch * config.epochs, 1)

    if config.optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=config.learning_rate)
    elif config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"unsupported optimizer: {config.optimizer}")

    if config.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
    else:
        scheduler = None

    use_fp16 = config.fp16 and policy_device.type == "cuda"
    scaler = make_grad_scaler(use_fp16)

    # ---- resume -------------------------------------------------------------
    start_epoch = 0
    consumed_examples = 0
    optimizer_step = 0
    best_accuracy = -1.0

    payload = load_resume_state(output_dir) if config.allow_resume else None
    if payload is not None:
        policy_model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None and scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler") is not None and use_fp16:
            scaler.load_state_dict(payload["scaler"])
        restore_rng(payload.get("rng"))
        progress = payload["progress"]
        start_epoch = progress["epoch"]
        consumed_examples = progress["consumed_examples"]
        optimizer_step = progress["optimizer_step"]
        best_accuracy = progress.get("best_accuracy", -1.0)
        print(
            f"[{config.run_name}] RESUMING at optimizer step {optimizer_step}/{total_steps} "
            f"(epoch {start_epoch}, {consumed_examples:,} pairs consumed, "
            f"best reward acc {best_accuracy:.3f})"
        )
    else:
        print(f"[{config.run_name}] starting a fresh run")

    trainable, total = count_parameters(policy_model)
    run_info = {
        "stage": "dpo",
        "run_name": config.run_name,
        "config": asdict(config),
        "reference_checkpoint": reference_path,
        "policy_device": str(policy_device),
        "reference_device": str(reference_device),
        "reference_dtype": str(reference_dtype),
        "effective_batch_size": config.effective_batch_size,
        "optimizer_steps": total_steps,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "prompt_template": format_prompt("{instruction}"),
        "environment": describe_environment(),
    }
    write_json(output_dir / "run_config.json", run_info)
    print(
        f"[{config.run_name}] {len(train_rows)} pairs | {total_steps} optimizer steps"
        f" | beta {config.beta} | policy {policy_device} | reference {reference_device}"
        + (f" | budget {config.max_seconds / 3600:.1f}h" if budget.enabled else "")
    )

    timer = Stopwatch()
    running = {"loss": 0.0, "accuracy": 0.0, "margin": 0.0, "n": 0}

    def log(record: dict) -> None:
        record["elapsed_s"] = round(timer.elapsed(), 1)
        append_jsonl(metrics_path, record)

    def run_validation(tag: str, loader) -> dict:
        nonlocal best_accuracy
        stats = evaluate(policy_model, reference_model, loader, config.beta, policy_device)
        log({"event": "eval", "step": optimizer_step, "tag": tag, **stats})
        print(
            f"[{config.run_name}] step {optimizer_step}/{total_steps} "
            f"val_loss {stats['val_loss']:.4f} "
            f"acc {stats['val_reward_accuracy']:.3f} "
            f"margin {stats['val_reward_margin']:+.4f}",
            flush=True,
        )
        if tag != "initial" and stats["val_reward_accuracy"] > best_accuracy:
            best_accuracy = stats["val_reward_accuracy"]
            save_checkpoint(policy_model, tokenizer, output_dir / "best", {
                "step": optimizer_step,
                "val_reward_accuracy": best_accuracy,
                "val_reward_margin": stats["val_reward_margin"],
                "val_loss": stats["val_loss"],
                "selection": "highest validation reward accuracy",
            }, save_dtype=config.save_dtype)
            print(f"[{config.run_name}] new best checkpoint (reward acc {best_accuracy:.3f})")
        return stats

    def write_resume(epoch: int, consumed: int) -> None:
        path = save_resume_state(
            output_dir, policy_model, optimizer, scheduler, scaler if use_fp16 else None,
            progress={
                "epoch": epoch,
                "consumed_examples": consumed,
                "optimizer_step": optimizer_step,
                "best_accuracy": best_accuracy,
            },
            extra={"stage": "dpo", "run_name": config.run_name, "total_steps": total_steps},
        )
        print(
            f"[{config.run_name}] resume checkpoint at step {optimizer_step} "
            f"({path.stat().st_size / 2**30:.2f} GiB, {free_space_gib(output_dir):.1f} GiB free)",
            flush=True,
        )

    # The initial evaluation documents the manual's observation that an
    # untouched policy ties with its reference, so strict reward accuracy is 0.
    if payload is None:
        run_validation("initial", quick_val_loader)

    optimizer.zero_grad(set_to_none=True)
    paused = False

    for epoch in range(start_epoch, config.epochs):
        order = epoch_order(len(train_dataset), config.seed, epoch)
        skip = consumed_examples if epoch == start_epoch else 0
        remaining = order[skip:]
        if len(remaining) < config.per_device_batch_size:
            consumed_examples = 0
            continue

        loader = DataLoader(
            Subset(train_dataset, remaining),
            batch_size=config.per_device_batch_size,
            shuffle=False,  # order already fixed by epoch_order
            collate_fn=collate,
            num_workers=config.num_workers,
            drop_last=True,
            pin_memory=False,
        )

        for micro_step, batches in enumerate(loader):
            with torch.autocast(
                device_type=policy_device.type, dtype=torch.float16, enabled=use_fp16
            ):
                logps = dpo_logps_from_batches(
                    policy_model, reference_model, batches, policy_device=policy_device
                )
                loss, metrics = dpo_loss_from_logps(*logps, beta=config.beta)

            scaler.scale(loss / config.grad_accum_steps).backward()
            running["loss"] += metrics["loss"]
            running["accuracy"] += metrics["reward_accuracy"]
            running["margin"] += metrics["reward_margin"]
            running["n"] += 1

            if (micro_step + 1) % config.grad_accum_steps != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), config.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            optimizer_step += 1
            consumed_examples = skip + (micro_step + 1) * config.per_device_batch_size

            if optimizer_step % config.log_every == 0:
                count = max(running["n"], 1)
                record = {
                    "event": "train",
                    "step": optimizer_step,
                    "epoch": epoch,
                    "train_loss": running["loss"] / count,
                    "train_reward_accuracy": running["accuracy"] / count,
                    "train_reward_margin": running["margin"] / count,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                }
                log(record)
                print(
                    f"[{config.run_name}] step {optimizer_step}/{total_steps} "
                    f"loss {record['train_loss']:.4f} "
                    f"acc {record['train_reward_accuracy']:.2f} "
                    f"| {timer.human()} "
                    f"{format_eta(optimizer_step, total_steps, timer.elapsed())}",
                    flush=True,
                )
                running = {"loss": 0.0, "accuracy": 0.0, "margin": 0.0, "n": 0}

            if config.eval_every and optimizer_step % config.eval_every == 0:
                run_validation("periodic", quick_val_loader)

            if config.resume_every and optimizer_step % config.resume_every == 0:
                write_resume(epoch, consumed_examples)

            if budget.exhausted():
                print(
                    f"[{config.run_name}] time budget reached after {timer.human()} "
                    f"at step {optimizer_step}/{total_steps} - saving and pausing",
                    flush=True,
                )
                write_resume(epoch, consumed_examples)
                paused = True
                break

        if paused:
            break
        consumed_examples = 0

    if paused:
        summary = {
            "stage": "dpo",
            "run_name": config.run_name,
            "status": "paused",
            "optimizer_steps": optimizer_step,
            "total_steps": total_steps,
            "progress": round(optimizer_step / max(total_steps, 1), 4),
            "best_val_reward_accuracy": best_accuracy,
            "wall_clock": timer.human(),
            "resume_file": str(output_dir / "resume.pt"),
        }
        write_json(output_dir / "summary.json", summary)
        print(f"[{config.run_name}] PAUSED at {summary['progress']:.1%}; rerun to continue")
        return summary

    # Checkpoint selection stays on the same validation slice used for every
    # periodic evaluation, so the accuracies being compared are comparable.
    final_stats = run_validation("final", quick_val_loader)
    save_checkpoint(policy_model, tokenizer, output_dir / "final", {
        "step": optimizer_step,
        "val_reward_accuracy": final_stats["val_reward_accuracy"],
        "val_reward_margin": final_stats["val_reward_margin"],
        "selection": "last optimizer step",
    }, save_dtype=config.save_dtype)

    # The full validation set is scored once, for the report only.
    full_stats = evaluate(
        policy_model, reference_model, full_val_loader, config.beta, policy_device
    )
    log({"event": "eval", "step": optimizer_step, "tag": "final_full_validation", **full_stats})
    print(
        f"[{config.run_name}] full validation ({full_stats['val_pairs']} pairs): "
        f"loss {full_stats['val_loss']:.4f} "
        f"acc {full_stats['val_reward_accuracy']:.3f} "
        f"margin {full_stats['val_reward_margin']:+.4f}"
    )

    clear_resume_state(output_dir)

    summary = {
        "stage": "dpo",
        "run_name": config.run_name,
        "status": "completed",
        "optimizer_steps": optimizer_step,
        "beta": config.beta,
        "selection_val_pairs": final_stats["val_pairs"],
        "best_val_reward_accuracy": max(best_accuracy, final_stats["val_reward_accuracy"]),
        "final_val_reward_accuracy": final_stats["val_reward_accuracy"],
        "final_val_reward_margin": final_stats["val_reward_margin"],
        "final_val_loss": final_stats["val_loss"],
        "full_val_pairs": full_stats["val_pairs"],
        "full_val_reward_accuracy": full_stats["val_reward_accuracy"],
        "full_val_reward_margin": full_stats["val_reward_margin"],
        "full_val_loss": full_stats["val_loss"],
        "reference_checkpoint": reference_path,
        "best_checkpoint": str(output_dir / "best"),
        "final_checkpoint": str(output_dir / "final"),
        "wall_clock": timer.human(),
        "output_size_gib": round(directory_size_gib(output_dir), 2),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[{config.run_name}] done in {timer.human()}")
    return summary
