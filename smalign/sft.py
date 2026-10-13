"""From-scratch supervised fine-tuning loop.

No Hugging Face ``Trainer`` and no TRL, per the assignment's library boundary:
only Transformers for the model/tokenizer and plain PyTorch for the loop.

Design notes
------------
* Weights stay in fp32 and the forward pass runs under fp16 autocast with a
  ``GradScaler``. That is the combination T4s need: fp16 master weights plus
  gradient scaling silently underflows at this learning rate.
* Loss is response-only (``token_mean`` reduction) - prompt and padding
  positions are masked to ``-100`` by the shared batch builder.
* Training batches pad to the longest row in the batch rather than to the full
  context length, which is a pure speed win and does not change the loss.
* The run is resumable. Example order is a deterministic function of
  ``(seed, epoch)``, and a resume checkpoint is written every
  ``resume_every`` optimizer steps, so a killed session continues from the last
  checkpoint instead of from zero. See :mod:`smalign.checkpointing`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from smalign.batching import build_lm_batch
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
from smalign.objectives import DEFAULT_SFT_REDUCTION, sft_loss_from_batch, token_logprobs
from smalign.prompts import format_prompt, format_response
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

TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    train_path: str = "data/processed/sft_train.json"
    val_path: str = "data/processed/sft_val.json"
    output_dir: str = "outputs/sft"
    max_length: int = 512
    per_device_batch_size: int = 2
    grad_accum_steps: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    epochs: int = 1
    eval_every: int = 100  # optimizer steps
    log_every: int = 10  # optimizer steps
    eval_batch_size: int = 4
    max_val_examples: int = 1000
    reduction: str = DEFAULT_SFT_REDUCTION
    fp16: bool = True
    save_dtype: str = "float16"  # fp16 on disk: six fp32 copies do not fit on Kaggle
    gradient_checkpointing: bool = False
    num_workers: int = 2
    seed: int = DEFAULT_SEED
    device: str = "cuda:0"
    max_train_examples: int = 0  # 0 means "use the whole file"
    resume_every: int = 150  # optimizer steps between resume checkpoints
    allow_resume: bool = True
    max_seconds: float = 0.0  # 0 disables the time budget
    extra: dict = field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.grad_accum_steps


class SFTDataset(Dataset):
    """Instruction/response rows formatted with the shared chat template."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "prompt": format_prompt(row["instruction"]),
            "response": format_response(row["response"]),
        }


def make_collator(tokenizer, max_length: int):
    def collate(samples: list[dict]) -> dict:
        return build_lm_batch(
            tokenizer,
            [sample["prompt"] for sample in samples],
            [sample["response"] for sample in samples],
            max_length=max_length,
            pad_to="longest",
        )

    return collate


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Response-only validation loss, aggregated over tokens and sequences.

    Both reductions are reported because they answer different questions: the
    per-token loss is comparable across batches, the per-sequence loss is the
    quantity the manual's alternative reduction defines.
    """

    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_rows = 0

    # Validation allocates the largest tensor in the whole run; start from a
    # clean allocator rather than whatever training left fragmented.
    release_cuda_memory()

    def score(part: dict) -> tuple[float, int, int]:
        per_token, mask = token_logprobs(
            model, part["input_ids"], part["attention_mask"], part["labels"]
        )
        return (
            float(-per_token.sum().item()),
            int(mask.sum().item()),
            int(part["input_ids"].shape[0]),
        )

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        for nll, tokens, rows in run_with_oom_backoff(score, batch):
            total_nll += nll
            total_tokens += tokens
            total_rows += rows

    if was_training:
        model.train()

    token_mean = total_nll / max(total_tokens, 1)
    return {
        "val_loss": token_mean,
        "val_loss_token_mean": token_mean,
        "val_loss_sequence_mean": total_nll / max(total_rows, 1),
        "val_perplexity": math.exp(min(token_mean, 20.0)),
        "val_tokens": total_tokens,
        "val_rows": total_rows,
    }


def train_sft(config: SFTConfig) -> dict:
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    budget = TimeBudget(config.max_seconds)

    train_rows = read_json(config.train_path)
    val_rows = read_json(config.val_path)
    if config.max_train_examples:
        train_rows = train_rows[: config.max_train_examples]
    val_rows = val_rows[: config.max_val_examples]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(config.model_name, torch.float32)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    collate = make_collator(tokenizer, config.max_length)
    train_dataset = SFTDataset(train_rows)
    val_loader = DataLoader(
        SFTDataset(val_rows),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    batches_per_epoch = len(train_dataset) // config.per_device_batch_size
    steps_per_epoch = batches_per_epoch // config.grad_accum_steps
    total_steps = max(steps_per_epoch * config.epochs, 1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    use_fp16 = config.fp16 and device.type == "cuda"
    scaler = make_grad_scaler(use_fp16)

    # ---- resume -------------------------------------------------------------
    start_epoch = 0
    consumed_examples = 0
    optimizer_step = 0
    best_val = float("inf")

    payload = load_resume_state(output_dir) if config.allow_resume else None
    if payload is not None:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler") is not None and use_fp16:
            scaler.load_state_dict(payload["scaler"])
        restore_rng(payload.get("rng"))
        progress = payload["progress"]
        start_epoch = progress["epoch"]
        consumed_examples = progress["consumed_examples"]
        optimizer_step = progress["optimizer_step"]
        best_val = progress.get("best_val", float("inf"))
        print(
            f"[sft] RESUMING at optimizer step {optimizer_step}/{total_steps} "
            f"(epoch {start_epoch}, {consumed_examples:,} examples consumed, "
            f"best val_loss {best_val:.4f})"
        )
    else:
        print("[sft] starting a fresh run")

    trainable, total = count_parameters(model)
    run_info = {
        "stage": "sft",
        "config": asdict(config),
        "effective_batch_size": config.effective_batch_size,
        "optimizer_steps": total_steps,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "prompt_template": format_prompt("{instruction}"),
        "environment": describe_environment(),
    }
    write_json(output_dir / "run_config.json", run_info)
    print(
        f"[sft] {len(train_rows)} train rows | {total_steps} optimizer steps"
        f" | effective batch {config.effective_batch_size} | fp16={use_fp16}"
        + (f" | budget {config.max_seconds / 3600:.1f}h" if budget.enabled else "")
    )

    timer = Stopwatch()

    def log(record: dict) -> None:
        record["elapsed_s"] = round(timer.elapsed(), 1)
        append_jsonl(metrics_path, record)

    def run_validation(tag: str) -> float:
        nonlocal best_val
        stats = evaluate(model, val_loader, device)
        log({"event": "eval", "step": optimizer_step, "tag": tag, **stats})
        print(
            f"[sft] step {optimizer_step}/{total_steps} "
            f"val_loss {stats['val_loss']:.4f} ppl {stats['val_perplexity']:.2f}",
            flush=True,
        )
        if stats["val_loss"] < best_val:
            best_val = stats["val_loss"]
            save_checkpoint(model, tokenizer, output_dir / "best", {
                "step": optimizer_step,
                "val_loss": best_val,
                "selection": "lowest response-only validation loss",
            }, save_dtype=config.save_dtype)
            print(f"[sft] new best checkpoint at step {optimizer_step} (val_loss {best_val:.4f})")
        return stats["val_loss"]

    def write_resume(epoch: int, consumed: int) -> None:
        path = save_resume_state(
            output_dir, model, optimizer, scheduler, scaler if use_fp16 else None,
            progress={
                "epoch": epoch,
                "consumed_examples": consumed,
                "optimizer_step": optimizer_step,
                "best_val": best_val,
            },
            extra={"stage": "sft", "total_steps": total_steps},
        )
        print(
            f"[sft] resume checkpoint at step {optimizer_step} "
            f"({path.stat().st_size / 2**30:.2f} GiB, {free_space_gib(output_dir):.1f} GiB free)",
            flush=True,
        )

    if payload is None:
        run_validation("initial")

    optimizer.zero_grad(set_to_none=True)
    paused = False
    running_loss = 0.0
    running_batches = 0

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
            pin_memory=(device.type == "cuda"),
        )

        for micro_step, batch in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                loss, _ = sft_loss_from_batch(model, batch, reduction=config.reduction)

            scaler.scale(loss / config.grad_accum_steps).backward()
            running_loss += float(loss.item())
            running_batches += 1

            if (micro_step + 1) % config.grad_accum_steps != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1
            consumed_examples = skip + (micro_step + 1) * config.per_device_batch_size

            if optimizer_step % config.log_every == 0:
                mean_loss = running_loss / max(running_batches, 1)
                log({
                    "event": "train",
                    "step": optimizer_step,
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                })
                print(
                    f"[sft] step {optimizer_step}/{total_steps} "
                    f"loss {mean_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                    f"| {timer.human()} {format_eta(optimizer_step, total_steps, timer.elapsed())}",
                    flush=True,
                )
                running_loss = 0.0
                running_batches = 0

            if config.eval_every and optimizer_step % config.eval_every == 0:
                run_validation("periodic")

            # Checkpoints are written only here, right after an optimizer step,
            # so a resumed run never lands mid-accumulation.
            if config.resume_every and optimizer_step % config.resume_every == 0:
                write_resume(epoch, consumed_examples)

            if budget.exhausted():
                print(
                    f"[sft] time budget reached after {timer.human()} "
                    f"at step {optimizer_step}/{total_steps} - saving and pausing",
                    flush=True,
                )
                write_resume(epoch, consumed_examples)
                paused = True
                break

        if paused:
            break
        consumed_examples = 0  # next epoch starts from the top of its own order

    if paused:
        summary = {
            "stage": "sft",
            "status": "paused",
            "optimizer_steps": optimizer_step,
            "total_steps": total_steps,
            "progress": round(optimizer_step / max(total_steps, 1), 4),
            "best_val_loss": best_val,
            "wall_clock": timer.human(),
            "resume_file": str(output_dir / "resume.pt"),
        }
        write_json(output_dir / "summary.json", summary)
        print(f"[sft] PAUSED at {summary['progress']:.1%}; rerun this stage to continue")
        return summary

    final_val = run_validation("final")
    save_checkpoint(model, tokenizer, output_dir / "final", {
        "step": optimizer_step,
        "val_loss": final_val,
        "selection": "last optimizer step",
    }, save_dtype=config.save_dtype)

    # The run finished, so the multi-GiB resume file is dead weight.
    clear_resume_state(output_dir)

    summary = {
        "stage": "sft",
        "status": "completed",
        "optimizer_steps": optimizer_step,
        "best_val_loss": best_val,
        "final_val_loss": final_val,
        "wall_clock": timer.human(),
        "best_checkpoint": str(output_dir / "best"),
        "final_checkpoint": str(output_dir / "final"),
        "reduction": config.reduction,
        "output_size_gib": round(directory_size_gib(output_dir), 2),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[sft] done in {timer.human()} | best val_loss {best_val:.4f}")
    return summary


def save_checkpoint(model, tokenizer, path, metadata: dict, save_dtype: str = "float16") -> None:
    """Write a checkpoint plus a small note about why it was kept.

    Only ``best`` and ``final`` are ever written. Even so, six fp32 copies of a
    0.5B model across the three training runs is roughly 12 GB, which crowds
    Kaggle's 20 GB working disk, so checkpoints are cast to fp16 on the way out.
    The cast is applied to a copy of the state dict - the live training weights
    stay fp32, which is what gradient scaling requires.

    A DPO run then loads this file for *both* the policy and the reference, so
    the two still start numerically identical and the initial reward margin is
    still exactly zero.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    dtype = TORCH_DTYPES.get(save_dtype)
    if dtype is not None and dtype != torch.float32:
        state_dict = {
            key: (value.detach().to(dtype) if value.is_floating_point() else value.detach())
            for key, value in model.state_dict().items()
        }
        model.save_pretrained(path, state_dict=state_dict, safe_serialization=True)
    else:
        model.save_pretrained(path, safe_serialization=True)

    tokenizer.save_pretrained(path)
    write_json(path / "checkpoint_info.json", {**metadata, "saved_dtype": save_dtype})
