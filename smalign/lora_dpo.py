"""Task 5: TRL ``DPOTrainer`` + PEFT/LoRA persona tuning.

This is the one track where TRL and PEFT are allowed. Two things make it worth
keeping separate from the from-scratch loops:

* no explicit reference model is constructed - with a PEFT policy, TRL obtains
  reference log probabilities by disabling the adapters, which halves memory;
* TRL's constructor keywords have moved around across releases (``tokenizer``
  became ``processing_class``, ``beta``/``max_length`` moved from the trainer to
  ``DPOConfig``). Rather than pinning one version, the builders below inspect
  the installed signatures and pass only what they accept.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from smalign.utils import (
    DEFAULT_SEED,
    Stopwatch,
    count_parameters,
    describe_environment,
    load_causal_lm,
    read_json,
    set_seed,
    write_json,
)


@dataclass
class LoRADPOConfig:
    run_name: str = "lora_dpo_persona"
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    train_path: str = "data/processed/persona_train.json"
    val_path: str = "data/processed/persona_val.json"
    output_dir: str = "outputs/lora_dpo_persona"
    beta: float = 0.1
    # Measured on the provided persona set: prompts reach ~33 tokens and
    # responses ~60, so 320/128 truncates nothing and wastes no compute.
    max_length: int = 320
    max_prompt_length: int = 128
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    epochs: int = 3
    logging_steps: int = 5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    fp16: bool = True
    seed: int = DEFAULT_SEED
    extra: dict = field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.grad_accum_steps


def to_conversational(rows: list[dict]) -> list[dict]:
    """TRL's conversational DPO shape: role/content lists on all three fields."""

    return [
        {
            "prompt": [{"role": "user", "content": row["instruction"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }
        for row in rows
    ]


def _supported_kwargs(target, candidates: dict) -> tuple[dict, list[str]]:
    """Filter ``candidates`` down to the keywords ``target`` actually accepts."""

    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return dict(candidates), []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(candidates), []
    accepted = {key: value for key, value in candidates.items() if key in parameters}
    dropped = sorted(set(candidates) - set(accepted))
    return accepted, dropped


def build_dpo_config(config: LoRADPOConfig):
    """A TRL ``DPOConfig`` populated with whatever this TRL version supports."""

    from trl import DPOConfig as TRLDPOConfig

    candidates = {
        "output_dir": str(Path(config.output_dir) / "trainer"),
        "beta": config.beta,
        "max_length": config.max_length,
        "max_prompt_length": config.max_prompt_length,
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "num_train_epochs": config.epochs,
        "logging_steps": config.logging_steps,
        "save_strategy": "no",
        "report_to": [],
        "fp16": config.fp16 and torch.cuda.is_available(),
        "bf16": False,
        "seed": config.seed,
        "remove_unused_columns": False,
    }
    kwargs, dropped = _supported_kwargs(TRLDPOConfig.__init__, candidates)
    if dropped:
        print(f"[lora-dpo] TRL DPOConfig ignores: {', '.join(dropped)}")
    return TRLDPOConfig(**kwargs), dropped


def build_lora_config(config: LoRADPOConfig):
    """LoRA on every attention and MLP projection of the Llama-style backbone.

    Rank 16 with alpha 32 is deliberately modest: the persona set is ~200 rows,
    and a larger rank mostly adds variance. Covering the MLP projections as well
    as attention matters for a *style* target, because tone shows up in token
    choice, not only in what the model attends to.
    """

    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_trainer(config: LoRADPOConfig, model, tokenizer, train_dataset, eval_dataset, peft_config):
    """Construct ``DPOTrainer`` across TRL versions."""

    from trl import DPOTrainer

    trl_config, _ = build_dpo_config(config)
    candidates = {
        "model": model,
        "ref_model": None,  # PEFT path: the adapter is disabled to score the reference
        "args": trl_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "beta": config.beta,
        "max_length": config.max_length,
        "max_prompt_length": config.max_prompt_length,
    }
    kwargs, _ = _supported_kwargs(DPOTrainer.__init__, candidates)
    # Newer TRL takes processing_class; older TRL takes tokenizer. Never both.
    if "processing_class" in kwargs:
        kwargs.pop("tokenizer", None)
    # beta/max_length live on DPOConfig in modern TRL; drop the duplicates.
    for key in ("beta", "max_length", "max_prompt_length"):
        if key in kwargs and hasattr(trl_config, key):
            kwargs.pop(key)
    print(f"[lora-dpo] DPOTrainer keywords: {sorted(kwargs)}")
    return DPOTrainer(**kwargs), trl_config


def train_lora_dpo(config: LoRADPOConfig) -> dict:
    from datasets import Dataset
    from transformers import AutoTokenizer

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_json(config.train_path)
    val_rows = read_json(config.val_path)
    train_dataset = Dataset.from_list(to_conversational(train_rows))
    eval_dataset = Dataset.from_list(to_conversational(val_rows)) if val_rows else None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(config.model_name, torch.float32)
    model.config.use_cache = False

    peft_config = build_lora_config(config)
    trainer, trl_config = build_trainer(
        config, model, tokenizer, train_dataset, eval_dataset, peft_config
    )

    trainable, total = count_parameters(trainer.model)
    run_info = {
        "stage": "lora_dpo",
        "run_name": config.run_name,
        "config": {**asdict(config), "lora_target_modules": list(config.lora_target_modules)},
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": round(trainable / max(total, 1), 6),
        "reference_model": "implicit: base model with LoRA adapters disabled",
        "environment": describe_environment(),
    }
    write_json(output_dir / "run_config.json", run_info)
    print(
        f"[lora-dpo] {len(train_rows)} pairs | LoRA r={config.lora_r} "
        f"alpha={config.lora_alpha} | trainable {trainable:,}/{total:,} "
        f"({100 * trainable / max(total, 1):.2f}%)"
    )

    timer = Stopwatch()
    result = trainer.train()
    history = getattr(trainer.state, "log_history", [])
    write_json(output_dir / "trl_log_history.json", history)

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    eval_metrics = {}
    if eval_dataset is not None:
        try:
            eval_metrics = trainer.evaluate()
        except Exception as error:  # pragma: no cover - version dependent
            print(f"[lora-dpo] evaluate() failed: {error}")

    summary = {
        "stage": "lora_dpo",
        "run_name": config.run_name,
        "base_model": config.model_name,
        "beta": config.beta,
        "adapter_dir": str(adapter_dir),
        "train_metrics": getattr(result, "metrics", {}),
        "eval_metrics": eval_metrics,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "wall_clock": timer.human(),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[lora-dpo] done in {timer.human()} | adapter -> {adapter_dir}")
    return summary
