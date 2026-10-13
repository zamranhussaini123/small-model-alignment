"""Batched generation on the fixed evaluation sets.

The decoding policy lives in :data:`DEFAULT_DECODING` and is applied to every
model in a comparison. That is the point: if SFT and DPO decode differently, the
judge is scoring decoding policies rather than checkpoints.

Greedy decoding is the main setting, as the manual asks. The one addition is a
mild repetition penalty, which is deterministic and is needed because a 0.5B
model falls into token loops often enough to dominate a 300-prompt comparison.
Any change here must be recorded in the report.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm.auto import tqdm

from smalign.prompts import clean_generation, format_prompt, stop_token_ids
from smalign.utils import read_json, write_json


def build_prompt(tokenizer, instruction: str, chat_template: str) -> str:
    """Render one instruction into the prompt string the model expects."""

    if chat_template == "tokenizer":
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as error:  # pragma: no cover - template-dependent
            print(f"[gen] tokenizer chat template unavailable ({error}); using the project template")
    return format_prompt(instruction)

#: Shared decoding policy. Deterministic: no sampling, no temperature.
DEFAULT_DECODING = {
    "max_new_tokens": 256,
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 0,
}


@dataclass
class GenerationConfig:
    model_path: str = "outputs/sft/best"
    adapter_path: str = ""  # set for the LoRA-DPO checkpoint
    prompts_path: str = "eval_sets/alpaca_eval_300_seed42.json"
    output_path: str = "outputs/eval/generations/sft_alpaca.json"
    generator: str = "qwen2.5-0.5b-sft"
    dataset: str = "alpaca_eval"
    batch_size: int = 8
    max_prompt_tokens: int = 512
    dtype: str = "float16"
    device: str = "cuda:0"
    decoding: dict = field(default_factory=lambda: dict(DEFAULT_DECODING))
    limit: int = 0  # 0 means "all prompts"; small values are for sanity checks
    resume: bool = True
    #: Which prompt format to use.
    #:
    #: ``"project"`` applies this repository's ChatML template and is what the
    #: Qwen SFT and DPO checkpoints were trained with.
    #:
    #: ``"tokenizer"`` calls ``tokenizer.apply_chat_template``. The persona
    #: track needs this: TRL formats conversational DPO data with the model's
    #: own template, and SmolLM2-Instruct's template injects a default system
    #: message. Generating without it would evaluate the LoRA model on a prompt
    #: shape it was never trained on.
    chat_template: str = "project"


def load_model_and_tokenizer(config: GenerationConfig):
    """Load a checkpoint for inference, attaching a LoRA adapter when given."""

    from transformers import AutoTokenizer

    from smalign.utils import load_causal_lm

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        config.dtype
    ]
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        dtype = torch.float32

    tokenizer_source = config.adapter_path or config.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation

    model = load_causal_lm(config.model_path, dtype)
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path, torch_dtype=dtype)
        model = model.merge_and_unload()

    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_batch(model, tokenizer, device, prompts: list[str], decoding: dict,
                   max_prompt_tokens: int) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )
    # Only the two tensors generate accepts: some tokenizers also return
    # token_type_ids, which causal LMs reject outright.
    encoded = {
        key: encoded[key].to(device)
        for key in ("input_ids", "attention_mask")
        if key in encoded
    }

    eos_ids = stop_token_ids(tokenizer)
    outputs = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids if eos_ids else None,
        **decoding,
    )
    # With left padding every row starts generating at the same offset.
    continuations = outputs[:, encoded["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(continuations, skip_special_tokens=True)
    return [clean_generation(text) for text in decoded]


def run_generation(config: GenerationConfig) -> dict:
    """Generate responses for every prompt, writing the file as it goes."""

    prompts = read_json(config.prompts_path)
    if config.limit:
        prompts = prompts[: config.limit]

    output_path = Path(config.output_path)
    done: dict[str, dict] = {}
    if config.resume and output_path.exists():
        for row in read_json(output_path):
            if row.get("output"):
                done[row["id"]] = row
        if done:
            print(f"[gen] resuming {config.generator}: {len(done)} responses already saved")

    pending = [row for row in prompts if row["id"] not in done]
    model, tokenizer, device = (None, None, None)
    if pending:
        model, tokenizer, device = load_model_and_tokenizer(config)

    results = dict(done)
    # Periodic progress lines rather than only a bar: in a committed Kaggle run
    # stderr is not a TTY, so a bar renders nothing and the stage looks hung -
    # and the orchestrator's quiet-timeout would eventually kill it.
    started_at = time.time()
    batches = max((len(pending) + config.batch_size - 1) // config.batch_size, 1)
    report_every = max(1, min(20, batches // 10 or 1))
    progress = tqdm(total=len(pending), desc=f"gen:{config.generator}",
                    dynamic_ncols=True, disable=not sys.stderr.isatty())

    for batch_index, start in enumerate(range(0, len(pending), config.batch_size)):
        chunk = pending[start : start + config.batch_size]
        texts = generate_batch(
            model,
            tokenizer,
            device,
            [build_prompt(tokenizer, row["instruction"], config.chat_template)
             for row in chunk],
            config.decoding,
            config.max_prompt_tokens,
        )
        for row, text in zip(chunk, texts):
            record = {
                "id": row["id"],
                "instruction": row["instruction"],
                "output": text,
                "generator": config.generator,
                "dataset": config.dataset,
            }
            for key in ("category", "expected_behavior"):
                if key in row:
                    record[key] = row[key]
            results[row["id"]] = record
        progress.update(len(chunk))
        # Rewrite after every batch so a disconnect never loses finished work.
        write_json(output_path, [results[row["id"]] for row in prompts if row["id"] in results])

        if (batch_index + 1) % report_every == 0 or (batch_index + 1) == batches:
            completed = min(start + config.batch_size, len(pending))
            elapsed = time.time() - started_at
            rate = elapsed / max(completed, 1)
            remaining = int(rate * (len(pending) - completed))
            print(
                f"[gen] {config.generator}: {completed}/{len(pending)} "
                f"({completed / max(len(pending), 1):.0%}) "
                f"| {int(elapsed // 60)}m elapsed "
                f"| eta {remaining // 60}m{remaining % 60:02d}s",
                flush=True,
            )
    progress.close()

    ordered = [results[row["id"]] for row in prompts if row["id"] in results]
    write_json(output_path, ordered)

    empty = sum(1 for row in ordered if not row["output"].strip())
    lengths = [len(row["output"].split()) for row in ordered]
    summary = {
        "generator": config.generator,
        "model_path": config.model_path,
        "adapter_path": config.adapter_path,
        "dataset": config.dataset,
        "prompts_path": config.prompts_path,
        "output_path": str(output_path),
        "responses": len(ordered),
        "empty_responses": empty,
        "mean_response_words": round(sum(lengths) / max(len(lengths), 1), 1),
        "decoding": config.decoding,
        "chat_template": config.chat_template,
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(
        f"[gen] {config.generator}: {len(ordered)} responses "
        f"| mean {summary['mean_response_words']} words | {empty} empty"
    )
    if model is not None:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return summary
