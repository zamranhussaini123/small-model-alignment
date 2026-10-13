"""Tokenisation and response-only label masking for SFT and DPO.

Contract implemented here (from the assignment manual, section 4.1):

* each response gets the tokenizer's EOS token appended *before* truncation, so
  a response that is cut at ``max_length`` legitimately loses its EOS;
* sequences longer than ``max_length`` are truncated, shorter ones are padded;
* prompt tokens and padding tokens are masked with ``-100`` in ``labels``;
* padding positions get ``attention_mask == 0``;
* row order is preserved exactly.

``pad_to`` exists because the grading contract and training efficiency want
different things. The public API pads to a fixed ``max_length`` (the literal
reading of "pad shorter examples ... to max_length"); the training collators
pass ``pad_to="longest"`` so short batches do not waste compute.
"""

from __future__ import annotations

from typing import Sequence

import torch

LABEL_IGNORE_INDEX = -100


def encode_without_specials(tokenizer, text: str) -> list[int]:
    """Token ids for ``text`` with no automatic special tokens.

    Uses ``tokenizer.encode`` because it behaves identically on Hugging Face
    tokenizers and on the lightweight tokenizer used by the public tests.
    """

    return list(tokenizer.encode(text, add_special_tokens=False))


def resolve_eos_id(tokenizer) -> int | None:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return None if eos_id is None else int(eos_id)


def resolve_pad_id(tokenizer) -> int:
    """Padding id, falling back to EOS and then 0.

    Padding positions are masked out of both the attention mask and the labels,
    so the exact value never reaches the loss; it only has to be in range.
    """

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = resolve_eos_id(tokenizer)
    if pad_id is None:
        pad_id = 0
    return int(pad_id)


def encode_prompt_response(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int | None = None,
    append_eos: bool = True,
) -> tuple[list[int], list[int]]:
    """Encode one example into ``(input_ids, labels)`` with prompt tokens masked.

    ``max_prompt_length`` truncates an over-long prompt from the *left*, which
    keeps the instruction tail and the assistant header - the part the model
    conditions on most directly - inside the window. It is used by DPO training
    and is left as ``None`` on the graded path.
    """

    prompt_ids = encode_without_specials(tokenizer, prompt)
    response_ids = encode_without_specials(tokenizer, response)

    eos_id = resolve_eos_id(tokenizer)
    if append_eos and eos_id is not None:
        response_ids = response_ids + [eos_id]

    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([LABEL_IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    pad_to: str = "max_length",
    max_prompt_length: int | None = None,
    append_eos: bool = True,
    device=None,
) -> dict:
    """Collate prompt/response pairs into a padded, response-masked LM batch.

    Returns exactly ``input_ids``, ``attention_mask``, and ``labels``.
    """

    prompts = list(prompts)
    responses = list(responses)
    if len(prompts) != len(responses):
        raise ValueError(f"got {len(prompts)} prompts but {len(responses)} responses")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if pad_to not in {"max_length", "longest"}:
        raise ValueError("pad_to must be 'max_length' or 'longest'")

    encoded = [
        encode_prompt_response(
            tokenizer,
            prompt,
            response,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            append_eos=append_eos,
        )
        for prompt, response in zip(prompts, responses)
    ]

    if pad_to == "max_length":
        width = max_length
    else:
        width = max((len(input_ids) for input_ids, _ in encoded), default=1)
        width = max(width, 1)

    batch_size = len(encoded)
    pad_id = resolve_pad_id(tokenizer)
    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    labels = torch.full((batch_size, width), LABEL_IGNORE_INDEX, dtype=torch.long)

    for row, (row_ids, row_labels) in enumerate(encoded):
        length = min(len(row_ids), width)
        if length == 0:
            continue
        input_ids[row, :length] = torch.tensor(row_ids[:length], dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(row_labels[:length], dtype=torch.long)

    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch


def build_preference_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
    pad_to: str = "max_length",
    max_prompt_length: int | None = None,
    append_eos: bool = True,
    device=None,
) -> dict:
    """Two response-masked LM batches sharing one prompt per row.

    The prompt is formatted and tokenised identically on both sides, so the
    chosen/rejected log-probability difference reflects the responses only.
    """

    shared = dict(
        max_length=max_length,
        pad_to=pad_to,
        max_prompt_length=max_prompt_length,
        append_eos=append_eos,
        device=device,
    )
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen_responses, **shared),
        "rejected": build_lm_batch(tokenizer, prompts, rejected_responses, **shared),
    }


def response_token_count(labels: torch.Tensor) -> torch.Tensor:
    """Number of label positions that contribute to a next-token loss.

    The first column can never be predicted from a previous token, so it drops
    out of the shifted comparison.
    """

    return labels[:, 1:].ne(LABEL_IGNORE_INDEX).sum()
