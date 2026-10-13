"""The single prompt format used for SFT, DPO, and evaluation.

Keeping one template everywhere matters more than which template it is: the DPO
reference model, the policy, and the generation code must all agree, or the
response-token masks and the implicit rewards stop meaning anything.

The format is ChatML, which is what the Qwen2.5 tokenizer family already uses::

    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<eos>

Responses are stored *without* a trailing marker. The batch builder appends the
tokenizer's EOS token, so the model learns where to stop.
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

PROMPT_TEMPLATE = f"{IM_START}user\n{{instruction}}{IM_END}\n{IM_START}assistant\n"
SYSTEM_TEMPLATE = f"{IM_START}system\n{{system}}{IM_END}\n"

#: Extra strings that mark the end of an assistant turn when decoding.
GENERATION_STOP_STRINGS = (IM_END, "<|endoftext|>", f"\n{IM_START}")


def format_prompt(instruction: str, system: str | None = None) -> str:
    """Wrap a raw instruction in the assignment's chat template."""

    prefix = SYSTEM_TEMPLATE.format(system=system.strip()) if system else ""
    return prefix + PROMPT_TEMPLATE.format(instruction=instruction.strip())


def format_response(response: str) -> str:
    """Normalise a target response; EOS is added later by the batch builder."""

    return response.strip()


def clean_generation(text: str) -> str:
    """Cut a decoded continuation at the first end-of-turn marker."""

    cut = len(text)
    for marker in GENERATION_STOP_STRINGS:
        position = text.find(marker)
        if position != -1:
            cut = min(cut, position)
    return text[:cut].strip()


def stop_token_ids(tokenizer) -> list[int]:
    """EOS ids to pass to ``generate``: the real EOS plus ``<|im_end|>``."""

    ids: list[int] = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.append(int(tokenizer.eos_token_id))
    try:
        extra = tokenizer.convert_tokens_to_ids(IM_END)
        unknown = getattr(tokenizer, "unk_token_id", None)
        if extra is not None and extra >= 0 and extra != unknown and extra not in ids:
            ids.append(int(extra))
    except Exception:
        pass
    return ids
