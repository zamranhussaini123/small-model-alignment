"""Build the fixed training and evaluation splits used by every run.

All four sources are reduced to the same two shapes:

* SFT rows   -> ``{"id", "instruction", "response"}``
* preference -> ``{"id", "instruction", "chosen", "rejected"}``

Selection is deterministic. The large instruction/preference corpora are read
in streaming mode and the first N rows that pass the filters are kept, which
avoids multi-gigabyte downloads on Kaggle and gives the same subset on every
run. The AlpacaEval subset is the one place the manual fixes exactly: 300
prompts with seed 42, so that one is shuffled with an explicit seed.
"""

from __future__ import annotations

import re
from pathlib import Path

from smalign.utils import DEFAULT_SEED, read_json, write_json

MIN_INSTRUCTION_CHARS = 8
MIN_RESPONSE_CHARS = 8
MAX_INSTRUCTION_CHARS = 2000
MAX_RESPONSE_CHARS = 6000


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"[ \t]+\n", "\n", str(text).strip())


def _usable(instruction: str, *responses: str) -> bool:
    if not (MIN_INSTRUCTION_CHARS <= len(instruction) <= MAX_INSTRUCTION_CHARS):
        return False
    for response in responses:
        if not (MIN_RESPONSE_CHARS <= len(response) <= MAX_RESPONSE_CHARS):
            return False
    return True


def _split(rows: list[dict], n_train: int, n_val: int) -> dict[str, list[dict]]:
    """Take the first ``n_train`` rows for training and the next ``n_val`` for validation."""

    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    return {"train": train, "validation": val}


# ----------------------------------------------------------------------------
# SFT data: UltraChat, reduced to its first turn
# ----------------------------------------------------------------------------


def prepare_sft(
    n_train: int = 30_000,
    n_val: int = 1_000,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft",
) -> dict[str, list[dict]]:
    """Single-turn instruction/response pairs from an UltraChat-style corpus."""

    from datasets import load_dataset

    stream = load_dataset(dataset_name, split=split, streaming=True)
    wanted = n_train + n_val
    rows: list[dict] = []

    for record in stream:
        messages = record.get("messages") or []
        instruction = ""
        response = ""
        for message in messages:
            role = message.get("role")
            if role == "user" and not instruction:
                instruction = _clean(message.get("content"))
            elif role == "assistant" and instruction and not response:
                response = _clean(message.get("content"))
                break
        if not instruction:
            instruction = _clean(record.get("prompt"))
        if not _usable(instruction, response):
            continue
        rows.append(
            {
                "id": f"sft-{len(rows):06d}",
                "instruction": instruction,
                "response": response,
                "source": dataset_name,
            }
        )
        if len(rows) >= wanted:
            break

    return _split(rows, n_train, n_val)


# ----------------------------------------------------------------------------
# HH-RLHF preference data
# ----------------------------------------------------------------------------

_HH_TURN = re.compile(r"\n\n(Human|Assistant):")


def parse_hh_transcript(transcript: str) -> tuple[str, str] | None:
    """Split an HH-RLHF transcript into ``(prompt, final_response)``.

    Returns ``None`` for anything that is not a clean single exchange, which is
    how multi-turn conversations are filtered out: the manual's reference setup
    is single-turn, and a multi-turn prompt format is not implemented here.
    """

    text = "\n\n" + transcript.strip()
    matches = list(_HH_TURN.finditer(text))
    if len(matches) != 2:
        return None

    if matches[0].group(1) != "Human" or matches[1].group(1) != "Assistant":
        return None

    instruction = text[matches[0].end() : matches[1].start()]
    response = text[matches[1].end() :]
    return _clean(instruction), _clean(response)


def prepare_hh(
    n_train: int = 15_000,
    n_val: int = 1_000,
    dataset_name: str = "Anthropic/hh-rlhf",
    split: str = "train",
) -> dict[str, list[dict]]:
    """Single-turn helpful/harmless preference pairs."""

    from datasets import load_dataset

    stream = load_dataset(dataset_name, split=split, streaming=True)
    wanted = n_train + n_val
    rows: list[dict] = []

    for record in stream:
        chosen = parse_hh_transcript(record.get("chosen") or "")
        rejected = parse_hh_transcript(record.get("rejected") or "")
        if chosen is None or rejected is None:
            continue
        instruction, chosen_response = chosen
        rejected_instruction, rejected_response = rejected
        if instruction != rejected_instruction:
            continue
        if chosen_response == rejected_response:
            continue
        if not _usable(instruction, chosen_response, rejected_response):
            continue
        rows.append(
            {
                "id": f"hh-{len(rows):06d}",
                "instruction": instruction,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "source": dataset_name,
            }
        )
        if len(rows) >= wanted:
            break

    return _split(rows, n_train, n_val)


# ----------------------------------------------------------------------------
# UltraFeedback preference data
# ----------------------------------------------------------------------------


def _last_assistant_message(value) -> str:
    """UltraFeedback stores chosen/rejected as message lists; take the reply."""

    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return _clean(message.get("content"))
        if value and isinstance(value[-1], dict):
            return _clean(value[-1].get("content"))
    return ""


def prepare_ultrafeedback(
    n_train: int = 15_000,
    n_val: int = 1_000,
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    split: str = "train_prefs",
) -> dict[str, list[dict]]:
    """General instruction-following preference pairs."""

    from datasets import load_dataset

    stream = load_dataset(dataset_name, split=split, streaming=True)
    wanted = n_train + n_val
    rows: list[dict] = []

    for record in stream:
        instruction = _clean(record.get("prompt"))
        chosen = _last_assistant_message(record.get("chosen"))
        rejected = _last_assistant_message(record.get("rejected"))
        if chosen == rejected:
            continue
        if not _usable(instruction, chosen, rejected):
            continue
        rows.append(
            {
                "id": f"uf-{len(rows):06d}",
                "instruction": instruction,
                "chosen": chosen,
                "rejected": rejected,
                "source": dataset_name,
            }
        )
        if len(rows) >= wanted:
            break

    return _split(rows, n_train, n_val)


# ----------------------------------------------------------------------------
# Fixed AlpacaEval evaluation subset
# ----------------------------------------------------------------------------


def prepare_alpaca_eval(n_prompts: int = 300, seed: int = DEFAULT_SEED) -> list[dict]:
    """``n_prompts`` AlpacaEval instructions sampled with an explicit seed.

    The raw JSON is pulled straight from the dataset repository rather than
    through a loading script, because ``datasets`` v3+ no longer executes
    dataset scripts.
    """

    records = _load_alpaca_eval_records()

    from datasets import Dataset

    dataset = Dataset.from_list(records).shuffle(seed=seed)
    selected = dataset.select(range(min(n_prompts, len(dataset))))

    rows = []
    for index, record in enumerate(selected):
        instruction = _clean(record.get("instruction"))
        if not instruction:
            continue
        rows.append(
            {
                "id": f"alpaca-{seed}-{index:04d}",
                "instruction": instruction,
                "dataset": "alpaca_eval",
                "reference_dataset": record.get("dataset", "alpaca_eval"),
            }
        )
    return rows


def _load_alpaca_eval_records() -> list[dict]:
    from huggingface_hub import hf_hub_download

    candidates = ("alpaca_eval.json", "alpaca_eval_all_outputs.json")
    errors = []
    for filename in candidates:
        try:
            path = hf_hub_download(
                repo_id="tatsu-lab/alpaca_eval",
                filename=filename,
                repo_type="dataset",
            )
            records = read_json(path)
            if isinstance(records, list) and records:
                return records
        except Exception as error:  # pragma: no cover - network dependent
            errors.append(f"{filename}: {error}")

    try:  # pragma: no cover - network dependent
        from datasets import load_dataset

        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        return [dict(record) for record in dataset]
    except Exception as error:
        errors.append(f"load_dataset: {error}")

    raise RuntimeError("could not load AlpacaEval prompts; tried " + " | ".join(errors))


# ----------------------------------------------------------------------------
# Persona preference data (shipped with the assignment)
# ----------------------------------------------------------------------------


def prepare_persona(
    path="data/persona_lora/ramsay_style_preferences.json",
    n_val: int = 20,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict]]:
    """Shuffle the provided persona pairs and hold out a small validation slice."""

    import random

    records = read_json(path)
    rows = [
        {
            "id": f"persona-{index:04d}",
            "instruction": _clean(record["prompt"]),
            "chosen": _clean(record["chosen"]),
            "rejected": _clean(record["rejected"]),
            "source": str(path),
        }
        for index, record in enumerate(records)
    ]
    random.Random(seed).shuffle(rows)
    n_val = min(n_val, max(len(rows) // 10, 1))
    return {"train": rows[:-n_val], "validation": rows[-n_val:]}


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

STAGE_FILES = {
    "sft": ("sft_train.json", "sft_val.json"),
    "hh": ("hh_train.json", "hh_val.json"),
    "ultrafeedback": ("uf_train.json", "uf_val.json"),
    "persona": ("persona_train.json", "persona_val.json"),
}


def save_split(output_dir, stage: str, split: dict[str, list[dict]]) -> dict[str, str]:
    train_name, val_name = STAGE_FILES[stage]
    output_dir = Path(output_dir)
    paths = {
        "train": str(write_json(output_dir / train_name, split["train"])),
        "validation": str(write_json(output_dir / val_name, split["validation"])),
    }
    print(
        f"[data] {stage}: {len(split['train'])} train / {len(split['validation'])} validation"
        f" -> {paths['train']}"
    )
    return paths
