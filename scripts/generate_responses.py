#!/usr/bin/env python
"""Generate responses from one checkpoint on one fixed evaluation set.

    python scripts/generate_responses.py \
        --model_path outputs/sft/best \
        --prompts_path eval_sets/alpaca_eval_300_seed42.json \
        --output_path outputs/eval/generations/sft_alpaca.json \
        --generator qwen2.5-0.5b-sft --dataset alpaca_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import run_and_exit
from smalign.generate import DEFAULT_DECODING, GenerationConfig, run_generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--dataset", default="alpaca_eval")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    parser.add_argument("--resume", type=lambda text: text.lower() == "true", default=True)
    parser.add_argument("--chat_template", default="project", choices=["project", "tokenizer"],
                        help="'tokenizer' uses the model's own chat template; "
                             "required for the SmolLM2 persona track")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_DECODING["max_new_tokens"])
    parser.add_argument("--repetition_penalty", type=float,
                        default=DEFAULT_DECODING["repetition_penalty"])
    parser.add_argument("--no_repeat_ngram_size", type=int,
                        default=DEFAULT_DECODING["no_repeat_ngram_size"])
    arguments = parser.parse_args()

    decoding = dict(DEFAULT_DECODING)
    decoding.update(
        max_new_tokens=arguments.max_new_tokens,
        repetition_penalty=arguments.repetition_penalty,
        no_repeat_ngram_size=arguments.no_repeat_ngram_size,
    )

    run_generation(
        GenerationConfig(
            model_path=arguments.model_path,
            adapter_path=arguments.adapter_path,
            prompts_path=arguments.prompts_path,
            output_path=arguments.output_path,
            generator=arguments.generator,
            dataset=arguments.dataset,
            batch_size=arguments.batch_size,
            device=arguments.device,
            dtype=arguments.dtype,
            decoding=decoding,
            limit=arguments.limit,
            resume=arguments.resume,
            chat_template=arguments.chat_template,
        )
    )


if __name__ == "__main__":
    run_and_exit(main)
