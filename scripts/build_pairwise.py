#!/usr/bin/env python
"""Build a blinded pairwise comparison file from two generation files.

    python scripts/build_pairwise.py \
        --sft_path outputs/eval/generations/sft_alpaca.json \
        --dpo_path outputs/eval/generations/dpo_uf_alpaca.json \
        --output_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import run_and_exit
from smalign.pairwise import build_pairwise
from smalign.utils import DEFAULT_SEED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft_path", required=True, help="baseline generation file")
    parser.add_argument("--dpo_path", required=True, help="comparison generation file")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip_empty", action="store_true",
                        help="drop pairs where either side produced nothing")
    arguments = parser.parse_args()

    build_pairwise(
        arguments.sft_path,
        arguments.dpo_path,
        arguments.output_path,
        seed=arguments.seed,
        skip_empty=arguments.skip_empty,
    )


if __name__ == "__main__":
    run_and_exit(main)
