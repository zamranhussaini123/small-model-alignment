#!/usr/bin/env python
"""Judge saved pairwise files with a Kaggle-hosted model.

    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/dpo_vs_sft_pairs.jsonl \
        --output_path outputs/eval/dpo_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash \
        --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

This script never generates model outputs; it only reads pairwise files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import run_and_exit
from smalign.judge import DEFAULT_JUDGE_MODEL, judge_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--failures_path", default="")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "kaggle_benchmarks", "direct", "dry_run"])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--safety_mode", action="store_true",
                        help="use the safety-focused judging criteria")
    parser.add_argument("--reuse_judgments", action="store_true",
                        help="skip pairs that already have a verdict on disk")
    parser.add_argument("--no_schema", action="store_true",
                        help="ask for plain-text JSON instead of structured output")
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    judge_pairs(
        pairs_path=arguments.pairs_path,
        output_path=arguments.output_path,
        failures_path=arguments.failures_path or None,
        backend=arguments.backend,
        judge_model=arguments.judge_model,
        batch_size=arguments.batch_size,
        sleep_seconds=arguments.sleep_seconds,
        max_retries=arguments.max_retries,
        safety_mode=arguments.safety_mode,
        reuse_judgments=arguments.reuse_judgments,
        use_schema=not arguments.no_schema,
        limit=arguments.limit,
    )


if __name__ == "__main__":
    run_and_exit(main)
