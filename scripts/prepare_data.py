#!/usr/bin/env python
"""Build every fixed split used by the pipeline.

    python scripts/prepare_data.py --stages all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import run_and_exit
from smalign.data_prep import (
    prepare_alpaca_eval,
    prepare_hh,
    prepare_persona,
    prepare_sft,
    prepare_ultrafeedback,
    save_split,
)
from smalign.utils import DEFAULT_SEED, read_json, set_seed, write_json

STAGES = ("sft", "hh", "ultrafeedback", "persona", "alpaca_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=["all"], choices=["all", *STAGES])
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--eval_sets_dir", default="eval_sets")
    parser.add_argument("--sft_train", type=int, default=30_000)
    parser.add_argument("--sft_val", type=int, default=1_000)
    parser.add_argument("--pref_train", type=int, default=15_000)
    parser.add_argument("--pref_val", type=int, default=1_000)
    parser.add_argument("--alpaca_prompts", type=int, default=300)
    parser.add_argument("--persona_path", default="data/persona_lora/ramsay_style_preferences.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    set_seed(arguments.seed)
    stages = STAGES if "all" in arguments.stages else tuple(arguments.stages)
    manifest: dict[str, object] = {"seed": arguments.seed, "stages": {}}

    if "sft" in stages:
        split = prepare_sft(n_train=arguments.sft_train, n_val=arguments.sft_val)
        manifest["stages"]["sft"] = save_split(arguments.output_dir, "sft", split)

    if "hh" in stages:
        split = prepare_hh(n_train=arguments.pref_train, n_val=arguments.pref_val)
        manifest["stages"]["hh"] = save_split(arguments.output_dir, "hh", split)

    if "ultrafeedback" in stages:
        split = prepare_ultrafeedback(n_train=arguments.pref_train, n_val=arguments.pref_val)
        manifest["stages"]["ultrafeedback"] = save_split(
            arguments.output_dir, "ultrafeedback", split
        )

    if "persona" in stages:
        split = prepare_persona(path=arguments.persona_path, seed=arguments.seed)
        manifest["stages"]["persona"] = save_split(arguments.output_dir, "persona", split)

    if "alpaca_eval" in stages:
        rows = prepare_alpaca_eval(n_prompts=arguments.alpaca_prompts, seed=arguments.seed)
        path = Path(arguments.eval_sets_dir) / f"alpaca_eval_{arguments.alpaca_prompts}_seed{arguments.seed}.json"
        write_json(path, rows)
        manifest["stages"]["alpaca_eval"] = {"eval": str(path), "prompts": len(rows)}
        print(f"[data] alpaca_eval: {len(rows)} prompts -> {path}")

    hh_eval = Path(arguments.eval_sets_dir) / "hh_rlhf_eval_300_seed42.json"
    if hh_eval.exists():
        manifest["stages"]["hh_eval"] = {"eval": str(hh_eval), "prompts": len(read_json(hh_eval))}

    write_json(Path(arguments.output_dir) / "data_manifest.json", manifest)
    print(f"\n[data] manifest -> {Path(arguments.output_dir) / 'data_manifest.json'}")


if __name__ == "__main__":
    run_and_exit(main)
