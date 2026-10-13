#!/usr/bin/env python
"""Run one from-scratch DPO training.

    python scripts/train_dpo.py --run_name dpo_hh \
        --train_path data/processed/hh_train.json \
        --val_path data/processed/hh_val.json \
        --output_dir outputs/dpo_hh
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import PAUSED_EXIT_CODE, run_and_exit
from smalign.dpo import DPOConfig, train_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DPOConfig()
    for field in dataclasses.fields(DPOConfig):
        if field.name == "extra":
            continue
        value = getattr(defaults, field.name)
        if isinstance(value, bool):
            parser.add_argument(f"--{field.name}", type=lambda text: text.lower() == "true",
                                default=value)
        elif field.name == "reference_path":
            parser.add_argument("--reference_path", type=str, default="")
        else:
            parser.add_argument(f"--{field.name}", type=type(value), default=value)
    arguments = parser.parse_args()

    config = DPOConfig(**{key: value for key, value in vars(arguments).items()})
    summary = train_dpo(config)

    # A paused run is not a failure: it stopped on its time budget with a
    # resume checkpoint written. The distinct exit code lets the caller tell
    # "rerun me to continue" apart from "this broke".
    if summary.get("status") == "paused":
        raise SystemExit(PAUSED_EXIT_CODE)


if __name__ == "__main__":
    run_and_exit(main)
