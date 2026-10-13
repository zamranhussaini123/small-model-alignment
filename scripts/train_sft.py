#!/usr/bin/env python
"""Run supervised fine-tuning.

    python scripts/train_sft.py --output_dir outputs/sft
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import PAUSED_EXIT_CODE, run_and_exit
from smalign.sft import SFTConfig, train_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = SFTConfig()
    for field in dataclasses.fields(SFTConfig):
        if field.name == "extra":
            continue
        value = getattr(defaults, field.name)
        if isinstance(value, bool):
            parser.add_argument(f"--{field.name}", type=lambda text: text.lower() == "true",
                                default=value)
        else:
            parser.add_argument(f"--{field.name}", type=type(value), default=value)
    arguments = parser.parse_args()

    config = SFTConfig(**{key: value for key, value in vars(arguments).items()})
    summary = train_sft(config)

    # A paused run is not a failure: it stopped on its time budget with a
    # resume checkpoint written. The distinct exit code lets the caller tell
    # "rerun me to continue" apart from "this broke".
    if summary.get("status") == "paused":
        raise SystemExit(PAUSED_EXIT_CODE)


if __name__ == "__main__":
    run_and_exit(main)
