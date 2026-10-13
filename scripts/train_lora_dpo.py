#!/usr/bin/env python
"""Run the TRL DPOTrainer + LoRA persona experiment.

    python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo_persona
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable

# TRL's DPOTrainer refuses to run under nn.DataParallel, which Hugging Face
# Trainer applies automatically whenever several GPUs are visible to one
# process - and the assignment's accelerator is GPU T4 x2. Restricting this
# stage to one GPU is the fix TRL itself suggests. It costs nothing here:
# SmolLM2-360M with LoRA needs only a few GiB, and the from-scratch SFT and DPO
# runs still use both GPUs. This must be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from smalign.utils import run_and_exit
from smalign.lora_dpo import LoRADPOConfig, train_lora_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = LoRADPOConfig()
    for field in dataclasses.fields(LoRADPOConfig):
        if field.name in {"extra", "lora_target_modules"}:
            continue
        value = getattr(defaults, field.name)
        if isinstance(value, bool):
            parser.add_argument(f"--{field.name}", type=lambda text: text.lower() == "true",
                                default=value)
        else:
            parser.add_argument(f"--{field.name}", type=type(value), default=value)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=list(defaults.lora_target_modules))
    arguments = parser.parse_args()

    values = vars(arguments)
    values["lora_target_modules"] = tuple(values["lora_target_modules"])
    train_lora_dpo(LoRADPOConfig(**values))


if __name__ == "__main__":
    run_and_exit(main)
