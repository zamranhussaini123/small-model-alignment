#!/usr/bin/env python
"""Render the training plots for the report.

    python scripts/make_plots.py --outputs_dir outputs --plots_dir outputs/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import run_and_exit
from smalign.plots import plot_dpo, plot_lora_history, plot_sft

DPO_RUNS = {
    "dpo_hh": "HH-RLHF DPO (helpful / harmless)",
    "dpo_uf": "UltraFeedback DPO (general helpfulness)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--plots_dir", default="outputs/plots")
    arguments = parser.parse_args()

    outputs = Path(arguments.outputs_dir)
    plots = Path(arguments.plots_dir)
    written = []

    sft_metrics = outputs / "sft" / "metrics.jsonl"
    if sft_metrics.exists():
        written.append(
            plot_sft(sft_metrics, plots / "sft_loss.png",
                     title="SFT on Qwen2.5-0.5B (response-only loss)")
        )

    for run_name, title in DPO_RUNS.items():
        metrics = outputs / run_name / "metrics.jsonl"
        if metrics.exists():
            written.append(plot_dpo(metrics, plots / f"{run_name}_training.png", title=title))

    lora_history = outputs / "lora_dpo_persona" / "trl_log_history.json"
    if lora_history.exists():
        written.append(
            plot_lora_history(lora_history, plots / "lora_dpo_persona_training.png",
                              title="LoRA-DPO persona tuning (SmolLM2-360M-Instruct)")
        )

    written = [path for path in written if path]
    print(f"\n[plots] {len(written)} figure(s) in {plots}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    run_and_exit(main)
