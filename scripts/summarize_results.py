#!/usr/bin/env python
"""Collect every run and judgment summary into one report-ready table.

    python scripts/summarize_results.py --outputs_dir outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make smalign importable
from smalign.utils import run_and_exit
from smalign.utils import read_json, write_json

TRAINING_RUNS = ("sft", "dpo_hh", "dpo_uf", "lora_dpo_persona")


def _markdown_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(row[index])) for row in [header, *rows]) for index in range(len(header))]
    lines = [
        "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(header)) + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--output_path", default="outputs/results_summary.json")
    parser.add_argument("--markdown_path", default="outputs/results_summary.md")
    arguments = parser.parse_args()

    outputs = Path(arguments.outputs_dir)
    collected: dict[str, object] = {"training": {}, "evaluation": {}}

    for run_name in TRAINING_RUNS:
        summary_path = outputs / run_name / "summary.json"
        config_path = outputs / run_name / "run_config.json"
        entry = {}
        if summary_path.exists():
            entry["summary"] = read_json(summary_path)
        if config_path.exists():
            entry["run_config"] = read_json(config_path)
        if entry:
            collected["training"][run_name] = entry

    for path in sorted((outputs / "eval").glob("*_judged.summary.json")):
        collected["evaluation"][path.name.replace("_judged.summary.json", "")] = read_json(path)

    write_json(arguments.output_path, collected)

    sections = ["# Results summary\n"]

    def cell(mapping, key, digits=4):
        """Render one table cell, or a dash when the run does not report it."""
        value = mapping.get(key)
        if value is None:
            return "-"
        return round(value, digits) if isinstance(value, float) else value

    training_rows = []
    for run_name, entry in collected["training"].items():
        summary = entry.get("summary", {})
        config = (entry.get("run_config") or {}).get("config", {})
        # SFT reports a validation loss; DPO reports reward accuracy and margin.
        validation_loss = summary.get("best_val_loss", summary.get("final_val_loss"))
        training_rows.append([
            run_name,
            cell(summary, "optimizer_steps"),
            cell(config, "learning_rate", digits=8),
            cell(config, "beta"),
            f"{config.get('per_device_batch_size', '-')}x{config.get('grad_accum_steps', '-')}",
            "-" if validation_loss is None else round(validation_loss, 4),
            cell(summary, "final_val_reward_accuracy"),
            cell(summary, "final_val_reward_margin"),
            summary.get("wall_clock", "-"),
        ])
    if training_rows:
        sections.append("## Training runs\n")
        sections.append(
            _markdown_table(
                training_rows,
                ["run", "steps", "lr", "beta", "bs x accum", "val loss",
                 "val reward acc", "val reward margin", "wall clock"],
            )
        )

    eval_rows = []
    for name, summary in collected["evaluation"].items():
        for model, entry in summary.get("per_model", {}).items():
            eval_rows.append([
                name,
                model,
                summary.get("samples", "-"),
                entry.get("wins", "-"),
                entry.get("losses", "-"),
                summary.get("ties", "-"),
                entry.get("win_rate_with_ties", "-"),
                entry.get("win_rate_excluding_ties", "-"),
                summary.get("failed", "-"),
                summary.get("judge_model", "-"),
            ])
    if eval_rows:
        sections.append("\n## Pairwise judging\n")
        sections.append(
            _markdown_table(
                eval_rows,
                ["comparison", "model", "n", "wins", "losses", "ties",
                 "win rate (w/ ties)", "win rate (excl. ties)", "failed", "judge"],
            )
        )

    markdown = "\n".join(sections) + "\n"
    Path(arguments.markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.markdown_path).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"[summary] wrote {arguments.output_path} and {arguments.markdown_path}")


if __name__ == "__main__":
    run_and_exit(main)
