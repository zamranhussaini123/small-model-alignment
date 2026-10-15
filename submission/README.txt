PA2 submission - Instruction Tuning and DPO on Qwen2.5-0.5B
Zamran Hussaini

GitHub repository (checklist item 1):
    https://github.com/zamranhussaini123/small-model-alignment

Contents of this bundle:
    REPORT.pdf                              item 2 - final report, 8 pages
    plots/sft_loss.png                      item 3 - SFT train/validation loss
    plots/dpo_hh_training.png               item 3 - HH-RLHF DPO loss + reward diagnostics
    plots/dpo_uf_training.png               item 3 - UltraFeedback DPO loss + reward diagnostics
    plots/lora_dpo_persona_training.png     (extra) LoRA persona DPO
    evaluation_summary.md                   item 4 - DPO vs SFT judging tables
    *_judged.summary.json                   item 4 - raw per-comparison judge output

Item 5 (submission_adapter.py + passing public tests) is in the repository root.
To verify:
    git clone https://github.com/zamranhussaini123/small-model-alignment
    cd small-model-alignment
    python -m pip install -q pytest torch
    PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
