PA2: Instruction Tuning and Direct Preference Optimization on a Small Language Model

Name:        Syed Zamran Hussaini
Roll Number: 27100403

Checklist item 1, GitHub repository URL:
    https://github.com/zamranhussaini123/small-model-alignment

Checklist item 2, final report (PDF, 8 pages):
    27100403_report.pdf

Checklist item 3, training plots:
    plots/27100403_sft_training.png                 SFT train and validation loss
    plots/27100403_dpo_hh_training.png              HH-RLHF DPO loss, reward accuracy, reward margin
    plots/27100403_dpo_uf_training.png              UltraFeedback DPO loss, reward accuracy, reward margin
    plots/27100403_lora_dpo_persona_training.png    LoRA persona DPO (additional)
  The same figures are embedded in section 4 of the report.

Checklist item 4, evaluation summary comparing DPO against SFT:
    evaluation/27100403_evaluation_summary.md       combined training and judging tables
    evaluation/*_judged.summary.json               raw per-comparison judge output
  The same results appear in section 5 of the report.

Checklist item 5, submission_adapter.py with the public tests passing:
    submission_adapter.py                          copy; the live file is at the repository root
  To verify:
      git clone https://github.com/zamranhussaini123/small-model-alignment
      cd small-model-alignment
      python -m pip install -q pytest torch
      PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py

Headline results:
    HH-RLHF DPO vs SFT          194 W / 22 T / 84 L    win rate 0.698 excluding ties
    UltraFeedback DPO vs SFT    113 W / 53 T / 134 L   win rate 0.458 excluding ties
    LoRA persona vs base          5 W /  0 T / 21 L    win rate 0.192
    Judge: google/gemini-3.6-flash, 626 pairs, 0 failures, 0 unknown verdicts
