# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

Post-training pipeline for PA2: supervised fine-tuning, two from-scratch DPO
runs, a TRL/LoRA persona-DPO run, and a blinded model-judged evaluation.

The SFT loop and both from-scratch DPO loops use **no** `Trainer`, **no** TRL,
and no prebuilt DPO loop - only Transformers for models and tokenizers and
plain PyTorch for the training loops. TRL and PEFT are used only in the persona
LoRA track, which is where the assignment allows them.

## Quick start on Kaggle

Two notebooks. The first does everything up to and including the blinded
pairwise files; the second judges them.

| Notebook | Accelerator | Needs | Produces |
|---|---|---|---|
| [`notebooks/01_pipeline_kaggle.ipynb`](notebooks/01_pipeline_kaggle.ipynb) | GPU T4 x2 | Internet on, project dataset attached | Checkpoints, metrics, plots, generations, pairwise files |
| [`notebooks/02_judge_kaggle.ipynb`](notebooks/02_judge_kaggle.ipynb) | none | Notebook 1's output attached | Judgments and result tables |

### Why two and not one

The manual sketches a three-notebook workflow, but as a recommendation - "a
practical workflow is" - and section 13.1 refers to "the training/generation
notebook" in the singular. So training and generation are merged.

Judging cannot be merged, for a hard technical reason. `kaggle_benchmarks` is
**not importable in a GPU notebook**; this was checked on the actual image:

```text
=== is kaggle_benchmarks usable in a GPU notebook? ===
kaggle_benchmarks NOT importable here: No module named 'kaggle_benchmarks'
```

The judge models live in Kaggle's benchmark/model-proxy environment, a different
notebook type. So notebook 2 must be created from
<https://www.kaggle.com/benchmarks/tasks/new>. This also satisfies the manual's
rule that judging reads saved pairwise files and never regenerates outputs.

### Creating the judging notebook

Notebook 2 needs Kaggle's benchmarks container, which is what carries
`kaggle_benchmarks` and the model proxy. Two ways to get it:

**Through the UI**: open <https://www.kaggle.com/benchmarks/tasks/new>. That URL
*creates* a notebook and redirects to it; upload `02_judge_kaggle.ipynb` into it.

**Through the API**: the only thing that distinguishes a benchmark notebook is
its container, and that field round-trips through `kaggle kernels push`:

```json
{
  "docker_image": "gcr.io/kaggle-private-byod/personal-benchmarks-new@sha256:<digest>",
  "enable_internet": true,
  "enable_gpu": false
}
```

Get the current digest by creating one task through the UI once and running
`kaggle kernels pull <slug> -m`. Kaggle rejects the accompanying
`"keywords": ["personal benchmark"]` on push; the container is what matters.

Judging needs no GPU, and `smalign.judge` deliberately avoids importing torch so
it runs in that container, which is not a training image.

### Running it

1. Upload this project as a Kaggle Dataset (or use the existing
   `pa2-small-model-alignment`), and attach it to the notebook.
2. Set `PROJECT_DATASET` in notebook 1 to its `/kaggle/input/...` path.
3. Run with `GPU T4 x2` and Internet on. A full run is 6-8 hours.
4. When it finishes, attach its output to notebook 2 and judge.

Set `SMOKE_TEST = True` in notebook 1 to run the whole pipeline end to end at
~1% scale in 15-25 minutes. Worth doing once before an overnight run: it
exercises every stage, checkpoint path and file format, though the resulting
models mean nothing.

## Surviving a dead session

Kaggle sessions end: they hit the limit, the kernel errors, or the tab closes.
Three mechanisms mean that costs minutes rather than hours.

**Resume checkpoints.** SFT and DPO write model, optimizer, scheduler, scaler,
RNG state and the exact data position every `resume_every` optimizer steps. The
write is atomic (temp file, then rename), so a process killed mid-write leaves
the previous checkpoint intact rather than a truncated one. Checkpoints are only
taken immediately after an optimizer step, so a resumed run never restarts
halfway through a gradient-accumulation window. Example order is a deterministic
function of `(seed, epoch)`, so a resumed run computes the examples it has not
seen instead of replaying the ones it has.

This is tested, not assumed: `tests/test_resume.py` trains a model straight
through, trains it again with an interruption, and asserts the **weights match**.

**Time budget.** `SESSION_BUDGET_HOURS` makes each stage stop voluntarily before
Kaggle kills it, writing a checkpoint first. A stage that stops this way exits
with code 42 and is recorded as `paused`, which is distinct from failed.

**Stage ledger.** `outputs/pipeline_state.json` records what finished. On a
rerun, completed stages are skipped, paused ones continue, failed ones retry. No
stage raises out of its cell: a failure is recorded and the notebook runs to the
end, so the commit completes and the outputs save.

To continue a run: attach the previous version's output and set `PREV_OUTPUT` to
it. Generation and judging are separately resumable too - both skip work that
already has a result on disk.

## Repository layout

```text
smalign/                     the implementation
  batching.py                tokenisation, truncation, padding, response-only masks
  objectives.py              response log probabilities, SFT loss, DPO loss + diagnostics
  sft.py                     from-scratch SFT loop
  dpo.py                     from-scratch DPO loop (used for both preference datasets)
  lora_dpo.py                TRL DPOTrainer + PEFT/LoRA persona track
  checkpointing.py           resume checkpoints, atomic writes, time budget
  pipeline.py                stage ledger, fail-soft stage runner
  prompts.py                 the one prompt template used everywhere
  data_prep.py               fixed splits from UltraChat / HH-RLHF / UltraFeedback / AlpacaEval
  generate.py                batched greedy generation
  pairwise.py                blinded pairwise file construction
  judge.py                   resumable Kaggle model-proxy judging
  plots.py                   training-behaviour figures
scripts/                     one CLI per stage (see "Commands" below)
notebooks/                   the three Kaggle notebooks
configs/                     the recipe for each run, as YAML, for the record
tests/                       public adapter tests plus edge-case and end-to-end tests
data/persona_lora/           provided persona preference data
eval_sets/                   fixed evaluation prompt sets
report/REPORT_TEMPLATE.md    the report skeleton with the required sections
submission_adapter.py        the graded interface
```

## The graded interface

`submission_adapter.py` exposes the six required functions as thin wrappers over
`smalign.batching` and `smalign.objectives`. Conventions, all documented in that
file and in the report:

- EOS is appended to each response **before** `max_length` truncation, so a
  truncated response may legitimately lose its EOS.
- Batches are padded to a fixed `max_length`; prompt and padding positions are
  `-100` in `labels` and `0` in `attention_mask`.
- `compute_response_logprobs` returns the **sum** of response-token log
  probabilities per row.
- `compute_sft_loss` uses the **`token_mean`** reduction: summed response
  negative log-likelihood divided by the number of predicted response tokens in
  the batch. This is the reduction behind every SFT number reported.
- Reference-model computation in DPO runs under `torch.no_grad()`; gradients
  reach the policy only.
- Input row order is preserved in every returned tensor.

## Tests

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q
```

| File | Covers |
|---|---|
| `tests/test_public_adapter_smoke.py` | the provided public suite: signatures, masking, scalar losses, gradient flow |
| `tests/test_adapter_edge_cases.py` | truncation boundaries, exact-fit sequences, tokenizers with no pad or EOS id, prompts longer than `max_length`, row ordering, dynamic vs fixed padding, both SFT reductions, the manual's reward-metric definitions, and a few real DPO optimization steps |
| `tests/test_data_and_eval.py` | HH transcript parsing and multi-turn filtering, pairwise blinding and reproducibility, judge-reply parsing, win/tie/failure accounting |
| `tests/test_end_to_end.py` | the whole pipeline on a tiny locally-built Qwen2: SFT, DPO, generation, pairwise, judging, plots |
| `tests/test_resume.py` | deterministic data order, pausing on the time budget, **a resumed run matching an uninterrupted one weight-for-weight**, cleanup of resume files, and tolerance of a corrupt checkpoint |

`tests/test_end_to_end.py` pins the two invariants the manual states about an
untouched DPO policy: with policy and reference identical, the loss is exactly
log 2, the reward margin is 0, and *strict* reward accuracy is 0 - not 0.5.

Skip the slow one with `-m "not slow"`.

## Commands

The notebooks call these; they also run standalone.

```bash
# fixed splits: 30k SFT, 2x15k preference, 300-prompt AlpacaEval subset (seed 42)
python scripts/prepare_data.py --stages all --seed 42

# supervised fine-tuning
python scripts/train_sft.py --output_dir outputs/sft

# from-scratch DPO, once per preference dataset
python scripts/train_dpo.py --run_name dpo_hh \
    --policy_path outputs/sft/best --reference_path outputs/sft/best \
    --train_path data/processed/hh_train.json --val_path data/processed/hh_val.json \
    --output_dir outputs/dpo_hh

python scripts/train_dpo.py --run_name dpo_uf \
    --policy_path outputs/sft/best --reference_path outputs/sft/best \
    --train_path data/processed/uf_train.json --val_path data/processed/uf_val.json \
    --output_dir outputs/dpo_uf

# TRL DPOTrainer + LoRA persona track
python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo_persona

# generation, one checkpoint and one evaluation set at a time
python scripts/generate_responses.py \
    --model_path outputs/sft/best \
    --prompts_path eval_sets/alpaca_eval_300_seed42.json \
    --output_path outputs/eval/generations/sft_alpaca.json \
    --generator qwen2.5-0.5b-sft --dataset alpaca_eval

# blinded pairwise file
python scripts/build_pairwise.py \
    --sft_path outputs/eval/generations/sft_alpaca.json \
    --dpo_path outputs/eval/generations/dpo_uf_alpaca.json \
    --output_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl --seed 42

# judging: reads saved pairwise files only, never generates
python scripts/judge_pairwise_kaggle.py \
    --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
    --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl \
    --judge_model google/gemini-3.6-flash \
    --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

python scripts/make_plots.py --outputs_dir outputs
python scripts/summarize_results.py --outputs_dir outputs
```

Add `--backend dry_run` to the judge script to exercise the plumbing without
spending any model calls.

## Models, data, and evaluation sets

| Track | Base | Train data | Evaluation set |
|---|---|---|---|
| SFT | `Qwen/Qwen2.5-0.5B` | 30,000 UltraChat-style single-turn pairs | validation loss on 1,000 held-out pairs |
| HH-RLHF DPO | SFT best checkpoint | 15,000 single-turn helpful/harmless pairs | `eval_sets/hh_rlhf_eval_300_seed42.json` |
| UltraFeedback DPO | SFT best checkpoint | 15,000 UltraFeedback-style pairs | `eval_sets/alpaca_eval_300_seed42.json` |
| LoRA persona DPO | `HuggingFaceTB/SmolLM2-360M-Instruct` | `data/persona_lora/ramsay_style_preferences.json` | `eval_sets/persona_style_eval.json` |

Both DPO runs start from the **same** SFT best checkpoint, with a frozen copy of
that same checkpoint as the reference model. Only the preference data differs,
which is what makes the two tracks comparable as experiments.

## Prompt format

One ChatML template, used identically for SFT, both DPO runs, and every
generation:

```text
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<eos>
```

Responses are stored without a trailing marker; the batch builder appends the
tokenizer's EOS token, so the model learns where to stop. `SmolLM2-Instruct`
uses the same template, so the persona track needs no special casing.

## Decoding

Greedy, `max_new_tokens=256`, `repetition_penalty=1.1`,
`no_repeat_ngram_size=0` - defined once in `smalign/generate.py` and applied to
**every** model in a comparison. The repetition penalty is deterministic and is
there because a 0.5B model falls into token loops often enough to dominate a
300-prompt comparison. If SFT and DPO decoded differently, the judge would be
scoring decoding policies instead of checkpoints.

## Evaluation integrity

- Side assignment in each pairwise file is randomised per prompt with seed 42,
  so position A is not a fixed model.
- The judge prompt contains only the instruction and "Response A" / "Response B".
  Model names, checkpoint paths, and the SFT/DPO distinction are kept in a
  `meta` block the prompt never reads, and verdicts are decoded back to
  identities only afterwards. Notebook 2 asserts this rather than assuming it.
- The safety-focused criteria score **over-refusal** as a failure, not a virtue;
  the helpful/harmless set includes 30 benign prompts so that is measurable.
- Judging is batched 5 at a time with a 5-second pause, 3 retries, per-batch
  checkpointing, resume, and a separate failure log.

## Notes on deviations from the guide configuration

The report has the full table with reasons. In short:

- **Transformers is not pinned below 5.** The manual suggests
  `"transformers>=4.43.0,<5"`, but the current Kaggle image ships 5.16.1 against
  torch 2.11, and forcing a downgrade risks breaking a working CUDA stack. Every
  call this project makes was verified against 5.16.1 on that image, and
  `smalign.utils.load_causal_lm` handles the v4/v5 `torch_dtype` to `dtype`
  rename either way, verifying the loaded dtype rather than trusting the keyword.
- **Checkpoints are saved as fp16.** Six fp32 copies of a 0.5B model is roughly
  12 GiB against a 19.5 GiB working disk. Live training weights stay fp32, which
  gradient scaling requires. A DPO run loads the same file for both policy and
  reference, so they still start numerically identical.
- **Periodic DPO evaluations use a fixed 300-pair slice**, with the full 1,000
  scored once at the end. Checkpoint selection compares only like with like.
- **Training data is selected by deterministic streaming**: the first N rows that
  pass the filters. Reproducible, and it avoids downloading gigabytes to keep 30k
  rows. The AlpacaEval subset is the one set the manual fixes exactly, so it is
  shuffled with seed 42.
- **Multi-turn HH conversations are dropped**, since the prompt format here is
  single-turn.
- **The LoRA persona stage runs on one GPU** (`CUDA_VISIBLE_DEVICES=0`). TRL's
  `DPOTrainer` refuses to run under `nn.DataParallel`, which Hugging Face
  `Trainer` applies automatically when several GPUs are visible - and the
  assignment's accelerator is GPU T4 x2. SmolLM2-360M with LoRA needs only a
  few GiB; the SFT and from-scratch DPO runs still use both GPUs.
- **The frozen DPO reference is fp32, not fp16**, whenever it has a GPU to
  itself. The manual permits fp16 ("the frozen reference *may* be loaded in
  fp16"), but with an fp32 policy that breaks the manual's own invariant: the
  initial reward margin came out at ~5e-4 and initial accuracy near 0.4 instead
  of exactly 0. Matching precision makes policy and reference bit-identical at
  step 0. fp16 is retained only for the single-GPU fallback.
- **The persona track generates with the model's own chat template**
  (`--chat_template tokenizer`). TRL formats conversational DPO data with that
  template and SmolLM2-Instruct's injects a default system message; generating
  without it would score the LoRA model on a prompt shape it never saw.
- **`max_prompt_length` and `warmup_ratio` are not passed to TRL.** `DPOConfig`
  in trl 1.15 does not accept them; the builder inspects the installed signature
  and reports what it drops.
