# Instruction Tuning and DPO on a Small Language Model

**Name:** _your name_ · **Course:** _course_ · **Date:** _date_

> Fill in every `_placeholder_`. Maximum 8 pages including figures, tables, and
> references. Each section below says where its numbers come from, so nothing
> has to be retyped from memory. Delete these quoted notes before submitting.
>
> The manual is explicit about two things that are easy to lose marks on:
> report the values **actually used** for the submitted checkpoints rather than
> the guide configuration, and separate **resource-driven** changes from
> **performance-driven** ones.

---

## 1. Repository, models, and data

**Repository:** `https://github.com/_your-username_/small-model-alignment`

**Base checkpoints**

| Track | Base checkpoint |
|---|---|
| SFT | `Qwen/Qwen2.5-0.5B` |
| HH-RLHF DPO | SFT best checkpoint (`outputs/sft/best`) |
| UltraFeedback DPO | SFT best checkpoint (`outputs/sft/best`) |
| LoRA persona DPO | `HuggingFaceTB/SmolLM2-360M-Instruct` |

**Data and evaluation sets**

| Track | Train data | Train / val size | Evaluation set | Prompts |
|---|---|---|---|---|
| SFT | UltraChat-style single-turn (`HuggingFaceH4/ultrachat_200k`, `train_sft`) | _30,000_ / _1,000_ | held-out validation loss | _1,000_ |
| HH-RLHF DPO | `Anthropic/hh-rlhf`, single-turn only | _15,000_ / _1,000_ | `eval_sets/hh_rlhf_eval_300_seed42.json` | 300 |
| UltraFeedback DPO | `HuggingFaceH4/ultrafeedback_binarized`, `train_prefs` | _15,000_ / _1,000_ | `eval_sets/alpaca_eval_300_seed42.json` | 300 |
| LoRA persona DPO | `data/persona_lora/ramsay_style_preferences.json` | _200_ / _20_ | `eval_sets/persona_style_eval.json` | 26 |

> Exact counts: `data/processed/data_manifest.json`, or the `train_examples` /
> `val_pairs` fields in each `outputs/<run>/run_config.json`.

**Seeds.** Single seed `42` throughout: data selection and splitting, training,
and the pairwise side assignment. Every number below is therefore single-seed;
see section 7.

**Prompt formatting.** One ChatML template for SFT, both DPO runs, and all
generation:

```text
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<eos>
```

Responses are stored without a trailing marker and the tokenizer's EOS token is
appended by the batch builder, so the model learns where to stop. The exact
string is recorded as `prompt_template` in every `run_config.json`.
`SmolLM2-Instruct` uses the same template.

**Filtering.** Multi-turn HH-RLHF conversations are dropped (one `Human` turn
and one `Assistant` turn required), since the prompt format is single-turn.
Pairs with identical chosen and rejected responses are dropped. Instructions and
responses outside _8-2,000_ and _8-6,000_ characters respectively are dropped.
Selection from the large corpora is deterministic streaming - the first N rows
that pass the filters - rather than a shuffled sample; see section 3.

**Loss reduction.** `token_mean`: summed response-token negative log-likelihood
divided by the number of predicted response tokens in the batch. Used
consistently, and exposed as `SFT_REDUCTION` in `submission_adapter.py`.

---

## 2. Experimental configuration

> All of this is in `outputs/<run>/run_config.json` under `config`. Report what
> the submitted checkpoints actually used.

| | SFT | HH-RLHF DPO | UltraFeedback DPO | LoRA-DPO persona |
|---|---|---|---|---|
| Trainable strategy | full fine-tune | full fine-tune | full fine-tune | LoRA |
| LoRA r / alpha / dropout | - | - | - | 16 / 32 / 0.05 |
| LoRA target modules | - | - | - | `q,k,v,o,gate,up,down_proj` |
| Trainable / total params | _494M / 494M_ | _494M / 494M_ | _494M / 494M_ | _from summary.json_ |
| Optimizer | AdamW | RMSprop | RMSprop | AdamW (TRL) |
| Peak learning rate | 2e-5 | 1e-6 | 1e-6 | 5e-5 |
| Scheduler / warmup | cosine / 3% | constant / none | constant / none | cosine / 10% |
| Weight decay | 0.1 | 0.0 | 0.0 | 0.0 |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 |
| Per-device batch x accum | 2 x 16 | 1 x 32 | 1 x 32 | 2 x 4 |
| Effective batch | 32 | 32 | 32 | 8 |
| Precision | fp16 autocast, fp32 weights | fp16 autocast, fp32 policy, fp16 frozen reference | same | fp16 autocast |
| Max sequence / prompt length | 512 / - | 512 / 256 | 512 / 256 | 320 / 128 |
| Epochs | 1 | 1 | 1 | 3 |
| Optimizer steps | _from summary.json_ | _~468_ | _~468_ | _~75_ |
| DPO beta | - | 0.1 | 0.1 | 0.1 |
| Reference checkpoint | - | `outputs/sft/best` (frozen copy) | `outputs/sft/best` (frozen copy) | base weights with adapters disabled |
| Checkpoint selection | lowest val loss | highest val reward accuracy | highest val reward accuracy | final |
| Seed | 42 | 42 | 42 | 42 |

**Generation settings** (identical for every model in a comparison, from
`smalign/generate.py` and recorded in each `*.summary.json`):

| Setting | Value |
|---|---|
| Decoding | greedy (`do_sample=False`, `num_beams=1`) |
| Temperature / top-p | n/a - deterministic |
| Max new tokens | 256 |
| Repetition penalty | 1.1 |
| No-repeat n-gram | 0 (disabled) |
| Chat template | the ChatML template in section 1 |
| Inference dtype | fp16 |

---

## 3. Parameter choices and deviations

> State the guide value, your value, and why. Keep resource-driven and
> performance-driven changes apart - they support different conclusions.

### Resource-driven

| Guide value | Value used | Why | Effect |
|---|---|---|---|
| `pip install "transformers>=4.43.0,<5"` | kept the image's Transformers 5.16.1; installed only `trl` | The current Kaggle image ships Transformers 5.16.1 against torch 2.11. Forcing the `<5` pin downgrades a working, CUDA-matched stack. Every call this project makes was verified against 5.16.1 on that image before the run. | None observed. The model loader handles the v4/v5 `torch_dtype` to `dtype` rename and verifies the loaded dtype rather than trusting the keyword. |
| Checkpoints in training precision (fp32) | saved as fp16 | Six fp32 copies of a 0.5B model across three runs is ~12 GiB on Kaggle's ~20 GiB working disk. Live training weights stay fp32, as gradient scaling requires. | None measurable. A DPO run loads the same file for policy and reference, so they still start identical: initial reward margin is exactly 0 and initial loss exactly log 2. |
| 1,000 validation pairs per DPO evaluation | 300-pair slice for periodic evals; full 1,000 scored once at the end | 1,000 pairs x 4 forward passes per evaluation, ~9 times per run, is a large fraction of the budget. | _compare `final_val_reward_accuracy` (300 pairs) with `full_val_reward_accuracy` (1,000) in `summary.json` and state whether they agree_ |
| Shuffled sample of the training corpora | deterministic streaming, first N rows passing the filters | Avoids downloading several GiB to keep 30k rows; reproducible without storing the sample. | Changes *which* examples are used, not how many. A possible confound: streaming order is not random with respect to the source. |
| _add any batch-size reduction you needed for OOM_ | | | |

### Performance-driven

| Guide value | Value used | Why | Effect |
|---|---|---|---|
| Pure greedy decoding | greedy + `repetition_penalty=1.1` | _state what you saw: a 0.5B model loops on some prompts, and looped outputs would dominate a 300-prompt comparison_ | _state whether it reduced degenerate outputs; the `empty` and `mean_words` columns from notebook 2 are the evidence_ |
| Optimizer "RMSprop or AdamW" | RMSprop for DPO, AdamW for SFT | RMSprop is what the DPO paper's implementation uses, and DPO here runs at a fixed 1e-6 where Adam's second-moment adaptation buys little. | _state whether you tried both_ |
| LoRA configuration: student choice | r=16, alpha=32, dropout=0.05, all attention + MLP projections, lr 5e-5, batch 8, 3 epochs | ~200 preference pairs: a larger rank mostly adds variance, and `alpha/r = 2` is standard scaling. The MLP projections are included because style lives in token choice, not only attention. lr is ~50x the full-finetune rate because far fewer weights are trainable. One epoch is too few steps to move style at this dataset size. | _state what the reward accuracy and margin did, and whether the style actually changed_ |
| LoRA sequence length 512 | 320 (prompt 128) | Measured on this dataset: prompts reach ~33 tokens and responses ~60, so nothing is truncated and no compute goes into padding. | None - no example is affected. |
| GPU T4 x2 for every stage | LoRA stage pinned to one GPU (`CUDA_VISIBLE_DEVICES=0`) | TRL's `DPOTrainer` raises `ValueError: DPOTrainer does not support nn.DataParallel`, which `Trainer` applies whenever several GPUs are visible. This is the fix TRL itself recommends. | None: SmolLM2-360M with LoRA needs only a few GiB. SFT and both from-scratch DPO runs still use both GPUs. |
| Frozen reference in fp16 (manual's memory tip) | fp32 reference when it has its own GPU | With an fp32 policy, an fp16 reference is not bit-identical, so the first evaluation reported margin ~5e-4 and accuracy ~0.4 instead of the exact 0 the manual describes. Matching precision restores the invariant. | Initial evaluation now reports exactly loss log 2, margin 0, strict accuracy 0. No memory pressure: the reference has a dedicated T4. |
| Project chat template everywhere | persona track generates with `--chat_template tokenizer` | TRL formats conversational DPO data with the model's own template, and SmolLM2-Instruct's injects a default system message. Generating without it would score the LoRA model on a prompt shape it never saw in training. | _state whether persona outputs look consistent with the training data_ |
| TRL `max_prompt_length` 256, `warmup_ratio` 0.1 | not passed | `DPOConfig` in trl 1.15 accepts neither; the builder inspects the installed signature and reports what it drops. Prompt truncation falls back to `max_length` (320), which is ample here - the measured worst case is ~91 tokens. | None: no persona example is truncated at either setting. |
| _add anything you tuned, e.g. beta_ | | | |

> If you tried several settings, put them in one small comparison table rather
> than describing them in chronological order.

---

## 4. Training behaviour

**Figures** (from `outputs/plots/`):

- `sft_loss.png` - training and validation response-only loss per optimizer
  step, with the selected checkpoint marked.
- `dpo_hh_training.png`, `dpo_uf_training.png` - DPO loss, reward accuracy, and
  reward margin, with the selected checkpoint marked.
- `lora_dpo_persona_training.png` - TRL log history for the persona run.

Smoothing is an exponential moving average with weight 0.6, drawn over the raw
series rather than replacing it.

**SFT.** _Describe the loss curve: where it fell fastest, whether validation
tracked training or separated, and which step was selected. Final best
validation loss: `_____` (from `outputs/sft/summary.json`)._

**DPO.** Both runs start with policy and reference identical, so the first
evaluation is loss exactly log 2 = 0.6931, reward margin 0, and strict reward
accuracy 0 - not 0.5, because `r_w > r_l` is false on a tie. _(This holds only
because the reference is loaded at the policy's precision; see section 3.)_ _Report where
accuracy and margin went from there, whether the margin grew steadily or
plateaued, and which step was selected._

| Run | Initial val loss | Final val loss | Final reward accuracy | Final reward margin | Selected step |
|---|---|---|---|---|---|
| HH-RLHF DPO | 0.6931 | _____ | _____ | _____ | _____ |
| UltraFeedback DPO | 0.6931 | _____ | _____ | _____ | _____ |

_Comment on convergence, any instability, and the checkpoint-selection decision.
Do not infer response quality from training loss alone - that is what section 5
is for._

---

## 5. DPO-versus-SFT results

> From `outputs/eval/*_judged.summary.json`, or straight out of
> `outputs/results_summary.md`.

**Judge model:** `google/gemini-3.6-flash` _(or whichever model was available;
record the actual one)_, via `kaggle_benchmarks`. Batches of 5 pairs, 5-second
pause between attempts, up to 3 attempts per batch.

**Invalid judgments** are recorded as `UNKNOWN` (reply parsed but no usable
verdict) or `FAILED` (batch exhausted its retries, logged to
`*_failures.jsonl`). Both are excluded from win rates and reported separately
rather than silently dropped.

### UltraFeedback DPO vs SFT, AlpacaEval 300 (seed 42)

| | n | Wins | Ties | Losses | Win rate (ties included) | Win rate (ties excluded) | Unknown | Failed |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-dpo-uf` | 300 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| `qwen2.5-0.5b-sft` | 300 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |

### HH-RLHF DPO vs SFT, helpful/harmless 300

| | n | Wins | Ties | Losses | Win rate (ties included) | Win rate (ties excluded) | Unknown | Failed |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-dpo-hh` | 300 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| `qwen2.5-0.5b-sft` | 300 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |

Per-category breakdown (ten categories, 30 prompts each) - from the
`by_category` block, printed by notebook 3:

| Category | DPO wins | SFT wins | Ties |
|---|---|---|---|
| harmful_instructions | | | |
| benign_helpfulness | | | |
| privacy_and_doxxing | | | |
| medical_safety | | | |
| self_harm_support | | | |
| legal_financial_safety | | | |
| hate_or_harassment | | | |
| cyber_abuse | | | |
| calibration_and_uncertainty | | | |
| safe_transformation | | | |

> `benign_helpfulness` is the over-refusal check. If HH-DPO wins the harmful
> categories but loses this one, that is an alignment tax, and it is the most
> interesting result in the table.

### LoRA persona DPO vs base (reported separately)

Style alignment is a different target from general helpfulness or safety, so
this is **not** combined with the numbers above.

| | n | Wins | Ties | Losses | Win rate |
|---|---|---|---|---|---|
| `smollm2-360m-lora-dpo-persona` | 26 | ___ | ___ | ___ | ___ |
| `smollm2-360m-instruct` | 26 | ___ | ___ | ___ | ___ |

_Answer the question the manual asks: did it become more stylistically faithful
**without** becoming less useful or less safe? Use the `coding`,
`general_help`, and `safety_sensitive` categories to answer the second half -
a persona win driven by a safety regression is a failure, not a success._

**Fairness controls.** Same prompts, same prompt order, same decoding settings,
same judge, same criteria within a comparison. Side assignment randomised per
prompt with seed 42, so position A is not a fixed model. The judge prompt
contains only the instruction and "Response A" / "Response B"; identities are
decoded afterwards.

---

## 6. Interpretation and error analysis

**Which model was better, and by how much.** _Answer directly for each
evaluation set, with the win rate and the margin. Keep measurement separate
from interpretation._

**Why you think that happened.** _Connect the result to concrete evidence from
your runs - the preference dataset, the strength of the SFT baseline, beta,
training length, and the reward diagnostics in section 4. If reward accuracy
rose but win rate did not, say so: that is the interesting case, and it means
the implicit reward learned something the judge does not reward._

**Consistency across prompt types.** _Use the per-category table. Is the effect
spread evenly, or concentrated in safety refusals, factuality,
instruction-following, verbosity, or persona style?_

**Verbosity.** _Mean response length per model, from the generation statistics
cell in notebook 2. A judge that prefers longer answers will reward length
regardless of quality, so if DPO changed length substantially, say so and treat
the win rate accordingly._

| Model | Evaluation set | Mean words | Median words | Empty |
|---|---|---|---|---|
| | | | | |

**Representative examples.** _Two or three where DPO improves, one or two where
it regresses or barely changes. Quote the prompt, both responses, and the
judge's stated reason. The `show_verdicts` cell in notebook 3 prints exactly
this. Use them to support the aggregate numbers, not to replace them._

1. **DPO improves:** _prompt, both responses, judge reason, your reading_
2. **DPO regresses:** _same_
3. **Little change:** _same_

**Candidate explanations.** _Pick the ones your evidence supports and say why:
preference-data mismatch, noisy pairs, over-optimization, insufficient training,
a strong SFT baseline, or decoding sensitivity. Each claim needs a number or a
quoted output behind it._

---

## 7. Limitations and reproducibility

**Limitations**

- **Judge bias.** A single model judge with known preferences for length and
  confident formatting. No human agreement check was run. _Note any pattern you
  saw in the judge's reasons._
- **Evaluation size.** 300 prompts per track, 26 for the persona track. At 300
  samples a win rate has roughly ±5-6 percentage points of binomial noise, so
  small gaps are not distinguishable from zero. _State whether your gap clears
  that bar._
- **Single seed.** One seed per run, so run-to-run variance is unmeasured and
  cannot be separated from the effect of DPO.
- **Scale.** 0.5B parameters, one epoch, 15k preference pairs. Conclusions here
  do not transfer directly to larger models or longer training.
- **Streaming data selection.** Deterministic, but not a random sample of the
  source corpora.
- **Interrupted sessions.** _If any stage paused on the time budget or failed and
  was resumed, say so and give the stage and step. Resume restores model,
  optimizer, scheduler, scaler, RNG and data position, and `tests/test_resume.py`
  asserts a resumed run matches an uninterrupted one weight-for-weight, so this
  should not affect results - but it should still be stated._
- _Any OOM retries or configuration changes forced mid-run._

**Reproducibility paths**

| What | Where |
|---|---|
| Recipe per run (as YAML) | `configs/*.yaml` |
| Values actually used | `outputs/<run>/run_config.json` |
| Metrics, step by step | `outputs/<run>/metrics.jsonl` |
| Run summaries | `outputs/<run>/summary.json` |
| Training plots | `outputs/plots/*.png` |
| Data manifest and split sizes | `data/processed/data_manifest.json` |
| Fixed evaluation prompt sets | `eval_sets/*.json` |
| Generations | `outputs/eval/generations/*.json` |
| Blinded pairwise files | `outputs/eval/*_pairs.jsonl` |
| Judgments and summaries | `outputs/eval/*_judged.jsonl`, `*_judged.summary.json` |
| Unresolved judge failures | `outputs/eval/*_failures.jsonl` |
| Combined tables | `outputs/results_summary.md` |
| Which stages ran, and their status | `outputs/pipeline_state.json` |

**Reproduce it.** Run `notebooks/01_pipeline_kaggle.ipynb` on Kaggle
`GPU T4 x2` with Internet on and the project dataset attached, then attach its
output to `notebooks/02_judge_kaggle.ipynb` and run that. Total runtime 6-8
hours.

Notebook 2 runs in Kaggle's benchmarks container
(`gcr.io/kaggle-private-byod/personal-benchmarks-new`), obtained either from
<https://www.kaggle.com/benchmarks/tasks/new> or by setting `docker_image` in the
kernel metadata. Judging and training cannot share a notebook: `kaggle_benchmarks`
is not importable in a GPU notebook, and even when installed by hand there it
reports zero available models - both verified on the images used for these runs. Training and generation *are* merged, which the manual permits - section
5.4 is phrased as a recommendation and 13.1 refers to "the training/generation
notebook" in the singular.

If a run is interrupted, attach its output as `PREV_OUTPUT` and run again:
completed stages are skipped and paused ones continue from their resume
checkpoints.

---

## References

1. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C.
   *Direct Preference Optimization: Your Language Model is Secretly a Reward
   Model.* NeurIPS, 2023.
2. Ouyang, L. et al. *Training language models to follow instructions with human
   feedback.* NeurIPS, 2022.
3. Dubois, Y. et al. *Length-Controlled AlpacaEval: A Simple Way to Debias
   Automatic Evaluators.* 2024.
4. Bai, Y. et al. *Training a Helpful and Harmless Assistant with Reinforcement
   Learning from Human Feedback.* 2022.
5. Hugging Face. Transformers, TRL, and PEFT documentation.
6. Kaggle. Notebooks and Models documentation; `kaggle-benchmarks` library.
