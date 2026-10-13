"""A whole-pipeline run on a tiny randomly initialised Qwen2 model.

SFT -> DPO -> generation -> pairwise -> judging -> plots, on CPU, in a few
seconds. It needs Transformers but no network access and no GPU: the model and
tokenizer are built locally. Marked slow; skipped if Transformers is absent.

This is the test that catches wiring mistakes the unit tests cannot see, and it
pins the two invariants the manual states about an untouched DPO policy.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

pytestmark = pytest.mark.slow

MAX_LENGTH = 64


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A local tokenizer plus a 2-layer random Qwen2, saved to disk."""

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    base = tmp_path_factory.mktemp("base")
    corpus = [
        "<|im_start|>user <|im_end|> assistant",
        "hello world how are you today please help me",
        "cooking coding safety refuse answer question number about",
        "a careful helpful answer that is safe and clear bad good",
    ]
    backend = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.train_from_iterator(
        corpus * 20,
        trainers.WordPieceTrainer(
            vocab_size=400,
            special_tokens=["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.save_pretrained(base)

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            tie_word_embeddings=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model.save_pretrained(base)
    return str(base)


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    """Synthetic files in the on-disk schema the real pipeline uses."""

    directory = tmp_path_factory.mktemp("data")
    sft = [
        {
            "id": f"s{index}",
            "instruction": f"question number {index} about cooking",
            "response": f"answer number {index} with helpful detail here",
        }
        for index in range(40)
    ]
    preferences = [
        {
            "id": f"p{index}",
            "instruction": f"question number {index} about safety",
            "chosen": f"a careful helpful answer {index} that is safe and clear",
            "rejected": f"bad {index}",
        }
        for index in range(32)
    ]
    prompts = [
        {
            "id": f"e{index}",
            "instruction": f"question number {index} about cooking",
            "dataset": "unit_test",
            "category": "cooking",
            "expected_behavior": "be helpful",
        }
        for index in range(6)
    ]

    paths = {}
    for name, rows in (
        ("sft_train", sft[:32]),
        ("sft_val", sft[32:]),
        ("pref_train", preferences[:24]),
        ("pref_val", preferences[24:]),
        ("eval", prompts),
    ):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths[name] = str(path)
    return paths


@pytest.fixture(scope="module")
def sft_run(tiny_model, datasets, tmp_path_factory):
    from smalign.sft import SFTConfig, train_sft

    output = tmp_path_factory.mktemp("sft_out") / "sft"
    summary = train_sft(
        SFTConfig(
            model_name=tiny_model,
            train_path=datasets["sft_train"],
            val_path=datasets["sft_val"],
            output_dir=str(output),
            max_length=MAX_LENGTH,
            per_device_batch_size=2,
            grad_accum_steps=2,
            epochs=2,
            eval_every=4,
            log_every=2,
            eval_batch_size=4,
            fp16=False,
            num_workers=0,
            device="cpu",
        )
    )
    return output, summary


def test_sft_writes_best_and_final_checkpoints_in_fp16(sft_run, tiny_model):
    import os

    output, summary = sft_run
    for name in ("best", "final"):
        assert (output / name / "model.safetensors").exists()
        info = json.loads((output / name / "checkpoint_info.json").read_text())
        assert info["saved_dtype"] == "float16"

    assert summary["best_val_loss"] <= summary["final_val_loss"] + 1e-6
    # fp16 on disk really is smaller than the fp32 source it came from.
    fp32_bytes = os.path.getsize(f"{tiny_model}/model.safetensors")
    fp16_bytes = os.path.getsize(output / "best" / "model.safetensors")
    assert fp16_bytes < fp32_bytes * 0.75


def test_sft_loss_decreases(sft_run):
    output, _ = sft_run
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    train_losses = [r["train_loss"] for r in records if r.get("event") == "train"]
    assert len(train_losses) >= 2
    assert train_losses[-1] < train_losses[0]


@pytest.fixture(scope="module")
def dpo_run(sft_run, datasets, tmp_path_factory):
    from smalign.dpo import DPOConfig, train_dpo

    sft_output, _ = sft_run
    output = tmp_path_factory.mktemp("dpo_out") / "dpo"
    summary = train_dpo(
        DPOConfig(
            run_name="dpo_unit_test",
            policy_path=str(sft_output / "best"),
            train_path=datasets["pref_train"],
            val_path=datasets["pref_val"],
            output_dir=str(output),
            beta=0.1,
            max_length=MAX_LENGTH,
            max_prompt_length=32,
            per_device_batch_size=1,
            grad_accum_steps=2,
            learning_rate=5e-4,
            epochs=3,
            eval_every=4,
            log_every=2,
            eval_batch_size=4,
            eval_subset=4,
            max_val_examples=8,
            fp16=False,
            num_workers=0,
            policy_device="cpu",
            reference_device="cpu",
        )
    )
    return output, summary


def test_an_untouched_policy_ties_its_reference(dpo_run):
    """The manual's statement: identical policy and reference give loss log 2,
    strict reward accuracy 0, and a reward margin of exactly 0."""

    output, _ = dpo_run
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    initial = next(r for r in records if r.get("tag") == "initial")

    assert initial["val_reward_accuracy"] == 0.0
    assert abs(initial["val_reward_margin"]) < 1e-6
    assert abs(initial["val_loss"] - 0.6931471805) < 1e-3


def test_dpo_widens_the_reward_margin_and_saves_checkpoints(dpo_run):
    output, summary = dpo_run
    assert (output / "best" / "model.safetensors").exists()
    assert (output / "final" / "model.safetensors").exists()
    assert summary["final_val_reward_margin"] > 0
    assert summary["final_val_reward_accuracy"] > 0
    assert summary["final_val_loss"] < 0.6931471805


def test_generation_pairwise_and_judging(dpo_run, sft_run, datasets, tmp_path):
    from smalign.generate import GenerationConfig, run_generation
    from smalign.judge import judge_pairs
    from smalign.pairwise import build_pairwise

    sft_output, _ = sft_run
    dpo_output, _ = dpo_run
    decoding = {
        "max_new_tokens": 12,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 0,
    }

    generations = {}
    for name, path in (("unit-sft", sft_output / "best"), ("unit-dpo", dpo_output / "best")):
        target = tmp_path / f"{name}.json"
        summary = run_generation(
            GenerationConfig(
                model_path=str(path),
                prompts_path=datasets["eval"],
                output_path=str(target),
                generator=name,
                dataset="unit_test",
                batch_size=3,
                device="cpu",
                dtype="float32",
                decoding=decoding,
            )
        )
        assert summary["responses"] == 6
        generations[name] = target

    rows = json.loads(generations["unit-sft"].read_text())
    assert [row["id"] for row in rows] == [f"e{index}" for index in range(6)]
    assert all(row["generator"] == "unit-sft" for row in rows)

    pairs = tmp_path / "pairs.jsonl"
    pair_summary = build_pairwise(generations["unit-sft"], generations["unit-dpo"], pairs, seed=42)
    assert pair_summary["pairs"] == 6

    judged = tmp_path / "judged.jsonl"
    summary = judge_pairs(
        pairs_path=str(pairs),
        output_path=str(judged),
        backend="dry_run",
        batch_size=5,
        sleep_seconds=0,
        max_retries=2,
        safety_mode=True,
        reuse_judgments=False,
    )
    assert summary["samples"] == 6
    assert summary["failed"] == 0

    # A rerun must reuse the saved verdicts rather than re-judging them.
    rerun = judge_pairs(
        pairs_path=str(pairs),
        output_path=str(judged),
        backend="dry_run",
        batch_size=5,
        sleep_seconds=0,
        max_retries=2,
        reuse_judgments=True,
    )
    assert rerun["samples"] == 6


def test_plots_are_written(dpo_run, sft_run, tmp_path):
    pytest.importorskip("matplotlib")
    from smalign.plots import plot_dpo, plot_sft

    sft_output, _ = sft_run
    dpo_output, _ = dpo_run

    sft_plot = plot_sft(sft_output / "metrics.jsonl", tmp_path / "sft.png")
    dpo_plot = plot_dpo(dpo_output / "metrics.jsonl", tmp_path / "dpo.png")
    assert sft_plot and (tmp_path / "sft.png").stat().st_size > 5000
    assert dpo_plot and (tmp_path / "dpo.png").stat().st_size > 5000
