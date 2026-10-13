"""Resume correctness: an interrupted run must continue, not restart.

The headline test is :func:`test_resumed_run_matches_an_uninterrupted_one`,
which trains the same tiny model twice - once straight through, once killed
partway and resumed - and compares the resulting weights. That is the property
that actually matters: resuming has to be equivalent to never stopping, not
merely "it starts again without crashing".
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from smalign.checkpointing import (  # noqa: E402
    epoch_order,
    load_resume_state,
    resume_path,
)
from smalign.utils import PAUSED_EXIT_CODE  # noqa: E402

pytestmark = pytest.mark.slow


# --- deterministic data ordering ----------------------------------------------


def test_epoch_order_is_deterministic_and_a_permutation():
    first = epoch_order(100, seed=42, epoch=0)
    again = epoch_order(100, seed=42, epoch=0)
    assert first == again
    assert sorted(first) == list(range(100))


def test_epoch_order_differs_across_epochs_and_seeds():
    base = epoch_order(100, seed=42, epoch=0)
    assert base != epoch_order(100, seed=42, epoch=1), "each epoch needs its own shuffle"
    assert base != epoch_order(100, seed=7, epoch=0), "seed must change the order"


def test_resuming_mid_epoch_covers_exactly_the_unseen_examples():
    order = epoch_order(50, seed=42, epoch=0)
    consumed = 20
    seen, remaining = order[:consumed], order[consumed:]
    assert len(remaining) == 30
    assert not set(seen) & set(remaining), "an example must not be replayed after resume"
    assert sorted(seen + remaining) == list(range(50))


# --- shared fixtures -----------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    base = tmp_path_factory.mktemp("resume_base")
    backend = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.train_from_iterator(
        ["hello world question answer number about cooking safety helpful detail"] * 40,
        trainers.WordPieceTrainer(
            vocab_size=200, special_tokens=["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
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
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            tie_word_embeddings=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).save_pretrained(base)
    return str(base)


@pytest.fixture(scope="module")
def sft_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("resume_data")
    train = [
        {
            "id": f"s{i}",
            "instruction": f"question number {i} about cooking",
            "response": f"answer number {i} with helpful detail",
        }
        for i in range(64)
    ]
    val = train[:8]
    paths = {}
    for name, rows in (("train", train), ("val", val)):
        path = directory / f"sft_{name}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths[name] = str(path)
    return paths


def _sft_config(tiny_model, sft_data, output_dir, **overrides):
    from smalign.sft import SFTConfig

    settings = dict(
        model_name=tiny_model,
        train_path=sft_data["train"],
        val_path=sft_data["val"],
        output_dir=str(output_dir),
        max_length=48,
        per_device_batch_size=2,
        grad_accum_steps=2,
        epochs=1,
        eval_every=0,  # keep the comparison about training, not checkpoint writes
        log_every=4,
        eval_batch_size=4,
        fp16=False,
        num_workers=0,
        device="cpu",
        seed=1234,
        resume_every=4,
    )
    settings.update(overrides)
    return SFTConfig(**settings)


def _weights(path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32)
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


# --- the real test -------------------------------------------------------------


def test_sft_pauses_on_the_time_budget_and_writes_a_resume_checkpoint(
    tiny_model, sft_data, tmp_path
):
    from smalign.sft import train_sft

    output = tmp_path / "paused"
    # A zero-length budget: the reserve means it is already exhausted, so the
    # run stops at the first optimizer step it reaches.
    summary = train_sft(_sft_config(tiny_model, sft_data, output, max_seconds=0.001))

    assert summary["status"] == "paused"
    assert 0 < summary["optimizer_steps"] < summary["total_steps"]
    assert resume_path(output).exists(), "a paused run must leave a resume checkpoint"

    payload = load_resume_state(output)
    assert payload is not None
    for key in ("model", "optimizer", "progress", "rng"):
        assert key in payload
    assert payload["progress"]["optimizer_step"] == summary["optimizer_steps"]
    # Checkpoints are only taken after an optimizer step, so the example count
    # must land on a whole accumulation window.
    consumed = payload["progress"]["consumed_examples"]
    assert consumed % (2 * 2) == 0


def test_resumed_run_matches_an_uninterrupted_one(tiny_model, sft_data, tmp_path):
    """Train straight through, then train with an interruption, and compare.

    If resume were dropping or replaying examples, losing optimizer state, or
    restarting the LR schedule, the two sets of weights would diverge.
    """

    from smalign.sft import train_sft

    straight_dir = tmp_path / "straight"
    straight = train_sft(_sft_config(tiny_model, sft_data, straight_dir))
    assert straight["status"] == "completed"

    # Now the same run, stopped partway and continued.
    interrupted_dir = tmp_path / "interrupted"
    first = train_sft(
        _sft_config(tiny_model, sft_data, interrupted_dir, max_seconds=0.001)
    )
    assert first["status"] == "paused"
    assert first["optimizer_steps"] < straight["optimizer_steps"]

    second = train_sft(_sft_config(tiny_model, sft_data, interrupted_dir))
    assert second["status"] == "completed"
    assert second["optimizer_steps"] == straight["optimizer_steps"], (
        "a resumed run must perform the same total number of optimizer steps"
    )

    straight_weights = _weights(straight_dir / "final")
    resumed_weights = _weights(interrupted_dir / "final")
    assert set(straight_weights) == set(resumed_weights)
    for name, expected in straight_weights.items():
        torch.testing.assert_close(
            resumed_weights[name], expected, rtol=1e-4, atol=1e-5,
            msg=lambda m, n=name: f"{n} diverged after resume:\n{m}",
        )


def test_resume_checkpoint_is_deleted_once_the_stage_completes(
    tiny_model, sft_data, tmp_path
):
    """Resume files carry optimizer state and are several GiB at real scale."""

    from smalign.sft import train_sft

    output = tmp_path / "cleanup"
    train_sft(_sft_config(tiny_model, sft_data, output, max_seconds=0.001))
    assert resume_path(output).exists()

    train_sft(_sft_config(tiny_model, sft_data, output))
    assert not resume_path(output).exists(), "a completed stage must free its resume file"


def test_a_corrupt_resume_checkpoint_is_ignored_rather_than_fatal(
    tiny_model, sft_data, tmp_path
):
    """A half-written file must not make every future run crash."""

    from smalign.sft import train_sft

    output = tmp_path / "corrupt"
    output.mkdir(parents=True)
    resume_path(output).write_bytes(b"this is not a torch checkpoint")

    assert load_resume_state(output) is None
    summary = train_sft(_sft_config(tiny_model, sft_data, output))
    assert summary["status"] == "completed"


def test_allow_resume_false_starts_over(tiny_model, sft_data, tmp_path):
    from smalign.sft import train_sft

    output = tmp_path / "ignore"
    paused = train_sft(_sft_config(tiny_model, sft_data, output, max_seconds=0.001))
    assert paused["status"] == "paused"

    fresh = train_sft(_sft_config(tiny_model, sft_data, output, allow_resume=False))
    assert fresh["status"] == "completed"
    assert fresh["optimizer_steps"] == 16  # 64 examples / (2 * 2) = a full run


# --- DPO resume ----------------------------------------------------------------


@pytest.fixture(scope="module")
def dpo_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("resume_pref")
    rows = [
        {
            "id": f"p{i}",
            "instruction": f"question number {i} about safety",
            "chosen": f"a careful helpful answer {i} that is safe",
            "rejected": f"bad answer {i}",
        }
        for i in range(32)
    ]
    paths = {}
    for name, data in (("train", rows), ("val", rows[:8])):
        path = directory / f"pref_{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[name] = str(path)
    return paths


def test_dpo_pauses_and_resumes_to_the_same_step_count(tiny_model, dpo_data, tmp_path):
    from smalign.dpo import DPOConfig, train_dpo

    def config(output_dir, **overrides):
        settings = dict(
            run_name="dpo_resume_test",
            policy_path=tiny_model,
            train_path=dpo_data["train"],
            val_path=dpo_data["val"],
            output_dir=str(output_dir),
            beta=0.1,
            max_length=48,
            max_prompt_length=24,
            per_device_batch_size=1,
            grad_accum_steps=2,
            learning_rate=1e-4,
            epochs=1,
            eval_every=0,
            log_every=4,
            eval_batch_size=4,
            eval_subset=4,
            max_val_examples=8,
            fp16=False,
            num_workers=0,
            policy_device="cpu",
            reference_device="cpu",
            seed=1234,
            resume_every=4,
        )
        settings.update(overrides)
        return DPOConfig(**settings)

    straight = train_dpo(config(tmp_path / "dpo_straight"))
    assert straight["status"] == "completed"

    interrupted_dir = tmp_path / "dpo_interrupted"
    paused = train_dpo(config(interrupted_dir, max_seconds=0.001))
    assert paused["status"] == "paused"
    assert resume_path(interrupted_dir).exists()

    finished = train_dpo(config(interrupted_dir))
    assert finished["status"] == "completed"
    assert finished["optimizer_steps"] == straight["optimizer_steps"]
    assert not resume_path(interrupted_dir).exists()

    straight_weights = _weights(tmp_path / "dpo_straight" / "final")
    resumed_weights = _weights(interrupted_dir / "final")
    for name, expected in straight_weights.items():
        torch.testing.assert_close(
            resumed_weights[name], expected, rtol=1e-3, atol=1e-4,
            msg=lambda m, n=name: f"{n} diverged after DPO resume:\n{m}",
        )


def test_paused_exit_code_is_distinct_from_success_and_failure():
    assert PAUSED_EXIT_CODE not in (0, 1)
