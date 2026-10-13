"""Extra adapter checks beyond the public smoke suite.

The manual warns that passing the public tests does not guarantee the private
grader passes, and names the areas worth covering: truncation boundaries,
unusual tokenizer configurations, mixed sequence lengths, row ordering, and
small optimization steps. Those are the cases below.
"""

from __future__ import annotations

import torch

import submission_adapter as adapter
from smalign.batching import build_lm_batch, response_token_count
from smalign.objectives import dpo_loss_from_logps, sft_loss
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    """A tokenizer that provides EOS but no padding id.

    ``pad_token`` stays a string because the toy tokenizer scans for it while
    encoding; only the id is missing, which is what the fallback handles.
    """

    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    """A tokenizer with neither an EOS id nor a padding id configured."""

    pad_token_id = None
    eos_token_id = None


def test_batch_is_padded_to_max_length_and_never_exceeds_it():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["short: ", "a"], ["x", "y"], max_length=24)
    assert batch["input_ids"].shape == (2, 24)
    for key in ("input_ids", "attention_mask", "labels"):
        assert batch[key].shape[1] == 24


def test_truncation_drops_eos_and_keeps_the_prompt_prefix():
    tokenizer = ToyTokenizer()
    prompt = "P" * 10
    response = "R" * 20
    max_length = 14
    batch = adapter.build_sft_batch(tokenizer, [prompt], [response], max_length=max_length)

    assert batch["input_ids"].shape == (1, max_length)
    assert batch["attention_mask"].eq(1).all(), "a truncated row has no padding"
    # Prompt is 10 tokens, so only 4 response tokens survive and EOS is cut.
    assert batch["labels"][0, :10].eq(-100).all()
    assert batch["labels"][0, 10:].ne(-100).all()
    assert tokenizer.eos_token_id not in batch["input_ids"][0].tolist()


def test_prompt_longer_than_max_length_yields_no_supervised_tokens():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q" * 40], ["answer"], max_length=8)
    assert batch["labels"].eq(-100).all()
    assert response_token_count(batch["labels"]).item() == 0

    # The loss must stay finite rather than dividing by zero.
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.3)
    loss = adapter.compute_sft_loss(model, tokenizer, ["Q" * 40], ["answer"], max_length=8)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_exact_fit_at_max_length_keeps_the_eos_token():
    tokenizer = ToyTokenizer()
    prompt, response = "abc", "de"  # 3 + 2 + 1 EOS = 6
    batch = adapter.build_sft_batch(tokenizer, [prompt], [response], max_length=6)
    assert batch["input_ids"][0, -1].item() == tokenizer.eos_token_id
    assert batch["labels"][0, -1].item() == tokenizer.eos_token_id


def test_missing_pad_token_falls_back_to_eos_and_still_masks_padding():
    tokenizer = NoPadTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["p: ", "p: "], ["short", "a"], max_length=16)
    assert batch["input_ids"].shape == (2, 16)
    # Row 1 is shorter, so it is padded; padding must be masked out everywhere.
    padded = batch["attention_mask"][1].eq(0)
    assert padded.any()
    assert batch["labels"][1][padded].eq(-100).all()


def test_missing_eos_token_appends_nothing():
    tokenizer = NoEosTokenizer()
    prompts, responses = ["q: "], ["ab"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=12)
    prompt_length = len(tokenizer.encode(prompts[0], add_special_tokens=False))
    response_length = len(tokenizer.encode(responses[0], add_special_tokens=False))
    assert batch["attention_mask"][0].sum().item() == prompt_length + response_length
    assert batch["labels"][0, prompt_length : prompt_length + response_length].ne(-100).all()
    assert batch["labels"][0, prompt_length + response_length :].eq(-100).all()


def test_row_order_is_preserved_for_mixed_lengths():
    tokenizer = ToyTokenizer()
    prompts = ["a: ", "bb: ", "ccc: "]
    responses = ["one", "two words here", "x"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=32)

    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        expected = tokenizer.encode(prompt, add_special_tokens=False) + tokenizer.encode(
            response, add_special_tokens=False
        )
        actual = batch["input_ids"][row, : len(expected)].tolist()
        assert actual == expected, f"row {row} is out of order"


def test_logprobs_follow_row_order_and_scale_with_response_length():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    logps = adapter.compute_response_logprobs(
        model, tokenizer, ["q: ", "q: "], ["a", "aaaaaa"], max_length=24
    )
    assert logps.shape == (2,)
    # A position-independent model gives a constant per-token log probability,
    # so a six-token response must score strictly lower than a one-token one.
    assert logps[0].item() > logps[1].item()


def test_dynamic_padding_matches_fixed_padding_for_logprobs():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.4)
    prompts = ["x: ", "yy: "]
    responses = ["alpha", "b"]

    fixed = build_lm_batch(tokenizer, prompts, responses, max_length=64, pad_to="max_length")
    longest = build_lm_batch(tokenizer, prompts, responses, max_length=64, pad_to="longest")
    assert longest["input_ids"].shape[1] < fixed["input_ids"].shape[1]

    from smalign.objectives import sequence_logprobs

    fixed_logps = sequence_logprobs(
        model, fixed["input_ids"], fixed["attention_mask"], fixed["labels"]
    )
    longest_logps = sequence_logprobs(
        model, longest["input_ids"], longest["attention_mask"], longest["labels"]
    )
    torch.testing.assert_close(fixed_logps, longest_logps)


def test_both_sft_reductions_are_consistent_with_each_other():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.25)
    prompts = ["do: ", "say: "]
    responses = ["a", "much longer answer"]

    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=32)
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=32)
    tokens = response_token_count(batch["labels"]).clamp_min(1)

    token_mean = sft_loss(model, tokenizer, prompts, responses, 32, reduction="token_mean")
    sequence_mean = sft_loss(model, tokenizer, prompts, responses, 32, reduction="sequence_mean")

    torch.testing.assert_close(token_mean, -logps.sum() / tokens)
    torch.testing.assert_close(sequence_mean, -logps.mean())


def test_dpo_batch_sides_share_the_prompt_tokens():
    tokenizer = ToyTokenizer()
    batch = adapter.build_dpo_batch(tokenizer, ["ask: "], ["good one"], ["bad"], max_length=20)
    prompt_length = len(tokenizer.encode("ask: ", add_special_tokens=False))
    chosen = batch["chosen"]["input_ids"][0, :prompt_length]
    rejected = batch["rejected"]["input_ids"][0, :prompt_length]
    assert torch.equal(chosen, rejected)
    assert batch["chosen"]["labels"][0, :prompt_length].eq(-100).all()
    assert batch["rejected"]["labels"][0, :prompt_length].eq(-100).all()


def test_dpo_loss_sign_and_monotonicity():
    beta = 0.1
    reference = torch.zeros(1)

    def loss_for(gap):
        return dpo_loss_from_logps(
            torch.tensor([gap]), torch.zeros(1), reference, reference, beta=beta
        )[0].item()

    # A bigger chosen-minus-rejected gap than the reference must lower the loss.
    assert loss_for(5.0) < loss_for(0.0) < loss_for(-5.0)
    assert abs(loss_for(0.0) - torch.log(torch.tensor(2.0)).item()) < 1e-6


def test_dpo_metrics_match_the_manual_definitions():
    beta = 0.1
    policy_chosen = torch.tensor([-1.0, -4.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -2.0, -2.0])
    reference_chosen = torch.tensor([-2.0, -2.0, -2.0])
    reference_rejected = torch.tensor([-2.0, -2.0, -2.0])

    _, metrics = dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=beta
    )
    # Rewards: chosen = [0.1, -0.2, 0.0], rejected = [-0.1, 0.0, 0.0].
    # Strict comparison means the tied third row does not count as correct.
    assert abs(metrics["reward_accuracy"] - 1 / 3) < 1e-6
    expected_margin = beta * ((2.0) + (-2.0) + 0.0) / 3
    assert abs(metrics["reward_margin"] - expected_margin) < 1e-6


def test_identical_policy_and_reference_give_zero_accuracy_and_zero_margin():
    values = torch.tensor([-3.0, -7.0, -1.0])
    _, metrics = dpo_loss_from_logps(values, values, values, values, beta=0.1)
    assert metrics["reward_accuracy"] == 0.0
    assert metrics["reward_margin"] == 0.0


def test_a_few_dpo_steps_increase_the_chosen_minus_rejected_gap():
    torch.manual_seed(0)
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.1)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.1)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    prompts = ["pick: ", "pick: "]
    chosen = ["aaa", "aaa"]
    rejected = ["zzz", "zzz"]
    optimizer = torch.optim.SGD(policy.parameters(), lr=5.0)

    def current_gap():
        with torch.no_grad():
            chosen_logps = adapter.compute_response_logprobs(
                policy, tokenizer, prompts, chosen, max_length=16
            )
            rejected_logps = adapter.compute_response_logprobs(
                policy, tokenizer, prompts, rejected, max_length=16
            )
        return (chosen_logps - rejected_logps).mean().item()

    before = current_gap()
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = adapter.compute_dpo_loss(
            policy, reference, tokenizer, prompts, chosen, rejected, beta=0.1, max_length=16
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    after = current_gap()

    assert after > before, "DPO should widen the chosen-over-rejected gap"
    assert losses[-1] < losses[0], "DPO loss should fall over a few steps"
    assert all(parameter.grad is None for parameter in reference.parameters())
