"""Response-only log probabilities, the SFT loss, and the DPO loss.

Everything downstream (the SFT loop, both from-scratch DPO loops, and the
graded adapter) goes through these functions, so there is exactly one place
where the training math lives.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from smalign.batching import LABEL_IGNORE_INDEX, build_lm_batch, build_preference_batch
from smalign.utils import model_device

SFT_REDUCTIONS = ("token_mean", "sequence_mean")

#: Reduction used for every reported SFT number in this project.
#:
#: ``token_mean`` divides the summed response negative log-likelihood by the
#: number of predicted response tokens in the batch, so the value is a
#: per-token loss comparable across batches with different response lengths.
DEFAULT_SFT_REDUCTION = "token_mean"


def token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position log probability of each label token, plus the valid mask.

    Teacher forcing shifts by one: position ``t`` of the logits predicts token
    ``t + 1``. Log probabilities are computed in float32 even under fp16
    autocast, because a 152k-way softmax in half precision loses accuracy
    exactly where the DPO log-ratios need it.
    """

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    targets = labels[:, 1:]

    mask = targets.ne(LABEL_IGNORE_INDEX)
    safe_targets = targets.masked_fill(~mask, 0)

    # log softmax written as (selected logit - logsumexp) to avoid allocating a
    # second (batch, time, vocab) tensor.
    selected = torch.gather(logits, dim=2, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    per_token = selected - torch.logsumexp(logits, dim=-1)
    per_token = torch.where(mask, per_token, torch.zeros_like(per_token))
    return per_token, mask


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    """One log probability per row, summed over response tokens only."""

    per_token, mask = token_logprobs(model, input_ids, attention_mask, labels)
    summed = per_token.sum(dim=-1)
    if average:
        return summed / mask.sum(dim=-1).clamp_min(1)
    return summed


def response_logprobs(
    model,
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    pad_to: str = "max_length",
    max_prompt_length: int | None = None,
) -> torch.Tensor:
    """Summed response-token log probability for each prompt/response pair."""

    batch = build_lm_batch(
        tokenizer,
        prompts,
        responses,
        max_length=max_length,
        pad_to=pad_to,
        max_prompt_length=max_prompt_length,
        device=model_device(model),
    )
    return sequence_logprobs(model, batch["input_ids"], batch["attention_mask"], batch["labels"])


def sft_loss_from_batch(
    model,
    batch: dict,
    reduction: str = DEFAULT_SFT_REDUCTION,
) -> tuple[torch.Tensor, dict]:
    """Scalar response-only SFT loss plus token statistics for logging."""

    if reduction not in SFT_REDUCTIONS:
        raise ValueError(f"reduction must be one of {SFT_REDUCTIONS}")

    per_token, mask = token_logprobs(
        model, batch["input_ids"], batch["attention_mask"], batch["labels"]
    )
    summed = per_token.sum(dim=-1)
    token_total = mask.sum().clamp_min(1)

    if reduction == "token_mean":
        loss = -summed.sum() / token_total
    else:
        loss = -summed.mean()

    stats = {
        "response_tokens": int(mask.sum().item()),
        "rows": int(batch["input_ids"].shape[0]),
        "sum_neg_logprob": float(-summed.sum().item()),
    }
    return loss, stats


def sft_loss(
    model,
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    reduction: str = DEFAULT_SFT_REDUCTION,
    pad_to: str = "max_length",
) -> torch.Tensor:
    """Scalar SFT loss straight from text."""

    batch = build_lm_batch(
        tokenizer,
        prompts,
        responses,
        max_length=max_length,
        pad_to=pad_to,
        device=model_device(model),
    )
    loss, _ = sft_loss_from_batch(model, batch, reduction=reduction)
    return loss


def dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """DPO objective and preference diagnostics from four log-probability vectors.

    ``loss = -log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))``, averaged
    over the batch. The implicit rewards are
    ``r = beta * (log pi(y|x) - log ref(y|x))``; reward accuracy is the fraction
    of rows with ``r_chosen > r_rejected`` (strict, as the manual specifies), and
    reward margin is the mean of ``r_chosen - r_rejected``.
    """

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps.to(policy_logratios.device) - (
        reference_rejected_logps.to(policy_logratios.device)
    )
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        chosen_rewards = beta * (
            policy_chosen_logps - reference_chosen_logps.to(policy_logratios.device)
        )
        rejected_rewards = beta * (
            policy_rejected_logps - reference_rejected_logps.to(policy_logratios.device)
        )
        metrics = {
            "loss": float(loss.item()),
            "reward_accuracy": float((chosen_rewards > rejected_rewards).float().mean().item()),
            "reward_margin": float((chosen_rewards - rejected_rewards).mean().item()),
            "chosen_reward": float(chosen_rewards.mean().item()),
            "rejected_reward": float(rejected_rewards.mean().item()),
            "policy_chosen_logp": float(policy_chosen_logps.mean().item()),
            "policy_rejected_logp": float(policy_rejected_logps.mean().item()),
            "logits_mean": float(logits.mean().item()),
        }
    return loss, metrics


def dpo_logps_from_batches(
    policy_model,
    reference_model,
    batches: dict,
    policy_device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four response-only log-probability vectors for one preference batch.

    The policy pass carries gradients; the reference pass never does. The two
    models may sit on different GPUs - the reference results are moved onto the
    policy device before the loss is formed.
    """

    policy_device = policy_device or model_device(policy_model)
    reference_device = model_device(reference_model)

    def side(model, name, device):
        batch = batches[name]
        return sequence_logprobs(
            model,
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )

    policy_chosen = side(policy_model, "chosen", policy_device)
    policy_rejected = side(policy_model, "rejected", policy_device)

    with torch.no_grad():
        reference_chosen = side(reference_model, "chosen", reference_device).to(policy_device)
        reference_rejected = side(reference_model, "rejected", reference_device).to(policy_device)

    return policy_chosen, policy_rejected, reference_chosen, reference_rejected


def dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    beta: float,
    max_length: int,
    pad_to: str = "max_length",
    max_prompt_length: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """End-to-end DPO loss from text, with the reference model frozen."""

    batches = build_preference_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length=max_length,
        pad_to=pad_to,
        max_prompt_length=max_prompt_length,
    )
    logps = dpo_logps_from_batches(policy_model, reference_model, batches)
    return dpo_loss_from_logps(*logps, beta=beta)
