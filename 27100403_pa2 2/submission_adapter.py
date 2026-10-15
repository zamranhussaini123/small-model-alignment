"""Stable grading interface for the SFT and DPO implementation.

Every function here is a thin wrapper around :mod:`smalign`, which is where the
real implementation lives:

* :mod:`smalign.batching` - tokenisation, truncation, padding, response masks;
* :mod:`smalign.objectives` - response log probabilities, SFT loss, DPO loss.

Documented conventions
----------------------
* The tokenizer's EOS token is appended to each response *before* the
  ``max_length`` truncation, so a truncated response may lose its EOS.
* Batches are padded to a fixed ``max_length``; prompt and padding positions are
  ``-100`` in ``labels`` and ``0`` in ``attention_mask``.
* ``compute_response_logprobs`` returns the *sum* of response-token log
  probabilities per row.
* ``compute_sft_loss`` uses the ``token_mean`` reduction: the summed response
  negative log-likelihood divided by the number of predicted response tokens in
  the batch. This is the reduction used for every SFT number in the report.
* Reference-model computations inside ``compute_dpo_loss`` run under
  ``torch.no_grad()``; gradients reach the policy model only.
* Input row order is preserved in every returned tensor.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):  # allow a bare clone to import smalign without installing
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from smalign import batching, objectives

#: Reduction reported for SFT, kept in one place so code and report agree.
SFT_REDUCTION = objectives.DEFAULT_SFT_REDUCTION


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return batching.build_lm_batch(
        tokenizer,
        prompts,
        responses,
        max_length=max_length,
        pad_to="max_length",
    )


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return objectives.response_logprobs(
        model,
        tokenizer,
        prompts,
        responses,
        max_length=max_length,
        pad_to="max_length",
    )


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return objectives.sft_loss(
        model,
        tokenizer,
        prompts,
        responses,
        max_length=max_length,
        reduction=SFT_REDUCTION,
        pad_to="max_length",
    )


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return batching.build_preference_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length=max_length,
        pad_to="max_length",
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    loss, _ = objectives.dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta,
    )
    return loss


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    loss, _ = objectives.dpo_loss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta=beta,
        max_length=max_length,
        pad_to="max_length",
    )
    return loss
