"""Validation must survive a CUDA out-of-memory and give the same answer.

A full run died at SFT step 930 of 937 - during the *final* validation, after
two hours of training. An 8x512 eval batch needs a float32 (rows x tokens x
vocab) tensor, which for a 152k vocabulary is 2.3 GiB, and by then the allocator
had 3.4 GiB reserved but fragmented. Ten earlier validations had fit; the
eleventh did not.

The fix retries a failed batch row by row. These tests check the part that
matters: that the fallback is numerically identical to the fast path, so a run
that degrades does not silently report different numbers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from smalign.utils import OOM_ERRORS, batch_rows, run_with_oom_backoff, split_batch


def _flat_batch(rows: int = 4):
    return {
        "input_ids": torch.arange(rows * 5).reshape(rows, 5),
        "attention_mask": torch.ones(rows, 5, dtype=torch.long),
        "labels": torch.arange(rows * 5).reshape(rows, 5),
    }


def _nested_batch(rows: int = 4):
    return {"chosen": _flat_batch(rows), "rejected": _flat_batch(rows)}


def test_batch_rows_handles_flat_and_nested_batches():
    assert batch_rows(_flat_batch(6)) == 6
    assert batch_rows(_nested_batch(3)) == 3


def test_split_batch_slices_rows_of_a_flat_batch():
    part = split_batch(_flat_batch(4), 1, 3)
    assert part["input_ids"].shape[0] == 2
    assert torch.equal(part["input_ids"][0], torch.arange(5, 10))


def test_split_batch_slices_both_sides_of_a_dpo_batch():
    part = split_batch(_nested_batch(4), 2, 3)
    for side in ("chosen", "rejected"):
        assert part[side]["input_ids"].shape[0] == 1
        assert part[side]["labels"].shape[0] == 1


def test_the_fast_path_runs_as_a_single_call():
    calls = []

    def score(part):
        calls.append(batch_rows(part))
        return float(part["input_ids"].sum())

    results = run_with_oom_backoff(score, _flat_batch(4))
    assert calls == [4], "no OOM, so the batch should not be split"
    assert len(results) == 1


def test_an_oom_falls_back_to_one_row_at_a_time_with_the_same_total():
    """The aggregate must not change just because the batch was split."""

    batch = _flat_batch(4)
    expected_total = float(batch["input_ids"].sum())

    state = {"failed": False}

    def score(part):
        if not state["failed"] and batch_rows(part) > 1:
            state["failed"] = True
            raise OOM_ERRORS[0]("CUDA out of memory (simulated)")
        return float(part["input_ids"].sum())

    results = run_with_oom_backoff(score, batch)
    assert len(results) == 4, "a 4-row batch should become four single-row calls"
    assert sum(results) == pytest.approx(expected_total)


def test_a_dpo_batch_also_degrades_row_by_row():
    batch = _nested_batch(3)
    state = {"failed": False}

    def score(part):
        if not state["failed"] and batch_rows(part) > 1:
            state["failed"] = True
            raise OOM_ERRORS[0]("CUDA out of memory (simulated)")
        return batch_rows(part)

    results = run_with_oom_backoff(score, batch)
    assert results == [1, 1, 1]


def test_an_oom_on_a_single_row_is_re_raised():
    """Splitting cannot help below one row, so the error must surface."""

    def always_fails(part):
        raise OOM_ERRORS[0]("CUDA out of memory (simulated)")

    with pytest.raises(OOM_ERRORS):
        run_with_oom_backoff(always_fails, _flat_batch(1))


def test_a_non_memory_error_is_not_retried():
    calls = []

    def broken(part):
        calls.append(1)
        raise ValueError("a real bug, not a memory problem")

    with pytest.raises(ValueError):
        run_with_oom_backoff(broken, _flat_batch(4))
    assert len(calls) == 1, "a genuine bug must not be retried row by row"


def test_release_cuda_memory_is_safe_without_a_gpu():
    from smalign.utils import release_cuda_memory

    release_cuda_memory()  # must be a no-op on CPU, not an error
