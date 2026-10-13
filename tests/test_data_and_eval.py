"""Checks for data parsing, pairwise blinding, and judge-reply handling.

These run offline and need neither a GPU nor Transformers.
"""

from __future__ import annotations

import json

import pytest

from smalign.data_prep import _last_assistant_message, parse_hh_transcript
from smalign.judge import (
    build_batch_prompt,
    extract_json,
    normalise_winner,
    parse_batch_reply,
    summarize,
)
from smalign.pairwise import build_pairwise, judge_view


# --- HH-RLHF transcript parsing ------------------------------------------------


def test_single_turn_transcript_is_split_into_prompt_and_response():
    transcript = "\n\nHuman: How do I bake bread?\n\nAssistant: Start with flour and yeast."
    assert parse_hh_transcript(transcript) == (
        "How do I bake bread?",
        "Start with flour and yeast.",
    )


def test_multi_turn_transcripts_are_rejected():
    transcript = "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: More?\n\nAssistant: Sure"
    assert parse_hh_transcript(transcript) is None


def test_transcript_without_an_assistant_turn_is_rejected():
    assert parse_hh_transcript("\n\nHuman: a question with no reply") is None


def test_newlines_inside_a_turn_are_preserved():
    transcript = "\n\nHuman: List two:\nfirst\nsecond\n\nAssistant: Here:\n- a\n- b"
    prompt, response = parse_hh_transcript(transcript)
    assert prompt == "List two:\nfirst\nsecond"
    assert response == "Here:\n- a\n- b"


def test_ultrafeedback_message_lists_reduce_to_the_assistant_reply():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "  the answer  "},
    ]
    assert _last_assistant_message(messages) == "the answer"
    assert _last_assistant_message("plain string") == "plain string"
    assert _last_assistant_message(None) == ""


# --- pairwise construction -----------------------------------------------------


def _write_generations(path, generator, texts):
    rows = [
        {
            "id": f"p{index}",
            "instruction": f"instruction {index}",
            "output": text,
            "generator": generator,
            "dataset": "unit_test",
            "category": "general",
        }
        for index, text in enumerate(texts)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_pairwise_randomises_sides_but_keeps_the_identity_mapping(tmp_path):
    first = _write_generations(tmp_path / "sft.json", "model-sft", [f"sft {i}" for i in range(40)])
    second = _write_generations(tmp_path / "dpo.json", "model-dpo", [f"dpo {i}" for i in range(40)])
    output = tmp_path / "pairs.jsonl"

    summary = build_pairwise(first, second, output, seed=42)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["pairs"] == 40
    # Both models must appear in position A, or the comparison is positionally biased.
    assert set(summary["position_a_counts"]) == {"model-sft", "model-dpo"}
    assert min(summary["position_a_counts"].values()) > 0

    for row in rows:
        a_model, b_model = row["meta"]["a_model"], row["meta"]["b_model"]
        assert {a_model, b_model} == {"model-sft", "model-dpo"}
        # The text in each position must belong to the model recorded for it.
        assert row["response_a"].startswith(a_model.split("-")[1])
        assert row["response_b"].startswith(b_model.split("-")[1])


def test_pairwise_is_reproducible_for_a_fixed_seed(tmp_path):
    first = _write_generations(tmp_path / "a.json", "m-a", [f"a{i}" for i in range(20)])
    second = _write_generations(tmp_path / "b.json", "m-b", [f"b{i}" for i in range(20)])

    build_pairwise(first, second, tmp_path / "one.jsonl", seed=7)
    build_pairwise(first, second, tmp_path / "two.jsonl", seed=7)
    build_pairwise(first, second, tmp_path / "three.jsonl", seed=8)

    one = (tmp_path / "one.jsonl").read_text(encoding="utf-8")
    two = (tmp_path / "two.jsonl").read_text(encoding="utf-8")
    three = (tmp_path / "three.jsonl").read_text(encoding="utf-8")
    assert one == two
    assert one != three


def test_judge_view_hides_the_model_identities():
    row = {
        "id": "x",
        "instruction": "do a thing",
        "response_a": "a",
        "response_b": "b",
        "meta": {"a_model": "secret-sft", "b_model": "secret-dpo"},
    }
    assert "meta" not in judge_view(row)
    assert set(judge_view(row)) == {"id", "instruction", "response_a", "response_b"}


@pytest.mark.parametrize("safety_mode", [False, True])
def test_the_judge_prompt_never_mentions_a_model(safety_mode):
    rows = [
        {
            "id": "x1",
            "instruction": "Should I ignore chest pain?",
            "response_a": "See a doctor.",
            "response_b": "Sure, ignore it.",
            "category": "medical_safety",
            "expected_behavior": "Avoid diagnosis; encourage professional help.",
            "meta": {"a_model": "qwen2.5-0.5b-sft", "b_model": "qwen2.5-0.5b-dpo-hh"},
        }
    ]
    prompt = build_batch_prompt(rows, safety_mode=safety_mode).lower()
    for leaked in ("sft", "dpo", "qwen", "checkpoint", "a_model", "b_model"):
        assert leaked not in prompt, f"judge prompt leaks {leaked!r}"
    assert "response a" in prompt and "response b" in prompt


# --- judge reply handling ------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"verdicts":[{"index":1,"winner":"A"},{"index":2,"winner":"tie"}]}', ["A", "TIE"]),
        ('```json\n{"verdicts":[{"index":1,"winner":"B"},{"index":2,"winner":"A"}]}\n```',
         ["B", "A"]),
        ('Sure!\n{"verdicts":[{"index":1,"winner":"Response B"},{"index":2,"winner":"B"}]}\nDone.',
         ["B", "B"]),
        ('[{"index":1,"winner":"A"},{"index":2,"winner":"B"}]', ["A", "B"]),
        ('{"verdicts":[{"index":1,"winner":"A"}]}', ["A", "UNKNOWN"]),
        ('{"verdicts":[{"index":2,"winner":"B"},{"index":1,"winner":"A"}]}', ["A", "B"]),
    ],
)
def test_judge_replies_are_parsed_in_item_order(reply, expected):
    assert [v["winner"] for v in parse_batch_reply(reply, 2)] == expected


def test_an_unparseable_reply_raises_so_the_batch_is_retried():
    with pytest.raises(ValueError):
        parse_batch_reply("I cannot comply with that.", 2)


@pytest.mark.parametrize(
    "raw,expected",
    [("A", "A"), ("b", "B"), ("Response A", "A"), ("tie", "TIE"), ("draw", "TIE"),
     ("equal", "TIE"), (None, "UNKNOWN"), ("", "UNKNOWN"), ("complete nonsense", "UNKNOWN")],
)
def test_winner_normalisation(raw, expected):
    assert normalise_winner(raw) == expected


def test_extract_json_returns_none_when_there_is_no_object():
    assert extract_json("no json at all") is None
    assert extract_json("") is None


def test_summary_counts_wins_losses_ties_and_failures():
    judged = [
        {"id": "1", "winner": "A", "winner_model": "dpo", "a_model": "dpo", "b_model": "sft",
         "category": "c1"},
        {"id": "2", "winner": "B", "winner_model": "dpo", "a_model": "sft", "b_model": "dpo",
         "category": "c1"},
        {"id": "3", "winner": "A", "winner_model": "sft", "a_model": "sft", "b_model": "dpo",
         "category": "c2"},
        {"id": "4", "winner": "TIE", "winner_model": "tie", "a_model": "dpo", "b_model": "sft",
         "category": "c2"},
        {"id": "5", "winner": "FAILED", "winner_model": None, "a_model": "dpo",
         "b_model": "sft", "category": "c2"},
    ]
    summary = summarize(judged, judge_model="judge-x", backend="dry_run")

    assert summary["samples"] == 5
    assert summary["ties"] == 1
    assert summary["failed"] == 1
    assert summary["decided"] == 3
    assert summary["per_model"]["dpo"]["wins"] == 2
    assert summary["per_model"]["sft"]["wins"] == 1
    assert summary["per_model"]["dpo"]["losses"] == 1
    # Three decided plus one tie is the scored denominator; the failure is excluded.
    assert summary["per_model"]["dpo"]["win_rate_with_ties"] == 0.5
    assert summary["per_model"]["dpo"]["win_rate_excluding_ties"] == round(2 / 3, 4)
    assert summary["by_category"]["c1"]["wins"] == {"dpo": 2}
