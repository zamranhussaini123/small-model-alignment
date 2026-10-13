"""Build blinded pairwise comparison files from two generation files.

Side assignment is randomised per prompt with a fixed seed, and the mapping
from position to model is kept in a ``meta`` block that the judge prompt never
sees. Decoding the A/B verdict back into a model identity happens after judging.
"""

from __future__ import annotations

import random
from pathlib import Path

from smalign.utils import DEFAULT_SEED, read_json, write_json, write_jsonl

#: Keys handed to the judge. Anything outside this set stays hidden.
JUDGE_VISIBLE_KEYS = ("id", "instruction", "response_a", "response_b", "category",
                      "expected_behavior")


def build_pairwise(
    first_path,
    second_path,
    output_path,
    seed: int = DEFAULT_SEED,
    skip_empty: bool = False,
) -> dict:
    """Pair two generation files by prompt id into a blinded JSONL file."""

    first = {row["id"]: row for row in read_json(first_path)}
    second = {row["id"]: row for row in read_json(second_path)}
    shared = [row_id for row_id in first if row_id in second]
    missing = sorted(set(first) ^ set(second))

    rng = random.Random(seed)
    rows = []
    dropped = 0
    for row_id in shared:
        left, right = first[row_id], second[row_id]
        if skip_empty and not (left["output"].strip() and right["output"].strip()):
            dropped += 1
            continue

        # Coin flip decides which model occupies position A.
        first_is_a = rng.random() < 0.5
        a_row, b_row = (left, right) if first_is_a else (right, left)

        record = {
            "id": row_id,
            "instruction": left["instruction"],
            "response_a": a_row["output"],
            "response_b": b_row["output"],
            "meta": {
                "a_model": a_row["generator"],
                "b_model": b_row["generator"],
                "dataset": left.get("dataset"),
                "first_is_a": first_is_a,
            },
        }
        for key in ("category", "expected_behavior"):
            if key in left:
                record[key] = left[key]
        rows.append(record)

    rows.sort(key=lambda row: row["id"])
    write_jsonl(output_path, rows)

    a_counts: dict[str, int] = {}
    for row in rows:
        name = row["meta"]["a_model"]
        a_counts[name] = a_counts.get(name, 0) + 1

    summary = {
        "pairs_path": str(output_path),
        "pairs": len(rows),
        "dropped_empty": dropped,
        "missing_ids": missing,
        "seed": seed,
        "models": sorted({row["meta"]["a_model"] for row in rows}
                         | {row["meta"]["b_model"] for row in rows}),
        "position_a_counts": a_counts,
    }
    write_json(Path(output_path).with_suffix(".pairs_summary.json"), summary)
    print(
        f"[pairs] {len(rows)} pairs -> {output_path}"
        f" | position A: {a_counts}"
        + (f" | dropped {dropped} empty" if dropped else "")
    )
    if missing:
        print(f"[pairs] warning: {len(missing)} ids present in only one file")
    return summary


def judge_view(row: dict) -> dict:
    """The subset of a pair that may appear in a judge prompt."""

    return {key: row[key] for key in JUDGE_VISIBLE_KEYS if key in row}
