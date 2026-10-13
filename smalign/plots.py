"""Training-behaviour plots for the report.

Reads the ``metrics.jsonl`` that each run appends to, so plots can be made from
a partial run or from an attached Kaggle output dataset without retraining.

Smoothing is an exponential moving average and is drawn *behind* the raw series
rather than instead of it, so instability stays visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from smalign.utils import read_jsonl  # noqa: E402

SMOOTHING = 0.6
RAW_STYLE = dict(alpha=0.25, linewidth=1.0)
SMOOTH_STYLE = dict(linewidth=2.0)


def _ema(values: list[float], weight: float = SMOOTHING) -> list[float]:
    smoothed: list[float] = []
    running = None
    for value in values:
        running = value if running is None else weight * running + (1 - weight) * value
        smoothed.append(running)
    return smoothed


def _series(records: list[dict], event: str, key: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for record in records:
        if record.get("event") == event and key in record and record[key] is not None:
            steps.append(record["step"])
            values.append(float(record[key]))
    return steps, values


def _draw(axis, steps, values, label, color=None, smooth=True):
    if not steps:
        return
    if smooth and len(values) > 3:
        axis.plot(steps, values, color=color, **RAW_STYLE)
        axis.plot(steps, _ema(values), label=f"{label} (EMA {SMOOTHING})", color=color,
                  **SMOOTH_STYLE)
    else:
        axis.plot(steps, values, marker="o", markersize=3, label=label, color=color,
                  **SMOOTH_STYLE)


def plot_sft(metrics_path, output_path, title: str = "SFT training") -> str | None:
    """Training and validation loss against optimizer steps."""

    records = read_jsonl(metrics_path)
    if not records:
        print(f"[plots] no records in {metrics_path}")
        return None

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=140)
    _draw(axis, *_series(records, "train", "train_loss"), label="train loss", color="#3b6ea5")
    val_steps, val_values = _series(records, "eval", "val_loss")
    _draw(axis, val_steps, val_values, label="validation loss", color="#c1532a", smooth=False)

    if val_values:
        best_index = min(range(len(val_values)), key=val_values.__getitem__)
        axis.axvline(val_steps[best_index], color="#444", linestyle="--", linewidth=1)
        axis.annotate(
            f"best checkpoint\nstep {val_steps[best_index]}, loss {val_values[best_index]:.3f}",
            xy=(val_steps[best_index], val_values[best_index]),
            xytext=(8, 14),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xlabel("optimizer step")
    axis.set_ylabel("response-only loss (per token)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    print(f"[plots] wrote {output_path}")
    return str(output_path)


def plot_dpo(metrics_path, output_path, title: str = "DPO training") -> str | None:
    """DPO loss, reward accuracy, and reward margin on one figure."""

    records = read_jsonl(metrics_path)
    if not records:
        print(f"[plots] no records in {metrics_path}")
        return None

    figure, axes = plt.subplots(3, 1, figsize=(8, 9), dpi=140, sharex=True)

    _draw(axes[0], *_series(records, "train", "train_loss"), label="train DPO loss",
          color="#3b6ea5")
    _draw(axes[0], *_series(records, "eval", "val_loss"), label="validation DPO loss",
          color="#c1532a", smooth=False)
    axes[0].axhline(0.6931, color="#777", linestyle=":", linewidth=1)
    axes[0].annotate("log 2: policy == reference", xy=(0.99, 0.6931),
                     xycoords=("axes fraction", "data"), xytext=(0, 3),
                     textcoords="offset points", fontsize=7, ha="right", va="bottom",
                     color="#555")
    axes[0].set_ylabel("DPO loss")
    axes[0].set_title(title)

    _draw(axes[1], *_series(records, "train", "train_reward_accuracy"),
          label="train reward accuracy", color="#3b6ea5")
    accuracy_steps, accuracy_values = _series(records, "eval", "val_reward_accuracy")
    _draw(axes[1], accuracy_steps, accuracy_values, label="validation reward accuracy",
          color="#c1532a", smooth=False)
    axes[1].axhline(0.5, color="#777", linestyle=":", linewidth=1)
    if accuracy_values:
        best_index = max(range(len(accuracy_values)), key=accuracy_values.__getitem__)
        axes[1].axvline(accuracy_steps[best_index], color="#444", linestyle="--", linewidth=1)
        axes[1].annotate(
            f"selected checkpoint\nstep {accuracy_steps[best_index]}, "
            f"acc {accuracy_values[best_index]:.3f}",
            xy=(accuracy_steps[best_index], accuracy_values[best_index]),
            xytext=(8, -24),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_ylabel("reward accuracy")

    _draw(axes[2], *_series(records, "train", "train_reward_margin"),
          label="train reward margin", color="#3b6ea5")
    _draw(axes[2], *_series(records, "eval", "val_reward_margin"),
          label="validation reward margin", color="#c1532a", smooth=False)
    axes[2].axhline(0.0, color="#777", linestyle=":", linewidth=1)
    axes[2].set_ylabel(r"reward margin  $r_w - r_l$")
    axes[2].set_xlabel("optimizer step")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    print(f"[plots] wrote {output_path}")
    return str(output_path)


def plot_lora_history(history_path, output_path, title: str = "LoRA-DPO training") -> str | None:
    """TRL log history: loss plus reward accuracy and margin where present."""

    from smalign.utils import read_json

    path = Path(history_path)
    if not path.exists():
        print(f"[plots] missing {history_path}")
        return None
    history = read_json(path)

    def pick(key):
        steps, values = [], []
        for record in history:
            if key in record and record[key] is not None:
                steps.append(record.get("step", len(steps)))
                values.append(float(record[key]))
        return steps, values

    figure, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=140, sharex=True)
    _draw(axes[0], *pick("loss"), label="train loss", color="#3b6ea5")
    _draw(axes[0], *pick("eval_loss"), label="eval loss", color="#c1532a", smooth=False)
    axes[0].set_ylabel("DPO loss")
    axes[0].set_title(title)

    for key, color in (("rewards/accuracies", "#3b6ea5"), ("eval_rewards/accuracies", "#c1532a")):
        _draw(axes[1], *pick(key), label=key, color=color, smooth=key.startswith("rewards"))
    for key, color in (("rewards/margins", "#2e7d32"), ("eval_rewards/margins", "#7b1fa2")):
        _draw(axes[1], *pick(key), label=key, color=color, smooth=key.startswith("rewards"))
    axes[1].set_ylabel("reward accuracy / margin")
    axes[1].set_xlabel("step")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    print(f"[plots] wrote {output_path}")
    return str(output_path)
