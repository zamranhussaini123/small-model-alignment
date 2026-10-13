"""Seeding, I/O, and small helpers shared by every stage of the pipeline."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

# Torch and NumPy are optional here. The judging stage runs in Kaggle's
# benchmarks container, which carries kaggle_benchmarks but not necessarily the
# training stack - and judging only needs the JSON/JSONL helpers below. Anything
# that genuinely needs torch raises a clear error when called.
try:
    import numpy as np
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the runtime image
    np = None
    torch = None
    TORCH_AVAILABLE = False

DEFAULT_SEED = 42

#: Exit code meaning "stopped cleanly on the time budget, rerun to continue".
#: Defined here rather than in :mod:`smalign.checkpointing` so the orchestrator
#: can import it without pulling in torch.
PAUSED_EXIT_CODE = 42


def _require_torch(what: str) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            f"{what} needs PyTorch, which is not installed in this runtime. "
            "This is expected in the benchmarks container, which only runs judging."
        )


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and Torch so a run can be repeated."""

    _require_torch("set_seed")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def read_jsonl(path) -> list:
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def append_jsonl(path, row) -> None:
    """Append one record and flush, so partial progress survives a disconnect."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


#: CUDA OOM is ``torch.OutOfMemoryError`` on recent Torch and
#: ``torch.cuda.OutOfMemoryError`` on older builds; catch whichever exists.
OOM_ERRORS = (
    tuple(
        error
        for error in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(error, type)
    )
    or (RuntimeError,)
) if TORCH_AVAILABLE else (RuntimeError,)


def release_cuda_memory() -> None:
    """Hand back cached-but-unused CUDA blocks.

    Long training runs fragment the allocator: a run that died here had 3.4 GiB
    reserved but unallocated while a 2.3 GiB validation tensor failed to fit.
    Emptying the cache before a big allocation costs a few milliseconds and
    reclaims exactly that space.
    """

    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_batch(batch: dict, start: int, end: int) -> dict:
    """Row slice of a batch, including the nested chosen/rejected DPO shape."""

    sliced = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            sliced[key] = {name: tensor[start:end] for name, tensor in value.items()}
        else:
            sliced[key] = value[start:end]
    return sliced


def batch_rows(batch: dict) -> int:
    for value in batch.values():
        if isinstance(value, dict):
            return next(iter(value.values())).shape[0]
        return value.shape[0]
    return 0


def run_with_oom_backoff(function, batch: dict):
    """Apply ``function`` to a batch, splitting it row-by-row if CUDA runs out.

    Validation allocates a (rows x tokens x vocab) float32 tensor, which for a
    152k vocabulary is gigabytes. Rather than pick an eval batch size small
    enough to be safe on the worst day, run at the fast size and degrade only
    when an allocation actually fails. Results are identical either way: the
    caller aggregates sums, and rows are independent.
    """

    try:
        return [function(batch)]
    except OOM_ERRORS:
        rows = batch_rows(batch)
        release_cuda_memory()
        if rows <= 1:
            raise
        print(f"[oom] validation batch of {rows} did not fit; retrying row by row", flush=True)
        results = []
        for index in range(rows):
            results.append(function(split_batch(batch, index, index + 1)))
            release_cuda_memory()
        return results


def run_and_exit(main) -> None:
    """Run a CLI entry point, then exit the process immediately.

    ``datasets`` leaves background threads behind and PyTorch ``DataLoader``
    workers are separate processes; either can keep the interpreter alive long
    after the work is finished. A parent that is streaming our output then waits
    on a pipe that never closes, and the whole pipeline hangs - which is exactly
    how a Kaggle run burns hours doing nothing.

    ``os._exit`` skips interpreter teardown. That is safe here because every
    output file is written and closed before this point; the streams are flushed
    explicitly first so no output is lost.
    """

    import os
    import traceback

    code = 0
    try:
        main()
    except SystemExit as requested:
        code = requested.code if isinstance(requested.code, int) else 0
    except BaseException:
        traceback.print_exc()
        code = 1

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def load_causal_lm(model_path, dtype=None, **kwargs):
    """Load a causal LM across the Transformers v4/v5 dtype-keyword change.

    v4 takes ``torch_dtype=``; v5 renamed it to ``dtype=``. Both versions accept
    ``**kwargs`` on ``from_pretrained``, so passing the wrong one does not raise -
    it is silently ignored and the model comes back in the wrong precision,
    which on a T4 shows up much later as an out-of-memory error or a dtype
    mismatch. So the result is verified and corrected rather than trusted.
    """

    from transformers import AutoModelForCausalLM

    if dtype is None:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    last_error = None
    for keyword in ("dtype", "torch_dtype"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, **{keyword: dtype}, **kwargs
            )
        except TypeError as error:
            last_error = error
            continue
        try:
            loaded_dtype = next(model.parameters()).dtype
        except StopIteration:
            return model
        if loaded_dtype == dtype:
            return model
        return model.to(dtype)

    raise RuntimeError(f"could not load {model_path} as {dtype}: {last_error}")


def model_device(model) -> torch.device:
    """Device of the first parameter, falling back to CPU for parameterless models."""

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def count_parameters(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def describe_environment() -> dict:
    """Versions and GPU inventory, recorded next to every run for the report."""

    if not TORCH_AVAILABLE:
        return {"torch": "not installed", "cuda_available": False,
                "device_count": 0, "devices": []}
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    for index in range(info["device_count"]):
        properties = torch.cuda.get_device_properties(index)
        info["devices"].append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / 1024**3, 2),
            }
        )
    for package in ("transformers", "datasets", "peft", "trl", "accelerate"):
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", "unknown")
        except Exception:
            info[package] = "not installed"
    return info


def make_grad_scaler(enabled: bool):
    """A gradient scaler that works across Torch versions.

    ``torch.cuda.amp.GradScaler`` is deprecated in favour of
    ``torch.amp.GradScaler("cuda", ...)``, but Kaggle images lag, so try the
    new spelling first and fall back.
    """

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older Torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Stopwatch:
    """Minimal wall-clock timer used in the training logs."""

    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def human(self) -> str:
        seconds = int(self.elapsed())
        return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"
