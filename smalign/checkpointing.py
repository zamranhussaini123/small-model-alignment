"""Mid-run checkpointing so a killed session resumes instead of restarting.

Kaggle sessions die: they hit the session limit, the kernel errors out, or the
browser tab is closed. Without this module a six-hour run that dies at hour five
starts again from zero.

What a resume checkpoint holds
------------------------------
Model weights, optimizer state, scheduler state, and gradient-scaler state, plus
the exact position in the data (epoch and number of examples consumed) and the
RNG states. Restoring all of it continues the run as if it had never stopped.

Two invariants make that work:

* **Checkpoints are only written immediately after an optimizer step**, so a
  resumed run never restarts halfway through a gradient-accumulation window.
* **Data order is a pure function of (seed, epoch)**, so the remaining examples
  can be computed on resume instead of replayed. Skipping is free, not a matter
  of spinning the dataloader forward.

Writes are atomic: the payload goes to a temporary file and is renamed over the
real one, so a process killed mid-write leaves the previous checkpoint intact
rather than a truncated file.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

RESUME_FILENAME = "resume.pt"
FORMAT_VERSION = 2


def epoch_order(num_examples: int, seed: int, epoch: int) -> list[int]:
    """The example order for one epoch: deterministic in ``(seed, epoch)``.

    This replaces ``DataLoader(shuffle=True)``. Because the permutation can be
    recomputed, a resumed run can slice off the examples it already saw rather
    than iterating past them.
    """

    generator = torch.Generator().manual_seed(seed * 100_003 + epoch)
    return torch.randperm(num_examples, generator=generator).tolist()


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    except Exception as error:  # pragma: no cover - RNG layout varies by version
        print(f"[resume] could not restore RNG state ({error}); continuing with a fresh one")


def resume_path(output_dir) -> Path:
    return Path(output_dir) / RESUME_FILENAME


def save_resume_state(
    output_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    progress: dict,
    extra: dict | None = None,
) -> Path:
    """Write a resume checkpoint atomically.

    The temporary file sits beside the target so the rename stays on one
    filesystem, which is what makes it atomic.
    """

    target = resume_path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")

    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": progress,
        "extra": extra or {},
        "rng": capture_rng(),
    }

    torch.save(payload, temporary)
    # Flush to disk before the rename, so a crash cannot leave a rename
    # pointing at data that never reached the platter.
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def load_resume_state(output_dir, map_location="cpu") -> dict | None:
    """Read a resume checkpoint, or return ``None`` when there is nothing to resume.

    A corrupt or unreadable file is reported and ignored rather than raising:
    starting over is bad, but crashing on every restart is worse.
    """

    target = resume_path(output_dir)
    if not target.exists():
        return None
    try:
        # weights_only defaults to True on recent Torch; this payload is our own
        # and holds optimizer/RNG objects, so it has to be False.
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except Exception as error:
        print(f"[resume] ignoring unreadable checkpoint {target}: {error}")
        return None

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        print(f"[resume] ignoring checkpoint written in format v{version} (expected v{FORMAT_VERSION})")
        return None
    return payload


def clear_resume_state(output_dir) -> None:
    """Delete the resume checkpoint once a stage has finished.

    Worth doing promptly: with optimizer state, one of these is several GiB, and
    Kaggle's working disk is about 20 GiB in total.
    """

    for path in (resume_path(output_dir), resume_path(output_dir).with_suffix(".pt.tmp")):
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            print(f"[resume] removed {path.name} ({size / 2**30:.2f} GiB reclaimed)")


class TimeBudget:
    """Stop a stage cleanly before the session is killed from under it.

    Kaggle terminates a session at its limit with no warning and no chance to
    save. Stopping voluntarily a little early, right after an optimizer step and
    a checkpoint write, converts a total loss into a resumable pause.
    """

    def __init__(self, max_seconds: float = 0.0, reserve_seconds: float = 420.0):
        import time

        self._time = time.time
        self.start = time.time()
        self.max_seconds = float(max_seconds or 0)
        # Headroom for the final checkpoint write, which can take a minute or
        # two once optimizer state is included.
        self.reserve_seconds = reserve_seconds

    @property
    def enabled(self) -> bool:
        return self.max_seconds > 0

    def elapsed(self) -> float:
        return self._time() - self.start

    def exhausted(self) -> bool:
        if not self.enabled:
            return False
        return self.elapsed() >= max(self.max_seconds - self.reserve_seconds, 0)

    def remaining(self) -> float:
        if not self.enabled:
            return float("inf")
        return max(self.max_seconds - self.elapsed(), 0.0)


def format_eta(done: int, total: int, elapsed: float) -> str:
    """A human-readable ETA line for long unattended runs."""

    if done <= 0:
        return "eta unknown"
    rate = elapsed / done
    remaining = max(total - done, 0) * rate
    return f"eta {int(remaining // 3600)}h{int((remaining % 3600) // 60):02d}m"


def directory_size_gib(path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    total = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total / 2**30


def free_space_gib(path="/kaggle/working") -> float:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 2**30
