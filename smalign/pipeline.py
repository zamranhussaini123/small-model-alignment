"""Stage orchestration: skip what is done, resume what was paused, never die.

A Kaggle commit run gets one shot. If an exception escapes a cell, the rest of
the notebook does not run and hours of finished work can go unsaved. So every
stage here is wrapped: failures are recorded and the notebook keeps going to the
end, where the outputs are written and the version is saved.

The ledger (``outputs/pipeline_state.json``) is what makes a rerun cheap. Each
stage records its status, so the next run skips what is ``completed``, continues
what is ``paused``, and retries what ``failed``. Combined with the resume
checkpoints in :mod:`smalign.checkpointing`, a session that dies at hour five
picks up near hour five.

Stage statuses
--------------
``completed``  finished, artifacts verified on disk
``paused``     stopped on its time budget; rerun continues from the checkpoint
``failed``     raised an error; rerun retries from its last resume checkpoint
``skipped``    a dependency was not complete, so it never started
"""

from __future__ import annotations

import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

from smalign.utils import PAUSED_EXIT_CODE, read_json, write_json

LEDGER_NAME = "pipeline_state.json"

DONE = "completed"
PAUSED = "paused"
FAILED = "failed"
SKIPPED = "skipped"


class Ledger:
    """Durable record of which stages are done, written after every change."""

    def __init__(self, path="outputs/pipeline_state.json"):
        self.path = Path(path)
        self.state: dict = {"stages": {}}
        if self.path.exists():
            try:
                loaded = read_json(self.path)
                if isinstance(loaded, dict) and "stages" in loaded:
                    self.state = loaded
            except Exception as error:
                print(f"[pipeline] ignoring unreadable ledger {self.path}: {error}")

    def get(self, name: str) -> dict:
        return self.state["stages"].get(name, {})

    def status(self, name: str) -> str:
        return self.get(name).get("status", "not_started")

    def is_done(self, name: str) -> bool:
        return self.status(name) == DONE

    def record(self, name: str, status: str, **fields) -> None:
        entry = dict(self.get(name))
        entry.update(status=status, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **fields)
        entry["attempts"] = entry.get("attempts", 0) + (1 if status != SKIPPED else 0)
        self.state["stages"][name] = entry
        write_json(self.path, self.state)

    def summary_rows(self) -> list[tuple[str, str, str, str]]:
        rows = []
        for name, entry in self.state["stages"].items():
            rows.append((
                name,
                entry.get("status", "?"),
                entry.get("detail", ""),
                entry.get("elapsed", ""),
            ))
        return rows


def _missing(paths) -> list[str]:
    return [str(p) for p in paths if not Path(p).exists()]


def stream(command: list[str], env: dict | None = None, timeout: float | None = None,
           quiet_timeout: float | None = None) -> int:
    """Run a command, echo its output live, and return the exit code.

    Output is streamed rather than captured because these stages run for hours
    and the log is the only way to see progress. ``PYTHONUNBUFFERED`` stops the
    child block-buffering into a pipe.

    The reading loop is deliberately not ``for line in process.stdout``. That
    form waits for the pipe to reach EOF, which happens only when *every* holder
    of the write end closes it - not just the child. A child that leaves a
    background thread running, or that spawned workers inheriting the pipe, can
    finish its work and still never close it, and the parent then waits forever.
    This loop watches the file descriptor and the child's exit status
    separately, and stops as soon as the child is gone.

    ``timeout`` caps total wall clock; ``quiet_timeout`` caps the gap between
    output lines. Either one firing kills the child rather than hanging the run.
    """

    child_env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        # Torch's own suggestion for the fragmentation pattern that killed a
        # run here: hours of training left 3.4 GiB reserved but unallocated,
        # and a 2.3 GiB validation tensor then failed to fit.
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    if env:
        child_env.update(env)

    print(f"$ {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env,
    )
    assert process.stdout is not None

    descriptor = process.stdout.fileno()
    os.set_blocking(descriptor, False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    started = time.time()
    last_output = started
    pending = b""

    def emit(chunk: bytes) -> bytes:
        *lines, remainder = (pending + chunk).split(b"\n")
        for line in lines:
            sys.stdout.write(line.decode("utf-8", "replace") + "\n")
        if lines:
            sys.stdout.flush()
        return remainder

    def drain() -> None:
        nonlocal pending
        while True:
            try:
                chunk = os.read(descriptor, 65536)
            except (BlockingIOError, OSError):
                break
            if not chunk:
                break
            pending = emit(chunk)

    killed_reason = None
    try:
        while True:
            events = selector.select(timeout=0.5)
            if events:
                try:
                    chunk = os.read(descriptor, 65536)
                except (BlockingIOError, OSError):
                    chunk = b""
                if chunk:
                    pending = emit(chunk)
                    last_output = time.time()

            if process.poll() is not None:
                drain()  # whatever the child wrote just before exiting
                break

            now = time.time()
            if timeout and now - started > timeout:
                killed_reason = f"exceeded its {timeout / 60:.0f} minute time limit"
            elif quiet_timeout and now - last_output > quiet_timeout:
                killed_reason = f"produced no output for {quiet_timeout / 60:.0f} minutes"
            if killed_reason:
                print(f"\n[pipeline] killing the stage: it {killed_reason}", flush=True)
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                break
    finally:
        selector.close()
        if pending:
            sys.stdout.write(pending.decode("utf-8", "replace") + "\n")
            sys.stdout.flush()
        try:
            process.stdout.close()
        except OSError:
            pass

    if killed_reason:
        return 124  # conventional timeout exit code
    return process.wait()


def run_stage(
    name: str,
    command: list[str],
    produces: list[str] | None = None,
    depends_on: list[str] | None = None,
    ledger: Ledger | None = None,
    force: bool = False,
    timeout: float | None = None,
    quiet_timeout: float = 3600.0,
) -> str:
    """Run one stage, honouring the ledger, and return its final status.

    Never raises. A stage that blows up is recorded and the caller continues, so
    one broken stage cannot cost the whole run its saved outputs.
    """

    ledger = ledger or Ledger()
    produces = produces or []
    depends_on = depends_on or []

    print("\n" + "=" * 78)
    print(f"STAGE: {name}")
    print("=" * 78, flush=True)

    # Already finished, and the artifacts are still there.
    if not force and ledger.is_done(name):
        missing = _missing(produces)
        if not missing:
            print(f"[pipeline] {name} already completed - skipping")
            for path in produces:
                print(f"  ok  {path}")
            return DONE
        print(f"[pipeline] {name} was marked complete but is missing {missing} - rerunning")

    unmet = [dependency for dependency in depends_on if not ledger.is_done(dependency)]
    if unmet:
        detail = f"waiting on {', '.join(unmet)}"
        print(f"[pipeline] SKIPPING {name}: {detail}")
        ledger.record(name, SKIPPED, detail=detail)
        return SKIPPED

    started = time.time()
    try:
        code = stream(command, timeout=timeout, quiet_timeout=quiet_timeout)
    except BaseException as error:  # noqa: BLE001 - a stage must not kill the notebook
        elapsed = _human(time.time() - started)
        detail = f"{type(error).__name__}: {error}"
        print(f"\n[pipeline] {name} FAILED after {elapsed}: {detail}")
        ledger.record(name, FAILED, detail=detail, elapsed=elapsed)
        return FAILED

    elapsed = _human(time.time() - started)

    if code == PAUSED_EXIT_CODE:
        print(f"\n[pipeline] {name} PAUSED after {elapsed} - rerun to continue from its checkpoint")
        ledger.record(name, PAUSED, detail="stopped on time budget", elapsed=elapsed)
        return PAUSED

    if code == 124:
        detail = "killed: timed out or went silent"
        print(f"\n[pipeline] {name} TIMED OUT after {elapsed}")
        ledger.record(name, FAILED, detail=detail, elapsed=elapsed)
        return FAILED

    if code != 0:
        detail = f"exit code {code}"
        print(f"\n[pipeline] {name} FAILED after {elapsed} ({detail})")
        ledger.record(name, FAILED, detail=detail, elapsed=elapsed)
        return FAILED

    missing = _missing(produces)
    if missing:
        detail = f"exited 0 but did not produce {missing}"
        print(f"\n[pipeline] {name} FAILED: {detail}")
        ledger.record(name, FAILED, detail=detail, elapsed=elapsed)
        return FAILED

    print(f"\n[pipeline] {name} completed in {elapsed}")
    for path in produces:
        print(f"  ok  {path}")
    ledger.record(name, DONE, detail="", elapsed=elapsed)
    return DONE


def _human(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def print_ledger(ledger: Ledger) -> None:
    """A compact status table; the first thing to read when a run comes back."""

    rows = ledger.summary_rows()
    if not rows:
        print("no stages have run yet")
        return

    marks = {DONE: "done", PAUSED: "PAUSED", FAILED: "FAILED", SKIPPED: "skipped"}
    width = max(len(name) for name, *_ in rows)
    print("\n" + "=" * 78)
    print("PIPELINE STATUS")
    print("=" * 78)
    for name, status, detail, elapsed in rows:
        line = f"  {name:<{width}}  {marks.get(status, status):<8} {elapsed:>10}"
        if detail:
            line += f"   {detail}"
        print(line)

    paused = [name for name, status, *_ in rows if status == PAUSED]
    failed = [name for name, status, *_ in rows if status == FAILED]
    skipped = [name for name, status, *_ in rows if status == SKIPPED]
    print("-" * 78)
    if paused:
        print(f"  {len(paused)} stage(s) paused on the time budget: {', '.join(paused)}")
        print("  -> save this version, attach it as PREV_OUTPUT, and run again to continue")
    if failed:
        print(f"  {len(failed)} stage(s) failed: {', '.join(failed)}")
        print("  -> read the log above; rerunning resumes from the last checkpoint")
    if skipped:
        print(f"  {len(skipped)} stage(s) skipped for unmet dependencies: {', '.join(skipped)}")
    if not (paused or failed or skipped):
        print("  every stage completed")
    print("=" * 78 + "\n")
