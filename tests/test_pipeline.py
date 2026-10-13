"""Orchestration: streaming, hang detection, and the stage ledger.

The first test here is a regression test for a real failure. A Kaggle run sat
idle for forty minutes because the parent read the child's output with
``for line in process.stdout``, which waits for the pipe to reach EOF. The child
had finished and exited, but something still held the write end open, so EOF
never came and the pipeline hung with no output and no error.
"""

from __future__ import annotations

import json
import sys
import time

from smalign.pipeline import DONE, FAILED, PAUSED, SKIPPED, Ledger, run_stage, stream

PYTHON = sys.executable


def _cmd(body: str) -> list[str]:
    return [PYTHON, "-c", body]


# --- streaming -----------------------------------------------------------------


def test_stream_returns_the_exit_code_and_echoes_output(capfd):
    code = stream(_cmd("print('hello from the child')"))
    assert code == 0
    assert "hello from the child" in capfd.readouterr().out


def test_stream_propagates_a_nonzero_exit_code():
    assert stream(_cmd("raise SystemExit(3)")) == 3


def test_stream_returns_when_a_grandchild_still_holds_the_pipe(capfd):
    """The forty-minute hang, reproduced.

    The child spawns a grandchild that inherits stdout and outlives it. The
    child then exits. A reader waiting for EOF would block until the grandchild
    died; this must return as soon as the child does.
    """

    body = (
        "import subprocess, sys\n"
        "print('child working', flush=True)\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(45)'])\n"
        "print('child done', flush=True)\n"
    )
    started = time.time()
    code = stream(_cmd(body), quiet_timeout=30.0)
    elapsed = time.time() - started

    assert code == 0
    assert elapsed < 20, f"stream waited {elapsed:.1f}s for a grandchild to die"
    out = capfd.readouterr().out
    assert "child working" in out and "child done" in out


def test_stream_kills_a_stage_that_goes_silent():
    """A hung child must be killed, not waited on forever."""

    started = time.time()
    code = stream(_cmd("import time; time.sleep(120)"), quiet_timeout=2.0)
    elapsed = time.time() - started

    assert code == 124, "a killed stage reports the timeout exit code"
    assert elapsed < 40, f"took {elapsed:.1f}s to give up"


def test_stream_kills_a_stage_that_exceeds_its_total_budget():
    body = (
        "import time, sys\n"
        "for _ in range(200):\n"
        "    print('still going', flush=True); time.sleep(0.2)\n"
    )
    started = time.time()
    # Chatty, so quiet_timeout never fires; only the total cap can stop it.
    code = stream(_cmd(body), timeout=3.0, quiet_timeout=60.0)
    assert code == 124
    assert time.time() - started < 40


def test_stream_does_not_lose_a_final_line_without_a_newline(capfd):
    code = stream(_cmd("import sys; sys.stdout.write('no trailing newline')"))
    assert code == 0
    assert "no trailing newline" in capfd.readouterr().out


# --- ledger and stage control ---------------------------------------------------


def test_completed_stage_is_skipped_on_a_rerun(tmp_path, capfd):
    ledger = Ledger(tmp_path / "state.json")
    marker = tmp_path / "artifact.txt"
    command = _cmd(f"open({str(marker)!r}, 'w').write('x')")

    assert run_stage("build", command, [str(marker)], ledger=ledger) == DONE
    marker.write_text("untouched by a second run")

    assert run_stage("build", command, [str(marker)], ledger=ledger) == DONE
    assert "already completed" in capfd.readouterr().out
    assert marker.read_text() == "untouched by a second run"


def test_a_stage_reruns_when_its_artifact_has_vanished(tmp_path):
    ledger = Ledger(tmp_path / "state.json")
    marker = tmp_path / "artifact.txt"
    command = _cmd(f"open({str(marker)!r}, 'w').write('rebuilt')")

    run_stage("build", command, [str(marker)], ledger=ledger)
    marker.unlink()

    assert run_stage("build", command, [str(marker)], ledger=ledger) == DONE
    assert marker.read_text() == "rebuilt"


def test_exit_zero_without_the_artifact_counts_as_failure(tmp_path):
    ledger = Ledger(tmp_path / "state.json")
    status = run_stage("build", _cmd("pass"), [str(tmp_path / "never.txt")], ledger=ledger)
    assert status == FAILED
    assert "did not produce" in ledger.get("build")["detail"]


def test_paused_exit_code_is_recorded_as_paused_not_failed(tmp_path):
    from smalign.utils import PAUSED_EXIT_CODE

    ledger = Ledger(tmp_path / "state.json")
    status = run_stage("train", _cmd(f"raise SystemExit({PAUSED_EXIT_CODE})"), ledger=ledger)
    assert status == PAUSED
    assert ledger.status("train") == PAUSED


def test_dependents_are_skipped_when_a_dependency_did_not_finish(tmp_path):
    ledger = Ledger(tmp_path / "state.json")
    assert run_stage("first", _cmd("raise SystemExit(1)"), ledger=ledger) == FAILED
    assert run_stage("second", _cmd("pass"), depends_on=["first"], ledger=ledger) == SKIPPED


def test_a_failing_stage_never_raises(tmp_path):
    """One broken stage must not stop the notebook and cost the run its outputs."""

    ledger = Ledger(tmp_path / "state.json")
    for command in (_cmd("raise RuntimeError('boom')"), ["definitely-not-a-real-binary"]):
        assert run_stage("boom", command, ledger=ledger) == FAILED


def test_the_ledger_survives_a_restart(tmp_path):
    path = tmp_path / "state.json"
    first = Ledger(path)
    first.record("sft", DONE, detail="", elapsed="1h00m00s")

    second = Ledger(path)
    assert second.is_done("sft")
    assert second.get("sft")["elapsed"] == "1h00m00s"


def test_a_corrupt_ledger_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    ledger = Ledger(path)
    assert ledger.status("anything") == "not_started"
    ledger.record("sft", DONE)
    assert json.loads(path.read_text())["stages"]["sft"]["status"] == DONE


def test_force_reruns_a_completed_stage(tmp_path):
    ledger = Ledger(tmp_path / "state.json")
    counter = tmp_path / "runs.txt"
    command = _cmd(
        f"p={str(counter)!r}\n"
        "import os\n"
        "n = int(open(p).read()) if os.path.exists(p) else 0\n"
        "open(p,'w').write(str(n+1))\n"
    )
    run_stage("build", command, [str(counter)], ledger=ledger)
    run_stage("build", command, [str(counter)], ledger=ledger)
    assert counter.read_text() == "1"

    run_stage("build", command, [str(counter)], ledger=ledger, force=True)
    assert counter.read_text() == "2"
