"""Does the mutation journal actually survive a kill? Prove it with a real kill.

A ``finally``-based restore looks correct in review and fails in exactly one
situation: the process does not get to run its ``finally``. That is not a
hypothetical -- it is what corrupted ``meshpool/controlplane/openai_api.py``
during a 78-gate sweep and silently invalidated every gate that ran afterwards.

So these tests do not assert that a journal file has the right shape. They
start a real child process, let it really mutate a file, kill it with
``TerminateProcess``/``SIGKILL`` so no cleanup can possibly run, and then check
that the next process puts the file back.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim import mutation_guard  # noqa: E402

ORIGINAL = b"the original bytes\nsecond line\n"
BROKEN = b"CORRUPTED\n"


@pytest.fixture
def victim(tmp_path_factory):
    """A file inside the repo, because the journal stores repo-relative paths."""
    d = ROOT / ".mutation-journal-test"
    d.mkdir(exist_ok=True)
    p = d / f"victim-{os.getpid()}-{time.time_ns()}.txt"
    p.write_bytes(ORIGINAL)
    yield p
    p.unlink(missing_ok=True)
    try:
        d.rmdir()
    except OSError:
        pass


@pytest.fixture(autouse=True)
def clean_journal():
    mutation_guard.restore_stale(force=True)
    yield
    mutation_guard.restore_stale(force=True)


def test_protect_restores_on_success(victim):
    with mutation_guard.protect(victim) as original:
        assert original == ORIGINAL
        victim.write_bytes(BROKEN)
        assert victim.read_bytes() == BROKEN
    assert victim.read_bytes() == ORIGINAL
    assert mutation_guard.pending() == []


def test_protect_restores_on_exception(victim):
    with pytest.raises(RuntimeError):
        with mutation_guard.protect(victim):
            victim.write_bytes(BROKEN)
            raise RuntimeError("gate blew up mid-mutation")
    assert victim.read_bytes() == ORIGINAL
    assert mutation_guard.pending() == []


def test_journal_exists_while_the_file_is_mutated(victim):
    with mutation_guard.protect(victim):
        victim.write_bytes(BROKEN)
        entries = [e for e in mutation_guard.pending()
                   if Path(e["path"]).name == victim.name]
        assert len(entries) == 1, "no journal written before the mutation"
        assert entries[0]["pid"] == os.getpid()
    assert mutation_guard.pending() == []


def _kill_child_mid_mutation(victim: Path, api: str) -> subprocess.Popen:
    """Start a child that mutates ``victim`` and hangs, then SIGKILL it."""
    body = {
        "protect": textwrap.dedent(f"""
            with mutation_guard.protect(victim) as original:
                victim.write_bytes({BROKEN!r})
                print("MUTATED", flush=True)
                time.sleep(600)
        """),
        "journal": textwrap.dedent(f"""
            original = victim.read_bytes()
            mutation_guard.journal(victim, original)
            victim.write_bytes({BROKEN!r})
            print("MUTATED", flush=True)
            time.sleep(600)
        """),
    }[api]

    script = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(ROOT)!r})
        from sim import mutation_guard
        victim = Path({str(victim)!r})
    """) + body

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = proc.stdout.readline().strip()
    assert line == "MUTATED", f"child never mutated the file: {line!r}"
    assert victim.read_bytes() == BROKEN

    # kill(), not terminate(): on Windows terminate() is TerminateProcess and on
    # POSIX kill() is SIGKILL. Neither runs `finally`. That is the whole point.
    proc.kill()
    proc.wait(timeout=30)
    return proc


@pytest.mark.parametrize("api", ["protect", "journal"])
def test_a_killed_process_leaves_recoverable_evidence(victim, api):
    _kill_child_mid_mutation(victim, api)

    # Non-vacuity: the damage is real and `finally` demonstrably did not run.
    assert victim.read_bytes() == BROKEN, (
        "the child was killed but the file is already clean -- this test is "
        "not exercising the crash path it claims to")
    entries = [e for e in mutation_guard.pending()
               if Path(e["path"]).name == victim.name]
    assert len(entries) == 1, "a killed mutation left no journal to recover from"

    restored = mutation_guard.restore_stale()
    assert any(Path(r).name == victim.name for r in restored)
    assert victim.read_bytes() == ORIGINAL
    assert mutation_guard.pending() == []


def test_restore_stale_leaves_live_processes_alone(victim):
    """Concurrent gates must not stomp each other's in-flight mutations."""
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            from pathlib import Path
            sys.path.insert(0, {str(ROOT)!r})
            from sim import mutation_guard
            victim = Path({str(victim)!r})
            with mutation_guard.protect(victim):
                victim.write_bytes({BROKEN!r})
                print("MUTATED", flush=True)
                time.sleep(30)
        """)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert proc.stdout.readline().strip() == "MUTATED"
        restored = mutation_guard.restore_stale()
        assert not [r for r in restored if Path(r).name == victim.name], (
            "restore_stale clobbered a mutation belonging to a live process")
        assert victim.read_bytes() == BROKEN
    finally:
        proc.kill()
        proc.wait(timeout=30)
    mutation_guard.restore_stale()
    assert victim.read_bytes() == ORIGINAL


def test_every_in_place_mutation_gate_uses_the_journal():
    """The fix is worthless if a future gate reintroduces the bare pattern.

    Any scenario that writes to a repo source file and restores it in a
    ``finally`` must go through this module, or it can corrupt a whole sweep
    the moment it is killed.
    """
    scenarios = ROOT / "sim" / "scenarios"
    offenders = []
    for path in sorted(scenarios.glob("verify_*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        mutates = any(
            f"{var}.write_bytes(original" in src
            for var in ("target", "path", "engine_path", "topo_path", "vpath",
                        "_VERIFICATION_PY"))
        if mutates and "from sim import mutation_guard" not in src:
            offenders.append(path.name)
    assert not offenders, (
        "these gates mutate real source files without a crash-safe journal; "
        f"a kill mid-run corrupts the checkout: {offenders}")


# --------------------------------------------------------------------------
# The mutation harness scores a mutation "caught" when its guarding command
# exits non-zero. That is only evidence if the same command exits ZERO on
# unmutated code -- otherwise a gate that is simply broken (missing file, import
# error, unrelated regression) is indistinguishable from a gate doing its job,
# and it would certify every mutation it guards. The harness had no such check
# until a mis-attributed mutation exposed the hole. These two tests hold the
# door shut, and they test both branches, because a detector that never fires
# and a detector that always fires are equally useless.
# --------------------------------------------------------------------------

_HARNESS_DRIVER = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "mc", r"{root}/sim/scenarios/verify_mutation_coverage.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
real = m.MUTATIONS[0]

def synth(mid, code):
    return {{"id": mid, "path": real["path"], "old": real["old"],
             "new": real["new"], "why": "synthetic",
             "cmd": [sys.executable, "-c", "raise SystemExit(%d)" % code]}}

m.MUTATIONS[:] = [synth("selftest-always-passes", 0),
                  synth("selftest-always-fails", 1)]
sys.argv = ["x"]
try:
    m.main()
except AssertionError as exc:
    print("VERDICT:" + " ".join(str(exc).split()))
    sys.exit(0)
print("VERDICT:main-returned-without-raising")
sys.exit(0)
"""


def _run_harness() -> str:
    """Drive the real harness with two synthetic mutations and return its verdict.

    Both synthetic guards ignore the mutated file entirely, so the only thing
    being measured is the harness's own bookkeeping.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         _HARNESS_DRIVER.format(root=ROOT.as_posix())],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=600)
    out = proc.stdout.decode("utf-8", "replace")
    for line in out.splitlines():
        if line.startswith("VERDICT:"):
            return line[len("VERDICT:"):]
    raise AssertionError(f"harness produced no verdict:\n{out[-3000:]}")


def test_harness_rejects_a_guard_that_fails_on_working_code():
    """A guard failing unmutated must be VACUOUS, never 'caught'."""
    verdict = _run_harness()
    assert "1 had a vacuous guard" in verdict, verdict


def test_harness_still_detects_a_real_survivor():
    """Non-vacuity of the test above: the harness has not simply stopped judging."""
    verdict = _run_harness()
    assert "1 of 2 mutations survived" in verdict, verdict
