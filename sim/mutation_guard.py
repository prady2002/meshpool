"""Crash-safe in-place source mutation for the non-vacuity checks.

Eleven acceptance gates prove they are not vacuous by breaking a real source
file, re-running themselves, and asserting the re-run fails. Every one of them
restored the file in a ``finally`` block, which is correct for exceptions and
useless for the case that actually happened:

    a gate exceeded the sweep timeout, the runner killed its process tree, and
    ``finally`` never ran. ``meshpool/controlplane/openai_api.py`` stayed
    mutated -- ``_sse_chunk`` returning ``"data: {not-json}"`` -- and every
    subsequent gate in a 78-gate sweep ran against a corrupted checkout.

The damage is silent and expensive: hours of green results that mean nothing,
and a working tree that looks like someone made an edit. ``finally`` cannot
survive SIGKILL/TerminateProcess, so the recovery information has to live
somewhere that does: on disk, written *before* the mutation.

This module keeps a journal directory. Each protected mutation writes a backup
of the original bytes plus a small manifest before touching the file, and
deletes both only after verifying the restore was byte-identical. Anything left
behind is, by construction, evidence of a crash -- and :func:`restore_stale`
turns that evidence back into a clean checkout.

Usage (the whole API):

    from sim import mutation_guard

    with mutation_guard.protect(target) as original:
        target.write_bytes(original.replace(anchor, broken, 1))
        ...run the gate and assert it fails...
    # restored and verified here, journal removed
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_DIR = ROOT / ".mutation-journal"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pid_alive(pid: int) -> bool:
    """Best effort: unknown means 'assume alive' so we never fight a live gate."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=20)
        except Exception:
            return True
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def journal(target: Path, original: bytes) -> str:
    """Record a crash-safe backup of ``target`` before it is mutated.

    :func:`protect` is the better API for new code. This exists because the
    eleven gates that predate this module already own their ``try``/``finally``
    restore, and rewrapping their bodies to change indentation would risk
    breaking working non-vacuity checks to fix a crash-only failure mode. Two
    lines -- ``journal`` before the mutation, ``clear`` after the restore --
    add the durable half without touching their control flow.
    """
    target = Path(target)
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    (JOURNAL_DIR / f"{token}.bak").write_bytes(original)
    try:
        rel = str(target.resolve().relative_to(ROOT))
    except ValueError:
        rel = str(target.resolve())
    (JOURNAL_DIR / f"{token}.json").write_text(json.dumps({
        "path": rel,
        "backup": f"{token}.bak",
        "sha256": _sha(original),
        "pid": os.getpid(),
        "created": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    _fsync_dir(JOURNAL_DIR)
    return token


def clear(token: str) -> None:
    """Drop a journal entry once its file is known to be restored."""
    (JOURNAL_DIR / f"{token}.json").unlink(missing_ok=True)
    (JOURNAL_DIR / f"{token}.bak").unlink(missing_ok=True)


@contextmanager
def protect(target: Path) -> Iterator[bytes]:
    """Yield ``target``'s current bytes, guaranteeing they come back.

    Restoration happens three ways, in descending order of preference:
    normal exit, ``finally`` on exception, and -- if the process is killed
    outright -- :func:`restore_stale` on the next run. The third is the only
    one that matters for the bug this exists to fix.
    """
    target = Path(target)
    original = target.read_bytes()
    digest = _sha(original)

    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    backup = JOURNAL_DIR / f"{token}.bak"
    manifest = JOURNAL_DIR / f"{token}.json"

    backup.write_bytes(original)
    try:
        rel = str(target.resolve().relative_to(ROOT))
    except ValueError:
        rel = str(target.resolve())
    manifest.write_text(json.dumps({
        "path": rel,
        "backup": backup.name,
        "sha256": digest,
        "pid": os.getpid(),
        "created": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    # The journal must be durable before the mutation, not after; otherwise a
    # kill in the window between them leaves a corrupted file with no record.
    _fsync_dir(JOURNAL_DIR)

    try:
        yield original
    finally:
        try:
            target.write_bytes(original)
            after = _sha(target.read_bytes())
            if after != digest:
                raise AssertionError(
                    f"{rel} was not restored byte-identically: "
                    f"{digest} != {after}")
        finally:
            manifest.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return  # Windows cannot open a directory for fsync; the write is enough.
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def pending() -> list[dict]:
    """Every journal entry currently on disk, newest last."""
    if not JOURNAL_DIR.is_dir():
        return []
    entries: list[dict] = []
    for manifest in sorted(JOURNAL_DIR.glob("*.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["_manifest"] = str(manifest)
        entries.append(data)
    entries.sort(key=lambda d: d.get("created", 0.0))
    return entries


def restore_stale(*, force: bool = False) -> list[str]:
    """Undo mutations abandoned by a killed process. Returns restored paths.

    Entries whose owning process is still alive are left alone unless ``force``
    is set -- a concurrently running gate is mid-mutation on purpose, and
    stomping it would corrupt exactly what this is meant to protect.
    """
    restored: list[str] = []
    for entry in pending():
        manifest = Path(entry["_manifest"])
        pid = int(entry.get("pid", 0))
        if not force and pid != os.getpid() and _pid_alive(pid):
            continue
        backup = JOURNAL_DIR / str(entry.get("backup", ""))
        rel = str(entry.get("path", ""))
        target = ROOT / rel
        if backup.is_file() and target.is_file():
            want = backup.read_bytes()
            if target.read_bytes() != want:
                target.write_bytes(want)
                restored.append(rel)
        manifest.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
    return restored


def main() -> int:
    import sys
    force = "--force" in sys.argv[1:]
    outstanding = pending()
    if not outstanding:
        print("[mutation-guard] no outstanding mutations")
        return 0
    for entry in outstanding:
        print(f"[mutation-guard] pending: {entry.get('path')} "
              f"(pid {entry.get('pid')}, {entry.get('created_iso')})")
    restored = restore_stale(force=force)
    for rel in restored:
        print(f"[mutation-guard] restored {rel}")
    if not restored:
        print("[mutation-guard] nothing restored "
              "(owning processes still alive; use --force to override)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
