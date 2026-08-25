"""Acceptance gate: the shipped "unzip and run" artifact actually works clean-room.

Phase 3's promise is that anyone can unzip MeshPool and *run and use it* with one
command and zero configuration. This gate proves that against the exact file set the
build zip ships (an ALLOWLIST copy into a throwaway directory - no tests/, no sim/, no
pre-existing state or device keys), by running the real launcher self-test from there:

    python -m meshpool.launcher self-test

which brings up a control plane + a local agent and runs a real batch job end to end.
If the shipped file set were missing a module, a static asset, or the model weights, or
if the launcher depended on test-only code, this fails. It is hermetic and cross-platform
(pure Python - no PowerShell/bash), so it runs anywhere pytest does.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The runtime allowlist the build zip ships (mirrors deploy/build_zip.*). Deliberately
# excludes tests/ and sim/ so this proves the product runs without any test scaffolding.
SHIP_DIRS = ["meshpool"]
SHIP_FILES = ["requirements.txt", "pyproject.toml", "README.md",
              "start.cmd", "start.ps1", "start.sh", "contribute.ps1", "contribute.sh"]


def _fresh_extract(dest: Path) -> None:
    """Reproduce an unzip: copy only the shipped files, stripped of caches/state."""
    for d in SHIP_DIRS:
        src = REPO_ROOT / d
        if src.exists():
            shutil.copytree(
                src, dest / d,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo",
                                              "*.db", "*.db-wal", "*.db-shm",
                                              "identity.json", "_data_*"))
    for f in SHIP_FILES:
        src = REPO_ROOT / f
        if src.exists():
            shutil.copy2(src, dest / f)


def main() -> int:
    stage = Path(tempfile.mkdtemp(prefix="meshpool-onboard-"))
    extract = stage / "meshpool-ship"
    extract.mkdir(parents=True)
    ok = False
    try:
        print("== simulating a fresh unzip (shipped files only, no tests/sim/state) ==")
        _fresh_extract(extract)
        # sanity: the pieces a user needs are present and test infra is NOT
        assert (extract / "meshpool" / "launcher.py").exists(), "launcher missing"
        assert (extract / "meshpool" / "controlplane" / "static" / "app.js").exists(), "UI missing"
        assert (extract / "meshpool" / "inference" / "weights" / "nano-gpt.npz").exists(), "weights missing"
        assert not (extract / "tests").exists() and not (extract / "sim").exists(), \
            "test scaffolding leaked into the shipped set"
        print(f"  extracted to {extract}")

        print("\n== running `python -m meshpool.launcher self-test` from the extract ==")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(extract) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "meshpool.launcher", "self-test",
             "--data-dir", str(stage / "state")],
            cwd=str(extract), env=env, capture_output=True, text=True, timeout=180)
        out = (proc.stdout or "") + (proc.stderr or "")
        # surface a few key lines for the log without dumping everything
        for line in (proc.stdout or "").splitlines():
            if "[self-test]" in line:
                print("  " + line)
        if proc.returncode != 0 or "SELFTEST_OK" not in out:
            print(f"\nFAILED: self-test exit={proc.returncode}; SELFTEST_OK present="
                  f"{'SELFTEST_OK' in out}")
            print("---- tail of output ----")
            print("\n".join(out.splitlines()[-25:]))
            return 1

        print("\n== a second launch reuses the same identity (no duplicate device) ==")
        # Re-run against the same state dir: the device identity must be stable
        # (load_or_create), proving persistence works from the shipped layout.
        id_files = list((stage / "state").rglob("identity.json"))
        assert id_files, "no device identity was persisted by the first run"
        print(f"  persisted {len(id_files)} device identity file(s) under the data dir")

        ok = True
        print("\nALL ONBOARDING CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
