"""Prove the CLI's --control-plane precedence against a REAL running control plane.

Unit tests can only show that a URL string is threaded through. This starts an
actual coordinator on a non-default port, points the ENV VAR at a dead port, and
checks that the flag still reaches the live server -- and that removing the flag
fails. That asymmetry is what proves the flag is really in control.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness  # noqa: E402

CP_PORT = 8491
DEAD = "http://127.0.0.1:9999"


def run_cli(args: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "meshpool.cli.main", *args],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )


def main() -> int:
    h = Harness(n_agents=1, cp_port=CP_PORT,
                data_dir=str(REPO_ROOT / "sim" / "scenarios" / "_data_clitest"))
    ok = True
    try:
        h.start()
        cp = h.cp_url()

        env = dict(os.environ)
        env["MESHPOOL_CONTROL_PLANE_URL"] = DEAD  # deliberately unreachable

        print("== flag points at the live CP; env var points at a dead port ==")
        r = run_cli(["--control-plane", cp, "nodes"], env)
        print(f"  exit={r.returncode}")
        print("  stdout:", r.stdout.strip()[:300])
        if r.returncode != 0:
            print("  stderr:", r.stderr.strip()[-400:])
        assert r.returncode == 0, "the flag did not reach the live control plane"
        assert "sim-node" in r.stdout, (
            f"no node listed, so the CLI did not really query the live CP: {r.stdout[:300]}")
        print("  PASS: the flag beat the environment variable")

        print("\n== env var alone (dead port) must fail ==")
        r2 = run_cli(["nodes"], env)
        print(f"  exit={r2.returncode}")
        assert r2.returncode != 0, (
            "the CLI succeeded against a dead URL — it is ignoring the env var too, "
            "so the test above proves nothing about precedence")
        print("  PASS: without the flag it fails, so the flag was genuinely in control")

        print("\n== env var alone, pointed at the live CP, must work ==")
        env2 = dict(os.environ)
        env2["MESHPOOL_CONTROL_PLANE_URL"] = cp
        r3 = run_cli(["nodes"], env2)
        print(f"  exit={r3.returncode}")
        assert r3.returncode == 0 and "sim-node" in r3.stdout, (
            f"env var alone did not work: exit={r3.returncode} {r3.stdout[:200]}")
        print("  PASS: env var still works when no flag is given")

        print("\nALL CLI PRECEDENCE CHECKS PASSED")
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
