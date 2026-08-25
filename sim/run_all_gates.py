"""Run every acceptance gate in sequence and report a pass/fail table.

Gates start real OS processes on fixed ports, so they must run one at a time.
Exit 0 only when every gate exits 0.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "sim" / "scenarios").glob("verify_*.py"))
LOGS = ROOT / "_gatelogs"


def main() -> int:
    only = set(sys.argv[1:])
    LOGS.mkdir(exist_ok=True)
    results: list[tuple[str, int, float]] = []
    for path in SCENARIOS:
        name = path.stem
        if only and name not in only:
            continue
        started = time.time()
        with open(LOGS / f"{name}.log", "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT),
                                  stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
        elapsed = time.time() - started
        results.append((name, proc.returncode, elapsed))
        flag = "PASS" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        print(f"{flag:<9} {name:<40} {elapsed:6.1f}s", flush=True)

    bad = [r for r in results if r[1] != 0]
    print(f"\n{len(results) - len(bad)}/{len(results)} gates passed "
          f"in {sum(r[2] for r in results):.0f}s")
    for name, code, _ in bad:
        print(f"  FAILED {name} (exit {code}) -> _gatelogs/{name}.log")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
