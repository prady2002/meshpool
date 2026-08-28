"""Gate: the published benchmark claims must still hold.

``bench_report.py`` writes ``docs/benchmarks.md`` and is therefore excluded
from the sweep by name -- a gate that rewrites a tracked file trips the
runner's dirty-checkout detection, and a timestamp changes on every run. But
the checks inside it are real invariants of the product thesis, not decoration:

  * SPD must beat ordinary decoding in every wide-area configuration, because
    if it does not, the mechanism the whole design rests on does not work.
  * The performance-model fit must not degenerate to its fallback value, which
    is how a full page of ``0.00 tok/s`` projections got published while the
    generator exited 0.
  * The ceiling table and the sensitivity table must agree where they describe
    the same deployment. They disagreed by 57% the first time anyone checked.

So this gate runs the identical code path in ``--check`` mode: everything is
measured and rendered, nothing is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.scenarios import bench_report  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(bench_report.main(check_only=True))
