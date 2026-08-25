"""The hand-written backprop is numerically correct (float64 central differences)."""
from __future__ import annotations

from meshpool.inference.train import gradient_check


def test_analytic_gradients_match_numeric():
    # worst relative error across sampled params must be tiny (<1e-4).
    worst = gradient_check(seed=0, n_samples=30)
    assert worst < 1e-4, f"gradient check failed, worst rel error = {worst}"
