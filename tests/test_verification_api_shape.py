"""``GET /api/v1/verification`` must return the same shape at every tier.

The endpoint has two branches. Below Tier 2 the Verifier is not wired in, so a
static snapshot is returned; at Tier 2 it delegates to ``Verifier.as_dict()``.
Those two payloads have to agree on their keys, and for a while they did not:
the inactive branch returned a ``classification`` block and the active one
silently dropped it.

That is the worst kind of API bug, because it is invisible in development. A
client written against a local Tier-0 mesh reads ``body["classification"]``
happily, ships, and then KeyErrors the first time it points at a real Tier-2
deployment -- the exact environment where verification data actually matters.
Nothing in the response signals that the shape is tier-dependent.

These tests compare the two branches directly rather than asserting a hardcoded
key list, so a field added to one branch and forgotten in the other fails here
instead of in production.
"""
from __future__ import annotations

from meshpool.controlplane.verification import (
    DIGEST_VERIFIABLE_FUNCS,
    NOT_VERIFIABLE_FUNCS,
    ORACLE_VERIFIABLE_FUNCS,
    Verifier,
)


def _inactive_payload(client) -> dict:
    r = client.get("/api/v1/verification")
    assert r.status_code == 200
    return r.json()


class TestShapeIsTierIndependent:
    def test_active_and_inactive_branches_expose_the_same_top_level_keys(self):
        active = Verifier().as_dict()

        # The inactive branch is a literal in app.py; reproduce it by asking the
        # app itself rather than copying the dict, so this test tracks the real
        # endpoint instead of a stale transcription of it.
        from fastapi.testclient import TestClient
        from meshpool.controlplane.app import build_app

        with TestClient(build_app()) as client:
            inactive = _inactive_payload(client)

        assert set(active) == set(inactive), (
            "the /api/v1/verification response shape changes with deployment "
            f"tier: active-only={sorted(set(active) - set(inactive))}, "
            f"inactive-only={sorted(set(inactive) - set(active))}"
        )

    def test_nested_blocks_agree_too(self):
        """A matching top level is not enough.

        ``policy``, ``stats`` and ``escrow`` are nested objects, and the same
        drift can happen one level down without changing the outer keys.
        """
        active = Verifier().as_dict()

        from fastapi.testclient import TestClient
        from meshpool.controlplane.app import build_app

        with TestClient(build_app()) as client:
            inactive = _inactive_payload(client)

        for block in ("policy", "stats", "escrow", "classification"):
            assert set(active[block]) == set(inactive[block]), (
                f"'{block}' differs between tiers: "
                f"active-only={sorted(set(active[block]) - set(inactive[block]))}, "
                f"inactive-only={sorted(set(inactive[block]) - set(active[block]))}"
            )

        assert set(active["stats"]["degraded"]) == set(
            inactive["stats"]["degraded"])


class TestClassificationContent:
    def test_active_branch_reports_the_real_function_classification(self):
        """The block must carry the actual policy, not an empty placeholder.

        A shape fix that returned three empty lists would satisfy the tests
        above while telling every caller that nothing is verifiable.
        """
        body = Verifier().as_dict()["classification"]

        assert body["digest_verifiable"] == sorted(DIGEST_VERIFIABLE_FUNCS)
        assert body["oracle_verifiable"] == sorted(ORACLE_VERIFIABLE_FUNCS)
        assert body["not_verifiable"] == sorted(NOT_VERIFIABLE_FUNCS)
        assert body["digest_verifiable"], "classification must not be empty"

    def test_the_three_buckets_are_disjoint(self):
        """A function in two buckets means an ambiguous verification path.

        The scheduler picks its check strategy from this classification, so an
        overlap would make the choice depend on iteration order.
        """
        d = set(DIGEST_VERIFIABLE_FUNCS)
        o = set(ORACLE_VERIFIABLE_FUNCS)
        n = set(NOT_VERIFIABLE_FUNCS)

        assert not (d & o), f"digest/oracle overlap: {sorted(d & o)}"
        assert not (d & n), f"digest/not-verifiable overlap: {sorted(d & n)}"
        assert not (o & n), f"oracle/not-verifiable overlap: {sorted(o & n)}"

    def test_every_registered_batch_function_is_classified(self):
        """An unclassified function would fall through to no verification.

        Silently treating an unknown function as unverifiable is how a
        verifiable workload ends up unchecked, so the mapping must be total
        over the registry rather than merely non-overlapping.
        """
        from meshpool.batch.functions import REGISTRY

        classified = (
            set(DIGEST_VERIFIABLE_FUNCS)
            | set(ORACLE_VERIFIABLE_FUNCS)
            | set(NOT_VERIFIABLE_FUNCS)
        )
        missing = set(REGISTRY) - classified
        assert not missing, (
            f"batch functions with no verification classification: "
            f"{sorted(missing)}"
        )
