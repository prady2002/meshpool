"""CLI surface for capability tokens.

A minted secret is shown exactly once and then only its hash is stored, so the CLI is
the only place a user ever sees it. That makes two things worth testing directly: the
secret must actually be printed on mint, and it must never appear anywhere else.

The HTTP client is patched the same way tests/test_cli.py does it -- the CLI reuses one
``httpx.Client``, so patching the module-level convenience wrappers would intercept
nothing.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import meshpool.cli.main as cli_module
from meshpool.cli.main import app

runner = CliRunner()
SECRET = "mpt_do-not-leak-me-anywhere-but-the-mint-line"
NOW = int(time.time() * 1000)


def _resp(payload):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = payload
    m.status_code = 200
    return m


@contextmanager
def _patch_http(**responses):
    client = MagicMock()
    for name, value in responses.items():
        getattr(client, name).return_value = value
    with patch.object(cli_module, "_http", return_value=client):
        yield client


@pytest.fixture(autouse=True)
def _reset_url_override():
    cli_module._override_url = None
    yield
    cli_module._override_url = None


MINTED = {
    "id": "tok-abc123456789", "label": "ci runner",
    "scopes": ["jobs:read", "jobs:write"], "owner": "dev:me",
    "expires_at_ms": NOW + 3600_000, "created_ms": NOW,
    "last_used_ms": None, "spend_cap": 12.5, "spent": 0.0,
    "token": SECRET,
}


def test_batch_submit_keeps_secret_names_in_its_original_position():
    """capability_token_id must stay keyword-only.

    It was originally added as the 7th *positional* parameter, ahead of secret_names.
    Every existing caller still passed secret_names positionally, so a list landed in
    a str field and every API job submission and every DAG stage launch died in
    Pydantic validation. Neither test_secrets.py nor test_batch_engine.py noticed,
    because both call submit() with keywords or bypass it entirely.

    Pinning the signature is the fix that generalises: a keyword-only parameter cannot
    be captured by an existing positional call site no matter who adds one next.
    """
    import inspect

    from meshpool.batch.engine import BatchManager

    sig = inspect.signature(BatchManager.submit)
    names = [p.name for p in sig.parameters.values()
             if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert names[:7] == ["self", "name", "func", "payloads", "cpu_pct",
                         "ram_mb", "owner"]
    assert names[7] == "secret_names", (
        f"secret_names moved out of position 7; positional callers now pass it to "
        f"{names[7]!r}")
    assert sig.parameters["capability_token_id"].kind is inspect.Parameter.KEYWORD_ONLY


def test_batch_submit_positional_secret_names_still_works():
    """The empirical half: the signature assertion above is only meaningful if a
    positional call actually round-trips."""
    from unittest.mock import MagicMock

    from meshpool.batch.engine import BatchManager

    mgr = BatchManager.__new__(BatchManager)
    mgr.jobs = {}
    mgr.store = MagicMock()
    mgr.events = MagicMock()
    mgr._notify = MagicMock()
    mgr.spend_caps = None
    job = BatchManager.submit(mgr, "n", "square", [{"x": 1}], 10, 64,
                              "owner", ["API_TOKEN"])
    assert job.secret_names == ["API_TOKEN"]
    assert job.capability_token_id is None

    with _patch_http(post=_resp(MINTED)) as client:
        r = runner.invoke(app, ["tokens", "mint", "ci runner",
                                "-s", "jobs:read", "-s", "jobs:write",
                                "--expires-in", "3600", "--spend-cap", "12.5"])
    assert r.exit_code == 0, r.output
    assert SECRET in r.output, "the one chance to copy the secret was not printed"
    assert "tok-abc123456789" in r.output
    assert "cannot be shown" in r.output

    body = client.post.call_args.kwargs["json"]
    assert body["scopes"] == ["jobs:read", "jobs:write"]
    assert body["expires_in_seconds"] == 3600
    assert body["spend_cap"] == 12.5


def test_mint_omits_spend_cap_when_not_requested():
    """Sending spend_cap: null is not the same as not sending it."""
    with _patch_http(post=_resp(MINTED)) as client:
        r = runner.invoke(app, ["tokens", "mint", "ci", "-s", "jobs:read"])
    assert r.exit_code == 0, r.output
    assert "spend_cap" not in client.post.call_args.kwargs["json"]


def test_list_never_prints_a_secret():
    listed = {"tokens": [{k: v for k, v in MINTED.items()
                          if k not in ("token", "owner")}]}
    with _patch_http(get=_resp(listed)):
        r = runner.invoke(app, ["tokens", "list"])
    assert r.exit_code == 0, r.output
    assert SECRET not in r.output
    assert "tok-abc" in r.output
    assert "jobs:read" in r.output


def test_list_shows_never_used_rather_than_a_fake_timestamp():
    """``last_used_ms: null`` is a real state -- a token minted and not yet used.

    Rendering it as an epoch date would tell the user the token had been used at the
    dawn of 1970, which is exactly the kind of confident wrong answer that makes an
    operator distrust the whole screen during an incident.
    """
    listed = {"tokens": [{**{k: v for k, v in MINTED.items() if k != "token"},
                          "last_used_ms": None}]}
    with _patch_http(get=_resp(listed)):
        r = runner.invoke(app, ["tokens", "list"])
    assert r.exit_code == 0, r.output
    assert "never" in r.output
    assert "1970" not in r.output


def test_list_marks_an_expired_token_as_expired():
    listed = {"tokens": [{**{k: v for k, v in MINTED.items() if k != "token"},
                          "expires_at_ms": NOW - 5_000}]}
    with _patch_http(get=_resp(listed)):
        r = runner.invoke(app, ["tokens", "list"])
    assert r.exit_code == 0, r.output
    assert "expired" in r.output


def test_list_on_an_empty_account_suggests_the_next_command():
    with _patch_http(get=_resp({"tokens": []})):
        r = runner.invoke(app, ["tokens", "list"])
    assert r.exit_code == 0, r.output
    assert "tokens mint" in r.output


def test_revoke_calls_delete_and_states_it_is_immediate():
    with _patch_http(request=_resp({"revoked": True, "id": "tok-abc123456789"})) as client:
        r = runner.invoke(app, ["tokens", "revoke", "tok-abc123456789"])
    assert r.exit_code == 0, r.output
    assert "revoked" in r.output
    assert "immediately" in r.output
    method, url = client.request.call_args.args[0], client.request.call_args.args[1]
    assert method == "DELETE", f"revoke must use DELETE, used {method}"
    assert "/api/v1/tokens/tok-abc123456789" in url
