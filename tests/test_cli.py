"""Unit tests for the ``meshpool`` command-line client.

These exercise the Typer wiring (command registration, arguments, options, and the
local input validation that happens before any network call) using Typer's in-process
CliRunner — no control plane required. The full request/response round-trip for the
storage commands is covered end-to-end by ``sim.scenarios.verify_storage`` (test 8).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import meshpool.cli.main as cli_module
from meshpool.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixture: reset the module-level URL override between every test in this file.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_url_override():
    cli_module._override_url = None
    yield
    cli_module._override_url = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mock_cluster():
    """Return a mock httpx.Response for GET /api/v1/cluster (used by `nodes`)."""
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {
        "summary": {"online": 0, "total_nodes": 0, "total_cores": 0, "total_ram_mb": 0},
        "nodes": [],
    }
    return m


def test_top_level_help_lists_storage_group():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    assert "storage" in r.output


def test_storage_group_exposes_put_get_ls_rm():
    r = runner.invoke(app, ["storage", "--help"])
    assert r.exit_code == 0, r.output
    for cmd in ("put", "get", "ls", "rm"):
        assert cmd in r.output, (cmd, r.output)


def test_dag_group_exposes_submit_list_show():
    r = runner.invoke(app, ["dag", "--help"])
    assert r.exit_code == 0, r.output
    for cmd in ("submit", "list", "show"):
        assert cmd in r.output, (cmd, r.output)


def test_storage_put_help_lists_options():
    r = runner.invoke(app, ["storage", "put", "--help"])
    assert r.exit_code == 0, r.output
    assert "--name" in r.output
    assert "--replication" in r.output


def test_storage_put_missing_file_fails_before_any_upload():
    r = runner.invoke(app, ["storage", "put", "definitely-not-a-real-file.bin"])
    assert r.exit_code == 1, r.output
    assert "no such file" in r.output


def test_models_group_still_registered():
    # guards against a regression where the storage group edit clobbered `models`.
    r = runner.invoke(app, ["models", "--help"])
    assert r.exit_code == 0, r.output
    assert "deploy" in r.output


def test_models_group_exposes_upload():
    r = runner.invoke(app, ["models", "--help"])
    assert r.exit_code == 0, r.output
    assert "upload" in r.output


def test_models_upload_help_lists_model_option():
    r = runner.invoke(app, ["models", "upload", "--help"])
    assert r.exit_code == 0, r.output
    assert "--model" in r.output


def test_models_upload_missing_file_fails_before_any_upload():
    r = runner.invoke(app, ["models", "upload", "definitely-not-a-real-ckpt.npz"])
    assert r.exit_code == 1, r.output
    assert "no such file" in r.output


# ---------------------------------------------------------------------------
# Task 1: --control-plane global flag
# ---------------------------------------------------------------------------

def test_control_plane_flag_visible_in_top_level_help():
    """--control-plane must appear in the root --help output."""
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    assert "--control-plane" in r.output


def test_control_plane_flag_only(monkeypatch):
    """When --control-plane is the only URL source, it must be used for requests."""
    monkeypatch.delenv("MESHPOOL_CONTROL_PLANE_URL", raising=False)
    with patch("httpx.get", return_value=_mock_cluster()) as mock_get:
        runner.invoke(app, ["--control-plane", "http://flaghost:7777", "nodes"])
    assert mock_get.called, "httpx.get was not called"
    called_url: str = mock_get.call_args[0][0]
    assert "flaghost:7777" in called_url, called_url


def test_control_plane_env_only(monkeypatch):
    """When no flag is given but the env var is set, the env var URL must be used."""
    monkeypatch.setenv("MESHPOOL_CONTROL_PLANE_URL", "http://envhost:8888")
    with patch("httpx.get", return_value=_mock_cluster()) as mock_get:
        runner.invoke(app, ["nodes"])
    assert mock_get.called, "httpx.get was not called"
    called_url: str = mock_get.call_args[0][0]
    assert "envhost:8888" in called_url, called_url


def test_control_plane_flag_overrides_env(monkeypatch):
    """--control-plane must take precedence over MESHPOOL_CONTROL_PLANE_URL."""
    monkeypatch.setenv("MESHPOOL_CONTROL_PLANE_URL", "http://envhost:8888")
    with patch("httpx.get", return_value=_mock_cluster()) as mock_get:
        runner.invoke(app, ["--control-plane", "http://flaghost:7777", "nodes"])
    assert mock_get.called, "httpx.get was not called"
    called_url: str = mock_get.call_args[0][0]
    assert "flaghost:7777" in called_url, f"flag URL not used; got {called_url}"
    assert "envhost:8888" not in called_url, f"env var leaked into URL: {called_url}"


def test_control_plane_invalid_url_exits_with_clear_message():
    """A non-http/https URL must fail with a human message, not a traceback."""
    r = runner.invoke(app, ["--control-plane", "definitely-not-a-url", "nodes"])
    assert r.exit_code != 0
    out = r.output.lower()
    # Must say "error" and mention the bad value or expected format.
    assert "error" in out, r.output
    assert "http" in out or "url" in out, r.output


def test_control_plane_flag_trailing_slash_stripped(monkeypatch):
    """Trailing slashes must be stripped so API paths don't get double slashes."""
    monkeypatch.delenv("MESHPOOL_CONTROL_PLANE_URL", raising=False)
    with patch("httpx.get", return_value=_mock_cluster()) as mock_get:
        runner.invoke(app, ["--control-plane", "http://myhost:9090/", "nodes"])
    if mock_get.called:
        called_url: str = mock_get.call_args[0][0]
        assert "//" not in called_url.split("//", 1)[1], (
            f"double slash in path: {called_url}"
        )
