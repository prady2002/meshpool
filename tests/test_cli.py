"""Unit tests for the ``meshpool`` command-line client.

These exercise the Typer wiring (command registration, arguments, options, and the
local input validation that happens before any network call) using Typer's in-process
CliRunner — no control plane required. The full request/response round-trip for the
storage commands is covered end-to-end by ``sim.scenarios.verify_storage`` (test 8).
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
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
    for cmd in ("login", "run", "serve", "mesh", "contribute", "store"):
        assert cmd in r.output


def test_storage_group_exposes_put_get_ls_rm():
    r = runner.invoke(app, ["storage", "--help"])
    assert r.exit_code == 0, r.output
    for cmd in ("put", "get", "ls", "rm"):
        assert cmd in r.output, (cmd, r.output)


def test_store_group_exposes_cp():
    r = runner.invoke(app, ["store", "--help"])
    assert r.exit_code == 0, r.output
    assert "cp" in r.output


def test_store_cp_rejects_s3_to_s3_as_unavailable():
    r = runner.invoke(app, ["store", "cp", "s3://bucket/a", "s3://bucket/b"])
    assert r.exit_code == 2, r.output
    assert "supports local -> s3:// and s3:// -> local" in r.output


def test_fn_deploy_python_fails_loudly():
    with patch("pathlib.Path.is_file", return_value=True):
        r = runner.invoke(app, ["fn", "deploy", "handler.py"])
    assert r.exit_code == 2, r.output
    assert "Python-to-WASM function packaging is not available" in r.output


def test_run_uses_streaming_openai_endpoint():
    with patch.object(cli_module, "_ensure_model") as ensure, \
         patch.object(cli_module, "_stream_openai_chat", return_value=(3, "hello")) as stream:
        r = runner.invoke(app, ["run", "nano-gpt", "hi", "--tokens", "3"])
    assert r.exit_code == 0, r.output
    ensure.assert_called_once_with("nano-gpt")
    stream.assert_called_once_with("nano-gpt", "hi", 3, 0.8)
    assert "stream chunks (measured)" in r.output


def test_login_persists_local_session_without_copy_paste():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"owner": "local-user", "session_token": "sess-123"}
    with patch("httpx.post", return_value=response) as post, \
         patch.object(cli_module, "_load_config", return_value={}), \
         patch.object(cli_module, "_save_config") as save:
        r = runner.invoke(app, ["--control-plane", "http://cp:8080", "login"])
    assert r.exit_code == 0, r.output
    assert post.call_args[0][0] == "http://cp:8080/api/v1/auth/local"
    save.assert_called_once()
    assert save.call_args[0][0]["session_token"] == "sess-123"


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


def test_models_list_labels_throughput_tier():
    with patch.object(cli_module, "_get", return_value={
        "deployed": True,
        "backend": "meshshard",
        "model": "nano-gpt",
        "n_layers": 8,
        "status": "ready",
        "tokens_per_sec": 12.3,
        "stages": [],
    }):
        r = runner.invoke(app, ["models", "list"])
    assert r.exit_code == 0, r.output
    assert "tokens/sec (measured)" in r.output


def test_ledger_labels_credit_totals_as_measured():
    with patch.object(cli_module, "_get", return_value={
        "by_node": [{"name": "node-a", "owner": "alice", "donated": 2.5, "consumed": 1.0}]
    }):
        r = runner.invoke(app, ["ledger"])
    assert r.exit_code == 0, r.output
    assert "mesh-seconds (measured)" in r.output


def test_deploy_custom_domain_fails_as_unavailable():
    r = runner.invoke(app, ["deploy", "hello", "--domain", "app.example.com"])
    assert r.exit_code == 2, r.output
    assert "custom domains are not available" in r.output


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


def test_control_plane_saved_login_used_when_no_flag_or_env(monkeypatch):
    """A prior `meshpool login` should make subsequent commands target that control plane."""
    monkeypatch.delenv("MESHPOOL_CONTROL_PLANE_URL", raising=False)
    with patch.object(cli_module, "_load_config",
                      return_value={"control_plane": "http://saved:9999"}), \
         patch("httpx.get", return_value=_mock_cluster()) as mock_get:
        runner.invoke(app, ["nodes"])
    called_url: str = mock_get.call_args[0][0]
    assert "saved:9999" in called_url, called_url


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


# ---------------------------------------------------------------------------
# Console width when stdout is not a terminal.
#
# Rich falls back to 80 columns whenever it cannot query a TTY -- which is every
# piped, redirected or captured run. The `nodes` table needs more than 80, so at
# 80 Rich shrinks columns and truncates cells: real node names turn into
# "gaming-…". That is worst in exactly the non-interactive cases people script
# against. These three tests are deliberately layered:
#
#   1. the module-level `console` really is widened (attacks the producer, so
#      reverting to a bare `Console()` fails this test);
#   2. 80 columns really does truncate (proves the fix is not a no-op);
#   3. the widened console does not pad output out to its full width (proves the
#      fix does not trade one piped-output bug for another).
# ---------------------------------------------------------------------------

_LONG_NODES = ("gaming-rig-basement", "gaming-rig-office", "threadripper-nas-01")


def _mock_cluster_named(names):
    """A GET /api/v1/cluster payload whose node names are realistically long."""
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {
        "summary": {"online": len(names), "total_nodes": len(names),
                    "total_cores": 8 * len(names), "total_ram_mb": 16000 * len(names)},
        "nodes": [
            {
                "name": n,
                "status": "online",
                "owner": "alice@example.com",
                "headroom_cpu_pct": 87.5,
                "trust_score": 0.93,
                "capability": {"cpu_cores": 8, "ram_mb": 16000},
            }
            for n in names
        ],
    }
    return m


def _render_nodes(console_obj, monkeypatch) -> str:
    """Run the real `nodes` command against `console_obj` and return what it drew."""
    buf = io.StringIO()
    console_obj.file = buf
    monkeypatch.setattr(cli_module, "console", console_obj)
    with patch("httpx.get", return_value=_mock_cluster_named(_LONG_NODES)):
        r = runner.invoke(app, ["nodes"])
    assert r.exit_code == 0, r.output
    return buf.getvalue()


def test_module_console_is_widened_when_stdout_is_not_a_tty():
    """The console the CLI actually prints with must not be clamped to 80 columns.

    Checked in a subprocess with stdout piped, so stdout is genuinely not a TTY
    regardless of how pytest itself was invoked (`-s` or not).
    """
    env = dict(os.environ)
    env.pop("COLUMNS", None)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import meshpool.cli.main as m; print(m.console.width)"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(cli_module.__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0, proc.stderr
    width = int(proc.stdout.strip())
    assert width >= 120, (
        f"module-level console is {width} columns wide when piped; the nodes "
        f"table needs more than 80 or it truncates node names"
    )


def test_eighty_columns_would_truncate_node_names(monkeypatch):
    """The oracle: at Rich's non-TTY default of 80 the table really does truncate.

    Without this, test_widened_console_shows_full_node_names could pass for a
    reason that has nothing to do with width.
    """
    monkeypatch.delenv("COLUMNS", raising=False)
    out = _render_nodes(Console(width=80), monkeypatch)
    assert "…" in out, f"expected truncation at 80 columns, got:\n{out}"
    truncated = [n for n in _LONG_NODES if n not in out]
    assert truncated, f"expected at least one truncated node name at 80 cols:\n{out}"


def test_widened_console_shows_full_node_names(monkeypatch):
    """With the real non-TTY console, every node name renders in full."""
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())  # isatty() -> False
    out = _render_nodes(cli_module._console(), monkeypatch)
    for n in _LONG_NODES:
        assert n in out, f"node name {n!r} was truncated:\n{out}"


def test_widened_console_does_not_pad_output_to_its_full_width(monkeypatch):
    """Widening the budget must not pad piped output out to 200 columns.

    Rich sizes a table to its content and only expands to the console width if
    the content needs it, so a three-node table should stay far short of 200.
    """
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    out = _render_nodes(cli_module._console(), monkeypatch)
    widest = max(len(line.rstrip()) for line in out.splitlines())
    assert widest < 150, f"output padded to {widest} columns:\n{out}"


def test_columns_env_var_still_wins_when_set(monkeypatch):
    """Honour an explicit COLUMNS, so users and CI can still pin the width."""
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert cli_module._console().width == 300
