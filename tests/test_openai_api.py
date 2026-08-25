"""Unit tests for the OpenAI-compatible inference API (/v1/*).

Tests run fully in-process against a FastAPI TestClient — no agents, no real model.
The inference engine is patched to return a predictable dict so tests stay fast and
deterministic.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from meshpool.common.config import Settings
from meshpool.controlplane.app import build_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def home_app(tmp_path):
    """Tier-0 home app: no auth gating, no session required."""
    s = Settings.load(scale="home", data_dir=str(tmp_path),
                      host="127.0.0.1", port=9080)
    return build_app(s)


@pytest.fixture()
def public_app(tmp_path):
    """Tier-2 public app in enforced auth mode: credentials required."""
    s = Settings.load(scale="public", data_dir=str(tmp_path),
                      host="127.0.0.1", port=9081, auth_mode="enforced")
    return build_app(s)


def _mock_deployment(model: str = "nano-gpt") -> MagicMock:
    dep = MagicMock()
    dep.id = "mdl-test"
    dep.model = model
    dep.status = "ready"
    return dep


def _fake_chat_result(text: str = "hello world", tokens: int = 2) -> dict:
    return {
        "model": "nano-gpt",
        "text": text,
        "tokens": tokens,
        "tokens_per_sec": 100.0,
        "stages": [],
        "elapsed_s": 0.01,
        "pipeline": {"hops": 0},
        "prompt": "test prompt",
        "deployment_id": "mdl-test",
    }


def _inject_mock_inference(app, text: str = "hello world", tokens: int = 2):
    """Inject a mock inference engine into the live CP after app startup.

    Call AFTER TestClient context is entered (lifespan has run).
    """
    import meshpool.controlplane.app as _appmod
    cp = _appmod.CP
    dep = _mock_deployment()
    cp.inference.list = MagicMock(return_value=[dep])
    cp.inference.chat = AsyncMock(return_value=_fake_chat_result(text, tokens))
    return cp


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_openai_shape(self, home_app, tmp_path):
        with TestClient(home_app) as client:
            r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) >= 1

    def test_models_have_required_fields(self, home_app):
        with TestClient(home_app) as client:
            r = client.get("/v1/models")
        for m in r.json()["data"]:
            assert "id" in m
            assert m["object"] == "model"
            assert "created" in m
            assert "owned_by" in m

    def test_builtin_nano_gpt_is_listed(self, home_app):
        with TestClient(home_app) as client:
            r = client.get("/v1/models")
        ids = {m["id"] for m in r.json()["data"]}
        assert "nano-gpt" in ids


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — non-streaming
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_basic_response_shape(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/chat/completions", json={
                "model": "nano-gpt",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion"
        assert "id" in body
        assert "created" in body
        assert "model" in body
        assert "choices" in body
        assert "usage" in body

    def test_choices_shape(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "test"}],
            })
        choices = r.json()["choices"]
        assert len(choices) >= 1
        assert choices[0]["message"]["role"] == "assistant"
        assert isinstance(choices[0]["message"]["content"], str)
        assert choices[0]["finish_reason"] in ("stop", "length")

    def test_usage_has_all_required_fields(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app, tokens=5)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "test message"}],
            })
        usage = r.json()["usage"]
        assert "prompt_tokens" in usage, "prompt_tokens missing from usage"
        assert "completion_tokens" in usage, "completion_tokens missing from usage"
        assert "total_tokens" in usage, "total_tokens missing from usage"
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_id_starts_with_chatcmpl(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert r.json()["id"].startswith("chatcmpl-")

    def test_no_model_deployed_returns_404(self, home_app):
        with TestClient(home_app) as client:
            import meshpool.controlplane.app as _appmod
            _appmod.CP.inference.list = MagicMock(return_value=[])
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert "message" in body["error"]

    def test_system_and_assistant_messages_flattened(self, home_app):
        """Multi-turn messages are included in the prompt (not just last user msg)."""
        captured: list[str] = []

        async def _capture_chat(model_id, prompt, max_new_tokens, **kw):
            captured.append(prompt)
            return _fake_chat_result()

        with TestClient(home_app) as client:
            import meshpool.controlplane.app as _appmod
            cp = _appmod.CP
            cp.inference.list = MagicMock(return_value=[_mock_deployment()])
            cp.inference.chat = _capture_chat
            client.post("/v1/chat/completions", json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ],
            })
        assert captured, "chat was never called"
        prompt = captured[0]
        assert "You are helpful" in prompt
        assert "Hello" in prompt
        assert "Hi there" in prompt
        assert "How are you?" in prompt


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — streaming (SSE)
# ---------------------------------------------------------------------------

class TestChatCompletionsStreaming:
    def test_stream_true_returns_event_stream(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_body_has_done_sentinel(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
        assert "data: [DONE]" in r.text

    def test_stream_chunks_have_correct_object_type(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
        chunks = []
        for line in r.text.splitlines():
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                chunks.append(json.loads(line[6:]))
        assert chunks, "no data chunks found in stream"
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk


# ---------------------------------------------------------------------------
# POST /v1/completions  (legacy text completion)
# ---------------------------------------------------------------------------

class TestCompletions:
    def test_basic_response_shape(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/completions", json={
                "model": "nano-gpt",
                "prompt": "The quick brown fox",
                "max_tokens": 16,
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion"
        assert "id" in body
        assert "choices" in body
        assert "usage" in body

    def test_choices_have_text_field(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app, text="jumped over")
            r = client.post("/v1/completions", json={
                "prompt": "The quick brown fox",
            })
        choices = r.json()["choices"]
        assert choices[0]["text"] == "jumped over"
        assert choices[0]["index"] == 0
        assert "finish_reason" in choices[0]

    def test_usage_fields_present(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app, tokens=3)
            r = client.post("/v1/completions", json={"prompt": "hello"})
        usage = r.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["completion_tokens"] == 3

    def test_id_starts_with_cmpl(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/completions", json={"prompt": "test"})
        assert r.json()["id"].startswith("cmpl-")

    def test_stream_returns_event_stream(self, home_app):
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post("/v1/completions", json={
                "prompt": "test", "stream": True,
            })
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        assert "data: [DONE]" in r.text


# ---------------------------------------------------------------------------
# Spend cap enforcement — must return 429 with OpenAI error envelope
# ---------------------------------------------------------------------------

class TestSpendCap:
    def test_capped_owner_gets_429_not_402(self, home_app):
        with TestClient(home_app) as client:
            import meshpool.controlplane.app as _appmod
            cp = _appmod.CP
            _inject_mock_inference(home_app)
            # Set a cap at zero so owner is immediately exceeded
            cp.spend_caps.set_cap("default", 0.0)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert r.status_code == 429, (
            f"capped owner must get 429 (OpenAI standard), got {r.status_code}: {r.text}")

    def test_capped_response_uses_openai_error_envelope(self, home_app):
        with TestClient(home_app) as client:
            import meshpool.controlplane.app as _appmod
            cp = _appmod.CP
            _inject_mock_inference(home_app)
            cp.spend_caps.set_cap("default", 0.0)
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
            })
        body = r.json()
        assert "error" in body, "response must use OpenAI error envelope"
        assert "message" in body["error"]
        assert "type" in body["error"]

    def test_completions_also_checks_spend_cap(self, home_app):
        with TestClient(home_app) as client:
            import meshpool.controlplane.app as _appmod
            cp = _appmod.CP
            _inject_mock_inference(home_app)
            cp.spend_caps.set_cap("default", 0.0)
            r = client.post("/v1/completions", json={"prompt": "hi"})
        assert r.status_code == 429
        assert "error" in r.json()


# ---------------------------------------------------------------------------
# Bearer token auth
# ---------------------------------------------------------------------------

class TestBearerAuth:
    def test_bearer_token_accepted_as_session(self, home_app):
        """Bearer <session-token> must be treated as X-MeshPool-Session."""
        with TestClient(home_app) as client:
            import meshpool.controlplane.app as _appmod
            cp = _appmod.CP
            _inject_mock_inference(home_app)
            # Obtain a real session token
            _, session_tok = cp.accounts.local_session(cp.identity)
            # Use it as a Bearer token
            r = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer {session_tok}"},
            )
        assert r.status_code == 200, (
            f"Bearer session token must be accepted, got {r.status_code}: {r.text}")

    def test_invalid_bearer_token_is_rejected(self, home_app):
        """An unrecognised API key must NOT be served as the 'default' owner.

        This test previously asserted the opposite (HTTP 200, silently falling
        back to the "default" owner). That billed another account for the work
        and bypassed the caller's own per-owner spend cap, so one typo in a key
        would spend someone else's credits. OpenAI answers 401 invalid_api_key.
        """
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer sk-definitely-not-a-valid-key"},
            )
        assert r.status_code == 401, r.text
        assert r.json()["error"]["code"] == "invalid_api_key"


# ---------------------------------------------------------------------------
# Error envelope shape
# ---------------------------------------------------------------------------

class TestErrorEnvelope:
    def test_missing_messages_returns_422(self, home_app):
        """FastAPI validation error is not OpenAI-shaped but that's acceptable for
        schema errors (the SDK validates before sending)."""
        with TestClient(home_app) as client:
            r = client.post("/v1/chat/completions", json={})
        # 422 from FastAPI's body validation
        assert r.status_code == 422

    def test_inference_error_uses_openai_envelope(self, home_app):
        """A runtime error from the inference engine must be wrapped."""
        with TestClient(home_app) as client:
            import meshpool.controlplane.app as _appmod
            cp = _appmod.CP
            cp.inference.list = MagicMock(return_value=[_mock_deployment()])

            async def _fail(model_id, prompt, max_new_tokens, **kw):
                raise RuntimeError("model not ready")

            cp.inference.chat = _fail
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert r.status_code in (503, 404)
        body = r.json()
        assert "error" in body
        assert "message" in body["error"]


# ---------------------------------------------------------------------------
# A wrong API key must be rejected, not silently billed to someone else
# ---------------------------------------------------------------------------

class TestWrongApiKeyIsRejected:
    """An unrecognised key used to fall through to the ``default`` owner.

    That is worse than it sounds: the request is served, the work is billed to
    another account, and because spend caps are per-owner the caller's own cap
    never applies. A single typo in an API key would spend someone else's
    credits. OpenAI answers 401 ``invalid_api_key``; so must we.
    """

    def test_chat_rejects_unknown_bearer(self, home_app):
        c = TestClient(home_app)
        r = c.post("/v1/chat/completions",
                   headers={"Authorization": "Bearer sk-not-a-real-key"},
                   json={"model": "nano-gpt",
                         "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401, r.text
        body = r.json()
        assert body["error"]["code"] == "invalid_api_key"

    def test_completions_rejects_unknown_bearer(self, home_app):
        c = TestClient(home_app)
        r = c.post("/v1/completions",
                   headers={"Authorization": "Bearer sk-not-a-real-key"},
                   json={"model": "nano-gpt", "prompt": "hi"})
        assert r.status_code == 401, r.text
        assert r.json()["error"]["code"] == "invalid_api_key"

    def test_no_header_at_all_is_still_allowed_in_open_mode(self, home_app):
        """Absent credential keeps existing behaviour; only a WRONG one fails.

        Without this control the tests above could pass simply because the
        endpoint rejects everything.
        """
        with TestClient(home_app) as client:
            _inject_mock_inference(home_app)
            r = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code != 401, r.text

    def test_error_body_uses_the_openai_envelope(self, home_app):
        c = TestClient(home_app)
        r = c.post("/v1/chat/completions",
                   headers={"Authorization": "Bearer nope"},
                   json={"model": "nano-gpt",
                         "messages": [{"role": "user", "content": "hi"}]})
        err = r.json()["error"]
        assert set(("message", "type", "code")).issubset(err.keys())
