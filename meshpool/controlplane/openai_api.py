"""OpenAI-compatible inference API — /v1/chat/completions, /v1/completions, /v1/models.

Wires the OpenAI wire protocol onto MeshPool's existing inference engine and
auth/billing infrastructure.  No parallel inference stack is built.

    POST /v1/chat/completions  — chat, streaming (SSE) and non-streaming
    POST /v1/completions       — legacy text completion
    GET  /v1/models            — model listing in OpenAI {"object":"list","data":[...]} shape

Auth:   Authorization: Bearer <token>  maps onto MeshPool's existing session token.
Errors: returned in OpenAI envelope {"error": {"message":..., "type":..., "code":...}}.
Spend:  enforced through SpendCapManager; capped owners receive HTTP 429.
Usage:  prompt_tokens + completion_tokens + total_tokens reported in every response.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..common.models import now_ms


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    status: int = 400,
) -> JSONResponse:
    """Return a JSONResponse in the OpenAI error envelope shape."""
    body: dict = {"error": {"message": message, "type": error_type}}
    if code is not None:
        body["error"]["code"] = code
    return JSONResponse(status_code=status, content=body)


# ---------------------------------------------------------------------------
# Request models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class _ChatMsg(BaseModel):
    role: str
    content: str


class OAChatRequest(BaseModel):
    model: str = "nano-gpt"
    messages: list[_ChatMsg]
    max_tokens: int = 48
    temperature: float = 0.8
    seed: int = 0
    stream: bool = False
    # MeshPool extensions — silently ignored by unmodified OpenAI clients
    speculative: bool = True
    draft_k: int = 4
    draft_layers: int = 2
    chunked_prefill: int = 0


class OACompletionRequest(BaseModel):
    model: str = "nano-gpt"
    prompt: str
    max_tokens: int = 48
    temperature: float = 0.8
    seed: int = 0
    stream: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_chat(messages: list[_ChatMsg]) -> str:
    """Render an OpenAI message list into a single prompt string."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        role = (m.role or "user").strip().lower()
        if role == "system":
            parts.append(m.content)
        elif role == "assistant":
            parts.append(f"Assistant: {m.content}")
        else:
            parts.append(f"User: {m.content}")
    if messages[-1].role != "assistant":
        parts.append("Assistant:")
    return "\n".join(p for p in parts if p)


def _bearer_token(request: Request) -> Optional[str]:
    """Extract the raw Bearer token from the Authorization header, or None."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        return tok or None
    return None


def _resolve_owner(request: Request, cp) -> str:
    """Resolve the consuming owner for an OpenAI request.

    Resolution order:
    1. Authorization: Bearer <session-token>
    2. X-MeshPool-Session header (same as existing endpoints)
    3. principal already stored by the auth middleware (enforced mode)
    4. Fallback to "default" (open mode / no credential)
    """
    # 1. Bearer token
    tok = _bearer_token(request)
    if tok:
        owner = cp.accounts.owner_for_session(tok)
        if owner:
            return owner

    # 2. Session header
    sess = request.headers.get("X-MeshPool-Session", "")
    if sess:
        owner = cp.accounts.owner_for_session(sess)
        if owner:
            return owner

    # 3. Principal from middleware
    principal = getattr(request.state, "principal", None)
    if principal and getattr(principal, "kind", "") not in ("anonymous", None):
        subj = getattr(principal, "subject", "")
        if subj and subj != "anonymous":
            return subj

    return "default"


def _reject_bad_key(request: Request, cp) -> Optional[JSONResponse]:
    """401 when a caller presents an API key that does not resolve to anyone.

    Falling back to the ``default`` owner for an unrecognised key is worse than
    it first looks. The request is served, the work is billed to somebody else,
    and — because spend caps are per-owner — the caller's own cap never applies.
    A typo in an API key would silently spend another account's credits.

    Only an *explicitly wrong* credential is rejected here. A request with no
    Authorization header at all keeps the existing open-mode behaviour, so
    local single-machine use is unaffected.
    """
    tok = _bearer_token(request)
    if not tok:
        return None
    if cp.accounts.owner_for_session(tok):
        return None
    principal = getattr(request.state, "principal", None)
    if principal and getattr(principal, "kind", "") not in ("anonymous", None):
        # A signed capability token presented as a bearer already resolved.
        subj = getattr(principal, "subject", "")
        if subj and subj != "anonymous":
            return None
    return openai_error(
        "Incorrect API key provided. Use a MeshPool session token as the API "
        "key; create one with `meshpool auth` or POST /api/v1/auth/local.",
        error_type="invalid_request_error",
        code="invalid_api_key",
        status=401,
    )


def _check_credits(cp, owner: str) -> Optional[JSONResponse]:
    """Return None if the owner may spend; return a 429 error response if capped.

    Cap and balance lookups are wrapped because they touch the datastore. The
    batch dispatcher's equivalent check deliberately fails **open** — an
    infrastructure error must never freeze the mesh — and this path follows the
    same rule, so a storage hiccup degrades to "allowed" rather than turning
    every inference request into a 500.
    """
    try:
        capped = cp.spend_caps.exceeded(owner)
    except Exception:
        capped = False
    if capped:
        st = cp.spend_caps.status(owner)
        return openai_error(
            f"spend cap reached for {owner}: "
            f"{st['spent']} of {st['cap']} compute-credits used. "
            "Raise the cap via PUT /api/v1/account/spend-cap",
            error_type="insufficient_quota",
            code="insufficient_quota",
            status=429,
        )
    try:
        can_spend = cp.accounts.can_spend(owner)
    except Exception:
        can_spend = True
    if not can_spend:
        return openai_error(
            f"insufficient compute-credits for {owner}: donate spare compute "
            "to earn credits, or link an account for a larger starter balance",
            error_type="insufficient_quota",
            code="insufficient_quota",
            status=429,
        )
    return None


def _openai_finish_reason(out: dict, completion_tokens: int, max_tokens: int) -> str:
    """Translate the engine's reason for stopping into OpenAI's vocabulary.

    The engine knows precisely why generation ended; a token count does not. Deriving
    the reason from ``completion_tokens >= max_tokens`` gets the important case exactly
    backwards: when the model's context window cuts a reply short, the count falls
    *below* max_tokens, so the naive rule reports ``"stop"`` -- which tells an OpenAI
    client the reply is complete when it was truncated mid-sentence. Agent loops and
    continuation logic then silently do the wrong thing.

    OpenAI's schema only admits stop/length/tool_calls/content_filter, so a context
    cut-off maps to ``"length"``: the completion was ended by a limit, not by the
    model. The precise reason stays available on MeshPool's native response.
    """
    reason = out.get("finish_reason")
    if reason in ("length", "context_window"):
        return "length"
    if reason == "stop":
        return "stop"
    return "length" if completion_tokens >= max_tokens else "stop"


def _sse_chunk(comp_id: str, model: str, delta: dict,
               finish_reason: Optional[str] = None) -> str:
    """Encode one Server-Sent Events chat-completion chunk."""
    chunk = {
        "id": comp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_openai_routes(app, cp_factory: Callable) -> None:  # noqa: C901
    """Register /v1/* OpenAI-compatible routes onto *app*.

    ``cp_factory`` is a zero-argument callable that returns the live
    ControlPlane instance (raising HTTPException 503 when not ready).
    Pass ``_cp`` from ``controlplane/app.py``.
    """

    # ------------------------------------------------------------------ #
    # GET /v1/models
    # ------------------------------------------------------------------ #
    @app.get("/v1/models")
    async def openai_list_models() -> JSONResponse:
        from ..inference import model as M
        try:
            cp = cp_factory()
        except Exception:
            return openai_error("service unavailable", "server_error", status=503)

        created_ts = int(time.time())
        data: list[dict] = []
        # Built-in sharded models always present
        for name in sorted(M.REGISTRY.keys()):
            data.append({
                "id": name,
                "object": "model",
                "created": created_ts,
                "owned_by": "meshpool",
            })
        # Ollama models advertised by currently-online nodes
        seen: set[str] = {d["id"] for d in data}
        for n in cp.registry.online():
            for m in (n.capability.ollama_models or []):
                if m not in seen:
                    seen.add(m)
                    data.append({
                        "id": m,
                        "object": "model",
                        "created": created_ts,
                        "owned_by": "meshpool-ollama",
                    })
        return JSONResponse({"object": "list", "data": data})

    # ------------------------------------------------------------------ #
    # POST /v1/chat/completions
    # ------------------------------------------------------------------ #
    @app.post("/v1/chat/completions")
    async def openai_chat_completions(  # noqa: C901
        req: OAChatRequest, request: Request,
    ) -> JSONResponse:
        try:
            cp = cp_factory()
        except Exception:
            return openai_error("service unavailable", "server_error", status=503)

        # Resolve owner and check spend cap
        bad_key = _reject_bad_key(request, cp)
        if bad_key is not None:
            return bad_key
        owner = _resolve_owner(request, cp)
        cap_err = _check_credits(cp, owner)
        if cap_err is not None:
            return cap_err

        # Resolve the deployed model to use
        deployed = cp.inference.list()
        if not deployed:
            return openai_error(
                "no model is deployed; POST /api/v1/models to deploy one",
                error_type="model_not_found",
                code="model_not_found",
                status=404,
            )
        model_id = deployed[0].id

        prompt = _flatten_chat(req.messages)
        # MeshPool's built-in tokenizer is character-based (1 char = 1 token).
        # Reported honestly; this is not GPT-4 tokenisation.
        prompt_tokens = len(prompt)
        comp_id = f"chatcmpl-{now_ms()}"
        gen_kw = dict(
            temperature=req.temperature,
            seed=req.seed,
            speculative=req.speculative,
            draft_k=req.draft_k,
            draft_layers=req.draft_layers,
            chunked_prefill=req.chunked_prefill,
        )

        if req.stream:
            agen = cp.inference.chat_stream(model_id, prompt, req.max_tokens, **gen_kw)
            # Pull the first item before committing to a 200. Generation failures
            # surface on the very first step (the deployment is resolved before any
            # token is produced), and an OpenAI client can act on a 404/503 body but
            # not on an exception buried inside an already-started event stream.
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                first = None
            except KeyError:
                await agen.aclose()
                return openai_error("model deployment not found",
                                    error_type="model_not_found",
                                    code="model_not_found", status=404)
            except RuntimeError as exc:
                await agen.aclose()
                return openai_error(str(exc), error_type="server_error", status=503)

            model_name = deployed[0].model

            async def _stream():
                yield _sse_chunk(comp_id, model_name, {"role": "assistant"})
                finish_reason = "stop"
                item = first
                try:
                    while item is not None:
                        if "delta" in item:
                            yield _sse_chunk(comp_id, model_name,
                                             {"content": item["delta"]})
                        elif item.get("done"):
                            out = item["result"]
                            finish_reason = _openai_finish_reason(
                                out, out["tokens"], req.max_tokens)
                        try:
                            item = await agen.__anext__()
                        except StopAsyncIteration:
                            item = None
                except Exception as exc:  # noqa: BLE001
                    # The status line is long gone. Report the truncation in-band so
                    # the client sees a cut-off answer as a cut-off answer.
                    yield _sse_chunk(comp_id, model_name,
                                     {"content": ""}, finish_reason="length")
                    yield ("data: " + json.dumps(
                        {"error": {"message": str(exc), "type": "server_error"}})
                        + "\n\n")
                    yield "data: [DONE]\n\n"
                    return
                yield _sse_chunk(comp_id, model_name, {}, finish_reason=finish_reason)
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(), media_type="text/event-stream")

        try:
            out = await cp.inference.chat(model_id, prompt, req.max_tokens, **gen_kw)
        except KeyError:
            return openai_error("model deployment not found",
                                error_type="model_not_found",
                                code="model_not_found", status=404)
        except RuntimeError as exc:
            return openai_error(str(exc), error_type="server_error", status=503)

        completion_tokens: int = out["tokens"]
        total_tokens = prompt_tokens + completion_tokens
        finish_reason = _openai_finish_reason(out, completion_tokens, req.max_tokens)

        return JSONResponse({
            "id": comp_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": out["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out["text"]},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        })

    # ------------------------------------------------------------------ #
    # POST /v1/completions  (legacy text completion)
    # ------------------------------------------------------------------ #
    @app.post("/v1/completions")
    async def openai_completions(  # noqa: C901
        req: OACompletionRequest, request: Request,
    ) -> JSONResponse:
        try:
            cp = cp_factory()
        except Exception:
            return openai_error("service unavailable", "server_error", status=503)

        bad_key = _reject_bad_key(request, cp)
        if bad_key is not None:
            return bad_key
        owner = _resolve_owner(request, cp)
        cap_err = _check_credits(cp, owner)
        if cap_err is not None:
            return cap_err

        deployed = cp.inference.list()
        if not deployed:
            return openai_error(
                "no model is deployed; POST /api/v1/models to deploy one",
                error_type="model_not_found",
                code="model_not_found",
                status=404,
            )
        model_id = deployed[0].id
        prompt_tokens = len(req.prompt)
        comp_id = f"cmpl-{now_ms()}"

        if req.stream:
            agen = cp.inference.chat_stream(
                model_id, req.prompt, req.max_tokens,
                temperature=req.temperature, seed=req.seed)
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                first = None
            except KeyError:
                await agen.aclose()
                return openai_error("model deployment not found",
                                    error_type="model_not_found",
                                    code="model_not_found", status=404)
            except RuntimeError as exc:
                await agen.aclose()
                return openai_error(str(exc), error_type="server_error", status=503)

            model_name = deployed[0].model

            def _text_chunk(text: str, finish: Optional[str] = None,
                            usage: Optional[dict] = None) -> str:
                body = {
                    "id": comp_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish,
                    }],
                }
                if usage is not None:
                    body["usage"] = usage
                return f"data: {json.dumps(body)}\n\n"

            async def _stream():
                finish_reason = "stop"
                completion_tokens = 0
                item = first
                try:
                    while item is not None:
                        if "delta" in item:
                            yield _text_chunk(item["delta"])
                        elif item.get("done"):
                            out = item["result"]
                            completion_tokens = out["tokens"]
                            finish_reason = _openai_finish_reason(
                                out, completion_tokens, req.max_tokens)
                        try:
                            item = await agen.__anext__()
                        except StopAsyncIteration:
                            item = None
                except Exception as exc:  # noqa: BLE001
                    yield _text_chunk("", finish="length")
                    yield ("data: " + json.dumps(
                        {"error": {"message": str(exc), "type": "server_error"}})
                        + "\n\n")
                    yield "data: [DONE]\n\n"
                    return
                yield _text_chunk("", finish=finish_reason, usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                })
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(), media_type="text/event-stream")

        try:
            out = await cp.inference.chat(
                model_id, req.prompt, req.max_tokens,
                temperature=req.temperature,
                seed=req.seed,
            )
        except KeyError:
            return openai_error("model deployment not found",
                                error_type="model_not_found",
                                code="model_not_found", status=404)
        except RuntimeError as exc:
            return openai_error(str(exc), error_type="server_error", status=503)

        completion_tokens = out["tokens"]
        total_tokens = prompt_tokens + completion_tokens
        finish_reason = _openai_finish_reason(out, completion_tokens, req.max_tokens)

        return JSONResponse({
            "id": comp_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": out["model"],
            "choices": [{
                "text": out["text"],
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        })
