"""Control-plane FastAPI application and background control loops."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..common.config import Settings
from ..common.db import Store
from ..common.identity import DeviceIdentity
from ..common.logging import get_logger
from ..common.models import DagStageSpec, DonationPolicy, Heartbeat, NodeStatus, RegisterRequest, new_id, now_ms
from ..common.netutil import TUNNEL_SENTINEL_URL
from ..common.secrets import SecretBroker, SecretError
from ..common.transport import HttpTransport, TransportError
from ..common.tracing import TraceContext, start_span
from ..common.trust import ContributionLedger, TrustEngine, fair_share_priority
from ..storage.engine import StorageError
from ..common.auth import (
    Scope, Principal, AuthzError, Authorizer,
    mint_anonymous_principal, mint_loopback_principal, mint_node_principal,
    mint_session_principal, TokenVerifier,
)
from ..batch.engine import BatchManager
from ..batch.dag import DagError, DagManager
from ..cloud.tunnel import CompositeTransport, TunnelHub, HDR_NODE_TOKEN
from ..cloud.quic import QuicFallbackTransport, QuicHub, start_quic_server
from .accounts import AccountManager, MAX_ACTIVE_JOBS_PER_OWNER
from .spend import PUBLIC_DEFAULT_SPEND_CAP, SpendCapManager, default_cap_from_env
from .registry import EventBus, NodeRegistry, schedulable
from .timeline import JobTimeline

log = get_logger("controlplane")

STATIC_DIR = Path(__file__).parent / "static"
# Built output of the React console in ``web/``. Optional: absent until someone runs
# ``npm --prefix web run build``.
CONSOLE_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"

# ---------------------------------------------------------------------------
# Authorization metadata — populated once by _register_routes.
#
# ROUTE_SCOPES: maps "METHOD /path/with/{params}" -> Scope (or None for public).
#   A route with no entry is denied by default in enforced mode.
#   The sim/scenarios/verify_authz_matrix.py scenario asserts every registered
#   route has an entry here, so this table can never silently rot.
#
# PUBLIC_ALLOWLIST: maps the same key -> a brief justification string.
#   Every public route must appear here with an explicit reason.
# ---------------------------------------------------------------------------
ROUTE_SCOPES: dict[str, Optional[Scope]] = {}
PUBLIC_ALLOWLIST: dict[str, str] = {}

# Header name for capability tokens issued by TokenIssuer.
HDR_CAPABILITY_TOKEN = "X-MeshPool-Token"


def _declare_route(method: str, path: str, scope: Optional[Scope],
                   public_justification: str = "") -> None:
    """Register a route's required scope (or None = public) in the global tables."""
    key = f"{method} {path}"
    ROUTE_SCOPES[key] = scope
    if scope is None:
        PUBLIC_ALLOWLIST[key] = public_justification or "explicitly public"


_authorizer = Authorizer()


class ControlPlane:
    """Holds all live control-plane services (assembled during lifespan)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = Store(settings.db_path())
        # This machine's own persistent identity. It lets the locally-served
        # dashboard mint a zero-signup "Use" session for the operator (see the
        # loopback-gated /api/v1/auth/local route) and gives the local user a
        # stable owner subject that accrues credits across restarts.
        self.identity = DeviceIdentity.load_or_create(settings.data_dir)
        self.events = EventBus()
        self.registry = NodeRegistry(self.store, settings.node_timeout_s, self.events)
        self.trust = TrustEngine(self.store, tier=settings.trust_tier)
        self.ledger = ContributionLedger(self.store)
        self.timeline = JobTimeline(self.store)
        # Optional accounts + credit gating (active only at Tier 2 / public mesh).
        # Anonymous devices work fully; this just resolves the owner that earns and
        # spends credits and enforces earn-to-spend for strangers.
        self.accounts = AccountManager(self.store, self.ledger, settings.trust_tier)
        # Hard spend caps (plan §6.3): a cap STOPS work rather than emailing about
        # it afterwards. Enforced at every tier -- protection from bill shock is a
        # user safety property, not a public-mesh billing feature. Owners are
        # uncapped unless a cap is set or MESHPOOL_DEFAULT_SPEND_CAP is configured.
        self.spend_caps = SpendCapManager(
            self.store,
            default_cap=(
                default_cap_from_env(PUBLIC_DEFAULT_SPEND_CAP)
                if settings.trust_tier >= 2 else None),
        )
        # Abuse reporting (spec §6.3)
        from ..common.egress import AbuseReportStore, RateLimiter
        self._abuse_reports = AbuseReportStore()
        self._abuse_rate_limiter = RateLimiter(limit=10, window_s=60.0)
        from ..common.billing import BillingManager
        self.billing = BillingManager(self.store, self.ledger)
        # S3-compatible API: access_key_id -> (secret_key, owner_subject).
        # Durable (see storage/s3_credentials.py) -- an in-memory dict silently
        # invalidated every boto3 client on control-plane restart, surfacing as
        # SignatureDoesNotMatch, which blames the client for a server-side loss.
        from ..storage.s3_credentials import S3CredentialStore
        self.s3_credentials = S3CredentialStore(self.store)
        # Donations credit the resolved owner (linked account, else the device).
        self.registry.owner_resolver = self.accounts.owner_for_device
        self.client = httpx.AsyncClient()
        # The reverse-tunnel hub tracks agents that dial in from behind NAT. The
        # CompositeTransport prefers a node's live tunnel and falls back to direct
        # HTTP for LAN nodes — so a single pool serves both, and in a pure-LAN run
        # the hub stays empty and behaviour is byte-for-byte the old HttpTransport.
        self.hub = TunnelHub()
        self.quic_hub = QuicHub(self.registry)
        self.tunnel_transport = CompositeTransport(
            self.hub, HttpTransport(self.registry, self.client))
        self.transport = QuicFallbackTransport(self.quic_hub, self.tunnel_transport)
        self.quic_addr = ""
        # A tunnel-only node (it advertises the unreachable sentinel URL) can be
        # reached only while its reverse tunnel is attached. Teaching the registry
        # this keeps the schedulers from dispatching to a node that is "online" by
        # heartbeat but whose tunnel has dropped or not yet (re)connected — e.g. the
        # brief window right after a coordinator restart — which would otherwise burn
        # retries against an unreachable address and risk dropping work. Directly
        # addressable LAN nodes are always considered reachable, so LAN behaviour is
        # unchanged and the hub stays out of the path.
        self.registry.reachability = self._node_reachable
        self.secrets = SecretBroker()
        self.store.recover_inflight_units()
        self.batch = BatchManager(self.registry, self.store, self.ledger,
                                  self.trust, self.events, self.transport, self.secrets)
        self.batch.billing = self.billing
        self.timeline.instrument_batch(self.batch)
        self.dags = DagManager(self.store, self.batch)
        # a one-time invite token powers the "magic" pairing onboarding
        self.invite_token = secrets.token_urlsafe(9)
        self.hosting = None   # set in app wiring (M4)
        self.inference = None  # set in app wiring (M5)
        self.storage = None    # set in app wiring (P3 pooled object store)
        self.storage_audit = None  # set in app wiring (proof-of-storage audit engine)

    def _node_reachable(self, node) -> bool:
        """Whether the scheduler can actually deliver work to ``node`` right now.

        A tunnel-only node advertises the unreachable sentinel URL and is reachable
        solely through its attached reverse tunnel; a directly addressable node (LAN)
        is always reachable. This prevents dispatch to a node that heartbeats as online
        but whose tunnel is not (yet) attached.
        """
        if node.url == TUNNEL_SENTINEL_URL:
            return self.quic_hub.is_connected(node.id) or self.hub.is_connected(node.id)
        return True

    async def reaper_loop(self) -> None:
        # Tell the registry how often we intend to call it, so it can recognise a
        # stall in this loop and defer evictions rather than blaming the nodes.
        self.registry.reap_interval_s = 2.0
        while True:
            try:
                gone = self.registry.reap()
                for node_id in gone:
                    self.hub.drop(node_id)   # fail in-flight tunnel requests promptly
                    self.batch.requeue_node_units(node_id)
                    if self.hosting:
                        self.hosting.on_node_offline(node_id)
                    if self.inference:
                        self.inference.on_node_offline(node_id)
                    if self.storage:
                        self.storage.on_node_offline(node_id)
            except Exception as exc:  # noqa: BLE001
                log.error("reaper error", err=str(exc))
            await asyncio.sleep(2.0)


CP: Optional[ControlPlane] = None


def build_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global CP
        CP = ControlPlane(settings)
        quic_server = None
        if settings.transport.lower() in ("auto", "quic"):
            bind_host = "0.0.0.0" if settings.host in ("0.0.0.0", "::") else settings.host
            quic_server = await start_quic_server(
                bind_host, settings.quic_port, settings.data_dir, CP.quic_hub)
            CP.quic_addr = f"{settings.advertise_host or settings.host}:{settings.quic_port}"
        # wire engines that depend on CP being constructed
        from ..hosting.engine import HostingManager
        from ..inference.engine import InferenceManager
        from ..storage.engine import StorageManager
        from ..storage.audit import StorageAuditEngine
        CP.hosting = HostingManager(CP.registry, CP.store, CP.ledger, CP.events, CP.transport)
        CP.inference = InferenceManager(CP.registry, CP.store, CP.ledger, CP.events, CP.transport)
        # Attach the proof-of-storage audit engine so verification material is recorded
        # at shard placement time and the audit loop can challenge nodes on demand.
        CP.storage_audit = StorageAuditEngine(CP.ledger)
        CP.storage = StorageManager(CP.registry, CP.store, CP.ledger, CP.events, CP.transport,
                                    audit=CP.storage_audit)
        # A LARGE app's bundle is staged in the pooled store and hydrated on each
        # replica start, so hosting needs a handle to storage.
        CP.hosting.storage = CP.storage
        # A model too big to pre-install on every node has its weights staged in the
        # pool and hydrated per-stage on load, so inference needs the same handle.
        CP.inference.storage = CP.storage
        # Node classification (§3.6) reads reputation, so placement can keep
        # Drifter-class machines out of pipeline stages when the model still fits
        # without them. A pipeline's availability is the product of its stages',
        # so one node that unplugs at 6pm degrades every request.
        CP.inference.trust = CP.trust
        # User WASM modules are distributed through the pooled object store, so the
        # batch engine needs the same handle.
        CP.batch.storage = CP.storage
        # Hard spend caps must reach the dispatcher, not just the API: a cap
        # enforced only at submission cannot stop a job that is already running.
        CP.batch.spend_caps = CP.spend_caps
        # Workload-matched result verification (spec §P6).  Only wired at Tier 2
        # (public/adversarial) where strangers donate compute and lying is profitable.
        # At lower tiers the verifier stays None and all guards in the engine fail-open,
        # so LAN behaviour is completely unchanged.
        if settings.trust_tier >= 2:
            from .verification import Verifier, VerificationPolicy
            _vp = VerificationPolicy()
            CP.batch.verifier = Verifier(ledger=CP.ledger, policy=_vp, trust=CP.trust)
        # Audit interval from env so tests can shrink it.
        _audit_interval = float(os.environ.get("MESHPOOL_AUDIT_INTERVAL_S", "60"))
        tasks = [
            asyncio.create_task(CP.reaper_loop()),
            asyncio.create_task(CP.batch.run_loop()),
            asyncio.create_task(CP.dags.run_loop()),
            asyncio.create_task(CP.hosting.run_loop()),
            asyncio.create_task(CP.storage.run_loop()),
            asyncio.create_task(
                CP.storage_audit.run_audit_loop(CP.storage, CP.transport, _audit_interval)),
        ]
        adv = settings.control_plane_url
        log.info("control plane up", scale=settings.scale, tier=settings.trust_tier,
                 bind=f"{settings.host}:{settings.port}", advertise=adv,
                 quic=CP.quic_addr or "disabled",
                 invite=CP.invite_token)
        if settings.host in ("0.0.0.0", "::"):
            print("\n" + "=" * 64, flush=True)
            print("  MeshPool control plane is LIVE", flush=True)
            print(f"  Dashboard   : {adv}", flush=True)
            print(f"  Add a machine (run on each laptop, in the meshpool folder):", flush=True)
            print(f"    Windows : powershell -ExecutionPolicy Bypass -File deploy\\join.ps1 "
                  f"-Server {adv} -Token {CP.invite_token}", flush=True)
            print(f"    macOS/Linux : bash deploy/join.sh --server {adv} "
                  f"--token {CP.invite_token}", flush=True)
            print("=" * 64 + "\n", flush=True)
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            if quic_server is not None:
                quic_server.close()
            await CP.client.aclose()
            CP.store.close()

    app = FastAPI(title="MeshPool Control Plane", version="0.1.0", lifespan=lifespan)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # The React console (``web/``) is an optional, separately built artefact. When
    # ``npm run build`` has produced it, serve it at /console; when it hasn't, the
    # route below returns a build hint instead of a bare 404. The classic dashboard
    # at "/" is untouched either way, so neither surface can break the other.
    if CONSOLE_DIR.exists():
        app.mount("/console/assets",
                  StaticFiles(directory=str(CONSOLE_DIR / "assets")),
                  name="console-assets")

    _register_routes(app)
    _add_auth_middleware(app, settings)
    return app


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class JobSubmit(BaseModel):
    name: str = "job"
    func: str
    payloads: list[dict]
    cpu_pct: int = 15
    ram_mb: int = 128
    owner: str = "default"
    secret_names: list[str] = []


class DagSubmit(BaseModel):
    name: str = "dag"
    stages: list[DagStageSpec]
    owner: str = "default"


class DeploySubmit(BaseModel):
    name: str
    app: str = "hello"
    replicas: int = 2
    owner: str = "default"
    site: Optional[dict[str, str]] = None
    start_cmd: Optional[str] = None
    env: Optional[dict[str, str]] = None
    autoscale_min: Optional[int] = None
    autoscale_max: Optional[int] = None


class ModelDeploySubmit(BaseModel):
    model: str = "nano-gpt"
    owner: str = "default"


class SpendCapBody(BaseModel):
    # None clears the cap (the owner becomes uncapped). Negative values are
    # rejected by SpendCapManager.set_cap.
    cap: Optional[float] = None


class CapabilityTokenCreate(BaseModel):
    label: str
    scopes: list[str]
    expires_in_seconds: Optional[float] = None
    expires_at_ms: Optional[int] = None
    spend_cap: Optional[float] = None


class RttReport(BaseModel):
    """Vivaldi RTT observation reported by an agent.

    ``from_node`` identifies the reporting node; ``to_node`` is the peer.
    In ``auth_mode=open`` the caller-supplied ids are trusted verbatim (same as
    heartbeat).  In ``auth_mode=enforced`` the handler verifies the
    X-MeshPool-Node-Token header and overrides ``from_node`` with the verified id.
    """
    from_node: str
    to_node: str
    rtt_ms: float


class InferSubmit(BaseModel):
    prompt: str
    max_new_tokens: int = 32
    model_id: Optional[str] = None
    temperature: float = 0.8
    seed: int = 0
    # Speculative Pipeline Decoding. Exact, not approximate: greedy output is
    # token-identical to ordinary decoding and sampled output has the same
    # distribution, so this is safe to leave on. Exposed mainly so benchmarks can
    # turn it off and measure the difference.
    speculative: bool = True
    draft_k: int = 4
    draft_layers: int = 2
    chunked_prefill: int = 0
    # Stream the answer as NDJSON deltas instead of one JSON object at the end.
    # Off by default so every existing caller keeps its current response shape.
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "nano-gpt"
    messages: list[ChatMessage]
    max_tokens: int = 48
    temperature: float = 0.8
    seed: int = 0
    # MeshPool extensions. Unknown fields are ignored by OpenAI clients, and these
    # all default to the same behaviour /api/v1/inference/chat uses, so an
    # unmodified OpenAI SDK gets speculative decoding without asking for it.
    speculative: bool = True
    draft_k: int = 4
    draft_layers: int = 2
    chunked_prefill: int = 0


class SecretPut(BaseModel):
    name: str
    value: str
    owner: str = "default"


class RedeemBody(BaseModel):
    lease_id: str
    node_id: str


class TunnelReply(BaseModel):
    """An agent's reply to a request pushed down its reverse tunnel."""
    id: str
    ok: bool = False
    status: Optional[int] = None
    body: Optional[dict] = None
    error: Optional[str] = None


class DeviceAuth(BaseModel):
    """Consumer-side proof-of-possession login (mirrors node enrollment)."""
    device_id: str
    public_key: str
    signature: str
    ts_ms: int


class LinkStart(BaseModel):
    session: str = ""     # optional; device is taken from the session when present


class LinkComplete(BaseModel):
    pending_id: str
    proof: str            # provider-specific (local provider: "<link_code>:<handle>")


class BucketGrant(BaseModel):
    grantee: str
    permissions: list[str]


def _cp() -> ControlPlane:
    if CP is None:
        raise HTTPException(status_code=503, detail="control plane not ready")
    return CP


# ---------------------------------------------------------------------------
# Principal extraction and middleware
# ---------------------------------------------------------------------------

def _token_verifier(cp: "ControlPlane") -> Optional[TokenVerifier]:
    """Return the CP-level TokenVerifier (lazily created, cached on the CP)."""
    if not hasattr(cp, "_token_verifier"):
        from cryptography.hazmat.primitives import serialization
        pub_raw = cp.identity._priv.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = cp.identity._priv.public_key()
        cp._token_verifier = TokenVerifier(pub)  # type: ignore[attr-defined]
    return cp._token_verifier  # type: ignore[attr-defined]


def _current_scopes_for_owner(owner: str) -> frozenset[Scope]:
    """Return the account's current scopes; capability tokens intersect with it."""
    return mint_session_principal(owner).scopes


def _principal_from_stored_capability(cp: "ControlPlane", secret: str) -> Optional[Principal]:
    row = cp.store.capability_token_by_secret(secret)
    if not row:
        return None
    now = now_ms()
    if row["revoked_ms"] is not None or int(row["expires_at_ms"]) < now:
        return None
    spend_cap = row.get("spend_cap")
    if spend_cap is not None and float(row.get("spent") or 0.0) >= float(spend_cap):
        return None
    stored = frozenset(
        Scope(s) for s in row["scopes"]
        if s in Scope._value2member_map_  # type: ignore[attr-defined]
    )
    effective = frozenset(s for s in stored if s in _current_scopes_for_owner(row["owner"]))
    cp.store.mark_capability_token_used(row["id"], now)
    return Principal(
        kind="user",
        subject=row["owner"],
        scopes=effective,
        via="token",
        expires_at_ms=row["expires_at_ms"],
        token_id=row["id"],
    )


def _request_token_id(request: Request) -> Optional[str]:
    principal = getattr(request.state, "principal", None)
    return getattr(principal, "token_id", None)


def _get_request_principal(request: Request, cp: "ControlPlane") -> Principal:
    """Extract and validate a Principal from the request headers.

    Resolution order:
    1. Capability token (``X-MeshPool-Token``) — signed, scoped, expiring.
    2. Consumer session (``X-MeshPool-Session``) — issued by /auth/device or /auth/local.
    3. Loopback — request comes from localhost and the CP is at Tier 0-1, or the
       session endpoint was already called; mints a full-access loopback Principal.
    4. No credential → returns None (caller decides whether to reject or continue).
    """
    # 1. Capability token
    cap = request.headers.get(HDR_CAPABILITY_TOKEN, "")
    if cap:
        stored = _principal_from_stored_capability(cp, cap)
        if stored is not None:
            return stored
        verifier = _token_verifier(cp)
        if verifier:
            p = verifier.verify(cap)
            if p and not p.is_expired():
                return p

    # 2. Per-node session token (used by agents for tunnel + secret redeem)
    node_tok = request.headers.get(HDR_NODE_TOKEN, "")
    if not node_tok:
        node_tok = request.headers.get("X-MeshPool-Node-Token", "")
    if node_tok:
        node_id = cp.registry.node_id_for_token(node_tok)
        if node_id:
            return mint_node_principal(node_id)

    # 3. Consumer session token
    sess = request.headers.get("X-MeshPool-Session", "")
    if sess:
        owner = cp.accounts.owner_for_session(sess)
        if owner:
            return mint_session_principal(owner)

    # 4. Authorization: Bearer <token> — OpenAI-compatible surface.
    #    Treat the Bearer value as a session token so existing OpenAI SDK callers
    #    can authenticate by setting their API key to a MeshPool session token.
    auth_hdr = request.headers.get("Authorization", "")
    if auth_hdr.lower().startswith("bearer "):
        bearer_tok = auth_hdr[7:].strip()
        if bearer_tok:
            owner = cp.accounts.owner_for_session(bearer_tok)
            if owner:
                return mint_session_principal(owner)
            stored = _principal_from_stored_capability(cp, bearer_tok)
            if stored is not None:
                return stored
            # Also accept a signed capability token delivered as a Bearer value
            verifier = _token_verifier(cp)
            if verifier:
                p = verifier.verify(bearer_tok)
                if p and not p.is_expired():
                    return p

    # 5. Loopback (Tier 0-1 only; Tier 2 requires an explicit credential)
    if cp.settings.trust_tier < 2 and _is_loopback(request):
        subject = cp.accounts.owner_for_device(cp.identity.device_id)
        return mint_loopback_principal(subject)

    # 6. AWS SigV4 (S3-compatible API callers such as boto3)
    auth_hdr2 = request.headers.get("Authorization", "")
    if auth_hdr2.startswith("AWS4-HMAC-SHA256 "):
        from ..storage.s3_api import verify_sigv4_request
        p = verify_sigv4_request(request, cp)
        if p is not None:
            return p

    return None  # type: ignore[return-value]  # no credential


import re as _re


def _path_to_regex(path: str) -> str:
    """Translate a Starlette path template into an anchored regex body.

    Two properties matter here and both have bitten us:

    * ``{name:path}`` is a *multi-segment* convertor — it must become ``.+``, not
      ``[^/]+``. Getting this wrong meant ``/app/{dep_id}/{path:path}`` only ever
      matched single-segment URLs, so a hosted app asking for
      ``/app/<dep>/static/css/main.css`` fell through to the deny-by-default
      branch and 404'd under ``auth_mode=enforced``.
    * Everything outside a ``{...}`` placeholder is a *literal* and must be
      escaped. Unescaped, the ``.`` in ``/openapi.json`` is a wildcard, so
      ``/openapiXjson`` matched it and inherited its public scope. Route
      classification must never be looser than the router's own matching.
    """
    out: list[str] = []
    pos = 0
    for m in _re.finditer(r"\{([^{}]+)\}", path):
        out.append(_re.escape(path[pos:m.start()]))
        out.append(".+" if m.group(1).endswith(":path") else "[^/]+")
        pos = m.end()
    out.append(_re.escape(path[pos:]))
    return "".join(out)


def _compile_route_patterns() -> list[tuple[_re.Pattern, Optional[Scope]]]:
    """Convert ROUTE_SCOPES keys into (method_path_regex, scope) for middleware matching.

    Ordered most-specific-first so that a broad ``{path:path}`` catch-all can never
    shadow a narrower sibling that was merely declared later. Specificity is
    approximated by literal length, which is exactly the signal that distinguishes
    ``/app/{dep_id}/{path:path}`` from ``/api/v1/nodes/{node_id}``.
    """
    entries = []
    for key, scope in ROUTE_SCOPES.items():
        method, _, path = key.partition(" ")
        full = f"^{_re.escape(method)} {_path_to_regex(path)}$"
        literal_len = len(_re.sub(r"\{[^{}]+\}", "", path))
        wildcards = path.count("{")
        entries.append((literal_len, -wildcards, _re.compile(full), scope))
    entries.sort(key=lambda e: (-e[0], e[1]))
    return [(pat, scope) for _, _, pat, scope in entries]


_route_patterns: list[tuple[_re.Pattern, Optional[Scope]]] = []


def _scope_for_request(method: str, path: str) -> Optional[Optional[Scope]]:
    """Return the required scope for a request (None = public, missing = deny).

    Returns a sentinel ``_MISSING`` if no pattern matched (deny by default).
    """
    key = f"{method} {path}"
    if _route_patterns:
        for pattern, scope in _route_patterns:
            if pattern.match(key):
                return scope
    return _MISSING


_MISSING = object()


def _add_auth_middleware(app: FastAPI, settings: Settings) -> None:  # noqa: C901
    """Add the authentication + authorisation middleware to ``app``.

    In ``auth_mode=open`` (the default): every request is given a full-access
    anonymous Principal, and scope checking is skipped entirely — this is the
    deliberate zero-friction LAN grant, not an accident.

    In ``auth_mode=enforced``: the middleware resolves a real Principal from the
    request headers, checks the route's required scope, and returns 401/403 if
    either is missing or insufficient. Public routes (scope=None in ROUTE_SCOPES)
    are explicitly exempted.
    """
    @app.middleware("http")
    async def _auth(request: Request, call_next):
        global _route_patterns
        if not _route_patterns and ROUTE_SCOPES:
            _route_patterns = _compile_route_patterns()

        cp_obj = CP
        if cp_obj is None or cp_obj.settings.auth_mode != "enforced":
            # open mode: explicitly grant a full-access anonymous principal
            request.state.principal = mint_anonymous_principal()
            return await call_next(request)

        method = request.method
        path = request.url.path

        # Determine the required scope for this route
        required = _scope_for_request(method, path)
        if required is _MISSING:
            # Static files and other Starlette mounts don't appear in ROUTE_SCOPES.
            # Static dashboard assets are public by nature (no sensitive data).
            if path.startswith("/static/") or path.startswith("/console/") or path == "/":
                request.state.principal = mint_anonymous_principal()
                return await call_next(request)
            # Unknown route: deny with a clear message.
            return Response(
                content='{"detail":"this endpoint is not reachable: check the path and method"}',
                status_code=404, media_type="application/json",
            )

        if required is None:
            # Explicitly public route (e.g. /health, /auth/device).
            request.state.principal = mint_anonymous_principal()
            return await call_next(request)

        # Need a real principal.
        principal = _get_request_principal(request, cp_obj)
        if principal is None:
            return Response(
                content=(
                    '{"detail":"authentication required: POST /api/v1/auth/device with '
                    'your device signature, or POST /api/v1/auth/local from the local '
                    'machine, then send the returned token as X-MeshPool-Session"}'
                ),
                status_code=401, media_type="application/json",
            )

        try:
            _authorizer.check_scope(principal, required)  # type: ignore[arg-type]
        except AuthzError as exc:
            return Response(
                content=f'{{"detail":"{exc}"}}',
                status_code=exc.status_code, media_type="application/json",
            )

        request.state.principal = principal
        return await call_next(request)


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _is_loopback(request: Request) -> bool:
    """True when the request originates from the same machine as the control plane.

    Used to gate zero-signup local sign-in: on the machine you installed MeshPool on
    the browser reaches the dashboard over loopback and "Use" just works; a browser
    hitting a *remote* public coordinator is not loopback and must link an account
    instead (so one hosted coordinator can't hand every visitor the same identity).
    """
    client = request.client
    if client is None:            # ASGI transports without a peer (test client)
        return True
    host = (client.host or "").strip()
    return host in _LOOPBACK_HOSTS


def _consumer_owner(cp: "ControlPlane", request: Request, body_owner: str) -> str:
    """Resolve who is *consuming* compute on this request.

    In ``auth_mode=enforced`` the owner is derived entirely from the authenticated
    Principal stored on ``request.state`` — the caller-supplied ``body_owner`` is
    accepted only as a *hint* and is silently overridden, fixing the hole where any
    caller could claim to be any owner. The ownership check is done by the authorizer
    so a principal can only act as its own subject (unless it holds cluster:admin).

    Below Tier 2 with ``auth_mode=open`` the original behaviour is preserved: the
    caller-supplied owner is used verbatim. At Tier 2 with ``auth_mode=open`` the
    existing session-based check still applies.
    """
    if cp.settings.auth_mode == "enforced":
        principal = getattr(request.state, "principal", None)
        if principal is not None and principal.kind != "anonymous":
            return principal.subject
        # anonymous in enforced mode shouldn't get here (middleware would have rejected),
        # but fall through to the legacy path for safety.

    if cp.settings.trust_tier < 2:
        return body_owner
    token = request.headers.get("X-MeshPool-Session", "")
    owner = cp.accounts.owner_for_session(token)
    if not owner:
        raise HTTPException(status_code=401, detail=(
            "authentication required: POST /api/v1/auth/device with your device "
            "signature, then send the returned token as X-MeshPool-Session"))
    return owner


def _require_credits(cp: "ControlPlane", owner: str) -> None:
    # A hard spend cap stops work before anything else is considered: the whole
    # point of the guarantee is that reaching the cap cannot be worked around by
    # having credits available. Checked here so all twelve work-creating
    # endpoints inherit it from one place.
    if cp.spend_caps.exceeded(owner):
        st = cp.spend_caps.status(owner)
        raise HTTPException(status_code=402, detail=(
            f"spend cap reached for {owner}: {st['spent']} of {st['cap']} "
            f"compute-credits used. Work is stopped, not just flagged. Raise the "
            f"cap or reset the spend window via PUT /api/v1/account/spend-cap"))
    if not cp.accounts.can_spend(owner):
        raise HTTPException(status_code=402, detail=(
            f"insufficient compute-credits for {owner}: donate spare compute to earn "
            f"credits, or link an account for a larger starter balance"))


async def _pool_bundle(cp: "ControlPlane", site: dict, owner: str, name: str) -> str:
    """Stage a large app's bundle in the pooled, content-addressed blob store and
    return its id. The bundle is chunked + replicated across the mesh, so an app far
    bigger than one control frame can be hosted; it is hydrated back on each replica
    start. Marked ``internal`` so it never shows up as one of the user's own files."""
    from ..storage.engine import StorageError
    bundle = json.dumps(site, separators=(",", ":")).encode("utf-8")
    try:
        meta = await cp.storage.put(bundle, owner=owner, name=f"bundle:{name}", internal=True)
    except StorageError as exc:
        raise HTTPException(status_code=503,
                            detail=f"could not stage the app in the pool: {exc}")
    return meta.id


# A model's uploaded checkpoint is buffered in memory to stage it in the pool, so cap
# it; overridable for genuinely large models on hosts with the RAM to marshal them.
MAX_WEIGHTS_BYTES = int(os.environ.get("MESHPOOL_WEIGHTS_MAX_BYTES", str(2_000_000_000)))  # 2 GB
# A WASM module is code, not data: real ones are KB-to-MB. A generous 64 MB ceiling
# stops an upload endpoint from becoming a way to park arbitrary bulk on the pool.
MAX_WASM_BYTES = int(os.environ.get("MESHPOOL_WASM_MAX_BYTES", str(64 * 1024 * 1024)))

# Version tag recorded in a weight manifest so a future format change is detectable.
WEIGHTS_MANIFEST_FORMAT = "meshpool/weights-manifest@1"


async def _pool_weights(cp: "ControlPlane", raw: bytes, owner: str, model: str) -> str:
    """Split a checkpoint into per-layer *group* blobs and return the id of a small JSON
    manifest mapping each group to its blob id.

    A deployment references only this manifest. Splitting up front means the control
    plane holds at most one layer-group in memory when it later places a stage — and a
    re-partition ships each survivor only the blob ids for its new layers — so a model
    whose full checkpoint dwarfs the coordinator's RAM still deploys and heals. The
    group blobs and the manifest are ``internal`` so raw weights never surface as one
    of the user's files."""
    import io
    import numpy as np
    from ..inference import model as M
    from ..storage.engine import StorageError

    cfg = M.REGISTRY[model]
    groups: dict[str, str] = {}
    meta_id: Optional[str] = None
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            for gname, keys in M.group_specs(cfg):
                sub = {k: data[k] for k in keys}     # materializes just this group
                bm = await cp.storage.put(M.dumps_npz(sub), owner=owner, name="",
                                          internal=True)
                groups[gname] = bm.id
            if "__itos__" in set(data.files):
                bm = await cp.storage.put(M.dumps_npz({}, itos=list(data["__itos__"])),
                                          owner=owner, name="", internal=True)
                meta_id = bm.id
        manifest = {"format": WEIGHTS_MANIFEST_FORMAT, "model": model,
                    "n_layer": cfg.n_layer, "groups": groups, "meta": meta_id}
        bm = await cp.storage.put(json.dumps(manifest).encode("utf-8"), owner=owner,
                                  name=f"weights:{model}", internal=True)
    except StorageError as exc:
        raise HTTPException(status_code=503,
                            detail=f"could not stage the weights in the pool: {exc}")
    return bm.id


def _validate_checkpoint(raw: bytes, cfg) -> None:
    """Reject a malformed/mismatched checkpoint at the upload boundary before staging.

    Reads only the ``.npz`` directory (not the weight arrays), so it stays cheap even
    for a multi-GB file, and confirms every parameter this architecture needs is
    present — a caller can't smuggle in arbitrary bytes or a checkpoint for a different
    model shape.

    ``allow_pickle`` is **off**: ``np.load`` with pickling enabled executes arbitrary
    code contained in the file, and this input arrives from an untrusted uploader, so
    enabling it would turn a model upload into remote code execution on the
    coordinator. Real checkpoints are plain float arrays and never need pickle.
    """
    import io
    import numpy as np
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            names = set(data.files)
    except Exception as exc:  # noqa: BLE001
        # numpy raises several unrelated types for garbage input (ValueError, OSError,
        # UnpicklingError, zipfile.BadZipFile...). Any of them means "not a valid
        # checkpoint", and letting one escape would turn a bad upload into a 500.
        raise ValueError(f"not a readable .npz checkpoint: {type(exc).__name__}: {exc}") from exc
    required = {"wte", "wpe", "lnf.g", "lnf.b"}
    for i in range(cfg.n_layer):
        for suf in ("ln1.g", "ln1.b", "attn.Wqkv", "attn.Wo", "ln2.g", "ln2.b",
                    "mlp.W1", "mlp.b1", "mlp.W2", "mlp.b2"):
            required.add(f"h{i}.{suf}")
    missing = required - names
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} parameter(s) for "
                         f"{cfg.name} (e.g. {sorted(missing)[:3]})")


def _rate_limit_jobs(cp: "ControlPlane", owner: str) -> None:
    if cp.settings.trust_tier < 2:
        return
    active = sum(1 for j in cp.batch.list()
                 if j.owner == owner and j.status in ("running", "pending"))
    if active >= MAX_ACTIVE_JOBS_PER_OWNER:
        raise HTTPException(status_code=429, detail=(
            f"too many active jobs for {owner} (limit {MAX_ACTIVE_JOBS_PER_OWNER}); "
            f"let some finish before submitting more"))


def _register_routes(app: FastAPI) -> None:
    # ---- Declare all route scopes up-front --------------------------------
    # These populate ROUTE_SCOPES (consulted by auth middleware + matrix test).
    # None = public (see PUBLIC_ALLOWLIST for the justification of each).
    # Ordering mirrors the handler definitions below so diffs stay readable.
    global _route_patterns
    _route_patterns = []   # reset so middleware rebuilds after re-registration

    # Public allowlist — every entry must have an explicit justification.
    _declare_route("GET",    "/api/v1/health",                None,
                   "health probe — must be callable without credentials by load-balancers")
    _declare_route("GET",    "/api/v1/probe/download",        None,
                   "link measurement — an agent probes bandwidth BEFORE it enrolls, so it "
                   "has no credential yet; abuse is bounded by a per-IP byte budget")
    _declare_route("HEAD",   "/api/v1/probe/download",         None,
                   "capability check for the download probe — same pre-enrollment rationale")
    _declare_route("POST",   "/api/v1/probe/upload",           None,
                   "uplink measurement sink — same pre-enrollment rationale; body size and "
                   "per-IP byte budget bound the cost")
    # FastAPI mounts these itself. They are real, reachable HTTP endpoints, so they
    # must carry an explicit decision rather than being silently exempt — an
    # undeclared route is exactly what this table exists to make impossible.
    for _docs_path in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"):
        for _docs_method in ("GET", "HEAD"):
            _declare_route(_docs_method, _docs_path, None,
                           "framework-generated API documentation — describes the public "
                           "API shape only and exposes no tenant data")
    _declare_route("GET",    "/api/v1/onboarding",            None,
                   "join-command info — must be readable before a node has any credential")
    _declare_route("POST",   "/api/v1/auth/device",           None,
                   "authentication entry point — cannot require a token to obtain a token")
    _declare_route("POST",   "/api/v1/auth/local",            None,
                   "loopback operator sign-in — gated by _is_loopback, not a pre-existing cred")
    _declare_route("POST",   "/api/v1/account/link/start",    None,
                   "account link initiation — requires no prior credential; proof presented at /complete")
    _declare_route("POST",   "/api/v1/nodes/register",        None,
                   "device enrollment — enroll_token (Tier 1+) is its own gate; no JWT needed")
    _declare_route("POST",   "/api/v1/nodes/heartbeat",       None,
                   "node keep-alive — verified by per-node session token inside the handler")
    _declare_route("POST",   "/api/v1/secrets/redeem",        None,
                   "node secret redemption — authenticated by X-MeshPool-Node-Token inside the handler")
    _declare_route("GET",    "/",                             None,
                   "dashboard root — static HTML served to anyone; no sensitive data")
    _declare_route("GET",    "/console",                      None,
                   "React console shell — static HTML/JS served to anyone; every API "
                   "call it makes is separately authorized")
    _declare_route("GET",    "/console/{path:path}",          None,
                   "React console client-side routes — same static shell")
    _declare_route("GET",    "/app/{dep_id}/{path:path}",     None,
                   "hosted-app proxy — app itself may require auth; MeshPool is the transport layer")
    _declare_route("GET",    "/app/{dep_id}",                 None,
                   "hosted-app proxy root — same rationale as above")

    # Authenticated routes — scope required.
    _declare_route("POST",   "/api/v1/account/link/complete", Scope.WALLET_READ)
    _declare_route("GET",    "/api/v1/wallet",                Scope.WALLET_READ)
    # Spend caps are the user's own safety limit on their own account, so the
    # owner may set them with wallet:write -- no admin scope. Raising a cap is
    # exactly as sensitive as spending, and it is already gated by the same
    # session-resolved owner as /api/v1/wallet.
    _declare_route("GET",    "/api/v1/account/spend-cap",       Scope.WALLET_READ)
    _declare_route("PUT",    "/api/v1/account/spend-cap",       Scope.WALLET_WRITE)
    _declare_route("POST",   "/api/v1/account/spend-cap/reset", Scope.WALLET_WRITE)
    # Managing your own capability tokens. tokens:write can mint only scopes the
    # caller already holds, so this can never be an escalation path.
    _declare_route("GET",    "/api/v1/tokens",                Scope.TOKENS_READ)
    _declare_route("POST",   "/api/v1/tokens",                Scope.TOKENS_WRITE)
    _declare_route("DELETE", "/api/v1/tokens/{token_id}",     Scope.TOKENS_WRITE)
    _declare_route("GET",    "/api/v1/ledger",                Scope.LEDGER_READ)
    _declare_route("GET",    "/api/v1/cluster",               Scope.NODES_READ)
    _declare_route("GET",    "/api/v1/nodes",                 Scope.NODES_READ)
    _declare_route("POST",   "/api/v1/nodes/{node_id}/reclaim", Scope.NODES_ADMIN)
    _declare_route("POST",   "/api/v1/nodes/{node_id}/resume",  Scope.NODES_ADMIN)
    _declare_route("POST",   "/api/v1/nodes/{node_id}/policy",  Scope.NODES_ADMIN)
    _declare_route("GET",    "/tunnel/stream",                Scope.NODES_READ)
    _declare_route("POST",   "/tunnel/reply",                 Scope.NODES_READ)
    _declare_route("POST",   "/api/v1/jobs",                  Scope.JOBS_WRITE)
    _declare_route("GET",    "/api/v1/jobs",                  Scope.JOBS_READ)
    _declare_route("GET",    "/api/v1/jobs/{job_id}",         Scope.JOBS_READ)
    _declare_route("GET",    "/api/v1/jobs/{job_id}/receipt", Scope.JOBS_READ)
    _declare_route("GET",    "/api/v1/jobs/{job_id}/timeline", Scope.JOBS_READ)
    _declare_route("GET",    "/api/v1/verification",           Scope.JOBS_READ)
    # ---- egress policy (spec §6.3) --------------------------------------- #
    _declare_route("GET",    "/api/v1/egress/policy",          Scope.JOBS_READ)
    _declare_route("POST",   "/api/v1/abuse/report",           None,
                   "public abuse reporting — anyone can report suspected abuse")
    _declare_route("POST",   "/api/v1/dags",                  Scope.JOBS_WRITE)
    _declare_route("GET",    "/api/v1/dags",                  Scope.JOBS_READ)
    _declare_route("GET",    "/api/v1/dags/{dag_id}",         Scope.JOBS_READ)
    _declare_route("POST",   "/api/v1/deployments",           Scope.DEPLOYMENTS_WRITE)
    _declare_route("PUT",    "/api/v1/deployments/{dep_id}", Scope.DEPLOYMENTS_WRITE)
    _declare_route("POST",   "/api/v1/deployments/upload",   Scope.DEPLOYMENTS_WRITE)
    _declare_route("GET",    "/api/v1/deployments",           Scope.DEPLOYMENTS_READ)
    _declare_route("GET",    "/api/v1/deployments/{dep_id}",  Scope.DEPLOYMENTS_READ)
    _declare_route("DELETE", "/api/v1/deployments/{dep_id}",  Scope.DEPLOYMENTS_WRITE)
    _declare_route("GET",    "/api/v1/deployments/{dep_id}/logs", Scope.DEPLOYMENTS_READ)
    _declare_route("POST",   "/api/v1/blobs",                 Scope.BLOBS_WRITE)
    _declare_route("GET",    "/api/v1/blobs",                 Scope.BLOBS_READ)
    _declare_route("GET",    "/api/v1/blobs/{blob_id}",       Scope.BLOBS_READ)
    _declare_route("GET",    "/api/v1/durability",            Scope.BLOBS_READ)
    _declare_route("DELETE", "/api/v1/blobs/{blob_id}",       Scope.BLOBS_WRITE)
    _declare_route("POST",   "/api/v1/buckets/{bucket}",      Scope.BLOBS_WRITE)
    _declare_route("POST",   "/api/v1/buckets/{bucket}/acl",  Scope.BLOBS_WRITE)
    _declare_route("POST",   "/api/v1/buckets/{bucket}/objects/{key:path}", Scope.BLOBS_WRITE)
    _declare_route("GET",    "/api/v1/buckets/{bucket}/objects/{key:path}", Scope.BLOBS_READ)
    _declare_route("GET",    "/api/v1/buckets/{bucket}/objects/{key:path}/versions", Scope.BLOBS_READ)
    _declare_route("POST",   "/api/v1/models",                Scope.MODELS_WRITE)
    _declare_route("POST",   "/api/v1/functions",             Scope.FUNCTIONS_WRITE)
    _declare_route("POST",   "/api/v1/models/upload",         Scope.MODELS_WRITE)
    _declare_route("POST",   "/api/v1/models/weights/group",  Scope.MODELS_WRITE)
    _declare_route("POST",   "/api/v1/models/weights/register", Scope.MODELS_WRITE)
    _declare_route("GET",    "/api/v1/models",                Scope.MODELS_READ)
    _declare_route("GET",    "/api/v1/models/available",      Scope.MODELS_READ)
    _declare_route("GET",    "/api/v1/models/{model_id}",     Scope.MODELS_READ)
    _declare_route("GET",    "/api/v1/models/{model_id}/durability", Scope.MODELS_READ)
    _declare_route("DELETE", "/api/v1/models/{model_id}",     Scope.MODELS_WRITE)
    _declare_route("POST",   "/api/v1/inference/chat",        Scope.INFERENCE_INVOKE)
    # OpenAI-compatible surface (/v1/ prefix to match the OpenAI wire protocol)
    _declare_route("GET",    "/v1/models",                    Scope.MODELS_READ)
    _declare_route("POST",   "/v1/chat/completions",          Scope.INFERENCE_INVOKE)
    _declare_route("POST",   "/v1/completions",               Scope.INFERENCE_INVOKE)
    _declare_route("GET",    "/api/v1/secrets",               Scope.SECRETS_READ)
    _declare_route("POST",   "/api/v1/secrets",               Scope.SECRETS_WRITE)
    _declare_route("DELETE", "/api/v1/secrets/{name}",        Scope.SECRETS_WRITE)
    _declare_route("GET",    "/api/v1/events",                Scope.EVENTS_SUBSCRIBE)
    # ---- Vivaldi network coordinates --------------------------------------- #
    # POST is public (like heartbeat) — nodes report RTTs before they have any
    # other credential; verified by X-MeshPool-Node-Token inside the handler.
    # GET is scoped: coordinate data is only interesting to authenticated operators
    # and the future live Mesh Map visualization.
    _declare_route("POST",   "/api/v1/mesh/rtt",              None,
                   "Vivaldi RTT observation from an agent — same pre-credential "
                   "rationale as /nodes/heartbeat; from_node verified by node token")
    _declare_route("GET",    "/api/v1/mesh/coordinates",      Scope.NODES_READ)
    # api_route with multiple methods (proxy): declare for each method
    for _m in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        _declare_route(_m, "/app/{dep_id}/{path:path}", None,
                       "hosted-app proxy — app itself may require auth")

    # ---- S3-compatible object-store protocol adapter (p5-s3) ----------- #
    # Every S3 operation maps onto the existing StorageManager: reads need
    # storage:read, writes need storage:write.  None of these are public so that
    # every PUT/POST/DELETE in the S3 surface is covered by the mutation gate.
    _declare_route("GET",    "/s3",                            Scope.STORAGE_READ)
    _declare_route("GET",    "/s3/",                           Scope.STORAGE_READ)
    _declare_route("PUT",    "/s3/{bucket}",                   Scope.STORAGE_WRITE)
    _declare_route("DELETE", "/s3/{bucket}",                   Scope.STORAGE_WRITE)
    _declare_route("GET",    "/s3/{bucket}",                   Scope.STORAGE_READ)
    _declare_route("PUT",    "/s3/{bucket}/{key:path}",         Scope.STORAGE_WRITE)
    _declare_route("GET",    "/s3/{bucket}/{key:path}",         Scope.STORAGE_READ)
    _declare_route("HEAD",   "/s3/{bucket}/{key:path}",         Scope.STORAGE_READ)
    _declare_route("DELETE", "/s3/{bucket}/{key:path}",         Scope.STORAGE_WRITE)
    _declare_route("POST",   "/s3/{bucket}/{key:path}",         Scope.STORAGE_WRITE)
    # S3 credential management (MeshPool-native; not an S3 wire operation)
    _declare_route("GET",    "/api/v1/s3/credentials",          Scope.BLOBS_READ)
    _declare_route("POST",   "/api/v1/s3/credentials",          Scope.BLOBS_WRITE)
    _declare_route("DELETE", "/api/v1/s3/credentials/{access_key}", Scope.BLOBS_WRITE)

    # ---- system ---------------------------------------------------------- #
    @app.get("/api/v1/health")
    async def health() -> dict:
        return {"ok": True, "ts": now_ms(),
                "scale": CP.settings.scale if CP else None}

    # ---- link probing (real bandwidth, not a health-check proxy) --------- #
    # Residential links are strongly asymmetric (e.g. 500 Mbps down / 20 Mbps up),
    # and a pipeline stage *sends* activations, so uplink is the binding constraint.
    # Both directions are measured; placement must never guess from a health ping.
    _PROBE_MAX_BYTES = 8 * 1024 * 1024
    # These two endpoints are necessarily unauthenticated (an agent measures its link
    # before it has any credential), which would otherwise make them a free bandwidth
    # amplifier: a stranger could pull 8 MB per request forever. A per-IP byte budget
    # keeps genuine probing (a few MB every few minutes) free while making the
    # endpoint useless as an amplifier.
    #
    # The budget is deliberately generous, because it is shared by *every* node
    # behind one public address. One full probe (ramped download + ramped upload)
    # costs ~25 MB, and the plan budgets 25-40% of the fleet behind CGNAT, where
    # thousands of subscribers share a carrier IP. A tight budget does not stop an
    # attacker — the response goes to the requesting TCP peer, so this is not a
    # reflection amplifier, only free egress — but it does silently stop honest
    # co-located nodes from ever measuring their link, which makes them
    # unplaceable. 512 MB/60 s caps sustained egress to one IP at ~68 Mbps while
    # leaving room for ~20 nodes to complete a probe each minute.
    _PROBE_BUDGET_BYTES = 512 * 1024 * 1024
    _PROBE_WINDOW_S = 60.0
    _probe_usage: dict[str, tuple[float, int]] = {}

    def _probe_charge(request: Request, nbytes: int) -> None:
        """Charge *nbytes* to the caller's IP budget; raise 429 once it is spent."""
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        start, used = _probe_usage.get(ip, (now, 0))
        if now - start >= _PROBE_WINDOW_S:
            start, used = now, 0
        if used >= _PROBE_BUDGET_BYTES:
            retry = max(1, int(_PROBE_WINDOW_S - (now - start)))
            raise HTTPException(
                status_code=429,
                detail=f"link-probe budget exhausted; retry in {retry}s",
                headers={"Retry-After": str(retry)})
        _probe_usage[ip] = (start, used + nbytes)
        if len(_probe_usage) > 4096:  # bound memory: drop entries whose window closed
            for k in [k for k, (s, _) in _probe_usage.items()
                      if now - s >= _PROBE_WINDOW_S]:
                _probe_usage.pop(k, None)

    def _probe_payload(size: int) -> bytes:
        n = max(1, min(int(size), _PROBE_MAX_BYTES))
        # Incompressible, so a proxy or Content-Encoding cannot inflate the result.
        return os.urandom(n)

    @app.head("/api/v1/probe/download")
    async def probe_download_head() -> Response:
        """Existence check — the agent HEADs this before running a bandwidth probe."""
        return Response(status_code=200)

    @app.get("/api/v1/probe/download")
    async def probe_download(request: Request, size: int = 65536) -> Response:
        body = _probe_payload(size)
        _probe_charge(request, len(body))
        return Response(content=body, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store",
                                 "X-MeshPool-Probe-Bytes": str(len(body))})

    @app.post("/api/v1/probe/upload")
    async def probe_upload(request: Request) -> dict:
        """Sink for uplink measurement. Reads the body and reports bytes received
        plus the server-side receive time, so the client can subtract its own
        serialisation cost from the round trip."""
        t0 = time.perf_counter()
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > _PROBE_MAX_BYTES:
                raise HTTPException(status_code=413, detail="probe payload too large")
        _probe_charge(request, total)
        return {"bytes": total,
                "server_recv_ms": round((time.perf_counter() - t0) * 1000.0, 3)}

    @app.get("/api/v1/onboarding")
    async def onboarding() -> dict:
        cp = _cp()
        join_url = f"{cp.settings.control_plane_url}"
        tok = cp.invite_token
        manual = (f"$env:MESHPOOL_CONTROL_PLANE_URL='{join_url}'; "
                  f"$env:MESHPOOL_ENROLL_TOKEN='{tok}'; python -m meshpool.agent.daemon")
        return {
            "control_plane_url": join_url,
            "invite_token": tok,
            # Friendliest one-command forms for each OS (run in the meshpool folder).
            "join_windows": f"powershell -ExecutionPolicy Bypass -File deploy\\join.ps1 "
                            f"-Server {join_url} -Token {tok}",
            "join_unix": f"bash deploy/join.sh --server {join_url} --token {tok}",
            "join_manual": manual,
            # Back-compat: the portable manual form used by older dashboard/CLI builds.
            "join_command": manual,
            "qr_payload": json.dumps({"url": join_url, "token": tok}),
        }

    @app.get("/api/v1/cluster")
    async def cluster() -> dict:
        cp = _cp()
        nodes = cp.registry.list()
        total_cores = sum(n.capability.cpu_cores for n in nodes if n.status == NodeStatus.ONLINE)
        total_ram = sum(n.capability.ram_mb for n in nodes if n.status == NodeStatus.ONLINE)
        return {
            "nodes": [_node_view(cp, n) for n in nodes],
            "summary": {
                "total_nodes": len(nodes),
                "online": len(cp.registry.online()),
                "total_cores": total_cores,
                "total_ram_mb": total_ram,
                "weight_group_pulls": getattr(cp.inference, "weight_group_pulls", 0),
                "weight_group_pushes": getattr(cp.inference, "weight_group_pushes", 0),
                "weight_group_cached": getattr(cp.inference, "weight_group_cached", 0),
                # Fraud detection, made observable: how many results were re-executed
                # against a trusted oracle and how many were caught wrong.
                "spot_checks": getattr(cp.batch, "spot_checks", 0),
                "spot_check_rejections": getattr(cp.batch, "spot_check_rejections", 0),
                # User-code distribution: proves a module is shipped once per node
                # (like a weight group) rather than once per work unit.
                "wasm_module_pushes": getattr(cp.batch, "wasm_module_pushes", 0),
                "speculative_attempts": getattr(cp.batch, "speculative_attempts", 0),
                "speculative_wins": getattr(cp.batch, "speculative_wins", 0),
                "speculative_discards": getattr(cp.batch, "speculative_discards", 0),
                "speculative_pairs": getattr(cp.batch, "speculative_pairs", []),
                "requeued_units": getattr(cp.batch, "requeued_units", 0),
            },
        }

    # ---- accounts / wallet (Tier 2 global mesh; harmless at lower tiers) -- #
    @app.post("/api/v1/auth/device")
    async def auth_device(body: DeviceAuth) -> dict:
        """Consumer login by device signature. Returns a session token used as
        ``X-MeshPool-Session`` on submit calls, plus the current balance. Grants the
        one-time starter balance on first login of a new owner."""
        cp = _cp()
        try:
            owner, token = cp.accounts.authenticate_device(
                body.device_id, body.public_key, body.signature, body.ts_ms)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"identity rejected: {exc}")
        return {"owner": owner, "session_token": token,
                "balance": round(cp.accounts.balance(owner), 2),
                "gating": cp.accounts.gating_enabled()}

    @app.post("/api/v1/auth/local")
    async def auth_local(request: Request) -> dict:
        """Zero-signup sign-in for the operator of the machine running MeshPool.

        The dashboard is served same-origin by this process, which holds the
        machine's device key, so we can issue a consumer session without the browser
        ever touching a private key. Loopback-gated at Tier 2 (public coordinator):
        a remote visitor must link an account instead. Harmless below Tier 2, where
        it simply represents the local ``default`` owner and unlocks the same UI.
        """
        cp = _cp()
        if cp.settings.trust_tier >= 2 and not _is_loopback(request):
            raise HTTPException(status_code=403, detail=(
                "local sign-in is only available on the machine running MeshPool; "
                "link an account to donate/consume on a remote mesh"))
        owner, token = cp.accounts.local_session(cp.identity)
        return {"owner": owner, "session_token": token,
                "device_id": cp.identity.device_id,
                "balance": round(cp.accounts.balance(owner), 2),
                "gating": cp.accounts.gating_enabled(),
                "tier": cp.settings.trust_tier}

    @app.post("/api/v1/account/link/start")
    async def account_link_start(body: LinkStart) -> dict:
        """Begin linking this device to an account. Returns provider instructions
        (local provider: a link code to echo back with a chosen handle)."""
        return _cp().accounts.begin_link()

    @app.post("/api/v1/account/link/complete")
    async def account_link_complete(body: LinkComplete, request: Request) -> dict:
        cp = _cp()
        token = request.headers.get("X-MeshPool-Session", "")
        info = cp.accounts.session_info(token)
        if not info:
            raise HTTPException(status_code=401, detail="authenticate the device first")
        try:
            return cp.accounts.complete_link(info["device_id"], body.pending_id, body.proof)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/v1/wallet")
    async def wallet(request: Request, owner: Optional[str] = None) -> dict:
        """Balance + donated/consumed + reputation for the caller. At Tier 2 the
        owner comes from the session; at lower tiers an ``owner`` query param (or
        'default') is used."""
        cp = _cp()
        if cp.settings.trust_tier >= 2:
            token = request.headers.get("X-MeshPool-Session", "")
            owner = cp.accounts.owner_for_session(token)
            if not owner:
                raise HTTPException(status_code=401, detail="authenticate the device first")
        else:
            owner = owner or "default"
        by_owner = cp.store.credits_by_owner()
        totals = cp.store.ledger_totals_by_owner().get(
            owner, {"donated": 0.0, "consumed": 0.0, "grant": 0.0})
        return {
            "owner": owner,
            "balance": round(by_owner.get(owner, 0.0), 2),
            "donated": round(totals["donated"], 2),
            "consumed": round(totals["consumed"], 2),
            "granted": round(totals["grant"], 2),
            "reputation": round(_owner_reputation(cp, owner), 3),
            "gating": cp.accounts.gating_enabled(),
            "nodes": [n.id for n in cp.registry.list() if n.owner == owner],
        }

    # ---- hard spend caps -------------------------------------------------- #
    def _cap_owner(cp: "ControlPlane", request: Request,
                   owner: Optional[str]) -> str:
        """Resolve the owner whose cap is being read/written (same rule as /wallet)."""
        if cp.settings.trust_tier >= 2:
            token = request.headers.get("X-MeshPool-Session", "")
            resolved = cp.accounts.owner_for_session(token)
            if not resolved:
                raise HTTPException(status_code=401,
                                    detail="authenticate the device first")
            return resolved
        return owner or "default"

    @app.get("/api/v1/account/spend-cap")
    async def get_spend_cap(request: Request, owner: Optional[str] = None) -> dict:
        """Current hard spend cap and spend inside the active window.

        ``enforced`` is false when no cap exists — the UI must show that rather
        than displaying a number nothing enforces.
        """
        cp = _cp()
        return cp.spend_caps.status(_cap_owner(cp, request, owner))

    @app.put("/api/v1/account/spend-cap")
    async def put_spend_cap(body: SpendCapBody, request: Request,
                            owner: Optional[str] = None) -> dict:
        """Set (``cap``) or clear (``cap: null``) the caller's hard spend cap.

        Changing the cap deliberately does **not** reset accrued spend; otherwise
        the cap could be bypassed indefinitely by re-setting it.
        """
        cp = _cp()
        try:
            return cp.spend_caps.set_cap(_cap_owner(cp, request, owner), body.cap)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/account/spend-cap/reset")
    async def reset_spend_cap(request: Request, owner: Optional[str] = None) -> dict:
        """Begin a new spend window. Ledger history is preserved for billing."""
        cp = _cp()
        return cp.spend_caps.reset_spend(_cap_owner(cp, request, owner))

    # ---- user-minted capability tokens ------------------------------------ #
    def _token_owner(cp: "ControlPlane", request: Request) -> str:
        """Owner whose token collection is being managed.

        Always derived from the authenticated principal: a capability token may
        manage only its own account's tokens, never another owner's.
        """
        principal = getattr(request.state, "principal", None)
        if principal is not None and principal.kind != "anonymous":
            return principal.subject
        session = request.headers.get("X-MeshPool-Session", "")
        resolved = cp.accounts.owner_for_session(session)
        if not resolved:
            raise HTTPException(status_code=401,
                                detail="authenticate the device first")
        return resolved

    def _public_token(row: dict) -> dict:
        """Listing shape. The bearer secret is *never* included — only its hash is
        stored, so disclosure is impossible rather than merely omitted."""
        return {
            "id": row["id"],
            "label": row["label"],
            "scopes": row["scopes"],
            "expires_at_ms": row["expires_at_ms"],
            "created_ms": row["created_ms"],
            "last_used_ms": row["last_used_ms"],
            "spend_cap": row["spend_cap"],
            "spent": row["spent"],
        }

    @app.post("/api/v1/tokens")
    async def mint_capability_token(body: CapabilityTokenCreate,
                                    request: Request) -> dict:
        """Mint a scoped, expiring capability token for the caller's own account.

        The secret is returned exactly once, here. Only its SHA-256 hash is
        persisted, so a later list can never leak it.

        Privilege escalation is impossible by construction: the requested scopes
        must be a subset of the scopes the *calling principal* already holds. A
        token minted by a token can therefore only ever narrow, never widen.
        """
        cp = _cp()
        owner = _token_owner(cp, request)
        principal = getattr(request.state, "principal", None)
        holder = principal.scopes if principal is not None else frozenset()

        unknown = [s for s in body.scopes if s not in Scope._value2member_map_]
        if unknown:
            raise HTTPException(status_code=400,
                                detail=f"unknown scopes: {sorted(unknown)}")
        requested = frozenset(Scope(s) for s in body.scopes)
        if not requested:
            raise HTTPException(status_code=400,
                                detail="a token must carry at least one scope")
        if principal is not None and not principal.has_scope(Scope.CLUSTER_ADMIN):
            excess = sorted(s.value for s in requested if s not in holder)
            if excess:
                raise HTTPException(status_code=403, detail=(
                    f"cannot grant scopes you do not hold: {excess}"))

        now = now_ms()
        if body.expires_at_ms is not None:
            expires_at = int(body.expires_at_ms)
        elif body.expires_in_seconds is not None:
            expires_at = now + int(float(body.expires_in_seconds) * 1000)
        else:
            expires_at = now + 30 * 24 * 3600 * 1000
        if expires_at <= now:
            raise HTTPException(status_code=400,
                                detail="expiry must be in the future")
        if body.spend_cap is not None and float(body.spend_cap) < 0.0:
            raise HTTPException(status_code=400,
                                detail="spend_cap must not be negative")

        secret = f"mpt_{secrets.token_urlsafe(32)}"
        row = {
            "id": new_id("tok"),
            "owner": owner,
            "token_hash": cp.store.capability_token_hash(secret),
            "label": body.label,
            "scopes": sorted(s.value for s in requested),
            "expires_at_ms": expires_at,
            "created_ms": now,
            "last_used_ms": None,
            "revoked_ms": None,
            "spend_cap": body.spend_cap,
            "spent": 0.0,
        }
        cp.store.put_capability_token(row)
        return {**_public_token(row), "owner": owner, "token": secret}

    @app.get("/api/v1/tokens")
    async def list_capability_tokens(request: Request) -> dict:
        """List the caller's live tokens. Never returns bearer secrets."""
        cp = _cp()
        owner = _token_owner(cp, request)
        return {"tokens": [_public_token(r)
                           for r in cp.store.list_capability_tokens(owner)]}

    @app.delete("/api/v1/tokens/{token_id}")
    async def revoke_capability_token(token_id: str, request: Request) -> dict:
        """Revoke a token. Revocation is a database write consulted on *every*
        request, so it takes effect on the immediately following call — there is
        no cached-validity window to wait out."""
        cp = _cp()
        owner = _token_owner(cp, request)
        if not cp.store.revoke_capability_token(token_id, owner, now_ms()):
            raise HTTPException(status_code=404, detail="unknown token")
        return {"revoked": True, "id": token_id}

    # ---- nodes ----------------------------------------------------------- #
    @app.get("/api/v1/ledger")
    async def ledger() -> dict:
        """Read-only credit ledger overview.

        Returns both shapes the product needs from one endpoint:

        * ``owners`` + ``summary`` — per-owner donated/consumed/granted + balance and
          a global roll-up (powers Wallet/Explore and lets acceptance tests assert the
          conservation invariant donated == consumed);
        * ``by_node`` + ``credits_by_owner`` + ``fair_share`` + ``recent`` — the
          per-node/legacy view consumed by the CLI ``ledger`` command and older UIs.

        (These were previously two routes registered on the same path, so the second
        was silently shadowed and its consumers broke — now unified.)
        """
        cp = _cp()
        totals = cp.store.ledger_totals_by_owner()
        bal = cp.store.credits_by_owner()
        owners = []
        tot_d = tot_c = tot_g = 0.0
        for owner, t in sorted(totals.items()):
            tot_d += t["donated"]; tot_c += t["consumed"]; tot_g += t["grant"]
            owners.append({
                "owner": owner,
                "donated": round(t["donated"], 3),
                "consumed": round(t["consumed"], 3),
                "granted": round(t["grant"], 3),
                "balance": round(bal.get(owner, 0.0), 3),
            })
        by_node = cp.store.ledger_by_node()
        node_views = []
        for n in cp.registry.list():
            stats = by_node.get(n.id, {"donated": 0.0, "consumed": 0.0})
            node_views.append({"node_id": n.id, "name": n.name, "owner": n.owner,
                               "donated": round(stats.get("donated", 0.0), 2),
                               "consumed": round(stats.get("consumed", 0.0), 2),
                               "trust": round(cp.trust.score(n.id), 3)})
        credits = {o: round(c, 2) for o, c in bal.items()}
        return {
            "owners": owners,
            "summary": {
                "owners": len(owners),
                "donated": round(tot_d, 3),
                "consumed": round(tot_c, 3),
                "granted": round(tot_g, 3),
            },
            "by_node": node_views,
            "credits_by_owner": credits,
            "fair_share": {o: round(fair_share_priority(c, cp.settings.trust_tier), 3)
                           for o, c in bal.items()},
            "recent": [e.model_dump(mode="json") for e in cp.store.recent_ledger(20)],
        }

    @app.post("/api/v1/nodes/register")
    async def register(req: RegisterRequest) -> dict:
        cp = _cp()
        if cp.settings.require_enroll_token:  # Tier 1+ admin-gated enrollment
            if req.enroll_token not in (cp.invite_token, cp.settings.enroll_token):
                raise HTTPException(status_code=403, detail="invalid or missing enroll token")
        try:
            resp = cp.registry.register(req, cp.settings.heartbeat_interval_s)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"identity rejected: {exc}")
        # A newly-seen contributor gets its one-time starter balance so it can also
        # consume from day one (idempotent per owner). Only meaningful at Tier 2.
        if cp.accounts.gating_enabled():
            node = cp.registry.get(resp.node_id)
            if node:
                cp.accounts.ensure_starter_grant(node.owner)
        out = resp.model_dump(mode="json")
        if cp.quic_addr:
            out["quic_addr"] = cp.quic_addr
        return out

    @app.post("/api/v1/nodes/heartbeat")
    async def heartbeat(hb: Heartbeat) -> dict:
        cp = _cp()
        ok = cp.registry.heartbeat(hb)
        if not ok:
            raise HTTPException(status_code=404, detail="unknown node; re-register")
        return {"ok": True}

    @app.get("/api/v1/nodes")
    async def list_nodes() -> dict:
        cp = _cp()
        return {"nodes": [_node_view(cp, n) for n in cp.registry.list()]}

    @app.post("/api/v1/nodes/{node_id}/reclaim")
    async def reclaim_node(node_id: str) -> dict:
        cp = _cp()
        node = cp.registry.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="unknown node")
        try:
            await cp.transport.call(node_id, "reclaim", timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"agent unreachable: {exc}")
        cp.registry.set_status(node_id, NodeStatus.DRAINING)
        if os.environ.get("MESHPOOL_DISABLE_RECLAIM_REQUEUE", "0") != "1":
            cp.batch.requeue_node_units(node_id)
        return {"status": "draining"}

    @app.post("/api/v1/nodes/{node_id}/resume")
    async def resume_node(node_id: str) -> dict:
        cp = _cp()
        node = cp.registry.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="unknown node")
        try:
            await cp.transport.call(node_id, "resume", timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"agent unreachable: {exc}")
        cp.registry.set_status(node_id, NodeStatus.ONLINE)
        return {"status": "online"}

    @app.post("/api/v1/nodes/{node_id}/policy")
    async def set_node_policy(node_id: str, body: dict) -> dict:
        cp = _cp()
        node = cp.registry.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="unknown node")
        try:
            data = await cp.transport.call(node_id, "policy", body, timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"agent unreachable: {exc}")
        # reflect the agent-confirmed policy in the registry now (no heartbeat lag)
        try:
            cp.registry.set_policy(node_id, DonationPolicy(**data["policy"]))
        except Exception:  # noqa: BLE001
            pass
        return data

    # ---- reverse tunnel (global agents dial out; CP pushes work down) ----- #
    @app.get("/tunnel/stream")
    async def tunnel_stream(request: Request):
        """Long-lived NDJSON stream an agent opens outbound; the CP writes request
        frames onto it. Authenticated by the per-node session token so only the
        owning agent can receive its node's traffic."""
        cp = _cp()
        token = request.headers.get(HDR_NODE_TOKEN, "")
        node_id = cp.registry.node_id_for_token(token)
        if not node_id:
            raise HTTPException(status_code=401, detail="invalid or missing node token")
        conn = cp.hub.attach(node_id)

        async def gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        frame = await asyncio.wait_for(conn.out.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield json.dumps({"op": "ping"}) + "\n"   # keepalive
                        continue
                    yield json.dumps(frame) + "\n"
            finally:
                cp.hub.detach(node_id, conn)

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/tunnel/reply")
    async def tunnel_reply(reply: TunnelReply, request: Request):
        """An agent posts the result of a tunnelled request back here; we hand it to
        the awaiting control-plane coroutine via the hub."""
        cp = _cp()
        token = request.headers.get(HDR_NODE_TOKEN, "")
        node_id = cp.registry.node_id_for_token(token)
        if not node_id:
            raise HTTPException(status_code=401, detail="invalid or missing node token")
        cp.hub.resolve(node_id, reply.id, reply.model_dump())
        return {"ok": True}

    # ---- batch ----------------------------------------------------------- #
    @app.post("/api/v1/jobs")
    async def submit_job(req: JobSubmit, request: Request) -> dict:
        cp = _cp()
        owner = _consumer_owner(cp, request, req.owner)
        _require_credits(cp, owner)
        _rate_limit_jobs(cp, owner)
        incoming_trace = TraceContext.from_headers(request.headers) or TraceContext.new()
        with start_span("controlplane.submit_job", context=incoming_trace,
                        attributes={"job_name": req.name, "func": req.func}):
            job = cp.batch.submit(req.name, req.func, req.payloads, req.cpu_pct,
                                  req.ram_mb, owner,
                                  capability_token_id=_request_token_id(request),
                                  secret_names=req.secret_names)
        trace_id = cp.timeline.trace_id_for_job(job.id)
        return {"job_id": job.id, "units": len(job.units), "owner": owner,
                "trace_id": trace_id}

    @app.get("/api/v1/jobs")
    async def list_jobs() -> dict:
        cp = _cp()
        def summary(j) -> dict:
            # `units` is deliberately not sent: it is unbounded and the list view
            # never needs the payloads. `active_nodes` is the one thing consumers
            # actually read off it (which machines are working right now), so it
            # is projected here instead of being silently absent -- the console
            # used to fall back to `job.units ?? []`, which made the mesh map's
            # live-work highlight and the jobs task grid permanently empty.
            active = sorted({u.assigned_node for u in j.units
                             if u.assigned_node
                             and str(getattr(u.status, "value", u.status))
                             in ("assigned", "running")})
            return {"id": j.id, "name": j.name, "func": j.func,
                    "status": j.status, "progress": j.progress(),
                    "owner": j.owner, "created_ms": j.created_ms,
                    "active_nodes": active, "units_total": len(j.units)}

        return {"jobs": [summary(j) for j in cp.batch.list()]}

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        cp = _cp()
        job = cp.batch.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="unknown job")
        progress = job.progress()
        receipt = cp.billing.receipt("batch", job_id)
        running = progress.get("running", 0) + progress.get("assigned", 0)
        return {
            **job.model_dump(mode="json"),
            "progress": progress,
            "receipt": receipt,
            "cost_meter": cp.billing.live_meter("batch", job_id, job.owner,
                                                running_units=running),
        }

    @app.get("/api/v1/jobs/{job_id}/receipt")
    async def get_job_receipt(job_id: str) -> dict:
        cp = _cp()
        receipt = cp.billing.receipt("batch", job_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail="receipt not found")
        return receipt

    @app.get("/api/v1/jobs/{job_id}/timeline")
    async def get_job_timeline(job_id: str) -> dict:
        cp = _cp()
        job = cp.batch.get(job_id) or cp.store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="unknown job")
        nodes = {n.id: n for n in cp.registry.list()}
        return cp.timeline.build(job, nodes, cp.events.recent_for_job(job_id))

    @app.get("/api/v1/verification")
    async def get_verification_stats() -> dict:
        """Workload-matched result verification statistics (spec \xa7P6).

        Returns policy config, observable counters (units verified, disagreements,
        nodes penalised), and compute escrow balances.  At Tier 0/1 the verifier
        is not wired in and an empty snapshot is returned so the endpoint is
        always callable without errors.
        """
        cp = _cp()
        verifier = getattr(cp.batch, "verifier", None)
        if verifier is not None:
            return verifier.as_dict()
        # Verifier not active (Tier 0/1 or build_app wiring not yet complete).
        from .verification import (
            DIGEST_VERIFIABLE_FUNCS, ORACLE_VERIFIABLE_FUNCS, NOT_VERIFIABLE_FUNCS,
        )
        return {
            "policy": {"redundant_rate": 0.0, "redundant_n": 2, "active": False},
            "stats": {
                "units_verified": 0,
                "redundant_verifications": 0,
                "disagreements": 0,
                "nodes_penalized": 0,
                "degraded": {"no_secondary": 0, "secondary_error": 0,
                             "no_tiebreak": 0},
                "stake_slashed": 0.0,
                "heldback_recorded": 0.0,
                "heldback_released": 0.0,
            },
            "escrow": {"total_held": 0.0, "total_released": 0.0,
                       "total_forfeited": 0.0},
            "staking": {"active": False, "destination_owner": "meshpool:slashing",
                        "total_slashed": 0.0},
            "heldback": {"active": False, "escrow_owner": "meshpool:heldback-escrow",
                         "total_held": 0.0, "total_recorded": 0.0,
                         "total_released": 0.0},
            "classification": {
                "digest_verifiable": sorted(DIGEST_VERIFIABLE_FUNCS),
                "oracle_verifiable": sorted(ORACLE_VERIFIABLE_FUNCS),
                "not_verifiable": sorted(NOT_VERIFIABLE_FUNCS),
            },
        }

    @app.get("/api/v1/egress/policy")
    async def get_egress_policy() -> dict:
        """Report the active egress policy for this cluster."""
        from ..common.egress import HARD_BLOCKED_PORTS, _BLOCKED_NETWORKS
        return {
            "hard_blocked_ports": sorted(HARD_BLOCKED_PORTS),
            "blocked_network_count": len(_BLOCKED_NETWORKS),
            "blocked_networks": [str(n) for n in _BLOCKED_NETWORKS],
            "default": "deny",
        }

    @app.post("/api/v1/abuse/report")
    async def submit_abuse_report(request: Request) -> dict:
        """Accept an abuse report from any caller (public endpoint).

        Rate-limited per IP: at most 10 reports per minute.
        Reports are queued in memory and logged for operator review.
        """
        cp = _cp()
        ip = request.client.host if request.client else "unknown"
        if not cp._abuse_rate_limiter.is_allowed(ip):
            raise HTTPException(status_code=429, detail="too many abuse reports from this IP")
        body = await request.json()
        from ..common.egress import AbuseReport
        report = AbuseReport(
            reporter_ip=ip,
            node_id=str(body.get("node_id", "")),
            description=str(body.get("description", ""))[:1000],
            evidence=str(body.get("evidence", ""))[:2000],
            ts=time.time(),
        )
        cp._abuse_reports.append(report)
        log.warning("abuse report received", reporter_ip=ip, node_id=report.node_id,
                    description=report.description[:100])
        return {"accepted": True, "queue_depth": cp._abuse_reports.count()}

    # ---- DAG workflows ---------------------------------------------------- #
    @app.post("/api/v1/dags")
    async def submit_dag(req: DagSubmit, request: Request) -> dict:
        cp = _cp()
        owner = _consumer_owner(cp, request, req.owner)
        _require_credits(cp, owner)
        try:
            dag = cp.dags.submit(req.name, req.stages, owner)
        except DagError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"dag_id": dag.id, "stages": len(dag.stages), "owner": owner}

    @app.get("/api/v1/dags")
    async def list_dags() -> dict:
        cp = _cp()
        return {"dags": [{"id": d.id, "name": d.name, "status": d.status,
                          "progress": d.progress()}
                         for d in cp.dags.list()]}

    @app.get("/api/v1/dags/{dag_id}")
    async def get_dag(dag_id: str) -> dict:
        cp = _cp()
        dag = cp.dags.get(dag_id)
        if not dag:
            raise HTTPException(status_code=404, detail="unknown dag")
        return {**dag.model_dump(mode="json"), "progress": dag.progress()}

    # ---- hosting --------------------------------------------------------- #
    @app.post("/api/v1/deployments")
    async def create_deployment(req: DeploySubmit, request: Request) -> dict:
        cp = _cp()
        owner = _consumer_owner(cp, request, req.owner)
        _require_credits(cp, owner)
        site = None
        site_blob_id = None
        if req.site:
            from ..hosting.site import MAX_BUNDLE_BYTES, SiteTooLarge, normalize_site
            try:
                site = normalize_site(req.site, require_index=req.start_cmd is None)
            except SiteTooLarge:
                # A large app is not rejected: validate it against the higher pooled
                # ceiling, then stage it in the mesh blob store (chunked + replicated).
                try:
                    site = normalize_site(req.site, require_index=req.start_cmd is None,
                                          max_bytes=MAX_BUNDLE_BYTES)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"invalid site: {exc}")
                site_blob_id = await _pool_bundle(cp, site, owner, req.name)
                site = None
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid site: {exc}")
        from ..hosting.site import clean_env
        try:
            env = clean_env(req.env)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid env: {exc}")
        dep = await cp.hosting.deploy(req.name, req.app, req.replicas, owner,
                                     site=site, site_blob_id=site_blob_id,
                                     start_cmd=req.start_cmd, env=env,
                                     autoscale_min=req.autoscale_min,
                                     autoscale_max=req.autoscale_max)
        return {"id": dep.id, "status": dep.status, "replicas": len(dep.replicas)}

    @app.put("/api/v1/deployments/{dep_id}")
    async def rollout_deployment(dep_id: str, req: DeploySubmit, request: Request) -> dict:
        cp = _cp()
        owner = _consumer_owner(cp, request, req.owner)
        _require_credits(cp, owner)
        site = None
        site_blob_id = None
        if req.site:
            from ..hosting.site import MAX_BUNDLE_BYTES, SiteTooLarge, normalize_site
            try:
                site = normalize_site(req.site, require_index=req.start_cmd is None)
            except SiteTooLarge:
                try:
                    site = normalize_site(req.site, require_index=req.start_cmd is None,
                                          max_bytes=MAX_BUNDLE_BYTES)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"invalid site: {exc}")
                site_blob_id = await _pool_bundle(cp, site, owner, req.name)
                site = None
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid site: {exc}")
        from ..hosting.site import clean_env
        try:
            env = clean_env(req.env)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid env: {exc}")
        dep = await cp.hosting.rollout(dep_id, req.name, req.app, req.replicas, owner,
                                      site=site, site_blob_id=site_blob_id,
                                      start_cmd=req.start_cmd, env=env,
                                      autoscale_min=req.autoscale_min,
                                      autoscale_max=req.autoscale_max)
        if dep is None:
            raise HTTPException(status_code=404, detail="unknown deployment")
        return {"id": dep.id, "status": dep.status, "replicas": len(dep.replicas)}

    @app.post("/api/v1/deployments/upload")
    async def upload_deployment(request: Request, name: str = "my-site",
                                replicas: int = 2, owner: str = "default",
                                env: Annotated[list[str], Query()] = []) -> dict:
        """Deploy a real multi-page app straight from the browser: the raw bytes of a
        ``.zip`` are POSTed as the request body (no multipart dependency) and unpacked
        server-side into a validated site map. This is the dashboard's drag-a-zip path,
        the browser equivalent of ``meshpool deploy ./mysite.zip``."""
        cp = _cp()
        owner = _consumer_owner(cp, request, owner)
        _require_credits(cp, owner)
        from ..hosting.site import (MAX_BUNDLE_BYTES, SiteTooLarge, clean_env,
                                    detect_start_cmd_from_bundle, pack_zip)
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload — attach a .zip of your site")
        if len(data) > MAX_BUNDLE_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"upload too large (over {MAX_BUNDLE_BYTES} bytes)")
        env_map: dict[str, str] = {}
        for item in env:
            key, sep, val = item.partition("=")
            if not sep or not key.strip():
                raise HTTPException(status_code=400,
                                    detail=f"bad env {item!r}; expected KEY=VALUE")
            env_map[key.strip()] = val
        try:
            env_map = clean_env(env_map)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid env: {exc}")
        large = False
        try:
            site = pack_zip(data, require_index=False)
        except SiteTooLarge:
            # A large app is not rejected: unpack against the higher pooled ceiling
            # (still zip-bomb-capped) and stage it in the mesh blob store.
            try:
                site = pack_zip(data, require_index=False, max_bytes=MAX_BUNDLE_BYTES)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid site zip: {exc}")
            large = True
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid site zip: {exc}")
        # A bundle carrying a Procfile/meshpool.json is a real server app; run it as a
        # process. Executing the deployer's command is still only ever done on a trusted
        # (non-sandbox) node -- a public donor refuses it -- exactly like the CLI path.
        start_cmd = detect_start_cmd_from_bundle(site)
        if not start_cmd and "index.html" not in site:
            raise HTTPException(
                status_code=400,
                detail=("a static site needs an 'index.html' at its root, or include a "
                        "Procfile ('web: ...') to host a server app"))
        app_kind = "proc" if start_cmd else "site"
        site_blob_id = None
        if large:
            site_blob_id = await _pool_bundle(cp, site, owner, name)
            site = None
        dep = await cp.hosting.deploy(name, app_kind, max(1, replicas), owner,
                                      site=site, site_blob_id=site_blob_id,
                                      start_cmd=start_cmd or None,
                                      env=env_map or None)
        return {"id": dep.id, "status": dep.status, "replicas": len(dep.replicas),
                "kind": app_kind}

    @app.get("/api/v1/deployments")
    async def list_deployments() -> dict:
        cp = _cp()
        return {"deployments": [d.model_dump(mode="json") for d in cp.hosting.list()]}

    @app.get("/api/v1/deployments/{dep_id}")
    async def get_deployment(dep_id: str) -> dict:
        cp = _cp()
        dep = cp.hosting.get(dep_id)
        if not dep:
            raise HTTPException(status_code=404, detail="unknown deployment")
        return dep.model_dump(mode="json")

    @app.delete("/api/v1/deployments/{dep_id}")
    async def delete_deployment(dep_id: str) -> dict:
        cp = _cp()
        ok = await cp.hosting.remove(dep_id)
        if not ok:
            raise HTTPException(status_code=404, detail="unknown deployment")
        return {"removed": dep_id}

    @app.get("/api/v1/deployments/{dep_id}/logs")
    async def deployment_logs(dep_id: str, tail: int = 200) -> dict:
        cp = _cp()
        reps = await cp.hosting.logs(dep_id, tail=max(1, min(tail, 1000)))
        if reps is None:
            raise HTTPException(status_code=404, detail="unknown deployment")
        return {"deployment": dep_id, "replicas": reps}

    # ---- pooled object store (content-addressed, replicated across nodes) ---- #
    @app.post("/api/v1/blobs")
    async def store_blob(request: Request, name: str = "", owner: str = "default",
                         replication: int = 2) -> dict:
        """Store an object in the pool: the raw bytes are POSTed as the body and
        replicated across several nodes' donated disk, keyed by their SHA-256 digest.
        The browser equivalent of dropping a file onto shared, fault-tolerant storage."""
        cp = _cp()
        owner = _consumer_owner(cp, request, owner)
        _require_credits(cp, owner)
        from ..storage.blobstore import MAX_OBJECT_BYTES
        from ..storage.engine import StorageError
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400,
                                detail="empty upload — attach the file bytes as the body")
        if len(data) > MAX_OBJECT_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"object too large (over {MAX_OBJECT_BYTES} bytes)")
        try:
            meta = await cp.storage.put(data, owner=owner, name=name,
                                        replication=max(1, min(replication, 5)))
        except StorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return meta.model_dump(mode="json")

    @app.get("/api/v1/blobs")
    async def list_blobs() -> dict:
        cp = _cp()
        return {"blobs": [b.model_dump(mode="json") for b in cp.storage.list()]}

    @app.get("/api/v1/blobs/{blob_id}")
    async def fetch_blob(blob_id: str) -> Response:
        cp = _cp()
        meta = cp.storage.get_meta(blob_id)
        data = await cp.storage.get(blob_id)
        if data is None:
            raise HTTPException(status_code=404, detail="unknown or unavailable object")
        raw_name = (meta.name if meta and meta.name else blob_id)
        # sanitise for a Content-Disposition header (strip anything that could inject
        # header syntax or a path); fall back to the always-safe digest.
        safe = "".join(c for c in raw_name if c.isalnum() or c in "._- ")[:100] or blob_id
        return Response(content=data, media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{safe}"'})

    @app.delete("/api/v1/blobs/{blob_id}")
    async def remove_blob(blob_id: str) -> dict:
        cp = _cp()
        ok = await cp.storage.delete(blob_id)
        if not ok:
            raise HTTPException(status_code=404, detail="unknown object")
        return {"removed": blob_id}

    @app.get("/api/v1/durability")
    async def get_durability(blob_id: Optional[str] = None) -> dict:
        """Report per-blob *effective* durability accounting for correlated placement.

        The classic failure: RS(6,3) promises 3-failure tolerance, but if all 9 shards
        live on machines in one subnet the real tolerance is 0 (one power cut loses
        everything). This endpoint reports what was actually achieved, not what RS
        mathematics would give under an independence assumption that does not hold.

        Without ``blob_id`` returns a cluster summary.
        With ``blob_id`` returns that object's full durability breakdown including a
        warning whenever correlated placement degrades the effective durability.
        """
        cp = _cp()
        from ..controlplane.topology import effective_durability as _eff_dur
        from ..storage.engine import _unpack_ec_params

        def _blob_durability(meta) -> dict:
            if meta is None:
                return {}
            # EC-coded object: compute domain-aware effective durability
            is_ec = (meta.id in (cp.storage._ec_params if cp.storage else {})
                     or (meta.chunks and meta.replication_desired >= 256))
            if not is_ec or not meta.chunks:
                return {
                    "blob_id": meta.id[:16], "ec": False,
                    "replicas": len(meta.replicas),
                    "desired": meta.replication_desired,
                    "status": meta.status,
                }
            stored_k_m = (cp.storage._ec_params.get(meta.id)
                          if cp.storage else None)
            k_val, m_val = stored_k_m if stored_k_m else _unpack_ec_params(
                meta.replication_desired)
            # Collect the failure domain of each shard holder
            domains: list[str] = []
            for sid in meta.chunks:
                sm = cp.storage.get_meta(sid) if cp.storage else None
                if sm and sm.replicas:
                    with cp.registry._lock:
                        node = cp.registry._nodes.get(sm.replicas[0])
                    if node:
                        domains.append(node.failure_domain or node.id)
            eff = _eff_dur(domains, k_val, m_val)
            eff["blob_id"] = meta.id[:16]
            eff["ec_k"] = k_val
            eff["ec_m"] = m_val
            eff["status"] = meta.status
            eff["ec"] = True
            return eff

        if blob_id is not None:
            if cp.storage is None:
                raise HTTPException(status_code=503, detail="storage not ready")
            meta = cp.storage.get_meta(blob_id)
            if meta is None or meta.internal:
                raise HTTPException(status_code=404, detail="unknown blob")
            return _blob_durability(meta)

        # Cluster summary
        if cp.storage is None:
            return {"blobs": [], "summary": {
                "total_blobs": 0, "ec_blobs": 0,
                "correlated_blobs": 0, "perfectly_diverse": 0}}
        results = [_blob_durability(m) for m in cp.storage.list() if m is not None]
        ec_results = [r for r in results if r.get("ec")]
        return {
            "summary": {
                "total_blobs": len(results),
                "ec_blobs": len(ec_results),
                "correlated_blobs": sum(1 for r in ec_results if r.get("correlated")),
                "perfectly_diverse": sum(
                    1 for r in ec_results if not r.get("correlated")),
                "any_warning": any(r.get("warning") for r in ec_results),
            },
            "blobs": results,
        }

    @app.post("/api/v1/buckets/{bucket}")
    async def create_bucket(bucket: str, request: Request, owner: str = "default") -> dict:
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        try:
            return cp.storage.create_bucket(bucket, actor)
        except StorageError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/buckets/{bucket}/acl")
    async def grant_bucket_acl(bucket: str, grant: BucketGrant, request: Request,
                               owner: str = "default") -> dict:
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        try:
            return cp.storage.grant_bucket(bucket, actor, grant.grantee, grant.permissions)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except StorageError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/v1/buckets/{bucket}/objects/{key:path}/versions")
    async def bucket_object_versions(bucket: str, key: str, request: Request,
                                     owner: str = "default") -> dict:
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        try:
            return {"versions": cp.storage.list_object_versions(
                actor=actor, bucket=bucket, key=key)}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except StorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/v1/buckets/{bucket}/objects/{key:path}")
    async def put_bucket_object(bucket: str, key: str, request: Request,
                                owner: str = "default", replication: int = 2) -> dict:
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        from ..storage.blobstore import MAX_OBJECT_BYTES
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400,
                                detail="empty upload — attach the object bytes as the body")
        if len(data) > MAX_OBJECT_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"object too large (over {MAX_OBJECT_BYTES} bytes)")
        try:
            return await cp.storage.put_object_version(
                data, actor=actor, bucket=bucket, key=key,
                name=f"{bucket}/{key}", replication=max(1, min(replication, 5)))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except StorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/api/v1/buckets/{bucket}/objects/{key:path}")
    async def get_bucket_object(bucket: str, key: str, request: Request,
                                owner: str = "default",
                                version_id: str | None = None) -> Response:
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        try:
            data = await cp.storage.get_object_version(
                actor=actor, bucket=bucket, key=key, version_id=version_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except StorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if data is None:
            raise HTTPException(status_code=404, detail="unknown object version")
        return Response(content=data, media_type="application/octet-stream")

    # ---- hosting load-balancer / reverse proxy --------------------------- #
    # Hop-by-hop / encoding headers that must NOT be relayed verbatim between the
    # client<->control-plane and control-plane<->replica hops (RFC 7230 §6.1 + the
    # encoding headers httpx re-derives from the body it re-sends).
    _HOP_REQ = frozenset({"host", "content-length", "connection", "keep-alive",
                          "proxy-authenticate", "proxy-authorization", "te", "trailer",
                          "transfer-encoding", "upgrade", "accept-encoding"})
    _HOP_RESP = frozenset({"content-length", "content-encoding", "transfer-encoding",
                           "connection", "keep-alive", "content-type"})
    _AFF_COOKIE = "__mesh_aff"   # session-affinity cookie (our own; scoped per app)

    @app.api_route("/app/{dep_id}/{path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def proxy_app(dep_id: str, path: str, request: Request):
        cp = _cp()
        from fastapi.responses import StreamingResponse
        t0 = time.perf_counter()
        body = await request.body()
        # Forward the client's real request headers (Cookie, Authorization,
        # Content-Type, Accept, ...) so a genuine stateful app behaves normally;
        # strip hop-by-hop + encoding headers that don't survive re-proxying.
        fwd_headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in _HOP_REQ}
        # Session affinity: a multi-page app keeps in-memory session state on ONE
        # replica, so pin each client (via cookie) to the replica it first landed on.
        aff = request.cookies.get(_AFF_COOKIE)
        # Try up to 3 healthy replicas so a replica dying mid-request is retried on
        # a survivor instead of surfacing as a dropped request (spec §9.5 "no
        # dropped requests beyond in-flight").
        tried: set[str] = set()
        last_err = "no healthy replica"
        for attempt in range(3):
            replica = None
            if attempt == 0 and aff:
                replica = cp.hosting.sticky_replica(dep_id, aff, exclude=tried)
            if replica is None:
                replica = cp.hosting.pick_replica(dep_id, exclude=tried)
            if replica is None:
                break
            tried.add(replica.id)
            try:
                # Open a *streaming* upstream: we get status+headers before the body, so
                # failover on connect-error / 5xx still works, but once we commit we relay
                # bytes incrementally (SSE, chunked, token streams) instead of buffering.
                upstream = await cp.transport.proxy_open(
                    replica.node_id, request.method, f"{replica.url}/{path}",
                    params=request.query_params, content=body,
                    headers=fwd_headers, timeout=15.0)
            except Exception as exc:  # noqa: BLE001 — delivery failure -> try next replica
                last_err = f"replica {replica.node_name} unreachable: {str(exc)[:80]}"
                continue
            if upstream.status_code >= 500:
                last_err = f"replica {replica.node_name} -> {upstream.status_code}"
                await upstream.aclose()
                continue
            ctype = upstream.headers.get("content-type", "application/octet-stream")
            # Relay the app's own response headers back (Set-Cookie, Cache-Control,
            # Location, ETag, ...) so logins/redirects/caching work end-to-end.
            out_headers = {k: v for k, v in upstream.headers.items()
                           if k.lower() not in _HOP_RESP}
            out_headers["X-MeshPool-Served-By"] = replica.node_name or replica.node_id

            async def _relay(stream=upstream):
                try:
                    async for chunk in stream.aiter_bytes():
                        yield chunk
                finally:
                    await stream.aclose()

            resp = StreamingResponse(_relay(), status_code=upstream.status_code,
                                     media_type=ctype, headers=out_headers)
            if aff != replica.id:   # (re)pin this client to the serving replica
                resp.set_cookie(_AFF_COOKIE, replica.id, path=f"/app/{dep_id}/",
                                httponly=True, samesite="lax", max_age=3600)
            cp.hosting.record_request(dep_id, upstream.status_code,
                                      (time.perf_counter() - t0) * 1000.0)
            return resp
        cp.hosting.record_request(dep_id, 502, (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=502, detail=last_err)

    @app.get("/app/{dep_id}")
    async def proxy_app_root(dep_id: str, request: Request):
        return await proxy_app(dep_id, "", request)

    # ---- inference (flagship: sharded model) ----------------------------- #
    def _current_model():
        """Newest deployment the dashboard's single-model panel should show."""
        models = _cp().inference.list()
        return models[0] if models else None

    def _resolve_model_id(explicit: Optional[str]):
        if explicit:
            return explicit
        cur = _current_model()
        return cur.id if cur else None

    def _flatten_chat(messages: list[ChatMessage]) -> str:
        """Render an OpenAI message list into a single prompt.

        The whole conversation is kept. Using only the last user message (as this
        endpoint previously did) silently drops all prior turns, which makes
        multi-turn clients appear to have no memory.
        """
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

    @app.post("/api/v1/models")
    async def deploy_model(req: ModelDeploySubmit, request: Request) -> dict:
        cp = _cp()
        owner = _consumer_owner(cp, request, req.owner)
        _require_credits(cp, owner)
        try:
            dep = await cp.inference.deploy(req.model, owner)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return cp.inference.view(dep.id)

    @app.post("/api/v1/functions")
    async def upload_function(request: Request, name: str = "",
                              owner: str = "default") -> dict:
        """Bring your own code: upload a WebAssembly module and get a function name.

        This is what turns MeshPool from "a pool that runs *our* six functions" into
        "a pool that runs *your* code". The module is validated here — at upload
        time, where the user is watching — rather than on a donor's machine after it
        has already wasted their CPU. It is then stored in the pooled object store,
        content-addressed, and referred to for the rest of its life as
        ``wasm:<sha256>``.

        The module runs inside the WASM sandbox: WASI preview 1 only, no filesystem,
        no sockets, no host FFI, with fuel, memory and wall-clock ceilings. That is
        what makes it safe to run a stranger's code on a volunteer's laptop.

        Guest ABI: read the JSON payload from stdin, write the JSON result to stdout
        (``wasm32-wasip1`` command modules; Rust, Go, Zig, C and others target it).
        """
        cp = _cp()
        owner = _consumer_owner(cp, request, owner)
        _require_credits(cp, owner)
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="empty module upload")
        raw = bytes(raw)
        if len(raw) > MAX_WASM_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"module exceeds the {MAX_WASM_BYTES}-byte limit")
        try:
            from ..functions import wasm as _wasm
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503,
                                detail=f"the wasm runtime is unavailable: {exc}")
        if not _wasm.WASM_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="the wasm runtime is unavailable on the control plane "
                       "(pip install wasmtime)")
        try:
            info = _wasm.preflight(raw)
        except _wasm.FunctionError as exc:
            raise HTTPException(status_code=400, detail=exc.message)
        except Exception as exc:  # noqa: BLE001 — never 500 on a bad upload
            raise HTTPException(status_code=400,
                                detail=f"module could not be validated: {exc}")
        if cp.storage is None:
            raise HTTPException(status_code=503, detail="pooled storage unavailable")
        meta = await cp.storage.put(raw, owner=owner,
                                    name=name or f"fn-{info['sha256'][:12]}",
                                    internal=True)
        if meta.id != info["sha256"]:
            raise HTTPException(
                status_code=500,
                detail="pooled store returned a different content id than the "
                       "module hash; refusing to register an unverifiable function")
        log.info("function uploaded", fn=info["sha256"][:12], owner=owner,
                 size=len(raw))
        return {
            "function": f"wasm:{info['sha256']}",
            "sha256": info["sha256"],
            "size_bytes": info["size_bytes"],
            "abi": info["abi"],
            "imports": info["imports"],
            "exports": info["exports"],
            "name": name or None,
            "owner": owner,
            "usage": {
                "submit": {"func": f"wasm:{info['sha256']}",
                           "payloads": [{"your": "input"}]},
                "note": "the module receives each payload as JSON on stdin and must "
                        "write its JSON result to stdout",
            },
        }

    @app.post("/api/v1/models/upload")
    async def upload_model(request: Request, model: str = "nano-gpt",
                           owner: str = "default") -> dict:
        """Bring your own weights: stage a trained checkpoint (.npz) in the pool and
        serve it sharded across the cluster. The checkpoint is split into per-layer
        group blobs (replicated over donor nodes) referenced by a small manifest, so a
        checkpoint too big for any single machine still deploys and each pipeline stage
        pulls only the layer blobs it needs — the control plane never holds the whole
        checkpoint. The model architecture (``model``) must be one the pool knows how
        to shard."""
        cp = _cp()
        owner = _consumer_owner(cp, request, owner)
        _require_credits(cp, owner)
        from ..inference import model as M
        cfg = M.REGISTRY.get(model)
        if cfg is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown model architecture {model!r}")
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="empty checkpoint upload")
        if len(raw) > MAX_WEIGHTS_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"checkpoint exceeds the {MAX_WEIGHTS_BYTES}-byte limit")
        try:
            _validate_checkpoint(raw, cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        blob_id = await _pool_weights(cp, raw, owner, model)
        try:
            dep = await cp.inference.deploy(model, owner, weights_blob_id=blob_id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return cp.inference.view(dep.id)

    @app.post("/api/v1/models/weights/group")
    async def stage_weight_group(request: Request, model: str = "nano-gpt",
                                 owner: str = "default") -> dict:
        """Stage ONE per-layer weight group in the pool and return its blob id.

        The streaming upload path: a client holding a checkpoint too big to buffer whole
        splits it locally and sends one layer group at a time, so the control plane
        never holds more than a single group. ``/api/v1/models/weights/register`` then
        stitches the staged groups into a deployable model."""
        cp = _cp()
        owner = _consumer_owner(cp, request, owner)
        _require_credits(cp, owner)
        from ..inference import model as M
        if model not in M.REGISTRY:
            raise HTTPException(status_code=404,
                                detail=f"unknown model architecture {model!r}")
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="empty weight-group upload")
        if len(raw) > MAX_WEIGHTS_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"weight group exceeds the {MAX_WEIGHTS_BYTES}-byte limit")
        import io
        import numpy as np
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                _ = data.files                 # cheap: reject a non-npz at the boundary
        except Exception as exc:  # noqa: BLE001 — numpy raises many types for garbage
            raise HTTPException(
                status_code=400,
                detail=f"not a readable .npz weight group: {type(exc).__name__}: {exc}")
        from ..storage.engine import StorageError
        try:
            bm = await cp.storage.put(raw, owner=owner, name="", internal=True)
        except StorageError as exc:
            raise HTTPException(status_code=503,
                                detail=f"could not stage the weight group in the pool: {exc}")
        return {"blob_id": bm.id, "bytes": len(raw)}

    @app.post("/api/v1/models/weights/register")
    async def register_weights(request: Request, model: str = "nano-gpt",
                               owner: str = "default") -> dict:
        """Assemble previously-staged weight groups into a deployable model.

        Given a manifest of ``{group name -> blob id}`` (as produced by streaming each
        group through ``/weights/group``), confirm every layer group is present and
        holds the right parameters -- fetching at most one group at a time so the
        control plane's memory stays bounded -- then record the manifest and deploy the
        model sharded across the pool."""
        cp = _cp()
        owner = _consumer_owner(cp, request, owner)
        _require_credits(cp, owner)
        from ..inference import model as M
        cfg = M.REGISTRY.get(model)
        if cfg is None:
            raise HTTPException(status_code=404,
                                detail=f"unknown model architecture {model!r}")
        body = await request.json()
        groups = body.get("groups") if isinstance(body, dict) else None
        meta_id = body.get("meta") if isinstance(body, dict) else None
        if not isinstance(groups, dict) or not groups:
            raise HTTPException(status_code=400, detail="manifest is missing 'groups'")
        expected = {name: keys for name, keys in M.group_specs(cfg)}
        if set(groups) != set(expected):
            missing = sorted(set(expected) - set(groups))
            extra = sorted(set(groups) - set(expected))
            raise HTTPException(status_code=400, detail=(
                f"manifest groups do not match {model}: missing {missing[:3]}, "
                f"unexpected {extra[:3]}"))
        import io
        import numpy as np
        for name, keys in expected.items():
            bid = groups[name]
            blob = await cp.storage.get(bid) if isinstance(bid, str) else None
            if blob is None:
                raise HTTPException(status_code=400,
                                    detail=f"weight group {name!r} is not in the pool")
            try:
                with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                    present = set(data.files)
            except Exception as exc:  # noqa: BLE001 — numpy raises many types
                raise HTTPException(
                    status_code=400,
                    detail=f"weight group {name!r} is not a readable .npz: "
                           f"{type(exc).__name__}: {exc}")
            if present != set(keys):
                raise HTTPException(status_code=400,
                                    detail=f"weight group {name!r} has the wrong parameters")
        if meta_id is not None and (not isinstance(meta_id, str)
                                    or await cp.storage.get(meta_id) is None):
            raise HTTPException(status_code=400,
                                detail="tokenizer metadata is not in the pool")
        manifest = {"format": WEIGHTS_MANIFEST_FORMAT, "model": model,
                    "n_layer": cfg.n_layer, "groups": groups, "meta": meta_id}
        from ..storage.engine import StorageError
        try:
            bm = await cp.storage.put(json.dumps(manifest).encode("utf-8"), owner=owner,
                                      name=f"weights:{model}", internal=True)
        except StorageError as exc:
            raise HTTPException(status_code=503,
                                detail=f"could not record the weight manifest: {exc}")
        try:
            dep = await cp.inference.deploy(model, owner, weights_blob_id=bm.id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return cp.inference.view(dep.id)

    @app.get("/api/v1/models")
    async def list_models() -> dict:
        cp = _cp()
        views = [cp.inference.view(m.id) for m in cp.inference.list()]
        views = [v for v in views if v]
        current = views[0] if views else {"deployed": False, "stages": []}
        return {**current, "models": views}

    @app.get("/api/v1/models/available")
    async def available_models() -> dict:
        """Models the pool can serve right now, for the deploy picker: the built-in
        nets plus every distinct Ollama model advertised by an online node (with the
        nodes that hold each, so the UI can show where a model would run)."""
        from ..inference import model as M
        cp = _cp()
        ollama: dict[str, list[str]] = {}
        for n in cp.registry.online():
            for m in (n.capability.ollama_models or []):
                ollama.setdefault(m, []).append(n.name)
        return {
            "builtin": sorted(M.REGISTRY.keys()),
            "ollama": [{"model": m, "nodes": sorted(set(hosts))}
                       for m, hosts in sorted(ollama.items())],
        }

    @app.get("/api/v1/models/{model_id}")
    async def get_model(model_id: str) -> dict:
        cp = _cp()
        v = cp.inference.view(model_id)
        if not v:
            raise HTTPException(status_code=404, detail="unknown model deployment")
        return v

    @app.get("/api/v1/models/{model_id}/durability")
    async def model_durability(model_id: str) -> dict:
        """How many machines hold each of this model's layer groups.

        On borrowed hardware the interesting question is not "is it up" but "what
        happens when these machines go away". ``survives_machine_losses`` answers it:
        that many nodes can leave before the model can no longer be rebuilt at all,
        governed by whichever layer group is held by fewest machines, because a
        pipeline needs every layer.

        Watching the counts climb as the pool serves the model is the visible form of
        weight-swarm growth: a node that serves a layer keeps it and becomes a source
        for the next node that needs it, so a model gets *more* durable the more it is
        used rather than less.
        """
        cp = _cp()
        d = await cp.inference.durability(model_id)
        if d is None:
            raise HTTPException(status_code=404, detail="unknown model deployment")
        return d

    @app.delete("/api/v1/models/{model_id}")
    async def remove_model(model_id: str) -> dict:
        cp = _cp()
        ok = await cp.inference.remove(model_id)
        if not ok:
            raise HTTPException(status_code=404, detail="unknown model deployment")
        return {"removed": model_id}

    @app.post("/api/v1/inference/chat")
    async def inference_chat(req: InferSubmit, request: Request):
        """Generate an answer, optionally streaming it.

        The non-streaming form answers with the engine's full result object. The
        console used to call that form, read the response body through a
        ByteStreamReader, and append the decoded bytes straight into the chat
        bubble -- so the user was shown the JSON envelope rather than the answer,
        and "streaming" was only however TCP happened to fragment one body. With
        ``stream: true`` this yields NDJSON: ``{"delta": "..."}`` per piece of new
        text, then ``{"done": true, "result": {...}}`` carrying exactly what the
        non-streaming form would have returned.
        """
        cp = _cp()
        owner = _consumer_owner(cp, request, "default")
        _require_credits(cp, owner)
        model_id = _resolve_model_id(req.model_id)
        if not model_id:
            raise HTTPException(status_code=409, detail="no model deployed; deploy one first")
        gen_kw = dict(temperature=req.temperature, seed=req.seed,
                      speculative=req.speculative, draft_k=req.draft_k,
                      draft_layers=req.draft_layers,
                      chunked_prefill=req.chunked_prefill)
        if not req.stream:
            try:
                return await cp.inference.chat(model_id, req.prompt,
                                               req.max_new_tokens, **gen_kw)
            except KeyError:
                raise HTTPException(status_code=404, detail="unknown model deployment")
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc))

        agen = cp.inference.chat_stream(model_id, req.prompt, req.max_new_tokens,
                                        **gen_kw)
        # Pull the first item before committing to a 200. Deployment failures
        # surface on the very first step, and a client can act on a 404/409 body
        # but not on an exception buried inside an already-started stream.
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:
            first = None
        except KeyError:
            await agen.aclose()
            raise HTTPException(status_code=404, detail="unknown model deployment")
        except RuntimeError as exc:
            await agen.aclose()
            raise HTTPException(status_code=409, detail=str(exc))

        async def _ndjson():
            item = first
            try:
                while item is not None:
                    yield json.dumps(item) + "\n"
                    try:
                        item = await agen.__anext__()
                    except StopAsyncIteration:
                        item = None
            except Exception as exc:  # noqa: BLE001
                # The status line is long gone, so report the truncation in-band:
                # a cut-off answer must be visible as one, not look complete.
                yield json.dumps({"error": str(exc)}) + "\n"

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    # OpenAI-compatible surface (/v1/*) — registered from openai_api.py so the
    # implementation lives in one place and the non-vacuity gate can mutate it.
    from .openai_api import register_openai_routes as _reg_oai
    _reg_oai(app, _cp)

    # ---- secrets (scoped, short-lived brokering; docs/security.md) ------- #
    @app.get("/api/v1/secrets")
    async def list_secrets() -> dict:
        """Names only — values are never returned by the API."""
        cp = _cp()
        return {"names": cp.secrets.list_names(),
                "active_leases": cp.secrets.active_lease_count()}

    @app.post("/api/v1/secrets")
    async def put_secret(body: SecretPut) -> dict:
        cp = _cp()
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        try:
            cp.secrets.put_secret(name, body.value, owner=body.owner)
        except SecretError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"name": name, "stored": True}

    @app.delete("/api/v1/secrets/{name}")
    async def delete_secret(name: str) -> dict:
        cp = _cp()
        return {"name": name, "deleted": cp.secrets.delete_secret(name)}

    @app.post("/api/v1/secrets/redeem")
    async def redeem_secret(body: RedeemBody, request: Request) -> dict:
        """Worker-facing: exchange a single-use, node-scoped lease for its value.

        Node identity is derived from the per-node session token (minted at
        registration) when presented, so a leaked lease id cannot be redeemed by a
        different node spoofing ``node_id``. At Tier 1+ a valid token is required.
        In ``auth_mode=enforced`` the caller-supplied ``node_id`` is **never** trusted —
        identity always comes from the authenticated token, fixing the spoofing hole."""
        cp = _cp()
        token = request.headers.get("X-MeshPool-Node-Token", "")
        verified = cp.registry.node_id_for_token(token)
        if cp.settings.require_enroll_token and not verified:
            raise HTTPException(status_code=403, detail="unauthenticated node")
        if cp.settings.auth_mode == "enforced" and not verified:
            raise HTTPException(status_code=403, detail=(
                "node authentication required: send a valid X-MeshPool-Node-Token "
                "obtained at registration — presenting a node_id in the body is "
                "not accepted in enforced mode (it can be forged)"))
        node_id = verified or body.node_id
        try:
            value = cp.secrets.redeem(body.lease_id, node_id)
        except SecretError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        return {"value": value}

    # ---- Vivaldi network coordinates ------------------------------------ #
    @app.post("/api/v1/mesh/rtt")
    async def report_rtt(body: RttReport, request: Request) -> dict:
        """Accept an RTT observation from an agent and update Vivaldi coordinates.

        ``from_node`` is overridden by the verified node token when present
        (enforced mode).  The symmetrical update means both endpoints' coordinates
        converge from each observation.
        """
        cp = _cp()
        # Resolve the reporter's identity: node token takes precedence.
        node_tok = request.headers.get("X-MeshPool-Node-Token", "")
        verified_id = cp.registry.node_id_for_token(node_tok) if node_tok else None
        from_node = verified_id or body.from_node
        if not from_node or not body.to_node:
            raise HTTPException(status_code=400, detail="from_node and to_node are required")
        if body.rtt_ms <= 0:
            raise HTTPException(status_code=400, detail="rtt_ms must be positive")
        # Nodes must be known to the registry; otherwise a caller could inject
        # coordinates for arbitrary IDs and pollute the coordinate space.
        if cp.registry.get(from_node) is None:
            raise HTTPException(status_code=404, detail=f"unknown node: {from_node!r}")
        if cp.registry.get(body.to_node) is None:
            raise HTTPException(status_code=404, detail=f"unknown node: {body.to_node!r}")
        cp.registry.update_vivaldi(from_node, body.to_node, body.rtt_ms)
        coord = cp.registry.get_coordinate(from_node)
        return {
            "ok": True,
            "from_node": from_node,
            "to_node": body.to_node,
            "rtt_ms": body.rtt_ms,
            "error": round(coord.error, 4) if coord else None,
            "n_samples": coord.n_samples if coord else 0,
        }

    @app.get("/api/v1/mesh/coordinates")
    async def mesh_coordinates() -> dict:
        """Return all nodes' Vivaldi coordinates plus a predicted RTT matrix.

        This is the data source for the live Mesh Map visualization and for the
        ``predicted_rtt(a, b)`` scheduler helper.  The matrix is O(n²) but the
        number of nodes in a MeshPool cluster is bounded in practice (10s–100s),
        so it is cheap to compute on demand.

        Response shape::

            {
              "nodes": {
                "<node_id>": {
                  "vec": [x, y],
                  "height": h,
                  "error": e,
                  "n_samples": n,
                  "trusted": bool
                }, ...
              },
              "predicted_rtt_ms": {
                "<node_a_id>": {"<node_b_id>": 42.1, ...},
                ...
              },
              "scheduler_uses_vivaldi": bool
            }
        """
        cp = _cp()
        coords = cp.registry.vivaldi_snapshot()
        # Build predicted RTT matrix for all known nodes (only pairs where both
        # have coordinates; absent from the matrix otherwise).
        from ..common.vivaldi import distance_to as _dist_to
        rtt_matrix: dict[str, dict[str, float]] = {}
        ids = list(coords.keys())
        for i, a in enumerate(ids):
            row: dict[str, float] = {}
            for b in ids:
                if a == b:
                    continue
                row[b] = round(_dist_to(coords[a], coords[b]), 3)
            if row:
                rtt_matrix[a] = row

        # Include nodes that are in the registry but have no coordinate yet
        # (so the consumer knows they exist but lack data).
        all_nodes = {n.id for n in cp.registry.list()}
        nodes_out: dict[str, dict] = {}
        for nid in all_nodes:
            c = coords.get(nid)
            if c is not None:
                nodes_out[nid] = {
                    "vec": [round(v, 4) for v in c.vec],
                    "height": round(c.height, 4),
                    "error": round(c.error, 4),
                    "n_samples": c.n_samples,
                    "trusted": c.is_trusted(),
                }
            else:
                nodes_out[nid] = {
                    "vec": None,
                    "height": None,
                    "error": 1.0,
                    "n_samples": 0,
                    "trusted": False,
                }
        return {
            "nodes": nodes_out,
            "predicted_rtt_ms": rtt_matrix,
            "scheduler_uses_vivaldi": any(
                c.is_trusted() for c in coords.values()
            ),
        }

    # ---- events (SSE) ---------------------------------------------------- #
    @app.get("/api/v1/events")
    async def events(request: Request) -> StreamingResponse:
        cp = _cp()
        q = cp.events.subscribe()

        async def gen():
            try:
                yield f"data: {json.dumps({'kind': 'hello', 'ts': now_ms()})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=10.0)
                        yield f"data: {json.dumps(evt)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                cp.events.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- dashboard ------------------------------------------------------- #
    @app.get("/")
    async def dashboard() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="dashboard not built")
        return FileResponse(str(index))

    @app.get("/console")
    @app.get("/console/{path:path}")
    async def console(path: str = "") -> FileResponse:
        """The React console. Client-side routed, so every path returns index.html
        and React Router resolves it — otherwise a browser refresh on /console/jobs
        would 404."""
        index = CONSOLE_DIR / "index.html"
        if not index.exists():
            raise HTTPException(
                status_code=404,
                detail="the console has not been built — run "
                       "`npm --prefix web install && npm --prefix web run build`")
        return FileResponse(str(index))

    # ---- S3-compatible API credentials management (p5-s3) --------------- #

    @app.get("/api/v1/s3/credentials")
    async def list_s3_credentials(request: Request, owner: str = "default") -> dict:
        """Return the caller's S3 access keys (never returns the secret key).

        Scoped to the caller. Returning every tenant's keys would hand any
        caller the exact input needed to revoke all of them via the endpoint
        below -- an enumeration step and a denial-of-service in one response.
        """
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        return {
            "credentials": [
                {"access_key": k, "owner": v[1]}
                for k, v in cp.s3_credentials.items()
                if v[1] == actor
            ]
        }

    @app.post("/api/v1/s3/credentials")
    async def create_s3_credentials(
            request: Request, owner: str = "default") -> dict:
        """Create a new S3 access-key pair and bind it to the owner.

        Returns ``access_key`` and ``secret_key`` once; save the secret now
        because this response is the only time it is displayed.

        Note that SigV4 is a shared-secret scheme: the server must be able to
        recompute the client's signature, so the secret key is necessarily
        stored in recoverable form. It lives in the control-plane datastore,
        the same trust boundary that already holds session tokens and the
        ledger. Treat control-plane storage accordingly.
        """
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        ak, sk = cp.s3_credentials.create(actor)
        return {"access_key": ak, "secret_key": sk, "owner": actor}

    @app.delete("/api/v1/s3/credentials/{access_key}")
    async def revoke_s3_credentials(
            request: Request, access_key: str, owner: str = "default") -> dict:
        """Revoke an S3 access key.

        Credentials that cannot be revoked are not really credentials; before
        they were persisted there was nothing to revoke them *from*.

        Only the key's own owner may revoke it. Without that check this is a
        cross-tenant kill switch: one call per key takes every other tenant's
        S3 clients offline, and the failure surfaces on the victim's side as
        ``InvalidAccessKeyId`` with no indication of who did it.

        A key belonging to someone else reports 404 rather than 403, so this
        endpoint cannot be used to probe which access keys exist.
        """
        cp = _cp()
        actor = _consumer_owner(cp, request, owner)
        holder = cp.s3_credentials.owner_of(access_key)
        if holder is None or holder != actor:
            raise HTTPException(status_code=404, detail="unknown access key")
        return {"access_key": access_key,
                "revoked": cp.s3_credentials.revoke(access_key)}

    # ---- S3 protocol router (includes at /s3 prefix) --------------------- #
    from ..storage import s3_api as _s3_mod
    _s3_mod.init(lambda: CP)
    app.include_router(_s3_mod.router, prefix="/s3")


def _owner_reputation(cp: ControlPlane, owner: str) -> float:
    """Average trust score across an owner's nodes (1.0 if they have none yet)."""
    scores = [cp.trust.score(n.id) for n in cp.registry.list() if n.owner == owner]
    return sum(scores) / len(scores) if scores else 1.0


def _node_view(cp: ControlPlane, n) -> dict:
    return {
        "id": n.id,
        "name": n.name,
        "owner": n.owner,
        "device_id": n.device_id,
        "url": n.url,
        "reachable_via": (
            cp.transport.path_for(n.id) if hasattr(cp.transport, "path_for")
            else ("tunnel" if cp.hub.is_connected(n.id) else "direct")
        ),
        "status": n.status.value if hasattr(n.status, "value") else n.status,
        "schedulable": schedulable(n),
        "capability": n.capability.model_dump(),
        "policy": n.policy.model_dump(mode="json"),
        "load": n.load.model_dump(),
        "link": n.link.model_dump(),
        "headroom_cpu_pct": round(n.headroom_cpu_pct(), 1),
        "headroom_ram_mb": n.headroom_ram_mb(),
        "trust_tier": int(n.trust_tier),
        "trust_score": round(cp.trust.score(n.id), 3),
        "last_heartbeat_ms": n.last_heartbeat_ms,
        "registered_ms": n.registered_ms,
    }


app = build_app()


def main() -> None:
    import uvicorn

    from meshpool.common import proactor_fix
    from meshpool.common.serve_guard import ListenerWatchdog

    settings = Settings.load()
    # Keep the listening socket alive through the transient accept failures
    # CPython treats as fatal on Windows (meshpool/common/proactor_fix.py).
    proactor_fix.install()
    # The control plane losing its listener is worse than a crash: every
    # agent keeps its tunnel open and reports healthy while no new work can
    # arrive. Fail loudly instead of silently.
    ListenerWatchdog(settings.host, settings.port).start()
    uvicorn.run("meshpool.controlplane.app:app", host=settings.host,
                port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
