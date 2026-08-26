"""Node Agent daemon: register, heartbeat, execute sandboxed work, kill-switch.

Run one per machine:  ``python -m meshpool.agent.daemon``
Config via env: MESHPOOL_CONTROL_PLANE_URL, MESHPOOL_AGENT_HOST/PORT, MESHPOOL_NAME,
MESHPOOL_OWNER, and MESHPOOL_SIM_* overrides used by the simulation harness to pin
heterogeneous fake capabilities/idle-time onto processes on a single machine.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import os
import re
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..common.logging import get_logger
from ..common.netutil import is_loopback
from ..common.models import (
    DonationPolicy, Heartbeat, NodeStatus, RegisterRequest, Schedule, now_ms,
)
from ..common.identity import enrollment_challenge
from .donation import DonationController
from .probe import measure_link, network_class_from_link, probe_capability
from ..common.faultinject import active_profile, corrupt_result
from .runtime import ProcessRuntime, free_port, start_app_process, LogBuffer
from ..inference import model as M
from ..inference.pipeline import CachedShard
from ..storage.blobstore import BlobStore, StorageFull
from ..batch.functions import is_wasm_function, wasm_module_id

log = get_logger("agent")

# A blob id is always a full SHA-256 hex digest; validate at the endpoint boundary so
# a caller can never coax the store into touching a path outside its shard layout.
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _envflag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _corrupt_result(v: Any) -> Any:
    """Deterministically turn a correct result into a plausible-but-wrong one.

    Used only when ``MESHPOOL_LIE=1`` to simulate a dishonest/faulty node in the
    Tier-2 verification acceptance test — the coordinator's spot-check oracle must
    catch this and reject the node's work. Never enabled in normal operation.
    """
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, str):
        return v + "X"
    if isinstance(v, list):
        return v[1:] if v else [1]
    if isinstance(v, dict):
        return {k: _corrupt_result(x) for k, x in v.items()}
    return v


def _sim_overrides() -> dict:
    ov: dict[str, Any] = {}
    for env, key, cast in [
        ("MESHPOOL_SIM_CORES", "cpu_cores", int),
        ("MESHPOOL_SIM_RAM_MB", "ram_mb", int),
        ("MESHPOOL_SIM_VRAM_MB", "vram_mb", int),
        ("MESHPOOL_SIM_FLOPS", "flops_estimate", float),
    ]:
        if os.environ.get(env):
            ov[key] = cast(os.environ[env])
    if os.environ.get("MESHPOOL_SIM_GPU"):
        ov["gpu"] = os.environ["MESHPOOL_SIM_GPU"] in ("1", "true", "yes")
    return ov


def _donate_disk_mb(capability) -> int:
    """How much disk (MB) this node donates to the pooled object store. An explicit
    MESHPOOL_DONATE_DISK_MB wins; otherwise donate a quarter of free disk with a
    256 MB floor so the pool is useful out of the box without threatening to fill the
    donor's disk. A ceiling, not a reservation."""
    env = os.environ.get("MESHPOOL_DONATE_DISK_MB")
    if env:
        return max(0, int(env))
    return max(256, int(capability.disk_mb * 0.25))


def detect_ollama_models(base_url: str) -> list[str]:
    """Models a local Ollama server already holds (empty if none / unreachable).

    Best-effort and fast (short timeout) so a node *without* Ollama still registers
    promptly. This is what lets the control plane route real-LLM inference to the
    nodes that already have a given model. Set ``MESHPOOL_OLLAMA=0`` to skip.
    """
    if os.environ.get("MESHPOOL_OLLAMA", "").strip() == "0":
        return []
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", []) if m.get("name")]
    except Exception:  # noqa: BLE001
        return []


class RunUnitRequest(BaseModel):
    job_id: str
    unit_id: str
    func: str
    payload: dict = {}
    cpu_pct: int = 15
    ram_mb: int = 128
    secret_leases: dict[str, str] = {}
    # For a ``wasm:<sha256>`` function: peer base URLs that already hold the module,
    # so a node seeing it for the first time can pull it instead of failing.
    wasm_sources: list[str] = []


class PolicyUpdate(BaseModel):
    cpu_pct: Optional[int] = None
    ram_mb: Optional[int] = None
    gpu_allowed: Optional[bool] = None
    schedule: Optional[Schedule] = None
    idle_minutes: Optional[int] = None


class StartReplicaRequest(BaseModel):
    deployment_id: str
    replica_id: str
    app: str = "hello"
    site: Optional[dict[str, str]] = None
    start_cmd: Optional[str] = None
    env: Optional[dict[str, str]] = None


class StopReplicaRequest(BaseModel):
    replica_id: str


class ReplicaLogsRequest(BaseModel):
    replica_id: str
    tail: int = 200


class ShardLoadRequest(BaseModel):
    model_id: str
    model_name: str = "nano-gpt"
    start_layer: int
    end_layer: int
    is_first: bool = False
    is_last: bool = False
    # base64 ``.npz`` of just this stage's parameters, hydrated by the control plane
    # from the pooled blob store. When None the node loads its slice from a checkpoint
    # it already has on disk (the built-in path, unchanged).
    params_b64: Optional[str] = None
    # Per-layer pull: sha256 ids of the weight-group blobs — already placed in this
    # node's local store — that make up this stage's slice. The node assembles its
    # params from these, so it holds only its own layers, never the whole checkpoint.
    weight_blobs: Optional[list[str]] = None


class ShardForwardRequest(BaseModel):
    model_id: str
    data: list          # token ids (is_tokens) or a [T, d_model] hidden-state tensor
    is_tokens: bool = False
    # --- KV-cached decoding (see meshpool/inference/pipeline.py) ------------ #
    # `session` selects a per-conversation KV cache on this node; omitting it keeps
    # the original stateless behaviour, where every call recomputes from scratch.
    session: Optional[str] = None
    pos_offset: int = 0
    use_cache: bool = False
    # Speculative decoding rolls rejected drafts back out of the cache. The rollback
    # rides along with the next forward instead of costing its own round trip, which
    # on a wide-area mesh would otherwise cancel most of speculation's benefit.
    truncate_to: Optional[int] = None


class ShardBatchItem(BaseModel):
    """One session's slice of a batched forward.

    ``rows`` says how many leading rows of the request's flat ``data`` belong to this
    item, so the whole batch travels as a single concatenated payload on one hop.
    """

    rows: int
    pos_offset: int = 0
    session: Optional[str] = None
    use_cache: bool = False
    truncate_to: Optional[int] = None


class ShardBatchForwardRequest(BaseModel):
    model_id: str
    data: list
    is_tokens: bool = False
    items: list[ShardBatchItem]


class ShardDropRequest(BaseModel):
    model_id: str
    session: str


class ShardUnloadRequest(BaseModel):
    model_id: str


class BlobPutRequest(BaseModel):
    blob_id: str
    data: str            # base64-encoded bytes


class BlobRequest(BaseModel):
    blob_id: str


class BlobChunkRef(BaseModel):
    id: str
    sources: list[str]         # base URLs of peer nodes holding this chunk


class BlobPullRequest(BaseModel):
    blob_id: str
    # Direct sources for a small (unchunked) object: base URLs of peer nodes holding it.
    sources: Optional[list[str]] = None
    # For a chunked object: its ordered chunks, each with its own peer sources. The node
    # pulls every chunk, verifies each against its id, and concatenates them.
    chunks: Optional[list[BlobChunkRef]] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8


class AgentState:
    def __init__(self) -> None:
        self.control_plane_url = os.environ.get(
            "MESHPOOL_CONTROL_PLANE_URL", "http://127.0.0.1:8080")
        # Global (tunnel) mode: the agent dials OUT to the coordinator and is reached
        # only through that reverse tunnel, so it never needs a routable inbound
        # address. It advertises an unreachable sentinel URL (so any control-plane →
        # agent op that still succeeds must have travelled through the tunnel) and
        # keeps its own server + hosted replicas on loopback, which the in-process
        # forwarder reaches. LAN mode is unchanged: advertise a routable LAN IP.
        self.tunnel = _envflag("MESHPOOL_TUNNEL")
        self.port = int(os.environ.get("MESHPOOL_AGENT_PORT", "9001"))
        if self.tunnel:
            self.host = "127.0.0.1"
            self.bind_host = os.environ.get("MESHPOOL_AGENT_BIND", "127.0.0.1")
            from ..common.netutil import TUNNEL_SENTINEL_URL
            self.url = os.environ.get("MESHPOOL_ADVERTISE_URL", TUNNEL_SENTINEL_URL)
        else:
            from ..common.netutil import lan_ip
            self.host = os.environ.get("MESHPOOL_AGENT_HOST") or lan_ip()
            self.bind_host = os.environ.get("MESHPOOL_AGENT_BIND", "0.0.0.0")
            self.url = f"http://{self.host}:{self.port}"
        # Where the tunnel forwarder (and health probes) reach this agent locally.
        self.local_url = f"http://127.0.0.1:{self.port}"
        self.name = os.environ.get("MESHPOOL_NAME", socket.gethostname())
        self.owner = os.environ.get("MESHPOOL_OWNER", "default")
        self.enroll_token = os.environ.get("MESHPOOL_ENROLL_TOKEN")
        # Persistent device identity (Ed25519, created on first run). A per-agent data
        # dir keeps simulated nodes on one machine cryptographically distinct.
        from ..common.identity import DeviceIdentity
        data_dir = os.environ.get("MESHPOOL_DATA_DIR") or os.path.join(
            os.path.expanduser("~"), ".meshpool")
        self.data_dir = data_dir
        self.identity = DeviceIdentity.load_or_create(data_dir)

        self.capability = probe_capability(_sim_overrides())
        # Detect a local Ollama server so the control plane can route real-LLM
        # inference to this node when it already holds the requested model. A ceiling,
        # not a reservation: absence just means this node won't be picked as a replica.
        self.ollama_url = os.environ.get("MESHPOOL_OLLAMA_URL", "http://127.0.0.1:11434")
        self.capability.ollama_models = detect_ollama_models(self.ollama_url)
        if self.capability.ollama_models:
            log.info("ollama detected", url=self.ollama_url,
                     models=self.capability.ollama_models)
        # Auto-scale the default RAM donation to the machine so the pool can host
        # real models out of the box. A fixed tiny cap (1 GB) is smaller than a
        # single model layer, which would make the flagship "deploy a model too big
        # for one machine" fail on first run. An explicit MESHPOOL_DONATE_RAM_MB
        # (the sim harness sets one) always wins; otherwise donate half of detected
        # RAM with a 2 GB floor. RAM cap is a ceiling, not a reservation.
        donate_ram_env = os.environ.get("MESHPOOL_DONATE_RAM_MB")
        donate_ram = (int(donate_ram_env) if donate_ram_env
                      else max(2048, int(self.capability.ram_mb * 0.5)))
        pol = DonationPolicy(
            cpu_pct=int(os.environ.get("MESHPOOL_DONATE_CPU_PCT", "50")),
            ram_mb=donate_ram,
            schedule=Schedule(os.environ.get("MESHPOOL_SCHEDULE", "always")),
        )
        sim_idle = os.environ.get("MESHPOOL_SIM_IDLE")
        self.donation = DonationController(
            pol, sim_idle_seconds=float(sim_idle) if sim_idle else None)
        # Public mesh: run strangers' units in the locked-down sandbox. Off by default
        # so LAN/office behaviour (fast shared pool) is unchanged.
        self.sandbox = _envflag("MESHPOOL_SANDBOX")
        # Test-only: emulate a dishonest node whose results must be caught by the
        # coordinator's Tier-2 spot-check. Never set in real deployments.
        self.lie = _envflag("MESHPOOL_LIE")
        self.runtime = ProcessRuntime(self.capability.cpu_cores, pol, sandbox=self.sandbox)
        self.node_id: Optional[str] = None
        self.node_token: str = ""
        self.replicas: dict[str, subprocess.Popen] = {}   # replica_id -> hosted app process
        self.replica_logs: dict[str, LogBuffer] = {}       # replica_id -> recent stdout/stderr
        self.shards: dict[str, CachedShard] = {}           # model_id -> pipeline stage
        # Receiver-side hop accounting. `shard_hops` counts wide-area round trips this
        # node actually served; `shard_rows` counts the sequence-steps carried inside
        # them. Their ratio is the real batching factor, measured where the packets
        # land rather than inferred from what the sender believes it sent -- which
        # makes it an independent oracle for the batching gate and the honest number
        # to surface on a node's activity panel.
        self.shard_hops = 0
        self.shard_rows = 0
        # Pooled object store: content-addressed blobs on donated disk (spec §P3).
        self.blobs = BlobStore(
            os.path.join(self.data_dir, "blobs"),
            cap_bytes=_donate_disk_mb(self.capability) * 1024 * 1024)
        self.status = NodeStatus.ONLINE
        self.heartbeat_interval_s = 3

    def policy(self) -> DonationPolicy:
        return self.donation.policy


# NOTE: constructed lazily (never at import time). AgentState builds a process
# pool + probes hardware; on Windows the pool uses "spawn", which re-imports this
# module in every worker. Building state at import would make each worker re-probe
# and build its own pool. get_state() ensures it is only built in the real agent
# process (main() / lifespan), never in a spawned pool worker.
state: "AgentState | None" = None


def get_state() -> "AgentState":
    global state
    if state is None:
        state = AgentState()
    return state


async def _register_once(client: httpx.AsyncClient, st: "AgentState | None" = None) -> bool:
    """One registration attempt. Signs an enrollment challenge so the coordinator can
    verify this device holds its private key. Returns True on success (state updated).

    ``st`` may be passed explicitly by a caller that reaches this function through a
    *different import* of this module. The process entrypoint runs the daemon as
    ``__main__`` (``python -m meshpool.agent.daemon``), so its populated global
    ``state`` lives on the ``__main__`` module object; the tunnel client, however,
    does ``from . import daemon`` and would otherwise see this package module's
    unpopulated ``state`` (``None``). Passing the live state object keeps the
    reverse-tunnel re-registration path (used when the coordinator restarts and the
    node's session token goes stale) working regardless of how the module was loaded.
    """
    st = st if st is not None else state
    link = measure_link(st.control_plane_url)
    if st.capability.network_class == "unknown":
        st.capability.network_class = network_class_from_link(link)
    ts = now_ms()
    signature = st.identity.sign(enrollment_challenge(st.identity.device_id, ts))
    req = RegisterRequest(
        name=st.name, url=st.url, owner=st.owner,
        capability=st.capability, link=link, policy=st.policy(),
        enroll_token=st.enroll_token,
        device_id=st.identity.device_id, public_key=st.identity.public_key_b64,
        signature=signature, ts_ms=ts,
        # MESHPOOL_FAILURE_DOMAIN lets the simulation harness inject synthetic
        # domain labels so gates can test correlated-failure scenarios with real
        # OS processes on a single host.  In production this env var is unset and
        # the control plane auto-infers the domain from the URL + owner.
        failure_domain=os.environ.get("MESHPOOL_FAILURE_DOMAIN") or None,
    )
    try:
        r = await client.post(
            f"{st.control_plane_url}/api/v1/nodes/register",
            json=req.model_dump(mode="json"), timeout=5.0)
        r.raise_for_status()
        body = r.json()
        st.node_id = body["node_id"]
        st.node_token = body.get("node_token", "")
        st.heartbeat_interval_s = body.get("heartbeat_interval_s", 3)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("register failed, retrying", err=str(exc)[:120])
        return False


async def _register(client: httpx.AsyncClient) -> None:
    while state.node_id is None:
        if await _register_once(client):
            log.info("registered with control plane", node=state.node_id,
                     cp=state.control_plane_url, device=state.identity.device_id)
            return
        await asyncio.sleep(2.0)


async def _heartbeat_loop(client: httpx.AsyncClient) -> None:
    while True:
        if state.node_id:
            load = state.donation.current_load(state.runtime._running)  # noqa: SLF001
            status = NodeStatus.DRAINING if state.donation.reclaimed else NodeStatus.ONLINE
            hb = Heartbeat(node_id=state.node_id, load=load, policy=state.policy(),
                           status=status)
            try:
                await client.post(
                    f"{state.control_plane_url}/api/v1/nodes/heartbeat",
                    json=hb.model_dump(mode="json"), timeout=5.0)
            except Exception as exc:  # noqa: BLE001
                log.debug("heartbeat failed", err=str(exc)[:120])
        await asyncio.sleep(state.heartbeat_interval_s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_state()  # build agent state in the real process (never in a pool worker)
    client = httpx.AsyncClient()
    app.state.client = client
    reg = asyncio.create_task(_register(client))
    hb = asyncio.create_task(_heartbeat_loop(client))
    tasks = [reg, hb]
    if state.tunnel:
        # Global mode: hold open an outbound reverse tunnel so the coordinator can
        # reach this agent without any inbound port.
        from .tunnel_client import run_tunnel_client
        tasks.append(asyncio.create_task(run_tunnel_client(state)))
    log.info("agent online", advertise=state.url, bind=f"{state.bind_host}:{state.port}",
             name=state.name, control_plane=state.control_plane_url,
             mode=("tunnel" if state.tunnel else "lan"),
             sandbox=state.sandbox)
    if active_profile().corrupt_results:
        log.warning("MESHPOOL_LIE is set: this node will return WRONG results "
                    "(test-only dishonest-node emulation; see faultinject.py)")
    if state.tunnel:
        log.info("global tunnel mode: dialing out to coordinator (no inbound port needed)")
    elif is_loopback(state.host):
        log.warning("advertising a loopback address — other machines cannot reach this "
                    "node; set MESHPOOL_AGENT_HOST to this machine's LAN IP", host=state.host)
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        for rid in list(state.replicas):
            _kill_replica(state.replicas.pop(rid, None))
        state.runtime.shutdown()
        await client.aclose()


app = FastAPI(title="MeshPool Node Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    load = state.donation.current_load(state.runtime._running)  # noqa: SLF001
    return {
        "node_id": state.node_id,
        "name": state.name,
        "status": ("draining" if state.donation.reclaimed else "online"),
        "capability": state.capability.model_dump(),
        "policy": state.policy().model_dump(mode="json"),
        "load": load.model_dump(),
        "cpu_slots": state.runtime.cpu_slots(),
    }


async def _redeem_secrets(leases: dict[str, str]) -> dict[str, str]:
    """Exchange single-use, node-scoped lease ids for their values, just-in-time.

    Values live only in this process (and the pool worker running the unit) for the
    unit's lifetime; they are never written to disk or logged (spec §5)."""
    out: dict[str, str] = {}
    if not leases:
        return out
    # Present the node session token so the control plane verifies our identity
    # server-side rather than trusting the node_id we send.
    headers = {"X-MeshPool-Node-Token": state.node_token} if state.node_token else {}
    async with httpx.AsyncClient() as c:
        for name, lease_id in leases.items():
            r = await c.post(
                f"{state.control_plane_url}/api/v1/secrets/redeem",
                json={"lease_id": lease_id, "node_id": state.node_id},
                headers=headers, timeout=5.0)
            r.raise_for_status()
            out[name] = r.json()["value"]
    return out


@app.post("/run_unit")
async def run_unit(req: RunUnitRequest) -> dict:
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")
    payload = req.payload
    if req.secret_leases:
        try:
            resolved = await _redeem_secrets(req.secret_leases)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502,
                                detail=f"secret redemption failed: {str(exc)[:150]}")
        payload = {**req.payload, "_secrets": resolved}
    if is_wasm_function(req.func):
        # Hydrate the user's module from the pooled, content-addressed blob store —
        # the same peer-pull path that distributes model weights. It is injected into
        # the payload rather than read from disk by the worker because the sandbox
        # child has no filesystem access of its own.
        try:
            module_id = wasm_module_id(req.func)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        blob = state.blobs.get(module_id)
        if blob is None and req.wasm_sources:
            async with httpx.AsyncClient(timeout=30.0) as c:
                try:
                    blob = await _pull_verified(c, module_id, req.wasm_sources)
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(
                        status_code=502,
                        detail=f"could not fetch wasm module {module_id[:12]}: "
                               f"{str(exc)[:150]}")
            try:
                state.blobs.put(module_id, blob)   # cache for the next unit
            except StorageFull:
                pass                                # run it anyway; just do not cache
        if blob is None:
            raise HTTPException(
                status_code=404,
                detail=f"wasm module {module_id[:12]} is not on this node and no "
                       f"sources were supplied")
        payload = {**payload, "_wasm_module": base64.b64encode(blob).decode("ascii")}
    try:
        result, compute_ms = await state.runtime.run_unit(req.func, payload)
        # Route through the fault-injection gate so the backdoor is never live in
        # a real deployment (see meshpool/common/faultinject.py for activation rules).
        result = corrupt_result(result)
        return {"unit_id": req.unit_id, "result": result, "compute_ms": compute_ms}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:200])


@app.post("/reclaim")
async def reclaim() -> dict:
    """Instant, local kill-switch — no control-plane approval required (spec §4.1)."""
    state.donation.reclaim()
    state.runtime.reclaim()
    for rid in list(state.replicas):
        _kill_replica(state.replicas.pop(rid, None))
    state.shards.clear()
    return {"status": "draining", "message": "all donated resources reclaimed"}


@app.post("/resume")
async def resume() -> dict:
    state.donation.resume()
    state.runtime.resume()
    return {"status": "online"}


@app.post("/policy")
async def update_policy(upd: PolicyUpdate) -> dict:
    pol = state.policy().model_copy(update={k: v for k, v in upd.model_dump().items()
                                            if v is not None})
    state.donation.set_policy(pol)
    state.runtime.apply_policy(pol)
    return {"policy": pol.model_dump(mode="json")}


# --- hosting: run web-app replicas as child processes (spec §9.5) ----------- #
def _kill_replica(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:  # noqa: BLE001
        pass


async def _wait_reachable(url: str, proc: subprocess.Popen, tries: int = 55) -> bool:
    """Wait until a dynamic app's own server accepts connections (any HTTP response
    means it's listening), or it exits / never binds. Kept under the coordinator's
    start_replica timeout so a slow/broken app fails fast instead of hanging."""
    async with httpx.AsyncClient() as c:
        for _ in range(tries):
            if proc.poll() is not None:
                return False
            try:
                await c.get(url + "/", timeout=0.4)
                return True
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.1)
    return False


@app.post("/start_replica")
async def start_replica(req: StartReplicaRequest) -> dict:
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")
    port = free_port()
    work_dir = ""
    if req.site:
        from ..hosting.site import MAX_BUNDLE_BYTES, write_site
        work_dir = os.path.join(state.data_dir, "sites", req.replica_id)
        try:
            write_site(work_dir, req.site, require_index=not req.start_cmd,
                       max_bytes=MAX_BUNDLE_BYTES)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid site: {exc}")
    url = f"http://{state.host}:{port}"

    if req.start_cmd:
        # Dynamic process app: run the deployer's own server. Executing an arbitrary
        # command is only safe on a trusted node, so a public/sandboxed donor refuses
        # it (static sites are still fine there).
        if state.sandbox:
            raise HTTPException(status_code=403,
                                detail="this node hosts static sites only, not process apps")
        if not work_dir:
            raise HTTPException(status_code=400, detail="a process app requires a bundle")
        from .runtime import start_proc_app
        logbuf = LogBuffer()
        try:
            proc = start_proc_app(work_dir, req.start_cmd, port,
                                  replica_id=req.replica_id, node_name=state.name,
                                  env_extra=req.env, log_buffer=logbuf)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"could not start app: {exc}")
        state.replicas[req.replica_id] = proc
        state.replica_logs[req.replica_id] = logbuf
        if not await _wait_reachable(url, proc):
            _kill_replica(proc)
            state.replicas.pop(req.replica_id, None)
            state.replica_logs.pop(req.replica_id, None)
            # Surface the app's own last output so the deployer sees the crash reason
            # (e.g. a traceback / "port in use") instead of an opaque "did not start".
            tail = logbuf.tail(15)
            detail = "app did not start listening on $PORT"
            if tail:
                detail += "; last output:\n" + "\n".join(tail)
            raise HTTPException(status_code=502, detail=detail)
        log.info("replica started", replica=req.replica_id, app="proc", port=port)
        return {"replica_id": req.replica_id, "url": url, "port": port}

    proc = start_app_process(req.app, port, replica_id=req.replica_id,
                             node_name=state.name, site_dir=work_dir)
    state.replicas[req.replica_id] = proc
    # briefly wait for the app to accept connections so the LB won't 502 first hit
    async with httpx.AsyncClient() as c:
        for _ in range(40):
            if proc.poll() is not None:
                raise HTTPException(status_code=500, detail="replica exited on start")
            try:
                r = await c.get(f"{url}/healthz", timeout=0.5)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.1)
    log.info("replica started", replica=req.replica_id, app=req.app, port=port)
    return {"replica_id": req.replica_id, "url": url, "port": port}


@app.post("/stop_replica")
async def stop_replica(req: StopReplicaRequest) -> dict:
    proc = state.replicas.pop(req.replica_id, None)
    state.replica_logs.pop(req.replica_id, None)
    _kill_replica(proc)
    shutil.rmtree(os.path.join(state.data_dir, "sites", req.replica_id), ignore_errors=True)
    return {"stopped": req.replica_id}


@app.post("/replica_logs")
async def replica_logs(req: ReplicaLogsRequest) -> dict:
    """Return the most recent captured stdout/stderr lines for one hosted replica."""
    buf = state.replica_logs.get(req.replica_id)
    if buf is None:
        return {"replica_id": req.replica_id, "known": False, "lines": []}
    return {"replica_id": req.replica_id, "known": True, "lines": buf.tail(req.tail)}


# --- pooled object store: hold content-addressed blobs on donated disk ------ #
@app.post("/blob_put")
async def blob_put(req: BlobPutRequest) -> dict:
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")
    if not _HEX64.match(req.blob_id):
        raise HTTPException(status_code=400, detail="blob_id must be a sha256 hex digest")
    try:
        raw = base64.b64decode(req.data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="data must be valid base64")
    try:
        state.blobs.put(req.blob_id, raw)
    except ValueError as exc:               # content-hash mismatch
        raise HTTPException(status_code=400, detail=str(exc))
    except StorageFull as exc:
        raise HTTPException(status_code=507, detail=str(exc))
    return {"blob_id": req.blob_id, "stored": True, "size": len(raw)}


@app.post("/blob_get")
async def blob_get(req: BlobRequest) -> dict:
    if not _HEX64.match(req.blob_id):
        raise HTTPException(status_code=400, detail="blob_id must be a sha256 hex digest")
    data = state.blobs.get(req.blob_id)
    if data is None:
        return {"blob_id": req.blob_id, "found": False, "data": ""}
    return {"blob_id": req.blob_id, "found": True,
            "data": base64.b64encode(data).decode("ascii")}


@app.post("/blob_has")
async def blob_has(req: BlobRequest) -> dict:
    return {"blob_id": req.blob_id, "has": state.blobs.has(req.blob_id)}


@app.post("/blob_delete")
async def blob_delete(req: BlobRequest) -> dict:
    state.blobs.delete(req.blob_id)
    return {"blob_id": req.blob_id, "deleted": True}


# --- proof-of-storage audit responder --------------------------------------- #
class BlobAuditRequest(BaseModel):
    """Coordinator-issued proof-of-storage challenge.

    The node must return the SHA-256 hex digest of the bytes at
    ``[chunk_index * chunk_size, (chunk_index + 1) * chunk_size)`` within
    the shard blob identified by ``shard_id``.  The ``challenge_id`` echoes
    back so the coordinator can match the response to its pending challenge.

    A fresh random ``chunk_index`` is chosen per challenge so a node that
    memorised only one chunk's answer will fail most challenges.
    """
    challenge_id: str
    shard_id: str            # SHA-256 hex digest of the full shard bytes
    chunk_index: int         # which AUDIT_CHUNK_SIZE block to prove
    chunk_size: int          # block size in bytes (must match coordinator's view)


@app.post("/blob_audit")
async def blob_audit(req: BlobAuditRequest) -> dict:
    """Respond to a coordinator proof-of-storage challenge.

    Reads the shard from local BlobStore, slices out the challenged chunk, and
    returns its SHA-256 as hex.  An honest node that still holds the bytes
    always passes; one that deleted them cannot compute the right hash.

    Returns ``{"challenge_id": ..., "hash": "<64-char hex>"}`` on success.
    Returns ``{"challenge_id": ..., "hash": ""}`` when the shard is not found
    (which the coordinator treats as a failed audit).
    """
    if not _HEX64.match(req.shard_id):
        raise HTTPException(status_code=400, detail="shard_id must be a sha256 hex digest")
    if req.chunk_index < 0 or req.chunk_size <= 0:
        raise HTTPException(status_code=400, detail="invalid chunk_index or chunk_size")
    data = state.blobs.get(req.shard_id)
    if data is None:
        # Shard not present — respond with an empty hash so the coordinator detects
        # the failure without needing a special error path.
        return {"challenge_id": req.challenge_id, "hash": ""}
    start = req.chunk_index * req.chunk_size
    end = start + req.chunk_size
    chunk = data[start:end]
    if not chunk:
        # chunk_index out of range for this shard's size
        return {"challenge_id": req.challenge_id, "hash": ""}
    response_hash = hashlib.sha256(chunk).hexdigest()
    return {"challenge_id": req.challenge_id, "hash": response_hash}


class _PullError(Exception):
    """No peer served a verified copy of a requested blob."""


async def _pull_verified(c: httpx.AsyncClient, blob_id: str, sources: list[str]) -> bytes:
    """Return the bytes for ``blob_id`` from the first peer that serves a copy whose
    SHA-256 matches the id. The digest check means a dishonest or misconfigured peer can
    never hand back the wrong bytes -- and doubles as SSRF-exfiltration protection: any
    response that does not hash to the expected id is discarded. Raises ``_PullError`` if
    no source yields verified bytes."""
    for src in sources:
        try:
            r = await c.post(f"{src.rstrip('/')}/blob_get", json={"blob_id": blob_id})
            if r.status_code != 200:
                continue
            body = r.json()
        except Exception:  # noqa: BLE001 -- peer unreachable / bad response, try the next
            continue
        if not body.get("found"):
            continue
        try:
            raw = base64.b64decode(body.get("data", ""), validate=True)
        except (binascii.Error, ValueError):
            continue
        if hashlib.sha256(raw).hexdigest() == blob_id:
            return raw
    raise _PullError(f"no peer served a valid copy of {blob_id[:12]}")


@app.post("/blob_pull")
async def blob_pull(req: BlobPullRequest) -> dict:
    """Pull a weight-group blob straight from a peer node that already holds it, so the
    control plane carries only ids + peer URLs and never the weight bytes themselves.
    Every piece is verified against its SHA-256 id and the reassembled object against
    ``blob_id`` before it is stored. Used for LAN pools; tunnel-attached nodes are served
    by a control-plane relay instead."""
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")
    if not _HEX64.match(req.blob_id):
        raise HTTPException(status_code=400, detail="blob_id must be a sha256 hex digest")
    if state.blobs.has(req.blob_id):
        return {"blob_id": req.blob_id, "stored": True, "bytes": 0, "cached": True}
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
        try:
            if req.chunks:
                parts: list[bytes] = []
                for ch in req.chunks:
                    if not _HEX64.match(ch.id):
                        raise HTTPException(status_code=400,
                                            detail="chunk id must be a sha256 hex digest")
                    parts.append(await _pull_verified(c, ch.id, ch.sources))
                whole = b"".join(parts)
                if hashlib.sha256(whole).hexdigest() != req.blob_id:
                    raise HTTPException(status_code=502,
                                        detail="reassembled object failed integrity check")
            else:
                whole = await _pull_verified(c, req.blob_id, req.sources or [])
        except _PullError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    try:
        state.blobs.put(req.blob_id, whole)
    except ValueError as exc:               # content-hash mismatch (already verified)
        raise HTTPException(status_code=400, detail=str(exc))
    except StorageFull as exc:
        raise HTTPException(status_code=507, detail=str(exc))
    return {"blob_id": req.blob_id, "stored": True, "bytes": len(whole)}


# --- inference: host a contiguous block of model layers (spec §9.6) --------- #
@app.post("/shard/load")
async def shard_load(req: ShardLoadRequest) -> dict:
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")
    cfg = M.REGISTRY.get(req.model_name)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"unknown model {req.model_name!r}")
    params: Optional[dict] = None
    if req.weight_blobs is not None:
        # Per-layer pull: assemble this stage's slice from the weight-group blobs already
        # placed in this node's local store, so the node holds only its own layers and
        # never the whole (possibly larger-than-one-box) checkpoint.
        merged: dict = {}
        for bid in req.weight_blobs:
            if not _HEX64.match(bid):
                raise HTTPException(status_code=400,
                                    detail="weight blob id must be a sha256 hex digest")
            blob = state.blobs.get(bid)
            if blob is None:
                raise HTTPException(status_code=404,
                                    detail=f"weight blob not on this node: {bid[:12]}")
            if hashlib.sha256(blob).hexdigest() != bid:
                raise HTTPException(status_code=400,
                                    detail=f"weight blob failed integrity check: {bid[:12]}")
            try:
                group, _ = M.loads_npz(blob)
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=400,
                                    detail=f"invalid weight blob {bid[:12]}: {exc}") from exc
            merged.update(group)
        params = merged
    elif req.params_b64 is not None:
        # Legacy inline slice: the control plane shipped the bytes directly.
        try:
            params, _ = M.loads_npz(base64.b64decode(req.params_b64, validate=True))
        except (binascii.Error, ValueError, OSError) as exc:
            raise HTTPException(status_code=400,
                                detail=f"invalid shard weights: {exc}") from exc
    shard = CachedShard(cfg, req.start_layer, req.end_layer, req.is_first, req.is_last,
                        params=params)
    state.shards[req.model_id] = shard
    n_params = sum(int(v.size) for v in shard.P.values())
    log.info("shard loaded", model=req.model_id, layers=f"{req.start_layer}-{req.end_layer}",
             first=req.is_first, last=req.is_last, params=n_params)
    return {"model_id": req.model_id, "layers": [req.start_layer, req.end_layer],
            "params": n_params}


@app.post("/shard/forward")
async def shard_forward(req: ShardForwardRequest) -> dict:
    shard = state.shards.get(req.model_id)
    if shard is None:
        raise HTTPException(status_code=404, detail="shard not loaded on this node")
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")

    session = req.session if req.use_cache else None
    state.shard_hops += 1
    state.shard_rows += 1
    if req.truncate_to is not None and req.session:
        # Applied before computing, so a rejected speculative branch is gone before
        # the next one is written. Costs no extra round trip.
        shard.truncate(req.session, int(req.truncate_to))

    inp = req.data if req.is_tokens else np.asarray(req.data, dtype="float32")
    try:
        out = await asyncio.get_running_loop().run_in_executor(
            None, shard.forward, inp, req.is_tokens, int(req.pos_offset), session)
    except ValueError as exc:
        # A cache miss is a protocol error the caller can recover from by replaying
        # the prompt, so report it as such rather than as a 500.
        raise HTTPException(status_code=409, detail=f"cache state error: {exc}") from exc
    return {"data": out.tolist()}


@app.post("/shard/forward_batch")
async def shard_forward_batch(req: ShardBatchForwardRequest) -> dict:
    """Run several sessions through this shard in one request.

    This is the node-side half of continuous batching. Each item is computed
    separately against its own KV cache -- exactly as an unbatched forward would be,
    so a tenant's output never depends on who shared the batch -- but they all arrive
    and leave on a single hop. On a wide-area mesh the hop is the expensive part, so B
    tenants cost one ``(N-1) * RTT`` between them instead of B.
    """
    shard = state.shards.get(req.model_id)
    if shard is None:
        raise HTTPException(status_code=404, detail="shard not loaded on this node")
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")
    if not req.items:
        raise HTTPException(status_code=400, detail="batch must contain at least one item")

    total = sum(int(i.rows) for i in req.items)
    if any(int(i.rows) < 1 for i in req.items):
        raise HTTPException(status_code=400, detail="every batch item needs rows >= 1")
    if len(req.data) != total:
        raise HTTPException(
            status_code=400,
            detail=f"payload has {len(req.data)} rows but items declare {total}")
    named = [i.session for i in req.items if i.session]
    if len(set(named)) != len(named):
        # Two rows sharing one KV cache in a single call would append both to the
        # same cache and interleave two conversations. Refuse rather than corrupt.
        raise HTTPException(status_code=400,
                            detail="batch items must use distinct sessions")

    for item in req.items:
        if item.truncate_to is not None and item.session:
            shard.truncate(item.session, int(item.truncate_to))

    # One hop carried len(items) sequence-steps. Counted after validation so a
    # rejected malformed batch is not credited as work.
    state.shard_hops += 1
    state.shard_rows += len(req.items)

    def _run() -> list:
        outs, cursor = [], 0
        for item in req.items:
            rows = int(item.rows)
            piece = req.data[cursor:cursor + rows]
            cursor += rows
            inp = piece if req.is_tokens else np.asarray(piece, dtype="float32")
            session = item.session if item.use_cache else None
            outs.append(shard.forward(inp, req.is_tokens, int(item.pos_offset), session))
        return outs

    try:
        outs = await asyncio.get_running_loop().run_in_executor(None, _run)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=f"cache state error: {exc}") from exc

    rows = [int(o.shape[0]) for o in outs]
    return {"rows": rows, "data": np.concatenate(outs, axis=0).tolist()}


@app.get("/shard/stats")
async def shard_stats() -> dict:
    """Hops this node actually served, and the work that rode inside them.

    Counted at the receiver, so it cannot be inflated by a sender's bookkeeping.
    ``mean_rows_per_hop`` above 1.0 means tenants really are sharing round trips.
    """
    hops, rows = state.shard_hops, state.shard_rows
    return {"shard_hops": hops, "shard_rows": rows,
            "mean_rows_per_hop": round(rows / hops, 4) if hops else 0.0,
            "loaded": sorted(state.shards)}


@app.post("/shard/drop")
async def shard_drop(req: ShardDropRequest) -> dict:
    """Release a session's KV cache. Idempotent."""
    shard = state.shards.get(req.model_id)
    if shard is not None:
        shard.drop(req.session)
    return {"dropped": req.session}


@app.post("/shard/unload")
async def shard_unload(req: ShardUnloadRequest) -> dict:
    state.shards.pop(req.model_id, None)
    return {"unloaded": req.model_id}


# --- inference: native real-LLM generation via a local Ollama server -------- #
@app.post("/ollama/generate")
async def ollama_generate(req: OllamaGenerateRequest) -> dict:
    """Proxy one generation to this node's local Ollama and report measured tok/s.

    Native-generate backends route a whole request here (the model runs entirely on
    this node), so we surface Ollama's own eval_count / eval_duration — the reported
    tokens/sec is measured on the box that did the work, never estimated.
    """
    if state.donation.reclaimed:
        raise HTTPException(status_code=503, detail="node draining (reclaimed by owner)")
    url = f"{state.ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": False,
        "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as c:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            body = r.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502,
                            detail=f"ollama generate failed: {str(exc)[:200]}")
    eval_count = int(body.get("eval_count", 0) or 0)
    eval_ns = int(body.get("eval_duration", 0) or 0)
    tps = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
    return {
        "text": body.get("response", ""),
        "tokens": eval_count,
        "tokens_per_sec": round(tps, 2),
        "node": state.name,
        "model": req.model,
    }


def main() -> None:
    import uvicorn
    st = get_state()
    uvicorn.run(app, host=st.bind_host, port=st.port, log_level="warning")


if __name__ == "__main__":
    main()
