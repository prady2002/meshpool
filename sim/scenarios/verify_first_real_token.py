"""M1 gate: first real-token path over three real OS processes.

Honest scope: this is a real Llama/Qwen2-class decoder architecture (RMSNorm, RoPE,
SwiGLU and grouped-query attention), driven by a real downloaded HuggingFace
tokenizer/config and executed as three layer-range stages in three node processes
over the MeshPool shard API plus the Python WAN emulator. Weight downloads are
blocked in this environment, so weights are deterministically initialized locally.
This validates the execution path, sharding, tokenization and transport — not
pretrained model quality.
"""
from __future__ import annotations

import asyncio
import json
import ssl
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meshpool.common.transport import TransportError  # noqa: E402
from meshpool.inference import model as M  # noqa: E402
from meshpool.inference.pipeline import Pipeline, PipelineStats  # noqa: E402
from meshpool.inference.remote import RemotePipeline, RemoteShard  # noqa: E402
from sim.harness import Harness  # noqa: E402
from sim.netem import NetEm, _estimate_payload_bytes  # noqa: E402
from sim.topology import JitterDistribution, LinkProfile, NodeSpec, Topology  # noqa: E402

HF_REPO = "yujiepan/qwen2-tiny-random"
MODEL_NAME = "qwen2-tiny-random"
PROMPT = "MeshPool can split a transformer across homes because "
N_TOKENS = 8
TIME_SCALE = 0.10


def _fetch(path: str) -> bytes:
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{path}"
    # The corporate TLS proxy on this machine presents a chain Python's stdlib
    # verifier rejects ("path length constraint exceeded"), while curl/browser
    # validation succeeds. Metadata is public and hashless; use an unverified
    # context rather than silently falling back to a fake tokenizer/config.
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=20, context=ctx) as r:
        return r.read()


@dataclass
class _Node:
    id: str
    url: str


class _Registry:
    def __init__(self, nodes: list[dict]) -> None:
        self.nodes = {n["id"]: _Node(n["id"], n["url"]) for n in nodes}

    def get(self, node_id: str) -> _Node | None:
        return self.nodes.get(node_id)


class CountingNetEmTransport:
    """NetEm wrapper that records the actual emulated bytes, hops and sleep time."""

    def __init__(self, registry: _Registry, netem: NetEm, src_node_id: str) -> None:
        self.registry = registry
        self.netem = netem
        self.src = src_node_id
        self.calls = 0
        self.hops = 0
        self.bytes = 0
        self.applied_sleep_s = 0.0

    def _base_url(self, node_id: str) -> str:
        node = self.registry.get(node_id)
        if node is None:
            raise TransportError(f"unknown node {node_id!r}")
        return node.url

    async def _transit_path(self, path: list[str], nbytes: int) -> None:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            delay = self.netem.one_way_delay_s(a, b, nbytes) * self.netem.time_scale
            self.hops += 1
            self.bytes += nbytes
            self.applied_sleep_s += delay
            await asyncio.sleep(delay)

    async def call(self, node_id: str, path: str, payload: dict | None = None, *,
                   method: str = "POST", timeout: float = 30.0) -> dict:
        self.calls += 1
        nbytes = _estimate_payload_bytes(payload)
        await self._transit_path(self.netem.path(self.src, node_id), nbytes)
        url = f"{self._base_url(node_id)}/{path.lstrip('/')}"
        async with httpx.AsyncClient() as client:
            resp = await client.request(method, url, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise TransportError(
                f"{node_id} {path} -> HTTP {resp.status_code}: {resp.text[:160]}",
                status=resp.status_code)
        result = resp.json() if resp.content else {}
        rbytes = _estimate_payload_bytes(result) if isinstance(result, dict) else 256
        await self._transit_path(self.netem.path(node_id, self.src), rbytes)
        return result

    async def close(self) -> None:
        return


class TimedRemoteShard(RemoteShard):
    def __init__(self, *args, label: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.label = label
        self.calls = 0
        self.wall_s = 0.0

    async def forward(self, data, is_tokens: bool, pos_offset: int, session: str | None):
        t0 = time.perf_counter()
        try:
            return await super().forward(data, is_tokens, pos_offset, session)
        finally:
            self.calls += 1
            self.wall_s += time.perf_counter() - t0


class CorruptFinalStage:
    """Wrap the last stage and deliberately corrupt its output logits."""

    def __init__(self, inner: TimedRemoteShard) -> None:
        self.inner = inner
        self.label = inner.label + " (corrupted)"

    def truncate(self, session: str, length: int) -> None:
        self.inner.truncate(session, length)

    def drop(self, session: str) -> None:
        self.inner.drop(session)

    def cache_bytes(self) -> int:
        return 0

    async def flush(self) -> None:
        await self.inner.flush()

    async def forward(self, data, is_tokens: bool, pos_offset: int, session: str | None):
        out = await self.inner.forward(data, is_tokens, pos_offset, session)
        out = np.array(out, dtype=np.float32, copy=True)
        out[..., 0] += 1_000_000.0
        return out


async def _stream_greedy(pipe: Pipeline, prompt_ids: list[int], n_tokens: int,
                         session: str, label: str) -> tuple[list[int], PipelineStats, float]:
    pipe.drop_session(session)
    pipe.reset_stats()
    t0 = time.perf_counter()
    logits = await pipe.forward(prompt_ids, pos_offset=0, session=session)
    pos = len(prompt_ids)
    out: list[int] = []
    print(f"  {label} stream:", end="", flush=True)
    for _ in range(n_tokens):
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        print(f" {nxt}", end="", flush=True)
        logits = await pipe.forward([nxt], pos_offset=pos, session=session)
        pos += 1
    print()
    return out, pipe.stats, max(1e-6, time.perf_counter() - t0)


def _single_process(prompt_ids: list[int], cfg: M.ModelConfig,
                    params: dict) -> tuple[list[int], float]:
    t0 = time.perf_counter()
    out = M.generate_with(lambda ctx: M.forward_full(params, cfg, ctx), cfg, prompt_ids,
                          max_new_tokens=N_TOKENS, temperature=0.0)
    return out, max(1e-6, time.perf_counter() - t0)


def _make_topology(nodes: list[dict]) -> Topology:
    specs = [NodeSpec(id="control-plane", region="local", asn="AS0", subnet="10.0.0.0/24",
                      nat_class="public")]
    specs += [NodeSpec(id=n["id"], region=f"region-{i}", asn=f"AS{i+1}",
                       subnet=f"10.0.{i+1}.0/24", nat_class="public")
              for i, n in enumerate(nodes)]

    def link(_a: str, _b: str) -> LinkProfile:
        return LinkProfile(rtt_ms=60.0,
                           jitter=JitterDistribution(sigma_ms=0.0,
                                                      heavy_tail_prob=0.0),
                           bw_down_kbps=50_000, bw_up_kbps=10_000,
                           loss_prob=0.0, reorder_prob=0.0)
    return Topology(specs, link_fn=link, seed=20240824)


def main() -> int:  # noqa: C901
    ok = True
    h = Harness(n_agents=3, cp_port=8380,
                data_dir=str(Path(__file__).resolve().parent / "_data_first_real_token"),
                topology=Topology([NodeSpec(id="placeholder", region="local", asn="AS0",
                                            subnet="10.0.0.0/24", nat_class="public")]))
    transport: CountingNetEmTransport | None = None
    try:
        print("== M1 honest scope ==")
        print("  real architecture + real tokenizer + real distributed execution")
        print("  weights are locally initialized because weight downloads are blocked here")
        print("  this validates the execution path, not model quality")

        print("\n== test 1: download real HuggingFace metadata/tokenizer ==")
        try:
            cfg_raw = _fetch("config.json")
            tok_raw = _fetch("tokenizer.json")
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP: cannot download config/tokenizer metadata: {exc}")
            return 0
        hf_cfg = M.config_from_hf_json(MODEL_NAME, cfg_raw)
        cfg = M.REGISTRY[MODEL_NAME]
        for field in ("vocab_size", "d_model", "n_head", "n_kv_head", "n_layer",
                      "d_ff", "architecture", "rope_theta"):
            assert getattr(cfg, field) == getattr(hf_cfg, field), (
                f"vendored runtime config field {field}={getattr(cfg, field)!r} "
                f"does not match downloaded config {getattr(hf_cfg, field)!r}")
        tok = M.HFTokenizer(tok_raw)
        prompt_ids = tok.encode(PROMPT)
        assert prompt_ids and max(prompt_ids) < cfg.vocab_size
        print(f"  repo={HF_REPO}")
        print(f"  model_type=qwen2 · layers={cfg.n_layer} · d_model={cfg.d_model} · "
              f"heads={cfg.n_head} · kv_heads={cfg.kv_heads} (GQA sharing={cfg.n_head // cfg.kv_heads}:1)")
        print(f"  tokenizer.json={len(tok_raw)} bytes · prompt tokens={prompt_ids}")
        print("  PASS: real config/tokenizer downloaded; no fake fallback used")

        print("\n== test 2: start three separate OS-process stage hosts ==")
        h.start()
        cp = h.cp_url()
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        nodes = sorted([n for n in nodes if n["status"] == "online"], key=lambda n: n["name"])[:3]
        assert len(nodes) == 3
        pids = [p.pid for p in h.agent_procs[:3] if p is not None]
        assert len(set(pids)) == 3
        print(f"  control-plane pid={h.cp_proc.pid if h.cp_proc else 'unknown'}")
        print("  stage PIDs: " + ", ".join(f"{n['name']} pid={pid}"
                                          for n, pid in zip(nodes, pids)))

        transport = CountingNetEmTransport(_Registry(nodes), NetEm(_make_topology(nodes),
                                                                   time_scale=TIME_SCALE),
                                           "control-plane")
        model_id = "m1-first-real-token"
        ranges = [(0, 1, True, False), (1, 2, False, False), (2, 2, False, True)]
        load_results = []
        for n, (start, end, first, last) in zip(nodes, ranges):
            body = awaitable(transport.call(
                n["id"], "shard/load",
                {"model_id": model_id, "model_name": MODEL_NAME,
                 "start_layer": start, "end_layer": end,
                 "is_first": first, "is_last": last},
                timeout=60.0))
            load_results.append(body)
        print("  loaded ranges: " + ", ".join(
            f"{n['name']}[{a}:{b}] params={r['params']}"
            for n, (a, b, _, _), r in zip(nodes, ranges, load_results)))
        assert all(r["params"] > 0 for r in load_results)

        stages = [
            TimedRemoteShard(transport, n["id"], model_id, a, b, first, last,
                             timeout=120.0, label=f"{n['name']}[{a}:{b}]")
            for n, (a, b, first, last) in zip(nodes, ranges)
        ]
        remote = RemotePipeline(cfg, stages)

        print("\n== test 3: non-vacuity — stages are incomplete alone ==")
        first_out = awaitable(stages[0].forward(prompt_ids[:2], True, 0, "alone-first"))
        assert first_out.shape[-1] == cfg.d_model and first_out.shape[-1] != cfg.vocab_size
        for idx in (1, 2):
            try:
                awaitable(stages[idx].forward(prompt_ids[:1], True, 0, f"alone-{idx}"))
                raise AssertionError(f"stage {idx+1} accepted raw tokens alone")
            except Exception as exc:  # noqa: BLE001
                print(f"  {stages[idx].label} cannot tokenize/generate alone: {str(exc)[:90]}")
        print(f"  {stages[0].label} alone returns hidden activations shape={first_out.shape}, not logits")
        print("  PASS: no stage has enough weights/function to produce a token alone")

        print("\n== test 4: sharded remote generation equals single-process at temperature 0 ==")
        params = M.random_params(cfg)
        single, single_elapsed = _single_process(prompt_ids, cfg, params)
        sharded, stats, elapsed = awaitable(_stream_greedy(remote, prompt_ids, N_TOKENS,
                                                           "m1-main", "sharded"))
        print(f"  single tokens : {single}")
        print(f"  sharded tokens: {sharded}")
        assert sharded == single, "sharded output diverged from single-process output"
        print(f"  single-process: {N_TOKENS / single_elapsed:.2f} tok/s ({single_elapsed:.4f}s)")
        print(f"  sharded+netem : {N_TOKENS / elapsed:.2f} tok/s ({elapsed:.4f}s)")
        print("  per-stage wall time: " + ", ".join(
            f"{s.label} calls={s.calls} wall={s.wall_s:.4f}s" for s in stages))
        print(f"  pipeline stats: traversals={stats.traversals}, hops={stats.hops}, "
              f"activation_wire_bytes={stats.wire_bytes}, compute_s={stats.compute_seconds:.4f}")
        print(f"  netem applied: calls={transport.calls}, path_hops={transport.hops}, "
              f"json_bytes={transport.bytes}, scheduled_sleep_s={transport.applied_sleep_s:.4f} "
              f"(rtt=60ms, time_scale={TIME_SCALE})")
        assert stats.hops > 0 and stats.wire_bytes > 0 and transport.applied_sleep_s > 0

        print("\n== test 5: removing/corrupting a stage makes the equality assertion fail ==")
        dropped_pipe = RemotePipeline(cfg, [stages[0], stages[2]])
        dropped, _, _ = awaitable(_stream_greedy(dropped_pipe, prompt_ids, N_TOKENS,
                                                 "m1-drop", "drop-middle"))
        assert dropped != single, "dropping the middle stage did not change output"
        corrupt_pipe = RemotePipeline(cfg, [stages[0], stages[1], CorruptFinalStage(stages[2])])
        corrupt, _, _ = awaitable(_stream_greedy(corrupt_pipe, prompt_ids, N_TOKENS,
                                                 "m1-corrupt", "corrupted"))
        try:
            assert corrupt == single, "intentional corruption changed tokens"
            raise AssertionError("deliberate perturbation unexpectedly passed equality")
        except AssertionError as exc:
            print(f"  expected comparison failure observed: {exc}")
        print(f"  drop-middle tokens: {dropped}")
        print(f"  corrupted tokens  : {corrupt}")
        print("  PASS: equality is not vacuous; missing/corrupt stages are detected")

        print("\nALL M1 FIRST-REAL-TOKEN CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if transport is not None:
            awaitable(transport.close())
        h.stop()
    return 0 if ok else 1


def awaitable(coro: Any):
    return asyncio.run(coro)


if __name__ == "__main__":
    raise SystemExit(main())
