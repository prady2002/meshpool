"""Native-generate inference path: the Ollama backend + whole-request routing.

Unlike ``meshshard`` (which splits a toy net into per-node layer blocks and is driven
token-by-token by the engine), a *native-generate* backend owns its own generation
loop: the engine picks one node that already holds the model and routes the whole
request to that node's ``/ollama/generate`` endpoint. These tests pin that behaviour
with fakes (no live Ollama needed), covering:

  * backend selection precedence (a node advertising the model -> ollama; otherwise
    the built-in meshshard net; otherwise a clean error),
  * deploy produces one full-model *replica* per capable node (no shard/load hops),
  * chat routes to a replica, reports Ollama's measured tok/s, and credits the node,
  * round-robin spreads concurrent requests and fails over past a dead replica.
"""
from __future__ import annotations

import asyncio

import pytest

from meshpool.common.models import Capability, DonationPolicy, Node, NodeStatus
from meshpool.inference.adapters import ollama_node_has_model
from meshpool.inference.engine import InferenceManager


# --------------------------------------------------------------------------- #
# Fakes (mirror the style of tests/test_verify.py)
# --------------------------------------------------------------------------- #
def _node(nid: str, models: list[str], owner: str = "alice") -> Node:
    return Node(id=nid, name=nid, url=f"http://{nid}:9001", owner=owner,
                capability=Capability(cpu_cores=4, ram_mb=32000, ollama_models=models),
                policy=DonationPolicy(cpu_pct=100, ram_mb=16000))


class _Registry:
    def __init__(self, nodes: list[Node]):
        self._nodes = {n.id: n for n in nodes}

    def schedulable_nodes(self) -> list[Node]:
        return [n for n in self._nodes.values() if n.status is NodeStatus.ONLINE]

    def online(self) -> list[Node]:
        return self.schedulable_nodes()

    def get(self, nid: str):
        return self._nodes.get(nid)

    def drop(self, nid: str) -> None:
        n = self._nodes.get(nid)
        if n:
            n.status = NodeStatus.OFFLINE


class _Store:
    def __init__(self):
        self._m: dict = {}

    def put_model(self, dep):
        self._m[dep.id] = dep

    def get_model(self, mid):
        return self._m.get(mid)

    def list_models(self):
        return list(self._m.values())

    def delete_model(self, mid):
        self._m.pop(mid, None)


class _Ledger:
    def __init__(self):
        self.donations: list[tuple] = []

    def record_donation(self, node_id, owner, units, detail=""):
        self.donations.append((node_id, owner, units, detail))


class _Events:
    def __init__(self):
        self.published: list[tuple] = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))


class _OllamaTransport:
    """Echoes a plausible Ollama result; can be told to fail specific nodes."""

    def __init__(self, fail_nodes=()):
        self.fail_nodes = set(fail_nodes)
        self.calls: list[tuple] = []

    async def call(self, node_id, path, payload, method="POST", timeout=None):
        self.calls.append((node_id, path, payload))
        if path != "ollama/generate":
            raise AssertionError(f"native path must be ollama/generate, got {path!r}")
        if node_id in self.fail_nodes:
            raise RuntimeError(f"{node_id} unreachable")
        return {"text": f"[{node_id}] hi", "tokens": payload["max_tokens"],
                "tokens_per_sec": 42.0, "node": node_id, "model": payload["model"]}


def _mgr(nodes: list[Node], fail_nodes=()) -> InferenceManager:
    return InferenceManager(_Registry(nodes), _Store(), _Ledger(), _Events(),
                            _OllamaTransport(fail_nodes))


# --------------------------------------------------------------------------- #
# Model-name matching
# --------------------------------------------------------------------------- #
def test_ollama_model_matched_exact_and_by_base_name():
    n = _node("n1", ["qwen2.5-coder:14b"])
    assert ollama_node_has_model(n, "qwen2.5-coder:14b")   # exact name:tag
    assert ollama_node_has_model(n, "qwen2.5-coder")       # base name matches the tag
    assert not ollama_node_has_model(n, "llama3")


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def test_select_backend_prefers_ollama_then_meshshard_then_errors():
    mgr = _mgr([_node("n1", ["qwen2.5-coder:7b"])])
    nodes = mgr.registry.schedulable_nodes()
    assert mgr._select_backend("qwen2.5-coder:7b", nodes).name == "ollama"
    assert mgr._select_backend("nano-gpt", nodes).name == "meshshard"
    with pytest.raises(ValueError):
        mgr._select_backend("no-such-model", nodes)


def test_deploy_unknown_model_raises():
    mgr = _mgr([_node("n1", [])])  # node has no Ollama models
    with pytest.raises(ValueError):
        asyncio.run(mgr.deploy("mystery-model", "bob"))


# --------------------------------------------------------------------------- #
# Deploy = replicate the whole model across capable nodes (no layer sharding)
# --------------------------------------------------------------------------- #
def test_deploy_replicates_whole_model_per_node_without_shard_load():
    nodes = [_node("n1", ["qwen2.5-coder:7b"]), _node("n2", ["qwen2.5-coder:7b"])]
    mgr = _mgr(nodes)
    dep = asyncio.run(mgr.deploy("qwen2.5-coder:7b", "bob"))
    assert dep.backend == "ollama"
    assert dep.status == "ready"
    assert len(dep.partition) == 2                          # one full-model replica per node
    assert all(a.is_first and a.is_last for a in dep.partition)
    assert mgr.transport.calls == []                        # native: never calls /shard/load
    view = mgr.view(dep.id)
    assert view["backend"] == "ollama" and view["replicas"] == 2


def test_deploy_only_places_on_nodes_that_have_the_model():
    nodes = [_node("n1", ["qwen2.5-coder:7b"]), _node("n2", ["llama3"])]
    mgr = _mgr(nodes)
    dep = asyncio.run(mgr.deploy("qwen2.5-coder:7b", "bob"))
    assert [a.node_id for a in dep.partition] == ["n1"]     # n2 doesn't hold it


# --------------------------------------------------------------------------- #
# Chat routes a whole request to one replica and reports measured tok/s
# --------------------------------------------------------------------------- #
def test_native_chat_routes_reports_tps_and_credits_node():
    mgr = _mgr([_node("n1", ["qwen2.5-coder:7b"], owner="carol")])
    dep = asyncio.run(mgr.deploy("qwen2.5-coder:7b", "bob"))
    out = asyncio.run(mgr.chat(dep.id, "hello", max_new_tokens=8, temperature=0.5))
    assert out["backend"] == "ollama"
    assert out["served_by"] == "n1"
    assert out["tokens"] == 8
    assert out["tokens_per_sec"] == 42.0
    assert "[n1]" in out["text"]
    assert [c[1] for c in mgr.transport.calls] == ["ollama/generate"]
    # the request payload forwarded temperature + token budget to the node
    _, _, payload = mgr.transport.calls[0]
    assert payload["temperature"] == 0.5 and payload["max_tokens"] == 8
    # the serving node's owner was credited for the compute it donated
    assert mgr.ledger.donations and mgr.ledger.donations[0][0] == "n1"
    assert mgr.ledger.donations[0][1] == "carol"


def test_round_robin_spreads_requests_across_replicas():
    nodes = [_node("n1", ["m"]), _node("n2", ["m"])]
    mgr = _mgr(nodes)
    dep = asyncio.run(mgr.deploy("m", "bob"))
    servers = [asyncio.run(mgr.chat(dep.id, "hi", max_new_tokens=2))["served_by"]
               for _ in range(4)]
    assert servers == ["n1", "n2", "n1", "n2"]              # alternates


def test_native_chat_fails_over_past_a_dead_replica():
    nodes = [_node("n1", ["m"]), _node("n2", ["m"])]
    mgr = _mgr(nodes, fail_nodes=("n1",))
    dep = asyncio.run(mgr.deploy("m", "bob"))
    out = asyncio.run(mgr.chat(dep.id, "hi", max_new_tokens=4))
    assert out["served_by"] == "n2"                         # n1 failed, fell over to n2
    assert [c[0] for c in mgr.transport.calls] == ["n1", "n2"]


def test_native_chat_errors_when_all_replicas_down():
    nodes = [_node("n1", ["m"]), _node("n2", ["m"])]
    mgr = _mgr(nodes, fail_nodes=("n1", "n2"))
    dep = asyncio.run(mgr.deploy("m", "bob"))
    with pytest.raises(RuntimeError):
        asyncio.run(mgr.chat(dep.id, "hi", max_new_tokens=4))


def test_native_chat_skips_offline_replica():
    nodes = [_node("n1", ["m"]), _node("n2", ["m"])]
    mgr = _mgr(nodes)
    dep = asyncio.run(mgr.deploy("m", "bob"))
    mgr.registry.drop("n1")                                 # n1 goes offline after deploy
    out = asyncio.run(mgr.chat(dep.id, "hi", max_new_tokens=4))
    assert out["served_by"] == "n2"
    assert all(c[0] == "n2" for c in mgr.transport.calls)   # never even tried n1


# --------------------------------------------------------------------------- #
# Remove is a no-op on the wire for native backends (nothing to unload)
# --------------------------------------------------------------------------- #
def test_remove_native_deployment_skips_shard_unload():
    mgr = _mgr([_node("n1", ["m"])])
    dep = asyncio.run(mgr.deploy("m", "bob"))
    assert asyncio.run(mgr.remove(dep.id)) is True
    assert mgr.transport.calls == []                        # no /shard/unload for ollama
    assert mgr.get(dep.id) is None
