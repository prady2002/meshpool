"""Inference engine adapters (spec §9.6: "engine adapter interface").

The distributed-inference control flow — topology-aware partitioning, pipeline
orchestration, auto-repartition on churn, tokens/sec measurement, the OpenAI API —
is **backend-agnostic**. Only the tensor kernels (how a layer block is stored and how
a forward hop is computed) differ between backends. This module is that seam.

* ``MeshShardBackend`` (default, active): the from-scratch NumPy transformer sharded
  across agent ``/shard/*`` endpoints. Always runnable, zero downloads, and the basis
  for the bit-for-bit correctness oracle.
* ``GGUFBackend`` (active): **real checkpoints, really layer-sharded.** Each node
  memory-maps only its own layer range, so the pool can serve a model no single
  member could hold. This is the flagship path.
* ``OllamaBackend`` (active): whole-model replicas on nodes that already have the
  model. Aggregate throughput and failover, but bounded by one machine's memory.
* ``PetalsBackend`` / ``ExoBackend`` (documented stubs): drop-in points for other
  large-model runtimes. They implement the same interface; enabling one is a
  dependency + config change, not an engine rewrite.

Keeping this interface stable is what lets "prove the concept on a laptop" and "serve a
70B model across the office" share one control plane.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Optional, Protocol

from ..common.models import LayerAssignment, Node
from ..controlplane.scheduler import topology_partition
from . import gguf_loader
from . import model as M
from .gguf_tokenizer import GGUFTokenizer


def ollama_node_has_model(node: Node, model_name: str) -> bool:
    """True if ``node`` advertises ``model_name`` in its local Ollama library.

    Matches the exact ``name:tag`` first, then falls back to a tag-insensitive match
    on the base name (so ``qwen2.5-coder`` matches an installed ``qwen2.5-coder:14b``).
    """
    models = list(getattr(node.capability, "ollama_models", None) or [])
    if model_name in models:
        return True
    base = model_name.split(":", 1)[0].lower()
    return any(m.split(":", 1)[0].lower() == base for m in models)


class InferenceBackend(Protocol):
    """Contract the :class:`InferenceManager` relies on. A backend owns *what a shard
    is* and *how a forward hop computes*; the engine owns *orchestration*.

    Shard load / forward / unload themselves happen over the agent's HTTP ``/shard/*``
    contract (mirroring ``Agent.LoadShard`` / ``Agent.Forward`` in the proto), so a new
    backend only needs to teach the agent worker how to execute its kernels.
    """

    name: str
    # How the engine drives this backend:
    #  * False (meshshard): the engine runs the token-by-token pipeline itself,
    #    hopping hidden states across per-node layer shards via /shard/forward.
    #  * True  (ollama, llama.cpp RPC): the backend owns its own generation loop.
    #    The engine routes a whole request to one replica node and calls that
    #    agent's native generate endpoint; ``partition`` entries are full-model
    #    replicas, not layer blocks. This is the seam real LLMs plug into.
    native_generate: bool

    def resolve_config(self, model_name: str) -> Optional[M.ModelConfig]:
        """Model metadata (layer count, per-layer memory, vocab) or ``None`` if unknown."""
        ...

    def plan_partition(self, nodes: list[Node], cfg: M.ModelConfig) -> list[LayerAssignment]:
        """Topology-aware layer partition across ``nodes`` (empty if it doesn't fit)."""
        ...

    def tokenizer(self, cfg: M.ModelConfig, meta: Optional[dict]):
        """Tokenizer for this model (checkpoint may ship its own vocab)."""
        ...

    def available(self) -> bool:
        """Whether this backend's dependencies are importable in this environment."""
        ...


class MeshShardBackend:
    """Default backend: the NumPy transformer, pipeline-sharded over agent processes."""

    name = "meshshard"
    native_generate = False

    def resolve_config(self, model_name: str) -> Optional[M.ModelConfig]:
        return M.REGISTRY.get(model_name)

    def plan_partition(self, nodes: list[Node], cfg: M.ModelConfig) -> list[LayerAssignment]:
        return topology_partition(nodes, cfg.n_layer, cfg.per_layer_mb)

    def tokenizer(self, cfg: M.ModelConfig, meta: Optional[dict]):
        if meta and meta.get("tokenizer_json"):
            return M.HFTokenizer(meta["tokenizer_json"])
        itos = (meta or {}).get("itos")
        return M.CharTokenizer(itos=itos, vocab_size=cfg.vocab_size)

    def available(self) -> bool:
        return True


class _UnavailableBackend:
    """Base for documented-but-not-installed backends (Petals, exo)."""

    name = "unavailable"
    native_generate = False
    _install_hint = ""

    def available(self) -> bool:
        return False

    def _fail(self):
        raise NotImplementedError(
            f"{self.name} backend is a documented adapter stub. To enable it: "
            f"{self._install_hint} Then implement resolve_config/plan_partition to map "
            f"its layer/memory metadata, and teach the agent worker to load and run its "
            f"layer blocks behind the same /shard/* contract. The control plane "
            f"(partitioning, pipeline, failover, OpenAI API) is unchanged.")

    def resolve_config(self, model_name):  # noqa: D102
        self._fail()

    def plan_partition(self, nodes, cfg):  # noqa: D102
        self._fail()

    def tokenizer(self, cfg, meta):  # noqa: D102
        self._fail()


class PetalsBackend(_UnavailableBackend):
    """hivemind/Petals: swarm-hosted blocks of a large model (spec §13.2 research)."""
    name = "petals"
    _install_hint = "pip install petals hivemind torch transformers, and pick a base model."


class ExoBackend(_UnavailableBackend):
    """exo: peer-to-peer topology-aware sharding across heterogeneous devices."""
    name = "exo"
    _install_hint = "install exo + a torch/MLX runtime; reuse its topology auto-partitioner."


class OllamaBackend:
    """Route real-LLM inference to nodes that already run the model under Ollama.

    Unlike ``meshshard`` (which splits a toy net into per-node layer blocks), this
    backend serves genuinely large open-weight models by treating **each node that
    has the model as a full replica**. A deployment's ``partition`` therefore holds
    one full-span :class:`LayerAssignment` per replica node, and the engine load-
    balances whole requests across them (aggregate throughput + failover) rather
    than pipelining a single request — the honest win on a LAN/Wi-Fi pool where a
    model that fits on one box can't be made faster per-request by sharding it.

    No Python dependency: the agent proxies to a local Ollama server over HTTP, so
    this backend is ``available()`` wherever the pool has at least one Ollama node.
    """

    name = "ollama"
    native_generate = True

    def available(self) -> bool:
        return True

    def resolve_config(self, model_name: str) -> Optional[M.ModelConfig]:
        # Native routing only carries the model *name* forward; Ollama owns the
        # weights + kernels, so layer geometry is irrelevant. n_layer=0 marks
        # "not layer-sharded" and keeps this out of the topology partitioner.
        return M.ModelConfig(name=model_name, n_layer=0, per_layer_mb=0)

    def nodes_with_model(self, nodes: list[Node], model_name: str) -> list[Node]:
        return [n for n in nodes if ollama_node_has_model(n, model_name)]

    def plan_partition(self, nodes: list[Node], cfg: M.ModelConfig) -> list[LayerAssignment]:
        # Each capable node is a whole-model replica. Encoding a replica as a
        # full-span LayerAssignment reuses the existing ModelDeployment / dashboard
        # data model unchanged (is_first == is_last == True == "owns everything").
        replicas = self.nodes_with_model(nodes, cfg.name)
        return [
            LayerAssignment(node_id=n.id, url=n.url, start_layer=0, end_layer=cfg.n_layer,
                            is_first=True, is_last=True)
            for n in replicas
        ]

    def tokenizer(self, cfg: M.ModelConfig, meta: Optional[dict]) -> M.CharTokenizer:
        # Unused on the native path (the agent/Ollama tokenizes); returned only to
        # satisfy the Protocol for callers that ask unconditionally.
        return M.CharTokenizer(vocab_size=cfg.vocab_size)


class GGUFBackend:
    """Real checkpoints, genuinely split by layer range across separate nodes.

    This is the backend the flagship depends on. Unlike ``ollama`` (whole-model
    replicas, so the model must fit on one box) and unlike ``meshshard`` (real
    sharding, toy weights), this one is real on both axes: a node assigned layers
    40-48 of a 600 GB file loads *only* those layers and never pages in the rest.
    That is the property that makes a model no single machine can hold runnable
    on a pool of machines that individually cannot hold it.

    ``native_generate`` is False on purpose: the engine drives the pipeline,
    which is what lets speculative pipeline decoding, hot spares and chunked
    prefill apply. Handing generation to the backend would forfeit all three.

    Model names are filesystem paths to ``.gguf`` files, resolved against
    ``MESHPOOL_GGUF_DIR`` when relative. Nothing is downloaded implicitly --
    fetching multi-hundred-gigabyte weights is the Weight Swarm's job, not a
    side effect of naming a model.
    """

    name = "gguf"
    native_generate = False

    def available(self) -> bool:
        try:
            import gguf  # noqa: F401
        except ImportError:
            return False
        return True

    def resolve_path(self, model_name: str) -> Optional[str]:
        """Model name -> checkpoint path, or ``None`` if there is no such file."""
        if not model_name:
            return None
        cand = [model_name]
        if not model_name.lower().endswith(".gguf"):
            cand.append(model_name + ".gguf")
        root = os.environ.get("MESHPOOL_GGUF_DIR")
        if root:
            cand += [os.path.join(root, c) for c in list(cand)]
        for c in cand:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def resolve_config(self, model_name: str) -> Optional[M.ModelConfig]:
        if not self.available():
            return None
        path = self.resolve_path(model_name)
        if path is None:
            return None
        try:
            info = gguf_loader.cached_info(path)
        except gguf_loader.GGUFUnsupported:
            # An unreadable or unsupported checkpoint is "unknown model", not a
            # crash: the manager tries other backends and reports cleanly.
            return None
        # The checkpoint's own ``general.name`` is usually the architecture
        # ("llama"), which cannot be resolved back to a file. Everything
        # downstream -- deployments, the tokenizer lookup, the dashboard -- keys
        # off ``cfg.name``, so it must stay the name the caller actually used.
        return dataclasses.replace(info.config, name=model_name)

    def plan_partition(self, nodes: list[Node], cfg: M.ModelConfig) -> list[LayerAssignment]:
        """Partition using the checkpoint's *measured* per-layer size.

        ``cfg.per_layer_mb`` here comes from summing real tensor byte counts in
        the file, not from a constant. Placement therefore fails honestly when
        the pool genuinely lacks the memory instead of succeeding on paper.
        """
        return topology_partition(nodes, cfg.n_layer, cfg.per_layer_mb)

    def tokenizer(self, cfg: M.ModelConfig, meta: Optional[dict]):
        """The checkpoint's own tokenizer -- never a substitute.

        Pairing real weights with an approximate tokenizer produces text that is
        wrong in a way that reads as poor model quality, so an unsupported vocab
        raises rather than falling back to something that merely runs.
        """
        path = (meta or {}).get("gguf_path") or self.resolve_path(cfg.name)
        if path is None:
            raise gguf_loader.GGUFUnsupported(
                f"cannot build a tokenizer for {cfg.name!r}: no GGUF file found. "
                f"Set MESHPOOL_GGUF_DIR or pass an absolute path.")
        return GGUFTokenizer.from_vocab(gguf_loader.read_vocab(path))

    def describe(self, model_name: str) -> Optional[str]:
        """One-line summary for logs and the console, or ``None`` if unresolvable."""
        path = self.resolve_path(model_name)
        if path is None or not self.available():
            return None
        try:
            return gguf_loader.describe(gguf_loader.cached_info(path))
        except gguf_loader.GGUFUnsupported:
            return None


DEFAULT_BACKEND = "meshshard"
BACKENDS: dict[str, InferenceBackend] = {
    "meshshard": MeshShardBackend(),
    "gguf": GGUFBackend(),
    "ollama": OllamaBackend(),
    "petals": PetalsBackend(),
    "exo": ExoBackend(),
}


def get_backend(name: str = DEFAULT_BACKEND) -> InferenceBackend:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown inference backend {name!r}; choose one of {sorted(BACKENDS)}")
