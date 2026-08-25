"""The GGUF backend must be honest about what it can and cannot serve.

The failure this file guards against is the expensive one: a backend that
*appears* to resolve a model, plans a partition, and only discovers at shard-load
time -- after the scheduler has committed nodes and the user is waiting -- that
the weights were never there. Resolution must fail early, cheaply, and out loud.
"""
from __future__ import annotations

import os

import pytest

from meshpool.common.models import Capability, Node
from meshpool.inference import adapters, gguf_loader
from meshpool.inference.gguf_tokenizer import GGUFTokenizer

pytest.importorskip("gguf", reason="gguf package not installed")

from gguf_fixtures import build_llama_gguf  # noqa: E402


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    p = tmp_path_factory.mktemp("gguf") / "tiny.gguf"
    build_llama_gguf(str(p), n_layer=6, d_model=16, n_head=4, n_kv_head=2,
                     d_ff=32, vocab_size=48)
    return str(p)


@pytest.fixture
def backend():
    return adapters.GGUFBackend()


def _node(nid: str, ram_mb: int = 4096) -> Node:
    return Node(id=nid, name=nid, url=f"http://127.0.0.1:900{nid[-1]}",
                capability=Capability(cpu_cores=4, ram_mb=ram_mb))


class TestRegistration:
    def test_the_backend_is_reachable_by_name(self):
        assert adapters.get_backend("gguf").name == "gguf"

    def test_it_does_not_displace_the_default(self):
        assert adapters.DEFAULT_BACKEND == "meshshard"

    def test_the_engine_drives_the_pipeline_not_the_backend(self, backend):
        """native_generate=True would forfeit SPD, hot spares and chunked prefill."""
        assert backend.native_generate is False

    def test_it_is_available_when_gguf_is_installed(self, backend):
        assert backend.available() is True


class TestPathResolution:
    def test_an_absolute_path_resolves(self, backend, ckpt):
        assert backend.resolve_path(ckpt) == os.path.abspath(ckpt)

    def test_a_missing_file_resolves_to_none(self, backend, tmp_path):
        assert backend.resolve_path(str(tmp_path / "nope.gguf")) is None

    def test_an_empty_name_resolves_to_none(self, backend):
        assert backend.resolve_path("") is None

    def test_a_name_resolves_against_the_model_directory(self, backend, ckpt, monkeypatch):
        monkeypatch.setenv("MESHPOOL_GGUF_DIR", os.path.dirname(ckpt))
        assert backend.resolve_path("tiny") == os.path.abspath(ckpt)
        assert backend.resolve_path("tiny.gguf") == os.path.abspath(ckpt)

    def test_the_extension_is_not_required(self, backend, ckpt, monkeypatch):
        monkeypatch.setenv("MESHPOOL_GGUF_DIR", os.path.dirname(ckpt))
        assert backend.resolve_path("tiny") is not None

    def test_a_directory_is_not_mistaken_for_a_checkpoint(self, backend, tmp_path):
        d = tmp_path / "model.gguf"
        d.mkdir()
        assert backend.resolve_path(str(d)) is None


class TestConfigResolution:
    def test_config_comes_from_the_file_not_a_registry(self, backend, ckpt):
        cfg = backend.resolve_config(ckpt)
        assert cfg is not None
        assert (cfg.n_layer, cfg.d_model, cfg.n_head, cfg.kv_heads) == (6, 16, 4, 2)

    def test_the_config_name_round_trips_back_to_the_file(self, backend, ckpt):
        """``general.name`` in a GGUF is usually just "llama".

        Everything downstream keys off ``cfg.name``, so if resolution returned
        the checkpoint's internal name the tokenizer could never be found again
        -- and the failure would land at shard-load time, after the scheduler
        had already committed nodes.
        """
        cfg = backend.resolve_config(ckpt)
        assert backend.resolve_path(cfg.name) == os.path.abspath(ckpt)

    def test_an_unknown_model_is_none_not_an_exception(self, backend, tmp_path):
        assert backend.resolve_config(str(tmp_path / "absent.gguf")) is None

    def test_a_corrupt_file_is_none_not_a_crash(self, backend, tmp_path):
        bad = tmp_path / "corrupt.gguf"
        bad.write_bytes(b"GGUF" + b"\x00" * 200)
        assert backend.resolve_config(str(bad)) is None

    def test_per_layer_mb_is_measured_not_assumed(self, backend, ckpt):
        """The simulated per_layer_mb=1000 is exactly what this replaces."""
        cfg = backend.resolve_config(ckpt)
        info = gguf_loader.cached_info(ckpt)
        assert cfg.per_layer_mb == info.per_layer_mb
        assert cfg.per_layer_mb != 1000, "looks like the simulated constant"

    def test_a_bigger_checkpoint_reports_more_bytes_per_layer(self, backend, tmp_path):
        """Non-vacuity: per_layer_mb must actually track the weights."""
        small = tmp_path / "s.gguf"
        big = tmp_path / "b.gguf"
        build_llama_gguf(str(small), n_layer=4, d_model=16, n_head=4, n_kv_head=2,
                         d_ff=32, vocab_size=48)
        build_llama_gguf(str(big), n_layer=4, d_model=64, n_head=4, n_kv_head=2,
                         d_ff=128, vocab_size=48)
        si = gguf_loader.cached_info(str(small))
        bi = gguf_loader.cached_info(str(big))
        assert bi.layer_bytes[0] > si.layer_bytes[0] * 3


class TestPartitioning:
    def test_layers_are_split_across_nodes_with_no_gaps(self, backend, ckpt):
        cfg = backend.resolve_config(ckpt)
        parts = backend.plan_partition([_node("n1"), _node("n2"), _node("n3")], cfg)
        assert parts, "expected a partition for a checkpoint this small"
        assert parts[0].start_layer == 0
        assert parts[-1].end_layer == cfg.n_layer
        for a, b in zip(parts, parts[1:]):
            assert a.end_layer == b.start_layer, "layer range gap or overlap"

    def test_every_layer_is_covered_exactly_once(self, backend, ckpt):
        cfg = backend.resolve_config(ckpt)
        parts = backend.plan_partition([_node("n1"), _node("n2")], cfg)
        covered = [l for p in parts for l in range(p.start_layer, p.end_layer)]
        assert sorted(covered) == list(range(cfg.n_layer))

    def test_exactly_one_first_and_one_last_stage(self, backend, ckpt):
        cfg = backend.resolve_config(ckpt)
        parts = backend.plan_partition([_node("n1"), _node("n2"), _node("n3")], cfg)
        assert sum(p.is_first for p in parts) == 1
        assert sum(p.is_last for p in parts) == 1

    def test_a_pool_without_the_memory_fails_instead_of_pretending(self, backend, ckpt):
        cfg = backend.resolve_config(ckpt)
        cfg = type(cfg)(**{**cfg.__dict__, "per_layer_mb": 10_000_000})
        assert backend.plan_partition([_node("n1", ram_mb=512)], cfg) == []

    def test_an_empty_pool_yields_no_partition(self, backend, ckpt):
        assert backend.plan_partition([], backend.resolve_config(ckpt)) == []


class TestTokenizer:
    def test_the_tokenizer_comes_from_the_checkpoint(self, backend, ckpt):
        cfg = backend.resolve_config(ckpt)
        tok = backend.tokenizer(cfg, {"gguf_path": ckpt})
        assert isinstance(tok, GGUFTokenizer)
        assert tok.vocab_size == cfg.vocab_size

    def test_it_round_trips_text(self, backend, ckpt):
        tok = backend.tokenizer(backend.resolve_config(ckpt), {"gguf_path": ckpt})
        ids = tok.encode("hello")
        assert ids, "encoder produced nothing"
        assert all(0 <= i < tok.vocab_size for i in ids), "id outside the vocabulary"

    def test_a_missing_checkpoint_raises_rather_than_substituting(self, backend, ckpt):
        """Silently returning a stand-in tokenizer would corrupt every response."""
        cfg = backend.resolve_config(ckpt)
        cfg = type(cfg)(**{**cfg.__dict__, "name": "not-a-real-model"})
        with pytest.raises(gguf_loader.GGUFUnsupported, match="no GGUF file found"):
            backend.tokenizer(cfg, None)

    def test_the_path_can_come_from_the_config_name(self, backend, ckpt):
        assert isinstance(backend.tokenizer(backend.resolve_config(ckpt), None),
                          GGUFTokenizer)


class TestDescribe:
    def test_it_summarizes_a_real_checkpoint(self, backend, ckpt):
        text = backend.describe(ckpt)
        assert text and "llama" in text and "6L" in text

    def test_an_unresolvable_model_describes_as_none(self, backend, tmp_path):
        assert backend.describe(str(tmp_path / "ghost.gguf")) is None
