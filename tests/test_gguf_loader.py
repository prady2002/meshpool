"""Real GGUF checkpoints, loaded and pipeline-sharded.

GGUF is what every published open-weight model ships as, so this is the seam
between "MeshPool runs our toy transformer" and "MeshPool runs the checkpoint
you downloaded". The fixtures are written with llama.cpp's own reference writer,
so the files are byte-exact GGUF; only the weight values are synthetic. See
``tests/gguf_fixtures.py`` for why that is the honest maximum here.

The assertions that matter most:

* the rope permutation is inverted correctly -- getting this wrong yields
  fluent-sounding garbage rather than a crash, so it is checked against a
  verbatim copy of the reference permutation rather than against itself;
* a stage loads *only* its layers, which is the property that lets a node hold a
  slice of a model far bigger than its RAM;
* sharded output is bit-identical to single-process output, which is what makes
  the distribution invisible to the caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shutil
import tempfile

from gguf_fixtures import build_llama_gguf, build_moe_llama_gguf, permute_rope_reference
from meshpool.inference import gguf_loader as G
from meshpool.inference import model as M

pytest.importorskip("gguf", reason="GGUF support needs the reference gguf package")


GEOM = dict(n_layer=6, d_model=32, n_head=4, n_kv_head=2, d_ff=64, vocab_size=64)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("gguf") / "model.gguf")
    truth = build_llama_gguf(path, **GEOM)
    return path, truth, G.read_info(path)


@pytest.fixture(scope="module")
def moe_ckpt(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("moe_gguf") / "moe.gguf")
    truth = build_moe_llama_gguf(
        path, n_layer=6, d_model=32, n_head=4, n_kv_head=2, d_ff=32,
        expert_count=4, expert_used_count=2, expert_ff=768, shared_ff=16,
        vocab_size=128, dense_layers=(0,), seed=20240820)
    return path, truth, G.read_info(path)


class TestRopePermutation:
    """llama.cpp rotates adjacent pairs; MeshPool rotates halves. The files carry
    a permutation to bridge that, and it has to be undone exactly."""

    @pytest.mark.parametrize("n_head,rows", [(4, 32), (2, 16), (8, 64), (1, 8)])
    def test_unpermute_inverts_the_reference_permutation(self, n_head, rows):
        rng = np.random.default_rng(7)
        w = rng.standard_normal((rows, 11)).astype(np.float32)
        assert np.array_equal(G.unpermute_rope(permute_rope_reference(w, n_head), n_head), w)

    def test_the_permutation_is_not_a_no_op(self):
        """Guards the test above: if permute did nothing, inverting it would
        pass trivially and prove nothing about the loader."""
        rng = np.random.default_rng(7)
        w = rng.standard_normal((32, 11)).astype(np.float32)
        assert not np.array_equal(permute_rope_reference(w, 4), w)

    def test_a_ragged_head_split_is_refused_not_guessed(self):
        with pytest.raises(G.GGUFUnsupported):
            G.unpermute_rope(np.zeros((10, 4), np.float32), 4)

    def test_q_and_k_are_unpermuted_with_their_own_head_counts(self, ckpt):
        """Under GQA, K has fewer rows than Q. Using the query head count for
        both would mis-group K and corrupt every key vector."""
        path, truth, info = ckpt
        P = G.load_params(path, 0, 1, True, False, info=info)
        assert np.array_equal(P["h0.attn.Wq"], truth["blk.0.attn_q.weight"].T)
        assert np.array_equal(P["h0.attn.Wk"], truth["blk.0.attn_k.weight"].T)
        assert info.config.kv_heads != info.config.n_head, "fixture must exercise GQA"


class TestMetadata:
    def test_geometry_comes_from_the_file(self, ckpt):
        _path, _truth, info = ckpt
        c = info.config
        assert (c.n_layer, c.d_model, c.n_head, c.kv_heads, c.d_ff) == (
            GEOM["n_layer"], GEOM["d_model"], GEOM["n_head"],
            GEOM["n_kv_head"], GEOM["d_ff"])
        assert c.vocab_size == GEOM["vocab_size"]
        assert c.architecture == "llama"

    def test_per_layer_mb_is_measured_not_assumed(self, ckpt):
        """Placement sizes nodes from this. The toy model hardcodes 1000 MB as a
        documented simulation; a real file must report its own footprint."""
        _path, _truth, info = ckpt
        assert info.per_layer_mb >= 1
        assert info.config.per_layer_mb == info.per_layer_mb
        assert info.config.per_layer_mb != M.ModelConfig().per_layer_mb
        assert len(info.layer_bytes) == GEOM["n_layer"]
        assert all(b > 0 for b in info.layer_bytes)

    def test_untied_output_head_is_detected(self, ckpt):
        _path, _truth, info = ckpt
        assert info.tied_embeddings is False

    def test_tied_output_head_is_detected(self, tmp_path):
        p = str(tmp_path / "tied.gguf")
        build_llama_gguf(p, tied=True, **GEOM)
        assert G.read_info(p).tied_embeddings is True

    def test_describe_is_informative(self, ckpt):
        _path, _truth, info = ckpt
        text = G.describe(info)
        assert "llama" in text and "6L" in text and "vocab=64" in text


class TestMoEMetadata:
    def test_dense_checkpoint_reports_no_experts_without_guessing(self, ckpt):
        _path, _truth, info = ckpt
        assert info.config.expert_count == 0
        assert info.config.expert_used_count == 0
        assert info.config.expert_feed_forward_length is None
        assert info.config.expert_shared_feed_forward_length is None
        assert all(not m.is_moe for m in info.moe_layers)

    def test_moe_metadata_comes_from_real_gguf_kv_fields(self, moe_ckpt):
        _path, _truth, info = moe_ckpt
        assert info.config.expert_count == 4
        assert info.config.expert_used_count == 2
        assert info.config.expert_feed_forward_length == 768
        assert info.config.expert_shared_feed_forward_length == 16
        assert [m.is_moe for m in info.moe_layers] == [False, True, True, True, True, True]

    def test_expert_tensor_names_are_attributed_to_their_layers(self, moe_ckpt):
        _path, _truth, info = moe_ckpt
        layer1 = info.moe_layers[1]
        assert layer1.expert_count == 4
        assert {
            "blk.1.ffn_gate_exps.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
            "blk.1.ffn_gate_shexp.weight",
            "blk.1.ffn_up_shexp.weight",
            "blk.1.ffn_down_shexp.weight",
        }.issubset(set(layer1.tensor_names))
        assert all(name.startswith("blk.1.") for name in layer1.tensor_names)

    def test_similar_tensor_names_are_not_misclassified_as_experts(self):
        assert G._is_moe_tensor("blk.1.ffn_gate_exps.weight")
        assert not G._is_moe_tensor("blk.1.not_ffn_gate_exps.weight")
        assert not G._is_moe_tensor("notblk.1.ffn_gate_exps.weight")

    def test_layer_bytes_include_expert_tensors(self, moe_ckpt):
        _path, truth, info = moe_ckpt
        dense = info.layer_bytes[0]
        moe = info.layer_bytes[1]
        expert_truth_bytes = sum(
            arr.nbytes for name, arr in truth.items()
            if name.startswith("blk.1.") and ("_exps.weight" in name or "_shexp.weight" in name))
        assert dense == 24_832
        assert moe == 1_198_848
        assert moe > dense * 40
        assert moe >= expert_truth_bytes
        assert G.layer_span_bytes(info, 1, 2) == moe

    def test_loading_moe_range_materializes_only_assigned_experts(self, moe_ckpt):
        path, _truth, info = moe_ckpt
        p = G.load_params(path, 1, 3, False, False, info=info)
        expected_suffixes = {"mlp.Wrouter", "mlp.Wgate_exps", "mlp.Wup_exps", "mlp.Wdown_exps",
                             "mlp.Wgate_shexp", "mlp.Wup_shexp", "mlp.Wdown_shexp"}
        for layer in (1, 2):
            loaded = {name.split(".", 1)[1] for name in p if name.startswith(f"h{layer}.")}
            assert expected_suffixes.issubset(loaded)
        assert not any(name.startswith("h0.") or name.startswith("h3.") for name in p)

    def test_declared_moe_without_expert_tensors_fails_loudly(self, tmp_path):
        p = str(tmp_path / "declared_but_dense.gguf")
        build_moe_llama_gguf(
            p, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=32,
            expert_count=4, expert_used_count=2, expert_ff=64, shared_ff=0,
            vocab_size=64, dense_layers=(0, 1))
        with pytest.raises(G.GGUFUnsupported, match="declares expert_count=4"):
            G.read_info(p)

    def test_loader_public_apis_release_windows_memmap_locks(self):
        tmpdir = tempfile.mkdtemp(prefix="meshpool_gguf_lock_")
        try:
            p = str(Path(tmpdir) / "lock.gguf")
            build_moe_llama_gguf(
                p, n_layer=3, d_model=32, n_head=4, n_kv_head=2, d_ff=32,
                expert_count=2, expert_used_count=1, expert_ff=64, shared_ff=16,
                vocab_size=64, dense_layers=(0,))
            info = G.read_info(p)
            params = G.load_params(p, 1, 2, False, False, info=info)
            vocab = G.read_vocab(p)
            assert params and vocab["tokens"]
            shutil.rmtree(tmpdir)
        except Exception:
            if Path(tmpdir).exists():
                shutil.rmtree(tmpdir, ignore_errors=True)
            raise


class TestPartialLoad:
    """The premise of the product: a node holds a slice, never the model."""

    @pytest.mark.parametrize("start,end", [(0, 2), (2, 4), (4, 6), (1, 5)])
    def test_only_the_requested_layers_are_materialised(self, ckpt, start, end):
        path, _truth, info = ckpt
        P = G.load_params(path, start, end, False, False, info=info)
        loaded = sorted({int(k.split(".")[0][1:]) for k in P if k.startswith("h")})
        assert loaded == list(range(start, end))

    def test_a_middle_stage_carries_no_embedding_or_head(self, ckpt):
        path, _truth, info = ckpt
        P = G.load_params(path, 2, 4, False, False, info=info)
        assert "wte" not in P and "lnf.g" not in P and "out" not in P

    def test_the_last_stage_carries_the_output_projection(self, ckpt):
        path, truth, info = ckpt
        P = G.load_params(path, 4, 6, False, True, info=info)
        assert np.array_equal(P["out"], truth["output.weight"])
        assert np.array_equal(P["lnf.g"], truth["output_norm.weight"])

    def test_a_range_outside_the_model_is_refused(self, ckpt):
        path, _truth, info = ckpt
        with pytest.raises(ValueError):
            G.load_params(path, 0, info.config.n_layer + 1, True, True, info=info)

    def test_layer_span_bytes_tracks_the_span(self, ckpt):
        _path, _truth, info = ckpt
        assert (G.layer_span_bytes(info, 0, 3) + G.layer_span_bytes(info, 3, 6)
                == G.layer_span_bytes(info, 0, 6))


class TestShardedEquivalence:
    """Distribution must be invisible: same tokens in, same logits out."""

    @staticmethod
    def _run(path, info, spans, tokens):
        h, is_tokens = tokens, True
        for i, (a, b) in enumerate(spans):
            P = G.load_params(path, a, b, i == 0, i == len(spans) - 1, info=info)
            shard = M.ShardModule(info.config, a, b, i == 0, i == len(spans) - 1, params=P)
            h = shard.forward(h, is_tokens=is_tokens)
            is_tokens = False
        return h

    @pytest.mark.parametrize("spans", [
        [(0, 6)],
        [(0, 3), (3, 6)],
        [(0, 2), (2, 4), (4, 6)],
        [(0, 1), (1, 5), (5, 6)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    ])
    def test_any_partitioning_gives_identical_logits(self, ckpt, spans):
        path, _truth, info = ckpt
        tokens = [1, 5, 9, 13, 21]
        full = G.load_full_params(path, info)
        reference = M.ShardModule(info.config, 0, info.config.n_layer, True, True,
                                  params=full).forward(tokens, is_tokens=True)
        got = self._run(path, info, spans, tokens)
        assert got.shape == (len(tokens), info.config.vocab_size)
        assert np.array_equal(reference, got), (
            f"{len(spans)}-stage split diverged, max |delta| = "
            f"{np.abs(reference - got).max()}")

    def test_the_logits_are_actually_model_output_not_zeros(self, ckpt):
        """Non-vacuity: an all-zero or constant tensor would satisfy every
        equality above while proving the model never ran."""
        path, _truth, info = ckpt
        out = self._run(path, info, [(0, 3), (3, 6)], [1, 5, 9, 13, 21])
        assert np.isfinite(out).all()
        assert out.std() > 1e-6
        assert len({int(r.argmax()) for r in out}) > 1


class TestQuantized:
    """Published checkpoints are quantized; this is the path they take."""

    QUANTS = ["Q8_0", "Q5_1", "Q4_0", "F16"]

    @pytest.mark.parametrize("quant", QUANTS)
    def test_a_quantized_checkpoint_loads_and_shards_identically(self, tmp_path, quant):
        p = str(tmp_path / f"{quant}.gguf")
        build_llama_gguf(p, quantize=quant, n_layer=4, d_model=64, n_head=8,
                         n_kv_head=4, d_ff=128, vocab_size=128)
        info = G.read_info(p)
        assert quant in info.quantizations, (
            f"fixture did not actually store {quant}: {sorted(info.quantizations)}")

        full = G.load_full_params(p, info)
        tokens = [1, 5, 9, 13]
        ref = M.ShardModule(info.config, 0, 4, True, True, params=full).forward(
            tokens, is_tokens=True)
        h, is_tokens = tokens, True
        for i, (a, b) in enumerate([(0, 2), (2, 4)]):
            P = G.load_params(p, a, b, i == 0, i == 1, info=info)
            h = M.ShardModule(info.config, a, b, i == 0, i == 1,
                              params=P).forward(h, is_tokens=is_tokens)
            is_tokens = False
        assert np.array_equal(ref, h)
        assert np.isfinite(ref).all()

    def test_dequantization_error_follows_the_bit_budget(self, tmp_path):
        """A codec that silently returned garbage, or one that ignored the
        quantization entirely, would not reproduce this ordering."""
        errs = {}
        for quant in ("Q8_0", "Q5_1", "Q4_0"):
            p = str(tmp_path / f"err_{quant}.gguf")
            truth = build_llama_gguf(p, quantize=quant, n_layer=2, d_model=64,
                                     n_head=8, n_kv_head=4, d_ff=128, vocab_size=128)
            P = G.load_params(p, 0, 1, False, False)
            want = truth["blk.0.attn_v.weight"].T           # V carries no permutation
            errs[quant] = float(np.abs(P["h0.attn.Wv"] - want).max() / np.abs(want).max())
        assert errs["Q8_0"] < errs["Q5_1"] < errs["Q4_0"], errs
        assert errs["Q8_0"] < 0.02, errs
        assert errs["Q4_0"] < 0.30, errs


class TestFailsLoudly:
    """Working agreement: a missing capability raises, it never guesses."""

    def test_a_file_without_an_architecture_is_refused(self, tmp_path):
        import gguf
        p = str(tmp_path / "noarch.gguf")
        w = gguf.GGUFWriter(p, "llama")
        w.add_tensor("t", np.zeros((2, 2), np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        import gguf as _g
        reader = _g.GGUFReader(p, "r")
        # The writer always stamps an architecture, so assert the guard directly
        # rather than pretending we can produce a file without one.
        assert "general.architecture" in reader.fields
        with pytest.raises(G.GGUFUnsupported):
            G._arch(type("R", (), {"fields": {}})())

    def test_a_missing_tensor_names_itself(self, ckpt):
        path, _truth, info = ckpt
        broken = M.ModelConfig(**{**info.config.__dict__, "n_layer": info.config.n_layer + 3})
        bad = G.GGUFInfo(path=path, architecture="llama", config=broken,
                         tied_embeddings=info.tied_embeddings,
                         quantizations=info.quantizations,
                         layer_bytes=info.layer_bytes + [0, 0, 0],
                         total_bytes=info.total_bytes)
        with pytest.raises(G.GGUFUnsupported, match="blk.7"):
            G.load_params(path, 7, 8, False, False, info=bad)


class TestArchitectureDetection:
    """slice_params used to identify the architecture by looking the dimensions
    up in a registry of models we ship, so any loaded checkpoint fell through to
    the GPT parameter names and died on a confusing KeyError."""

    def test_an_unregistered_llama_checkpoint_slices_correctly(self, ckpt):
        path, _truth, info = ckpt
        full = G.load_full_params(path, info)
        assert M.REGISTRY.get(info.config.name) is None, "fixture must be unregistered"
        sub = M.slice_params(full, 1, 3, False, False)
        assert sorted(sub) == sorted(
            f"h{i}.{s}" for i in (1, 2) for s in M._LLAMA_LAYER_SUFFIXES)

    def test_a_gpt_checkpoint_still_slices_with_gpt_names(self):
        cfg = M.REGISTRY["nano-gpt"]
        full = M.load_params(cfg)[0]
        sub = M.slice_params(full, 0, 2, False, False)
        assert any(k.endswith(".ln1.g") for k in sub)
        assert not any(k.endswith(".rms1.g") for k in sub)
