"""Gate: MoE-aware placement keeps every expert of a layer co-resident.

Honest scope (measured in this process):
  * real GGUF container written by llama.cpp's reference writer;
  * synthetic weights because model downloads are blocked in this environment;
  * validates MoE metadata/tensor parsing, byte-aware placement, and bit-exact
    sharded NumPy execution -- not pretrained model quality or WAN throughput.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from meshpool.common.models import Capability, DonationPolicy, LayerAssignment, Node  # noqa: E402
from meshpool.controlplane.scheduler import explain_layer_bytes_fit, topology_partition_by_bytes  # noqa: E402
from meshpool.inference import gguf_loader as G  # noqa: E402
from meshpool.inference import model as M  # noqa: E402


def _node(nid: str, ram_mb: int = 2) -> Node:
    return Node(
        id=nid, name=nid, url=f"http://127.0.0.1:91{nid}",
        capability=Capability(cpu_cores=4, ram_mb=ram_mb, flops_estimate=1.0),
        policy=DonationPolicy(ram_mb=ram_mb),
    )


def _run_partition(path: str, info: G.GGUFInfo, parts: list[LayerAssignment], tokens: list[int]) -> np.ndarray:
    h: object = tokens
    is_tokens = True
    for p in parts:
        params = G.load_params(path, p.start_layer, p.end_layer, p.is_first, p.is_last, info=info)
        shard = M.ShardModule(info.config, p.start_layer, p.end_layer, p.is_first, p.is_last, params=params)
        h = shard.forward(h, is_tokens=is_tokens)
        is_tokens = False
    return np.asarray(h, dtype=np.float32)


def main() -> int:  # noqa: C901
    ok = True
    work = ROOT / ".meshpool_moe_gate"
    try:
        print("== honest scope ==")
        print("  real GGUF format + measured tensor bytes + layer-range sharding")
        print("  synthetic deterministic weights; no model-quality or WAN-throughput claim")

        print("\n== test 1: write and parse a real MoE GGUF checkpoint ==")
        from gguf_fixtures import build_moe_llama_gguf
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        path = str(work / "moe.gguf")
        build_moe_llama_gguf(
            path, n_layer=6, d_model=32, n_head=4, n_kv_head=2, d_ff=32,
            expert_count=4, expert_used_count=2, expert_ff=768, shared_ff=16,
            vocab_size=64, dense_layers=(0,))
        info = G.read_info(path)
        print(f"  {G.describe(info)}")
        print(f"  layer_bytes={info.layer_bytes}")
        assert info.moe_layers[0].is_moe is False
        assert all(m.is_moe for m in info.moe_layers[1:])
        assert info.config.expert_count == 4 and info.config.expert_used_count == 2
        print("  PASS: MoE metadata and expert tensors were detected from GGUF")

        print("\n== test 2: byte-aware partition keeps each MoE layer indivisible ==")
        nodes = [_node("n1"), _node("n2"), _node("n3"), _node("n4"), _node("n5")]
        parts = topology_partition_by_bytes(nodes, info.layer_bytes)
        assert parts, "checkpoint should fit this pool"
        G.validate_moe_expert_colocation(info, parts)
        for p in parts:
            used = G.layer_span_bytes(info, p.start_layer, p.end_layer)
            print(f"  {p.node_id}: layers [{p.start_layer}:{p.end_layer}) -> {used} bytes")
        print("  PASS: every MoE layer's router, experts, and shared experts have exactly one node owner")

        print("\n== test 3: placement is byte-balanced, not layer-balanced ==")
        cap = int(2 * 0.8 * 1024 * 1024)
        naive = [(0, 2), (2, 4), (4, 5), (5, 6)]
        naive_bytes = [G.layer_span_bytes(info, a, b) for a, b in naive]
        ours = [G.layer_span_bytes(info, p.start_layer, p.end_layer) for p in parts]
        assert any(b > cap for b in naive_bytes), naive_bytes
        assert all(b <= cap for b in ours), ours
        print(f"  naive uniform-layer bytes={naive_bytes}, node cap={cap}")
        print(f"  byte-aware bytes={ours}")
        print("  PASS: a uniform layer-count split would overload a node; measured-byte placement does not")

        print("\n== test 4: non-vacuity guards reject bad placements ==")
        split = {name: "n1" for name in info.moe_layers[1].tensor_names}
        split[info.moe_layers[1].tensor_names[0]] = "n2"
        try:
            G.validate_moe_expert_colocation(info, [LayerAssignment(node_id="n1", url="u", start_layer=0, end_layer=6)], split)
            raise AssertionError("expert-splitting placement was accepted")
        except ValueError as exc:
            print(f"  deliberate expert split rejected: {exc}")
        too_small = [_node("tiny", ram_mb=1)]
        fit = explain_layer_bytes_fit(too_small, info.layer_bytes)
        assert topology_partition_by_bytes(too_small, info.layer_bytes) == []
        assert not fit.fits
        print(f"  too-small pool rejected: {fit.reason}; needed={fit.bytes_needed}, available={fit.bytes_available}")
        print("  PASS: the gate would fail if bad placements were accepted")

        print("\n== test 5: sharded MoE generation is bit-exact vs single process at temperature 0 ==")
        prompt = [1, 5, 9, 13]
        full = G.load_full_params(path, info)
        single = M.generate_with(lambda ctx: M.forward_full(full, info.config, ctx), info.config,
                                 prompt, max_new_tokens=4, temperature=0.0)
        sharded = M.generate_with(lambda ctx: _run_partition(path, info, parts, ctx), info.config,
                                  prompt, max_new_tokens=4, temperature=0.0)
        print(f"  single={single}")
        print(f"  sharded={sharded}")
        assert sharded == single
        logits = _run_partition(path, info, parts, prompt)
        assert np.isfinite(logits).all() and logits.std() > 1e-6
        print("  PASS: MoE layer-range sharding is mathematically identical")

        print("\nALL MOE PLACEMENT CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
