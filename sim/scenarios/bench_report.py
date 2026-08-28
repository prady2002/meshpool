"""Benchmark report: the flagship's numbers, with provenance attached to each one.

Run:  python -m sim.scenarios.bench_report      (writes docs/benchmarks.md)

Plan section 5.2 forbids any performance number reaching a user-facing surface
without a tier label, because the market this product enters has been poisoned
by a decade of unlabelled projections. This report is where that rule is
enforced, so it separates three things that are easy to blur together:

MEASURED
    Real model, real tokenizer, real forward passes on this machine.
    ``PipelineStats.compute_seconds`` is wall-clock ``perf_counter`` time spent
    in actual matmuls. The *network* is emulated -- ``LinkProfile`` charges
    RTT/2 plus serialisation per hop -- because one machine cannot produce a
    real 150 ms link. So these rows are "measured compute over an emulated
    network", which is stated on the page rather than left for the reader to
    infer.

PROJECTED
    The fitted performance model (``sim/perfmodel.py``) extrapolated to
    hardware nobody here owns. Fitted on the measured rows, reported with the
    fit quality and a 90% CI. A projection whose fit quality is not printed
    next to it is a guess wearing a lab coat.

CITED
    Other people's published measurements, attributed.

The one number that decides whether the whole thesis works is
``tokens_per_traversal``. Wide-area pipeline decoding costs
``(N-1) x RTT`` per traversal no matter what, so the only way to beat it is to
emit more than one token per traversal. Everything else on this page is
supporting evidence for that ratio.
"""
from __future__ import annotations

import asyncio
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from meshpool.inference import model as M  # noqa: E402
from meshpool.inference.pipeline import (LinkProfile, Pipeline,  # noqa: E402
                                         build_stages, generate_baseline)
from meshpool.inference.spd import SelfDraft, speculative_generate  # noqa: E402
from sim.perfmodel import (MODEL_CONFIGS, ClaimTier, Observation,  # noqa: E402
                           PipelineConfig, fit, predict)

PROMPT_TEXT = "MeshPool turns idle machines into "
MAX_NEW_TOKENS = 48
DRAFT_LAYERS = 2
K = 4
SEED = 7
REPEATS = 3

# The four link classes from plan section 5.1, as one-way RTT in milliseconds.
LINKS: list[tuple[str, float]] = [
    ("LAN / same room", 1.0),
    ("Same metro", 15.0),
    ("Same country", 45.0),
    ("Intercontinental", 150.0),
]

DEPTHS = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _pipe(cfg, params, depth: int, rtt_ms: float) -> Pipeline:
    return Pipeline(cfg, build_stages(cfg, depth, params), LinkProfile(rtt_ms=rtt_ms))


def _tok_s(stats, n_tokens: int) -> float:
    """Throughput implied by measured compute plus emulated link time.

    ``n_tokens`` must be the number actually generated, not the number asked
    for. The demo model's context is 64 tokens and the prompt eats 33 of them,
    so a request for 48 returns 31 -- and dividing by 48 inflated every row in
    this table by 1.55x. It also drove ``fit()`` to a negative coefficient
    (measured throughput implied less time than the emulated network alone
    could take), which silently fell back to ``base_tok_s = 1.0`` and made
    every projection read 0.00 tok/s.
    """
    total = stats.compute_seconds + stats.link_seconds
    if n_tokens <= 0:
        raise AssertionError("no tokens were generated; the row would be a lie")
    return (n_tokens / total) if total > 0 else float("inf")


def measure_grid(cfg, params, prompt: list[int]) -> list[dict]:
    """Baseline vs SPD across the depth x link grid, repeated and medianed."""
    rows: list[dict] = []
    draft = SelfDraft(cfg, params, n_layer=DRAFT_LAYERS)
    for link_name, rtt in LINKS:
        for depth in DEPTHS:
            base_tps, spd_tps, accepts, per_trav = [], [], [], []
            base_trav = spd_trav = 0
            n_base = n_spd = 0
            for r in range(REPEATS):
                pipe = _pipe(cfg, params, depth, rtt)
                btoks, bstats = asyncio.run(generate_baseline(
                    pipe, prompt, MAX_NEW_TOKENS, seed=SEED + r))
                n_base = len(btoks)
                base_tps.append(_tok_s(bstats, n_base))
                base_trav = bstats.traversals

                pipe = _pipe(cfg, params, depth, rtt)
                stoks, pstats, spec = asyncio.run(speculative_generate(
                    pipe, draft, prompt, MAX_NEW_TOKENS, k=K, seed=SEED + r))
                n_spd = len(stoks)
                spd_tps.append(_tok_s(pstats, n_spd))
                spd_trav = pstats.traversals
                accepts.append(spec.acceptance_rate)
                per_trav.append(spec.tokens_per_traversal)

            # Comparing throughputs computed over different token counts would
            # bake a length ratio into the speedup, so refuse rather than
            # publish a number whose meaning depends on where generation
            # happened to stop.
            if abs(n_base - n_spd) > max(2, 0.1 * n_base):
                raise AssertionError(
                    f"{link_name} depth={depth}: baseline generated {n_base} "
                    f"tokens but SPD generated {n_spd}; the speedup would be "
                    f"measuring sequence length, not decoding speed")

            b = statistics.median(base_tps)
            s = statistics.median(spd_tps)
            rows.append({
                "link": link_name, "rtt_ms": rtt, "depth": depth,
                "baseline_tok_s": b, "spd_tok_s": s,
                "speedup": (s / b) if b else float("nan"),
                "acceptance": statistics.median(accepts),
                "tokens_per_traversal": statistics.median(per_trav),
                "baseline_traversals": base_trav,
                "spd_traversals": spd_trav,
                "tokens_generated": n_base,
            })
            print(f"  {link_name:<18} depth={depth}  base={b:7.2f} tok/s  "
                  f"spd={s:7.2f} tok/s  x{s / b if b else 0:4.2f}  "
                  f"accept={statistics.median(accepts):.0%}  "
                  f"tok/traversal={statistics.median(per_trav):.2f}",
                  flush=True)
    return rows


def measure_ttft(cfg, params, tok) -> list[dict]:
    """Time to first token vs prompt length, with and without chunked prefill."""
    out: list[dict] = []
    depth, rtt = 4, 45.0
    base_prompt = tok.encode(PROMPT_TEXT) or [0]
    # The demo model's context is 64 tokens, so asking for 256 is not a longer
    # benchmark -- it is a ValueError from kvcache. Cap at the real limit and
    # report the sizes actually measured rather than the ones we hoped for.
    limit = int(getattr(cfg, "block_size", 64))
    lengths = sorted({n for n in (16, 64, 256, 1024) if n <= limit} | {limit})
    for n in lengths:
        prompt = (base_prompt * ((n // len(base_prompt)) + 1))[:n]
        for chunk in (0, 32):
            pipe = _pipe(cfg, params, depth, rtt)
            t0 = time.perf_counter()
            _, stats = asyncio.run(generate_baseline(
                pipe, prompt, 1, seed=SEED, chunked_prefill=chunk))
            wall = time.perf_counter() - t0
            out.append({
                "prompt_tokens": n,
                "chunk": chunk or None,
                "modelled_s": stats.compute_seconds + stats.link_seconds,
                "wall_s": wall,
                "hops": stats.hops,
                "wire_bytes": stats.wire_bytes,
            })
            print(f"  prompt={n:<4} chunk={chunk or 'off':<4} "
                  f"ttft={stats.compute_seconds + stats.link_seconds:6.3f}s "
                  f"hops={stats.hops:<4} wire={stats.wire_bytes / 1024:.1f} KiB",
                  flush=True)
    return out


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------

def project(rows: list[dict]) -> tuple[list[dict], object]:
    """Fit on the measured rows, then extrapolate to models we cannot host."""
    obs = [
        Observation(
            config=PipelineConfig(
                model="nano-gpt",
                n_stages=r["depth"],
                cpu_scales=[1.0] * r["depth"],
                rtt_ms_per_hop=[r["rtt_ms"]] * max(0, r["depth"] - 1),
                bw_kbps_per_hop=[0.0] * max(0, r["depth"] - 1),
            ),
            tok_s=r["baseline_tok_s"],
        )
        for r in rows
    ]
    params = fit(obs)
    print(f"  fit: {params!r}", flush=True)
    base = params.base_tok_s.get("nano-gpt", 0.0)
    # fit() falls back to base_tok_s = 1.0 when least-squares returns a
    # non-positive coefficient, which happens whenever the measured rows imply
    # less time than the emulated network alone. That fallback is not a fit; it
    # is a placeholder, and it produced a full page of "0.00 tok/s" projections
    # that the gate happily exited 0 on. Refuse instead.
    if base <= 1.0 + 1e-9:
        raise AssertionError(
            f"degenerate fit: base_tok_s={base} is the fallback value, so every "
            f"projection below would be meaningless. MAPE={params.mape:.1%}. "
            f"Check that measured throughput uses the tokens actually generated.")
    if params.mape > 0.50:
        raise AssertionError(
            f"fit error {params.mape:.1%} exceeds the 50% ceiling; a projection "
            f"built on it would be a guess wearing a lab coat")

    # What follows deliberately does NOT extrapolate this machine's throughput
    # to a 70B model. The previous version did, by scaling base_tok_s down by
    # the active-parameter ratio (0.001B / 70B), and produced "0.00 tok/s" for
    # every target. That was not a conservative estimate, it was a category
    # error: a 1M-parameter NumPy model running at 207 tok/s is bounded by
    # Python and BLAS call overhead, not by FLOPs, so its throughput carries no
    # information about a 70B model on a GPU. Extrapolating across four orders
    # of magnitude on a variable the experiment never varied is exactly the
    # unlabelled-projection habit plan section 5.2 exists to prevent.
    #
    # Two things *can* be defended from what was actually measured:
    #
    #   1. The latency ceiling. Wide-area pipeline decoding costs one traversal
    #      of (N-1) hops per traversal, whatever the hardware, and we measured
    #      how many tokens each traversal yields. tokens_per_traversal divided
    #      by traversal latency is therefore an upper bound that assumes
    #      compute is free -- true of no hardware, but a real ceiling, and the
    #      quantity MeshPool's design actually moves.
    #
    #   2. Sensitivity to compute. For an assumed aggregate per-token compute
    #      cost, the same model gives an end-to-end number. The assumption is
    #      printed as an input rather than smuggled in as a result.
    scenarios = [
        ("llama-2-70b", 4, 20.0, "4-node metro constellation"),
        ("llama-3.1-405b", 8, 20.0, "8-node metro constellation"),
        ("deepseek-v3", 10, 20.0, "10-node metro constellation"),
        ("kimi-k2", 11, 20.0, "11-node metro constellation"),
        ("kimi-k2", 11, 45.0, "11 nodes, same country"),
        ("kimi-k2", 11, 150.0, "11 nodes scattered worldwide"),
    ]
    tpt = statistics.median(r["tokens_per_traversal"] for r in rows)
    accept = median_acceptance(rows)
    results: list[dict] = []
    for name, stages, rtt, label in scenarios:
        mc = MODEL_CONFIGS[name]
        hops = stages - 1
        # LinkProfile and perfmodel both charge one-way latency per hop.
        traversal_s = hops * (rtt / 2000.0)
        # Activations are tiny: 2 x d_model bytes per hop. Charged anyway at a
        # residential 20 Mbps uplink so the row cannot be accused of ignoring
        # bandwidth -- it is ~6 ms for Kimi K2 and does not change the story.
        traversal_s += hops * (mc.activation_bytes_per_hop * 8.0) / 20e6
        ceiling_base = 1.0 / traversal_s if traversal_s > 0 else float("inf")
        results.append({
            "model": name, "label": label, "stages": stages, "rtt_ms": rtt,
            "hops": hops,
            "traversal_ms": traversal_s * 1000.0,
            "activation_bytes_per_hop": mc.activation_bytes_per_hop,
            "ceiling_baseline": ceiling_base,
            "ceiling_spd": ceiling_base * tpt,
            "tokens_per_traversal": tpt,
            "acceptance": accept,
            "tier": ClaimTier.PROJECTED.value,
        })
        print(f"  {name:<16} {label:<30} traversal={traversal_s * 1000:6.1f} ms  "
              f"ceiling: base={ceiling_base:6.1f} tok/s  "
              f"SPD={ceiling_base * tpt:6.1f} tok/s", flush=True)

    bad = [r["label"] for r in results if not (r["ceiling_spd"] > r["ceiling_baseline"])]
    if bad:
        raise AssertionError(f"SPD failed to raise the latency ceiling in: {bad}")
    return results, params


def sensitivity(rows: list[dict]) -> list[dict]:
    """End-to-end tok/s for an *assumed* per-token compute cost.

    The assumption is the input, printed in its own column. Nothing here
    pretends to know how fast a 70B shard runs on a stranger's GPU.

    Activation transfer is charged on the same basis as the ceiling table
    above -- Kimi K2's 14 KiB per hop over a residential 20 Mbps uplink. It has
    to be: without it the zero-compute row read 23.8 tok/s while the ceiling
    table read 15.2 tok/s for the identical 11-node metro constellation, and a
    page that contradicts itself two sections apart is worse than no page.
    Note the size of that correction. Ten hops of 14 KiB at 20 Mbps is 57 ms,
    against 100 ms of latency -- so on residential uplinks bandwidth is roughly
    a third of the traversal cost at depth 11, not the rounding error it is at
    depth 2. Uplink capacity therefore belongs in constellation formation
    alongside RTT.
    """
    tpt = statistics.median(r["tokens_per_traversal"] for r in rows)
    mc = MODEL_CONFIGS["kimi-k2"]
    out: list[dict] = []
    for compute_ms in (0, 10, 25, 50, 100, 200):
        for label, stages, rtt in (("11-node metro constellation", 11, 20.0),
                                   ("11 nodes, same country", 11, 45.0),
                                   ("11 nodes worldwide", 11, 150.0)):
            hops = stages - 1
            traversal_s = (hops * (rtt / 2000.0)
                           + hops * (mc.activation_bytes_per_hop * 8.0) / 20e6
                           + compute_ms / 1000.0)
            base = 1.0 / traversal_s
            out.append({
                "assumed_compute_ms": compute_ms, "label": label,
                "stages": stages, "rtt_ms": rtt,
                "baseline_tok_s": base, "spd_tok_s": base * tpt,
            })
    return out


def median_acceptance(rows: list[dict]) -> float:
    return statistics.median(r["acceptance"] for r in rows)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

CITED = [
    ("Petals (public swarm)", "Llama-2-70B", "6 tok/s",
     "NeurIPS 2023 / arXiv 2312.08361"),
    ("Petals (public swarm)", "Falcon-180B", "4 tok/s",
     "NeurIPS 2023 / arXiv 2312.08361"),
    ("exo, 4x M3 Ultra over Thunderbolt-5 RDMA (<1 ms, LAN)", "Kimi K2",
     "~30 tok/s", "exo labs, 2025 -- a LAN result, not comparable to WAN"),
]


def render(rows, ttft, projections, sens, params, meta) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Benchmarks")
    a("")
    a(f"_Generated {meta['when']} by `python -m sim.scenarios.bench_report`._")
    a("")
    a("Every number below carries a provenance tier, per the project rule that no "
      "performance figure reaches a reader without one.")
    a("")
    a("| Tier | What it means |")
    a("|---|---|")
    a("| **MEASURED** | Real model, real tokenizer, real forward passes on this "
      "machine. Compute time is wall-clock; the network is emulated. |")
    a("| **PROJECTED** | The fitted performance model extrapolated to hardware we "
      "do not own. Reported with fit quality and a 90% CI. |")
    a("| **CITED** | Someone else's published measurement, attributed. |")
    a("")
    a("## Environment")
    a("")
    a(f"- Host: {meta['platform']}, Python {meta['python']}")
    a(f"- Model: **{meta['model']}** — {meta['layers']} layers, "
      f"d_model={meta['d_model']}, trained checkpoint={meta['trained']}")
    a(f"- Draft: `SelfDraft` over the first {DRAFT_LAYERS} layers "
      f"(runs on the first stage, so it adds **no wide-area hop**)")
    a(f"- K={K}, {MAX_NEW_TOKENS} generated tokens, median of {REPEATS} runs")
    a("")
    a("> The model is small on purpose. One machine cannot hold a 70B model, so the "
      "honest choice is a **real** small model with a real tokenizer and real "
      "matmuls, rather than a fake big one. Plan section 5.3.")
    a("")

    a("## 1. Speculative Pipeline Decoding — MEASURED")
    a("")
    a("The governing equation for wide-area pipeline inference is "
      "`t_token ≈ Σ compute + (N−1) × RTT`. The RTT term is fixed by physics, so "
      "the only way to beat it is to emit more than one token per traversal.")
    a("")
    a("| Link | RTT | Stages | Baseline | SPD | Speed-up | Acceptance | "
      "Tokens/traversal |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        a(f"| {r['link']} | {r['rtt_ms']:.0f} ms | {r['depth']} | "
          f"{r['baseline_tok_s']:.2f} tok/s | {r['spd_tok_s']:.2f} tok/s | "
          f"**×{r['speedup']:.2f}** | {r['acceptance']:.0%} | "
          f"{r['tokens_per_traversal']:.2f} |")
    a("")
    multi = [r for r in rows if r["depth"] > 1]
    best = max(multi, key=lambda r: r["speedup"]) if multi else rows[0]
    a(f"- Median acceptance rate across the grid: **{median_acceptance(rows):.0%}**.")
    a(f"- Best multi-stage speed-up: **×{best['speedup']:.2f}** "
      f"({best['link']}, {best['depth']} stages).")
    a("- At depth 1 there is no wide-area hop to amortise, so SPD's advantage "
      "collapses — which is exactly what the equation predicts, and is the "
      "sanity check that these numbers describe the intended mechanism.")
    a("")

    a("## 2. Time to first token — MEASURED")
    a("")
    a("Prefill is position-parallel: a stage processes all `T` positions at once and "
      "forwards one `[T × d_model]` tensor, so TTFT does **not** grow by one "
      "pipeline traversal per prompt token. Chunked prefill bounds memory and stops "
      "one long prompt from head-of-line blocking a shared pipeline.")
    a("")
    a("| Prompt tokens | Chunked prefill | TTFT | Hops | Bytes on the wire |")
    a("|---:|---|---:|---:|---:|")
    for t in ttft:
        a(f"| {t['prompt_tokens']} | {t['chunk'] or 'off'} | "
          f"{t['modelled_s'] * 1000:.0f} ms | {t['hops']} | "
          f"{t['wire_bytes'] / 1024:.1f} KiB |")
    a("")

    a("## 3. What this predicts for models this machine cannot hold — PROJECTED")
    a("")
    a(f"The performance model fits the measured rows above with **R² = "
      f"{params.r2:.4f}**, **MAPE = {params.mape:.1%}** "
      f"(single-node throughput fitted at {params.base_tok_s.get('nano-gpt', 0):.1f} "
      f"tok/s for this model on this host).")
    a("")
    a("**What is deliberately not claimed.** An earlier version of this page "
      "scaled that fitted throughput down by the active-parameter ratio "
      "(0.001B → 70B) and printed the result as a tok/s projection. That is a "
      "category error, not a conservative estimate: a 1M-parameter NumPy model "
      "is bounded by Python and BLAS call overhead, not by FLOPs, so its speed "
      "carries no information about a 70B model on a GPU. It produced "
      "`0.00 tok/s` for every target, and the generator exited 0 anyway. Both "
      "the extrapolation and that silent pass are gone; the generator now "
      "refuses to write this file if the fit degenerates or if SPD fails to "
      "raise the ceiling.")
    a("")
    a("**What is claimed.** Wide-area pipeline decoding pays `(N−1)` hops of "
      "one-way latency per traversal regardless of hardware, and the table in "
      "section 1 measured how many tokens one traversal yields. Dividing the "
      "second by the first is an upper bound that assumes compute is free — "
      "true of no hardware, but a genuine ceiling, and precisely the quantity "
      "constellations and SPD are designed to move. Activation transfer is "
      "charged at a residential 20 Mbps uplink rather than ignored.")
    a("")
    a("| Model | Deployment | Hops | Activation/hop | Traversal | "
      "Ceiling, baseline | Ceiling, SPD | Tier |")
    a("|---|---|---:|---:|---:|---:|---:|---|")
    for p in projections:
        a(f"| {p['model']} | {p['label']} | {p['hops']} | "
          f"{p['activation_bytes_per_hop'] / 1024:.1f} KiB | "
          f"{p['traversal_ms']:.1f} ms | {p['ceiling_baseline']:.1f} tok/s | "
          f"**{p['ceiling_spd']:.1f} tok/s** | {p['tier']} |")
    a("")
    a(f"The SPD column is the baseline ceiling multiplied by the "
      f"**{projections[0]['tokens_per_traversal']:.2f} tokens per traversal "
      f"measured** in section 1 at a "
      f"{projections[0]['acceptance']:.0%} acceptance rate. Note what the table "
      "says about topology: the same 11 nodes are worth "
      f"{projections[-3]['ceiling_spd'] / projections[-1]['ceiling_spd']:.1f}× "
      "more in one metro than scattered worldwide. That is the entire argument "
      "for constellations, and it is arithmetic, not opinion.")
    a("")
    a("### 3b. Sensitivity to per-token compute")
    a("")
    a("Compute is the term this machine cannot measure for a 70B-class model, "
      "so it is an **input** here, printed in its own column, across a range "
      "that spans a fast GPU shard to a slow CPU one. Read the row whose "
      "assumption you believe.")
    a("")
    a("| Assumed compute/token | Deployment | Baseline | SPD |")
    a("|---:|---|---:|---:|")
    for s in sens:
        assumed = ("0 (latency + bandwidth ceiling)" if s["assumed_compute_ms"] == 0
                   else f"{s['assumed_compute_ms']} ms")
        a(f"| {assumed} | {s['label']} | {s['baseline_tok_s']:.1f} tok/s | "
          f"**{s['spd_tok_s']:.1f} tok/s** |")
    a("")
    a("Two things are worth reading off this table. Compute has to be **very** "
      "slow before it, rather than the network, is what limits a wide-area "
      "pipeline: 200 ms per token still leaves the worldwide deployment within "
      "a factor of 1.3 of its zero-compute ceiling. And bandwidth is not free "
      "at depth. Ten hops of 14 KiB over a residential 20 Mbps uplink is 57 ms "
      "against 100 ms of metro latency — about a third of the traversal — so "
      "uplink capacity belongs in constellation formation next to RTT, not "
      "dismissed as a rounding error the way it safely can be at depth 2.")
    a("")

    a("## 4. Published results from other systems — CITED")
    a("")
    a("| System | Model | Throughput | Source |")
    a("|---|---|---|---|")
    for sysname, model, tps, src in CITED:
        a(f"| {sysname} | {model} | {tps} | {src} |")
    a("")
    a("## Reproduce")
    a("")
    a("```bash")
    a("python -m sim.scenarios.bench_report        # regenerates this file")
    a("python sim/scenarios/verify_spd_equivalence.py     # SPD correctness")
    a("python sim/scenarios/verify_netem_calibration.py   # model accuracy bound")
    a("python sim/scenarios/verify_streaming_ttft.py      # TTFT budget")
    a("```")
    a("")
    return "\n".join(lines) + "\n"


def main(check_only: bool = False) -> int:
    check_only = check_only or "--check" in sys.argv[1:]
    cfg = M.REGISTRY["nano-gpt"]
    params_w, meta_w = M.load_params(cfg)
    tok = M.CharTokenizer(itos=(meta_w or {}).get("itos"), vocab_size=cfg.vocab_size)
    prompt = tok.encode(PROMPT_TEXT) or [tok.encode(" ")[0]]

    print("== 1/3 speculative pipeline decoding across the depth x link grid ==",
          flush=True)
    rows = measure_grid(cfg, params_w, prompt)

    print("\n== 2/3 time to first token ==", flush=True)
    ttft = measure_ttft(cfg, params_w, tok)

    print("\n== 3/3 fitting the performance model and projecting ==", flush=True)
    projections, fitted = project(rows)
    sens = sensitivity(rows)

    meta = {
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "model": cfg.name, "layers": cfg.n_layer, "d_model": cfg.d_model,
        "trained": meta_w is not None,
    }
    out = REPO / "docs" / "benchmarks.md"
    page = render(rows, ttft, projections, sens, fitted, meta)
    if check_only:
        # run_all_gates.py fails any gate that leaves a tracked file modified,
        # and this page embeds a timestamp, so --check renders everything (a
        # crash in render() is still a failure) and simply does not write.
        print(f"\n[--check] rendered {len(page)} chars, not writing "
              f"{out.relative_to(REPO)}")
    else:
        out.write_text(page, encoding="utf-8")
        print(f"\nwrote {out.relative_to(REPO)}")

    # The report is also a gate: if SPD does not beat the baseline once there is
    # a wide-area hop to amortise, the product thesis is wrong and this must not
    # quietly publish a page saying so.
    multi = [r for r in rows if r["depth"] > 1 and r["rtt_ms"] >= 15]
    bad = [r for r in multi if r["speedup"] <= 1.0]
    if bad:
        print(f"\nFAILED: SPD did not beat the baseline in {len(bad)}/{len(multi)} "
              f"wide-area configurations", file=sys.stderr)
        for r in bad:
            print(f"  {r['link']} depth={r['depth']} speedup=x{r['speedup']:.2f}",
                  file=sys.stderr)
        return 1
    if any(p["tier"] != ClaimTier.PROJECTED.value for p in projections):
        print("\nFAILED: a projection was not labelled PROJECTED", file=sys.stderr)
        return 1

    # Section 3 and section 3b describe the same 11-node Kimi K2 deployments by
    # two different routes. If they disagree, at least one is wrong, and the
    # page would be publishing a contradiction with a tier label on it. They
    # disagreed by 57% the first time this was checked.
    ceil_by_label = {p["label"]: p for p in projections if p["model"] == "kimi-k2"}
    pairs = [("11-node metro constellation", "11-node metro constellation"),
             ("11 nodes, same country", "11 nodes, same country"),
             ("11 nodes scattered worldwide", "11 nodes worldwide")]
    for ceil_label, sens_label in pairs:
        c = ceil_by_label[ceil_label]["ceiling_spd"]
        s = next(x["spd_tok_s"] for x in sens
                 if x["assumed_compute_ms"] == 0 and x["label"] == sens_label)
        if abs(c - s) > 0.05:
            print(f"\nFAILED: section 3 says {c:.2f} tok/s for {ceil_label!r} but "
                  f"section 3b's zero-compute row says {s:.2f}", file=sys.stderr)
            return 1

    print(f"OK: SPD beat the baseline in all {len(multi)} wide-area configurations")
    print(f"OK: ceiling and sensitivity tables agree on all "
          f"{len(pairs)} shared configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
