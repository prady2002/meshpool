"""Acceptance gate: many tenants must SHARE wide-area round trips, not queue for them.

``verify_spd_equivalence.py`` proves one tenant can amortise ``(N-1) * RTT`` across
tokens. This gate proves many tenants amortise the same term across *each other* --
on a real multi-process cluster, over real HTTP, with weights really sharded.

Why this is the difference between a demo and a service
-------------------------------------------------------
A single-tenant pipeline that reaches 20 tok/s serves exactly one person. Every other
decentralised-inference project stops there, and it is why they read as demos. The
governing cost is

    t_token  ~=  sum(compute_i)  +  (N - 1) * RTT

and the RTT term is paid once per *traversal*, regardless of how many rows ride inside
it. So B concurrent users cost B traversals per step when served one at a time and ONE
when batched. On a mesh -- where RTT dominates and always will -- extra tenants are
close to free until compute binds.

What is asserted, and why each assertion can fail
-------------------------------------------------
  1. **Exactness.** Concurrent output must be token-identical to the same prompt run
     alone, against a single-process oracle. If batching changed anyone's answer it
     would be a correctness bug wearing a performance costume. Checked at temperature 0
     where the answer is deterministic and an oracle exists.
  2. **Sharing really happens.** The server must report ``mean_batch > 1``: requests
     genuinely rode traversals together. Serving them serially would report 1.0.
  3. **Round trips actually collapse.** Total wire traversals for B concurrent requests
     must be far below B times the solo cost. This is measured from the server's own
     per-request attribution, which is exact for traversals and hops.
  4. **Throughput rises with tenants.** Aggregate tokens/sec under concurrency must
     beat the same work done sequentially. A counter that improves while the clock
     does not is an accounting change, not a speedup.
  5. **Nobody is starved.** A tenant asking for many tokens must not stop short
     requests from completing -- the property that separates continuous batching from
     static batching, and the one users actually feel.
  6. **It composes with speculation.** SPD amortises across tokens, batching across
     tenants; run together they must both still hold and still be exact.

Exit code: 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.inference import model as M  # noqa: E402

PROMPTS = [
    "MeshPool turns idle machines into ",
    "the quick brown fox jumps over ",
    "a decentralised computer is made of ",
    "every idle laptop in the world ",
    "pipeline parallelism over the internet ",
    "speculative decoding amortises ",
]
N_TOKENS = 16


def single_machine_greedy(prompt: str, n: int) -> str:
    """Reference: the whole model in one process, greedy. The independent oracle."""
    cfg = M.REGISTRY["nano-gpt"]
    P, meta = M.load_params(cfg)
    tok = M.CharTokenizer(itos=(meta or {}).get("itos"), vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    out = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg, ids,
                          max_new_tokens=n, temperature=0.0)
    return tok.decode(out)


def chat(cp: str, prompt: str, **kw) -> dict:
    body = {"prompt": prompt, "max_new_tokens": N_TOKENS, "temperature": 0.0}
    body.update(kw)
    r = httpx.post(f"{cp}/api/v1/inference/chat", json=body, timeout=300)
    r.raise_for_status()
    return r.json()


def fire_concurrently(cp: str, prompts: list[str], **kw) -> tuple[list[dict], float]:
    """Issue every request at the same moment from real OS threads.

    Threads, not asyncio: the requests must genuinely overlap inside the server, which
    is the only way the batcher can see them together.
    """
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futures = [ex.submit(chat, cp, p, **kw) for p in prompts]
        out = [f.result() for f in futures]
    return out, time.perf_counter() - t0


def main() -> int:  # noqa: C901
    h = Harness(n_agents=3, cp_port=8143,
                data_dir=str(Path(__file__).resolve().parent / "_data_batching"))
    ok = True
    try:
        print("== starting a real 3-node cluster (separate processes) ==")
        h.start()
        cp = h.cp_url()

        r = httpx.post(f"{cp}/api/v1/models", json={"model": "nano-gpt"}, timeout=180)
        r.raise_for_status()
        stages = r.json()["stages"]
        assert len(stages) >= 2, "model must be sharded across >= 2 machines"
        print("  sharded: " + ", ".join(
            f"{s['node_name']}[{s['first']}:{s['last']}]" for s in stages))

        oracles = {p: single_machine_greedy(p, N_TOKENS) for p in PROMPTS}

        # ------------------------------------------------------------------ #
        print("\n== test 1: concurrent output is identical to solo output ==")
        solo = [chat(cp, p, speculative=False) for p in PROMPTS[:3]]
        for d in solo:
            assert d["text"] == oracles[d["prompt"]], "solo mesh output is already wrong"
        conc, _ = fire_concurrently(cp, PROMPTS, speculative=False)
        for d in conc:
            assert d["text"] == oracles[d["prompt"]], (
                f"concurrent output diverged for {d['prompt']!r}:\n"
                f"  got      {d['text']!r}\n  expected {oracles[d['prompt']]!r}")
        print(f"  {len(conc)} concurrent requests, all token-identical to the "
              f"single-process oracle")
        print("  PASS: a tenant's answer does not depend on who shared its batch")

        # ------------------------------------------------------------------ #
        print("\n== test 2: requests genuinely shared traversals ==")
        shared = [d["batch"]["mean_batch"] for d in conc]
        best = max(shared)
        print(f"  mean_batch per request: {[round(s, 2) for s in shared]}")
        print(f"  solo requests reported: {[d['batch']['mean_batch'] for d in solo]}")
        assert all(d["batch"]["mean_batch"] == 1.0 for d in solo), (
            "a request served alone should report mean_batch 1.0")
        assert best > 1.0, (
            "no request ever shared a traversal with another; the server served them "
            "one at a time")
        print(f"  PASS: peak sharing {best:.2f} requests per traversal")

        # ------------------------------------------------------------------ #
        print("\n== test 3: round trips collapse, measured at the receiving nodes ==")

        def node_hops() -> tuple[int, int]:
            """Total hops and sequence-steps served, summed across every real node.

            This is counted inside the agent processes, on arrival. It is not derived
            from anything the requester recorded, so it cannot be talked up by
            sender-side bookkeeping -- if the mesh really paid B round trips, this
            number really shows B.
            """
            hops = rows = 0
            for port in h.agent_ports:
                s = httpx.get(f"http://{h.host}:{port}/shard/stats", timeout=30).json()
                hops += s["shard_hops"]
                rows += s["shard_rows"]
            return hops, rows

        h0, r0 = node_hops()
        chat(cp, PROMPTS[0], speculative=False)
        h1, r1 = node_hops()
        solo_hops = h1 - h0
        assert solo_hops > 0, "no shard hop was recorded; the gate is measuring nothing"

        conc3, _ = fire_concurrently(cp, PROMPTS, speculative=False)
        h2, r2 = node_hops()
        burst_hops, burst_rows = h2 - h1, r2 - r1
        naive = solo_hops * len(conc3)
        print(f"  one request alone cost {solo_hops} hops across the mesh")
        print(f"  {len(conc3)} served one-at-a-time would cost ~{naive} hops")
        print(f"  {len(conc3)} served concurrently actually cost {burst_hops} hops "
              f"carrying {burst_rows} sequence-steps")
        print(f"  receiver-measured batching factor: "
              f"{burst_rows / max(burst_hops, 1):.2f} steps per hop")
        assert burst_rows >= naive * 0.9, (
            f"only {burst_rows} sequence-steps were computed where ~{naive} were "
            f"expected; work went missing rather than being shared")
        assert burst_hops < naive, (
            f"{len(conc3)} concurrent requests still cost {burst_hops} wide-area hops, "
            f"the same as serving them one at a time ({naive}); nothing was shared")
        print(f"  PASS: {naive - burst_hops} wide-area round trips eliminated "
              f"({100 * (1 - burst_hops / naive):.0f}% fewer)")

        for d in conc3:
            assert d["text"] == oracles[d["prompt"]], "output changed while collapsing hops"

        # ------------------------------------------------------------------ #
        print("\n== test 4: concurrency beats sequential on the clock ==")
        t0 = time.perf_counter()
        seq = [chat(cp, p, speculative=False) for p in PROMPTS]
        seq_s = time.perf_counter() - t0
        conc2, conc_s = fire_concurrently(cp, PROMPTS, speculative=False)
        seq_tps = sum(d["tokens"] for d in seq) / seq_s
        conc_tps = sum(d["tokens"] for d in conc2) / conc_s
        print(f"  sequential: {seq_s:.2f}s  ->  {seq_tps:.1f} tok/s aggregate")
        print(f"  concurrent: {conc_s:.2f}s  ->  {conc_tps:.1f} tok/s aggregate")
        print(f"  aggregate throughput: {conc_tps / max(seq_tps, 1e-9):.2f}x")
        assert conc_tps > seq_tps, (
            f"serving {len(PROMPTS)} tenants concurrently ({conc_tps:.1f} tok/s) was "
            f"no faster in aggregate than serving them one after another "
            f"({seq_tps:.1f} tok/s)")
        for d in conc2:
            assert d["text"] == oracles[d["prompt"]], "output changed under load"
        print("  PASS: more tenants, more aggregate throughput, same answers")

        # ------------------------------------------------------------------ #
        print("\n== test 5: a long request does not starve short ones ==")
        # nano-gpt's window is 64 positions and the prompts are ~34 tokens, so "long"
        # stays inside the window: this test is about scheduling fairness, and must
        # not be able to fail for the unrelated reason of hitting the context limit.
        results: dict[str, float] = {}

        def timed(label: str, prompt: str, n: int) -> dict:
            t = time.perf_counter()
            d = chat(cp, prompt, max_new_tokens=n, speculative=False)
            results[label] = time.perf_counter() - t
            return d

        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            long_f = ex.submit(timed, "long", PROMPTS[0], 28)
            time.sleep(0.3)
            short_fs = [ex.submit(timed, f"short{i}", PROMPTS[i + 1], 3)
                        for i in range(3)]
            shorts = [f.result() for f in short_fs]
            long_d = long_f.result()
        worst_short = max(results[f"short{i}"] for i in range(3))
        print(f"  long request (28 tokens): {results['long']:.2f}s")
        print(f"  slowest short request (3 tokens): {worst_short:.2f}s")
        assert long_d["tokens"] == 28, (
            f"long request produced {long_d['tokens']} of 28 tokens "
            f"(finish_reason={long_d.get('finish_reason')})")
        for i, d in enumerate(shorts):
            assert d["tokens"] == 3, (
                f"short{i} produced {d['tokens']} of 3 tokens "
                f"(finish_reason={d.get('finish_reason')})")
        assert worst_short < results["long"], (
            f"a 3-token request took {worst_short:.2f}s while a 28-token request took "
            f"{results['long']:.2f}s -- short requests were made to wait for the long "
            f"one, which is static batching, not continuous batching")
        print("  PASS: short requests finished while the long one was still generating")

        # ------------------------------------------------------------------ #
        print("\n== test 6: a truncated answer is never reported as a complete one ==")
        # The window is 64 positions; asking for far more than fits forces a cut-off.
        # A client must be able to tell "the model finished" from "we ran out of
        # room", because agent loops continue on one and stop on the other.
        short_d = chat(cp, PROMPTS[0], max_new_tokens=4, speculative=False)
        cut = chat(cp, PROMPTS[0], max_new_tokens=512, speculative=False)
        print(f"  asked for 4   -> got {short_d['tokens']:3d}, "
              f"finish_reason={short_d.get('finish_reason')}")
        print(f"  asked for 512 -> got {cut['tokens']:3d}, "
              f"finish_reason={cut.get('finish_reason')}")
        assert short_d.get("finish_reason") == "length", (
            "a request that simply reached its token budget should report 'length'")
        assert cut["tokens"] < 512, "expected the context window to cut this short"
        assert cut.get("finish_reason") == "context_window", (
            f"generation stopped at {cut['tokens']} of 512 tokens because the context "
            f"window ran out, but the response reported "
            f"{cut.get('finish_reason')!r} -- a truncated answer is being presented "
            f"as a finished one")

        oai = httpx.post(f"{cp}/v1/chat/completions", json={
            "model": "nano-gpt",
            "messages": [{"role": "user", "content": PROMPTS[0]}],
            "max_tokens": 512, "temperature": 0.0}, timeout=300)
        oai.raise_for_status()
        choice = oai.json()["choices"][0]
        used = oai.json()["usage"]["completion_tokens"]
        print(f"  OpenAI surface: completion_tokens={used}, "
              f"finish_reason={choice['finish_reason']!r}")
        assert choice["finish_reason"] == "length", (
            f"the OpenAI-compatible surface reported {choice['finish_reason']!r} for a "
            f"reply that was cut off at {used} tokens; an unmodified OpenAI client "
            f"would treat this truncated reply as a complete one")
        print("  PASS: truncation is reported as truncation on both surfaces")

        # ------------------------------------------------------------------ #
        print("\n== test 7: batching composes with speculative decoding ==")
        spd_conc, spd_s = fire_concurrently(cp, PROMPTS, speculative=True, draft_k=4)
        for d in spd_conc:
            assert d["text"] == oracles[d["prompt"]], (
                "SPD under concurrency diverged -- the two amortisations must compose "
                "without changing the answer")
            assert "speculative" in d, "speculation telemetry vanished under batching"
        spd_trav = max(d["pipeline"]["traversals"] for d in spd_conc)
        plain_trav = max(d["pipeline"]["traversals"] for d in conc2)
        accept = [d["speculative"]["acceptance_rate"] for d in spd_conc]
        print(f"  deepest request: {plain_trav} traversals plain -> {spd_trav} with SPD")
        print(f"  acceptance rates: {[round(a, 2) for a in accept]}")
        print(f"  mean_batch under SPD: "
              f"{[round(d['batch']['mean_batch'], 2) for d in spd_conc]}")
        assert spd_trav < plain_trav, (
            "speculation stopped reducing traversals once batching was switched on; "
            "the two optimisations are interfering rather than composing")
        print("  PASS: tokens and tenants are amortised at the same time")

    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        h.stop()

    print("\nALL CONTINUOUS BATCHING CHECKS PASSED" if ok else "\nCONTINUOUS BATCHING GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
