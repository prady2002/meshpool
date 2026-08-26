"""Honest throughput benchmark: single-box Ollama vs the pooled MeshPool mesh.

The physics MeshPool does NOT break (and this tool is careful not to pretend it does):
pooling does not make a *single* request faster. A model that fits on one machine is
fastest served directly from that machine — routing it through the coordinator, or
sharding it across boxes, only adds a network hop per request. What pooling DOES buy
is **aggregate throughput under concurrency** (every extra machine adds parallel
serving capacity) and the ability to run models **too big for any one box**.

So this harness measures *aggregate tokens/sec at a configurable concurrency*, where
adding machines adds capacity — the metric that actually improves. It reports the
single-box baseline and the pooled result side by side, plus per-request latency
(which pooling does not improve, and we say so).

Typical use on the head machine of a 2-box pool::

    # 1. baseline: this box's own Ollama, hit directly
    python -m meshpool.inference.bench --mode ollama --model qwen2.5-coder:7b -c 8

    # 2. pooled: a running coordinator that routes across every node with the model
    python -m meshpool.inference.bench --mode pool --cp http://HEAD:8080 -c 8

    # 3. both, side by side, deploying the model first
    python -m meshpool.inference.bench --mode both --cp http://HEAD:8080 \
        --model qwen2.5-coder:7b -c 8 --tokens 64 --deploy

Exit code is 0 as long as the requested measurements completed (it is a benchmark,
not a pass/fail gate).
"""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

_client: Optional[httpx.Client] = None


def _http() -> httpx.Client:
    """One shared, thread-safe client for the whole benchmark run.

    This is a correctness fix for the *measurement*, not an optimisation of it.
    ``httpx.post`` builds a fresh ``httpx.Client`` per call, and building one loads the
    system TLS trust store -- **~1.0 s on Windows**, measured (the same construction
    with ``verify=False`` takes 5 ms, so it is all certificate loading). Both
    generators below start their stopwatch *before* that call, so every latency and
    every tokens/sec figure this tool printed carried roughly a second of client
    construction that has nothing to do with the server, the model, or the mesh.

    It corrupted the headline comparison in both directions. A 16-token generation
    that really took 0.5 s was reported at ~1.5 s, understating throughput ~3x. And
    because the same constant landed on the Ollama baseline and the pooled path alike,
    it compressed the ratio between them -- flattering whichever side was slower and
    making a genuine difference look like noise. A benchmark whose docstring promises
    honesty must not measure its own HTTP library's start-up cost.

    A real client builds its trust store once at start-up, so building it once here is
    also the faithful model of a real caller. Per-request TCP connection setup is left
    inside the timed window, because that cost *is* genuinely paid per request.
    """
    global _client
    if _client is None:
        _client = httpx.Client(timeout=600.0,
                               limits=httpx.Limits(max_connections=256,
                                                   max_keepalive_connections=256))
    return _client

# A spread of distinct prompts so a server-side prompt cache can't flatter the
# numbers — every request does real generation work.
_PROMPT_BANK = [
    "Explain what a hash map is in two sentences.",
    "Write a haiku about distributed systems.",
    "List three trade-offs of microservices.",
    "Describe TCP vs UDP in plain language.",
    "What is a race condition? Give one example.",
    "Summarize the CAP theorem in two sentences.",
    "Write a one-line Python function to reverse a string.",
    "Explain eventual consistency to a beginner.",
    "What does a load balancer do? Keep it short.",
    "Give two reasons to use a message queue.",
    "Explain idempotency with a real-world analogy.",
    "What is backpressure in streaming systems?",
]


def build_prompts(n: int) -> list[str]:
    return [f"{_PROMPT_BANK[i % len(_PROMPT_BANK)]} (variation {i})" for i in range(n)]


# --------------------------------------------------------------------------- #
# Single-request generators (return tokens produced + wall seconds)
# --------------------------------------------------------------------------- #
def gen_ollama(base_url: str, model: str, prompt: str, num_predict: int,
               temperature: float) -> tuple[int, float]:
    """One generation straight to a local Ollama server (the single-box baseline)."""
    t0 = time.perf_counter()
    r = _http().post(f"{base_url.rstrip('/')}/api/generate",
                     json={"model": model, "prompt": prompt, "stream": False,
                           "options": {"temperature": temperature, "num_predict": num_predict}},
                     timeout=600.0)
    r.raise_for_status()
    body = r.json()
    dt = time.perf_counter() - t0
    return int(body.get("eval_count", 0) or 0), dt


def gen_pool(cp: str, prompt: str, max_tokens: int, temperature: float) -> tuple[int, float]:
    """One generation through the MeshPool coordinator (routes to a replica node)."""
    t0 = time.perf_counter()
    r = _http().post(f"{cp.rstrip('/')}/api/v1/inference/chat",
                     json={"prompt": prompt, "max_new_tokens": max_tokens,
                           "temperature": temperature},
                     timeout=600.0)
    r.raise_for_status()
    body = r.json()
    dt = time.perf_counter() - t0
    return int(body.get("tokens", 0) or 0), dt


# --------------------------------------------------------------------------- #
# Load driver
# --------------------------------------------------------------------------- #
@dataclass
class BenchResult:
    label: str
    concurrency: int
    n_requests: int
    ok: int = 0
    failed: int = 0
    total_tokens: int = 0
    wall_s: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def aggregate_tps(self) -> float:
        return self.total_tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
        return s[idx]


def run_load(label: str, one: Callable[[str], tuple[int, float]], prompts: list[str],
             concurrency: int) -> BenchResult:
    """Fire ``len(prompts)`` requests at the given concurrency; ``one(prompt)`` runs a
    single request and returns (tokens, latency_s)."""
    res = BenchResult(label=label, concurrency=concurrency, n_requests=len(prompts))

    def _do(p: str):
        try:
            return one(p), None
        except Exception as exc:  # noqa: BLE001
            return None, exc

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for out, err in pool.map(_do, prompts):
            if err is not None or out is None:
                res.failed += 1
                continue
            tokens, latency = out
            res.ok += 1
            res.total_tokens += tokens
            res.latencies.append(latency)
    res.wall_s = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(r: BenchResult) -> str:
    return (f"  {r.label:<22} ok={r.ok:<3} fail={r.failed:<3} "
            f"tokens={r.total_tokens:<6} wall={r.wall_s:6.1f}s  "
            f"AGG={r.aggregate_tps:6.1f} tok/s  "
            f"lat p50={r.p50_latency:5.1f}s p95={r.p95_latency:5.1f}s")


def print_report(baseline: Optional[BenchResult], pooled: Optional[BenchResult],
                 concurrency: int) -> None:
    print("\n" + "=" * 78)
    print(f"MeshPool throughput benchmark  (concurrency={concurrency})")
    print("=" * 78)
    if baseline:
        print(_fmt(baseline))
    if pooled:
        print(_fmt(pooled))
    if baseline and pooled and baseline.aggregate_tps > 0:
        speedup = pooled.aggregate_tps / baseline.aggregate_tps
        print("-" * 78)
        print(f"  aggregate throughput  pooled / single-box  =  {speedup:.2f}x")
        if concurrency <= 1:
            print("  (at concurrency=1 expect ~1x or slightly below: one request can't be")
            print("   split usefully on a LAN — the honest win needs concurrent load.)")
        elif speedup >= 1.15:
            print("  pooling added real aggregate capacity under concurrent load.")
        else:
            print("  little/no aggregate gain here — likely the replicas share one Ollama")
            print("  (co-located test) or the single box wasn't saturated. Use separate")
            print("  machines and raise --concurrency to see the pool pull ahead.")
    print("=" * 78)
    print("Note: per-request latency is NOT improved by pooling (often slightly worse by")
    print("one network hop). Pooling scales AGGREGATE throughput and runs models too big")
    print("for one box. That is the honest value proposition.\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _maybe_deploy(cp: str, model: str) -> None:
    print(f"  deploying {model!r} on the pool ...")
    r = _http().post(f"{cp.rstrip('/')}/api/v1/models", json={"model": model}, timeout=180.0)
    if r.status_code >= 400:
        raise RuntimeError(f"deploy failed [{r.status_code}]: {r.text[:200]}")
    view = r.json()
    print(f"  deployed: backend={view.get('backend')} status={view.get('status')} "
          f"replicas={view.get('replicas')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Honest single-box vs pooled throughput benchmark")
    ap.add_argument("--mode", choices=["ollama", "pool", "both"], default="both",
                    help="ollama=single-box baseline, pool=MeshPool coordinator, both=side by side")
    ap.add_argument("--model", default="qwen2.5-coder:7b",
                    help="model name (Ollama tag) for the baseline and --deploy")
    ap.add_argument("--cp", default="http://127.0.0.1:8080", help="MeshPool coordinator URL")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="local Ollama URL (baseline)")
    ap.add_argument("-c", "--concurrency", type=int, default=8, help="concurrent requests")
    ap.add_argument("-n", "--requests", type=int, default=0,
                    help="total requests (default: 2x concurrency)")
    ap.add_argument("--tokens", type=int, default=64, help="tokens to generate per request")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--deploy", action="store_true", help="deploy the model on the pool first")
    args = ap.parse_args()

    n = args.requests or (2 * args.concurrency)
    prompts = build_prompts(n)
    print(f"model={args.model}  requests={n}  concurrency={args.concurrency}  "
          f"tokens/req={args.tokens}")

    baseline: Optional[BenchResult] = None
    pooled: Optional[BenchResult] = None

    if args.mode in ("ollama", "both"):
        print(f"\n[baseline] hitting single Ollama at {args.ollama} ...")
        baseline = run_load(
            "single-box Ollama",
            lambda p: gen_ollama(args.ollama, args.model, p, args.tokens, args.temperature),
            prompts, args.concurrency)

    if args.mode in ("pool", "both"):
        if args.deploy:
            _maybe_deploy(args.cp, args.model)
        print(f"\n[pooled] hitting MeshPool coordinator at {args.cp} ...")
        pooled = run_load(
            "MeshPool (pooled)",
            lambda p: gen_pool(args.cp, p, args.tokens, args.temperature),
            prompts, args.concurrency)

    print_report(baseline, pooled, args.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
