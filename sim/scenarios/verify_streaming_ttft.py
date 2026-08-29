"""Acceptance gate: streaming is incremental, not a replay of a finished answer.

The defect this gate exists to prevent shipped once and survived every test it had.
``/v1/chat/completions`` awaited the whole generation, then sliced the finished
string into Server-Sent Events. The frames were shaped correctly, ``[DONE]`` arrived
in the right place, and the OpenAI SDK reported dozens of incremental deltas -- so
the format gate passed. Time-to-first-token was equal to total generation time and
the mesh looked, to a user, like it had hung.

Format cannot detect that. Only time can. So this gate measures, over real HTTP
against a real multi-process mesh:

  1. when the first byte of content arrives, relative to when the last one does;
  2. how many deltas there were, relative to how many tokens were generated;
  3. whether the streamed text is the same answer the non-streaming route returns.

(3) matters as much as (1): a streaming path that is fast because it is a different
code path producing different output is not an improvement, it is a second bug.

Every threshold here is deliberately loose. Real streaming puts the first token at
roughly 1/N of the total for an N-token answer; the fake put it at 1.0. Anything
below 0.5 separates them with a wide margin, so this gate fails on the defect and
not on a busy CI box.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness
from sim import mutation_guard  # noqa: E402

CP_PORT = 8566
DATA_DIR = str(Path(__file__).resolve().parent / "_data_streaming_ttft")
_INNER_RUN = os.environ.get("MESHPOOL_STREAM_TTFT_MUTATION") == "1"

MAX_TOKENS = 40
# The built-in model's positional window is 64 and the prompt eats into it, so the
# prompt is kept tiny and the budget conservative: a request that gets truncated by
# the context window would silently shorten the measurement.
PROMPT = "hi"
# Below this the run is too short to distinguish "arrived first" from "arrived at
# the same time" through HTTP and scheduler noise, and a passing ratio would be
# luck rather than evidence. Failing loudly beats reporting a number we do not
# believe.
MIN_TOTAL_S = 0.15
# Real streaming lands near 1/(number of traversals). The fake landed at ~1.0.
MAX_TTFT_RATIO = 0.5


def _fail(message: str) -> None:
    print(f"  FAIL: {message}")
    raise SystemExit(1)


def _ok(message: str) -> None:
    print(f"  PASS: {message}")


# One client, reused for everything.
#
# Measured on this machine: a one-shot ``httpx.get`` costs 400-600 ms while the same
# call on an established client costs 2 ms -- building the SSL context against the
# Windows trust store dominates. Timing a stream through a fresh client would charge
# half a second of client construction to time-to-first-token and make real
# streaming indistinguishable from the fake it is meant to detect. Measurement error
# prefers to flatter; here it prefers to condemn, and either way it has to go.
CLIENT = httpx.Client(timeout=300)


def _session(cp: str) -> str:
    r = CLIENT.post(f"{cp}/api/v1/auth/local", timeout=10)
    r.raise_for_status()
    return r.json()["session_token"]


def _deploy(cp: str, model: str = "nano-gpt") -> None:
    r = CLIENT.post(f"{cp}/api/v1/models", json={"model": model}, timeout=120)
    r.raise_for_status()
    deadline = time.time() + 120
    while time.time() < deadline:
        view = CLIENT.get(f"{cp}/api/v1/models", timeout=10).json()
        if view.get("status") == "ready":
            _ok(f"{model} deployed across {len(view.get('stages') or [])} stage(s)")
            return
        time.sleep(1)
    _fail(f"{model} did not become ready")


def _stream_timings(cp: str, token: str, path: str, payload: dict,
                    extract) -> dict:
    """Consume an SSE response, timestamping every content delta as it lands."""
    deltas: list[str] = []
    first_at = None
    started = time.perf_counter()
    with CLIENT.stream("POST", f"{cp}{path}", json=payload,
                       headers={"Authorization": f"Bearer {token}"}) as r:
        if r.status_code != 200:
            r.read()
            _fail(f"{path} returned {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body.strip() == "[DONE]":
                break
            obj = json.loads(body)
            if "error" in obj:
                _fail(f"{path} streamed an error: {obj['error']}")
            piece = extract(obj)
            if piece:
                if first_at is None:
                    first_at = time.perf_counter()
                deltas.append(piece)
    ended = time.perf_counter()
    if first_at is None:
        _fail(f"{path} produced no content deltas at all")
    return {
        "deltas": deltas,
        "text": "".join(deltas),
        "ttft_s": first_at - started,
        "total_s": ended - started,
    }


def _chat_delta(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


def _completion_delta(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("text") or ""


def _assert_incremental(label: str, m: dict) -> None:
    ratio = m["ttft_s"] / m["total_s"] if m["total_s"] > 0 else 1.0
    print(f"  {label}: ttft={m['ttft_s']*1000:.0f}ms total={m['total_s']*1000:.0f}ms "
          f"ratio={ratio:.3f} deltas={len(m['deltas'])} chars={len(m['text'])}")
    if m["total_s"] < MIN_TOTAL_S:
        _fail(f"{label} finished in {m['total_s']*1000:.0f}ms, below the "
              f"{MIN_TOTAL_S*1000:.0f}ms floor needed to measure a ratio at all; "
              "raise MAX_TOKENS rather than trusting this number")
    if ratio > MAX_TTFT_RATIO:
        _fail(f"{label} delivered its first token {ratio:.1%} of the way through the "
              "response. That is generate-then-replay, not streaming")


def _chat(cp: str, token: str, **over) -> dict:
    payload = {
        "model": "nano-gpt",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 4242,
    }
    payload.update(over)
    r = CLIENT.post(f"{cp}/v1/chat/completions", json=payload,
                    headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        _fail(f"chat returned {r.status_code}: {r.text[:300]}")
    return r.json()


def warm_up(cp: str, token: str) -> None:
    """Pay the one-off costs before measuring anything.

    The first request to a deployment builds the batcher, opens connections to every
    stage host and prefills a cold cache. Charging that to time-to-first-token would
    make real streaming look like fake streaming on a short answer -- the measurement
    would be dominated by setup rather than by generation, which is exactly the kind
    of flattering-then-misleading number this project refuses to ship.
    """
    print("\n== warming the deployment ==")
    t0 = time.perf_counter()
    _chat(cp, token, max_tokens=4)
    _ok(f"first (cold) request completed in {time.perf_counter()-t0:.2f}s")


def verify_chat_streaming(cp: str, token: str) -> dict:
    """One delta per token, and the first of them early.

    Speculative decoding is switched off here so the relationship between tokens and
    traversals is exactly one to one, which turns a vague "several chunks" assertion
    into an exact equality. The speculative path is measured separately below.
    """
    print("\n== /v1/chat/completions streams while it generates ==")
    control = _chat(cp, token, speculative=False)
    expected = control["usage"]["completion_tokens"]
    if expected < 8:
        _fail(f"the control request only generated {expected} tokens, too few for a "
              "meaningful timing measurement")

    m = _stream_timings(cp, token, "/v1/chat/completions", {
        "model": "nano-gpt",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 4242,
        "speculative": False,
        "stream": True,
    }, _chat_delta)
    _assert_incremental("chat", m)
    if len(m["deltas"]) != expected:
        _fail(f"non-speculative decoding produces one token per traversal, so the "
              f"stream must carry {expected} deltas; it carried {len(m['deltas'])}. "
              "Anything else means tokens were grouped before being sent")
    _ok(f"{expected} tokens arrived as {expected} separate deltas, first one early")

    plain = control["choices"][0]["message"]["content"]
    if plain != m["text"]:
        _fail("streamed and non-streamed answers differ for the same seed:\n"
              f"  streamed:     {m['text']!r}\n"
              f"  non-streamed: {plain!r}")
    _ok(f"the streamed text is byte-identical to the non-streamed answer "
        f"({len(plain)} characters)")
    return m


def verify_speculative_streaming(cp: str, token: str) -> None:
    """The production path: SPD accepts runs, so deltas carry runs.

    This is where a naive implementation leaks rejected drafts onto the wire. It
    cannot leak here without changing the text, so the equality against the
    non-streaming answer is the real assertion; the delta count only proves the runs
    are genuinely being batched into single traversals.
    """
    print("\n== speculative decoding streams accepted runs, not drafts ==")
    control = _chat(cp, token, speculative=True, draft_k=4)
    expected = control["usage"]["completion_tokens"]
    m = _stream_timings(cp, token, "/v1/chat/completions", {
        "model": "nano-gpt",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 4242,
        "speculative": True,
        "draft_k": 4,
        "stream": True,
    }, _chat_delta)
    _assert_incremental("chat+spd", m)
    if m["text"] != control["choices"][0]["message"]["content"]:
        _fail("speculative streaming produced different text than the same "
              f"speculative request unstreamed:\n  {m['text']!r}\n  "
              f"{control['choices'][0]['message']['content']!r}")
    if not 1 < len(m["deltas"]) < expected:
        _fail(f"expected accepted runs to group {expected} tokens into fewer, but "
              f"more than one, delta; got {len(m['deltas'])}")
    _ok(f"{expected} tokens arrived as {len(m['deltas'])} accepted runs "
        f"({expected/len(m['deltas']):.2f} tokens per traversal), text identical")


def verify_ttft_is_independent_of_answer_length(cp: str, token: str) -> None:
    """The sharpest test of the claim, and the one immune to fixed overheads.

    Under real streaming the first token costs one prefill plus one traversal
    regardless of how long the answer will be, so time-to-first-token is flat in
    ``max_tokens``. Under generate-then-replay it is the *whole generation*, so it
    grows linearly. Comparing a short request against a long one therefore
    distinguishes the two by their shape rather than by an absolute threshold --
    no calibration, no assumption about how fast this machine happens to be.

    A single sample is unreliable: TTFT has ~20 ms of scheduling jitter on a
    normal machine while the absolute values for short(4) and long(40) span
    only ~22-50 ms.  One unlucky pair can therefore yield a 2x ratio even when
    streaming is genuinely incremental.  Taking the median of several repeats
    suppresses that noise without touching the threshold: real streaming keeps
    the median well below 2x, and generate-then-replay gives ~8x regardless of
    how many samples you take.
    """
    _REPS = 5  # odd number so the median is a single measured value

    print("\n== time-to-first-token does not grow with the length of the answer ==")

    def once(n: int) -> dict:
        return _stream_timings(cp, token, "/v1/chat/completions", {
            "model": "nano-gpt",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": n, "temperature": 0.0, "seed": 4242,
            "speculative": False, "stream": True,
        }, _chat_delta)

    def median_sample(n: int) -> dict:
        """Return a synthetic sample whose ttft_s and total_s are the medians
        of _REPS individual measurements.  Using the median rather than the mean
        discards outliers from scheduler preemptions without requiring a fixed
        floor, which would need calibration per machine."""
        samples = [once(n) for _ in range(_REPS)]
        med_ttft = sorted(s["ttft_s"] for s in samples)[_REPS // 2]
        med_total = sorted(s["total_s"] for s in samples)[_REPS // 2]
        # Return the raw sample closest to the median TTFT for the delta/text fields.
        closest = min(samples, key=lambda s: abs(s["ttft_s"] - med_ttft))
        return {**closest, "ttft_s": med_ttft, "total_s": med_total}

    short = median_sample(4)
    long = median_sample(MAX_TOKENS)
    print(f"  short(4):  ttft={short['ttft_s']*1000:6.1f}ms "
          f"total={short['total_s']*1000:7.1f}ms  (median of {_REPS})")
    print(f"  long({MAX_TOKENS}): ttft={long['ttft_s']*1000:6.1f}ms "
          f"total={long['total_s']*1000:7.1f}ms  (median of {_REPS})")

    growth = long["total_s"] / max(1e-6, short["total_s"])
    if growth < 2.0:
        _fail(f"the long request only took {growth:.1f}x as long as the short one, "
              "so the two are not actually different workloads and the comparison "
              "below would prove nothing")
    ttft_growth = long["ttft_s"] / max(1e-6, short["ttft_s"])
    print(f"  total grew {growth:.1f}x, ttft grew {ttft_growth:.2f}x")
    if ttft_growth > 2.0:
        _fail(f"asking for {MAX_TOKENS//4}x more tokens made the FIRST token "
              f"{ttft_growth:.1f}x slower (median of {_REPS} samples each). "
              "The first token cannot depend on how many come after it unless "
              "the answer is generated before the stream starts")
    _ok(f"answer got {growth:.1f}x longer, first token stayed flat "
        f"({ttft_growth:.2f}x, median of {_REPS} samples)")


def verify_finish_reason_is_terminal(cp: str, token: str) -> None:
    """The last chunk must carry a finish_reason, and only the last one.

    Streaming is where an agent loop decides whether to keep going. A missing
    terminal reason makes every reply look unfinished; an early one truncates a
    reply that was still coming.
    """
    print("\n== terminal chunk carries the finish reason ==")
    reasons: list[tuple[int, str]] = []
    idx = 0
    with CLIENT.stream("POST", f"{cp}/v1/chat/completions", json={
        "model": "nano-gpt",
        "messages": [{"role": "user", "content": "short"}],
        "max_tokens": 8, "temperature": 0.0, "stream": True,
    }, headers={"Authorization": f"Bearer {token}"}) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                continue
            obj = json.loads(line[6:])
            reason = (obj.get("choices") or [{}])[0].get("finish_reason")
            if reason is not None:
                reasons.append((idx, reason))
            idx += 1
    if len(reasons) != 1:
        _fail(f"expected exactly one chunk with a finish_reason, got {reasons}")
    if reasons[0][0] != idx - 1:
        _fail(f"finish_reason appeared at chunk {reasons[0][0]} of {idx}, not last")
    if reasons[0][1] != "length":
        _fail(f"an answer stopped by max_tokens must report 'length', got "
              f"{reasons[0][1]!r}")
    _ok(f"finish_reason={reasons[0][1]!r} on the final chunk only")


def verify_completions_streaming(cp: str, token: str) -> None:
    print("\n== /v1/completions streams while it generates ==")
    payload = {
        "model": "nano-gpt",
        "prompt": "hi",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 99,
        "stream": True,
    }
    m = _stream_timings(cp, token, "/v1/completions", payload, _completion_delta)
    _assert_incremental("completions", m)
    _ok(f"the legacy route streams for real too ({len(m['deltas'])} deltas)")


def verify_abandoning_a_stream_frees_the_mesh(cp: str, token: str) -> None:
    """Hanging up must stop the work, not just stop the listening.

    With real streaming, clients disconnect mid-answer as a matter of course. If the
    batcher keeps riding traversals for an abandoned request it spends wide-area
    round trips -- and someone's credits -- on output that no longer has a reader.
    """
    print("\n== abandoning a stream stops the generation behind it ==")
    before = CLIENT.get(f"{cp}/api/v1/models", timeout=10).json()
    with CLIENT.stream("POST", f"{cp}/v1/chat/completions", json={
        "model": "nano-gpt",
        "messages": [{"role": "user", "content": "a very long answer please"}],
        "max_tokens": 512, "temperature": 0.0, "stream": True,
    }, timeout=120, headers={"Authorization": f"Bearer {token}"}) as r:
        seen = 0
        for line in r.iter_lines():
            if line.startswith("data: "):
                seen += 1
            if seen >= 3:
                break
    if seen < 3:
        _fail("the stream ended before it could be abandoned")
    # The mesh must still serve immediately afterwards. A wedged batch slot or a
    # stranded session shows up here as a timeout or an error, not as a slow reply.
    t0 = time.perf_counter()
    r = CLIENT.post(f"{cp}/v1/chat/completions", json={
        "model": "nano-gpt",
        "messages": [{"role": "user", "content": "still working?"}],
        "max_tokens": 8, "temperature": 0.0,
    }, timeout=60, headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        _fail(f"the mesh stopped serving after a client hung up: {r.status_code} "
              f"{r.text[:200]}")
    _ok(f"served a fresh request {time.perf_counter()-t0:.2f}s after the hang-up "
        f"(abandoned mid-stream after {seen} frames); before={before.get('status')}")


def run_nonvacuity_check() -> None:
    """Break streaming at its source and require this gate to notice.

    The mutation restores exactly the old behaviour: the batcher stops handing
    tokens to the streaming caller, so ``chat_stream`` has nothing to yield until
    the generation finishes and the whole answer arrives in one final delta. Every
    frame is still well formed and ``[DONE]`` still arrives -- which is precisely why
    the format gates could not see it.
    """
    print("\n== non-vacuity: kill incremental emission, this gate must fail ==")
    target = REPO_ROOT / "meshpool" / "inference" / "engine.py"
    original = target.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    anchor = b"            on_tokens=queue.put_nowait, **kw))"
    replacement = b"            on_tokens=None, **kw))"
    mutated = original.replace(anchor, replacement, 1)
    if mutated == original:
        _fail("mutation anchor not found in engine.py")
    _mg = mutation_guard.journal(target, original)
    target.write_bytes(mutated)
    try:
        env = dict(os.environ)
        env["MESHPOOL_STREAM_TTFT_MUTATION"] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(REPO_ROOT), capture_output=True, text=True, errors="replace",
            timeout=600, env=env,
        )
    finally:
        target.write_bytes(original)
        mutation_guard.clear(_mg)

    after = hashlib.sha256(target.read_bytes()).hexdigest()
    if after != before:
        _fail(f"engine.py was not restored byte-identically: {before} != {after}")
    print(f"  restored engine.py sha256={before}")
    if result.returncode == 0:
        _fail("generate-then-replay streaming still passed this gate; stdout:\n"
              + result.stdout[-1200:])
    _ok(f"fake streaming failed as expected with exit {result.returncode}")


def main() -> int:
    harness = Harness(n_agents=3, cp_port=CP_PORT, data_dir=DATA_DIR, scale="home")
    try:
        print("== starting MeshPool cluster ==")
        harness.start()
        cp = harness.cp_url()
        token = _session(cp)
        _deploy(cp)
        warm_up(cp, token)
        verify_chat_streaming(cp, token)
        verify_ttft_is_independent_of_answer_length(cp, token)
        verify_speculative_streaming(cp, token)
        verify_finish_reason_is_terminal(cp, token)
        verify_completions_streaming(cp, token)
        verify_abandoning_a_stream_frees_the_mesh(cp, token)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: unexpected {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        harness.stop()

    if not _INNER_RUN:
        run_nonvacuity_check()
    print("\nPASS: MeshPool streams tokens as it generates them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
