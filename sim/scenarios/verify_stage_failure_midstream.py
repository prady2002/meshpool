"""Acceptance gate: a pipeline stage's node dies MID-GENERATION and the user's
answer is unchanged (plan §3.4, ``verify_stage_failure_midstream``).

This is a strictly stronger property than the one ``verify_inference.py`` test 4
proves. That gate kills a node *between* generations and shows the control plane
re-partitions and serves again. Valuable, but it does not help the request that was
in flight when the node died -- that one fails, and re-partitioning takes seconds
the user is sitting through.

Hot spares close that gap. Each stage is loaded onto a warm standby where the pool
can afford one; a transport failure rotates to it and replays the session's history
through the new node (O(T) through ONE stage) before continuing. What must be true:

  1. The pool actually places warm spares, and says how many stages are protected.
  2. Killing a spare-protected node while tokens are streaming does not fail the
     stream.
  3. The completed text is **byte-identical** to a run where nothing was killed.
     "The stream kept flowing" is not evidence of correctness; a spare that answered
     from a cold cache would also keep flowing.
  4. The failover really happened -- the deployment reports the spare is now serving
     and the dead node is retired. Without this the gate would pass just as happily
     if the kill had silently done nothing.
  5. The kill targets the FIRST stage whenever that stage has a spare, because it
     is the one no activation cache can rebuild (boundary 0 is raw tokens, which
     the cache deliberately does not store) — it is recovered from the token
     journal instead. When stage 0 is unprotected the gate falls back to whichever
     stage does have a standby and says so.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402

_CLIENT: httpx.Client | None = None


def _http() -> httpx.Client:
    """One client for the whole gate.

    httpx.get/post build a fresh Client per call and Client construction loads the
    system TLS trust store (~0.7 s on this machine). A gate that kills a node at a
    wall-clock moment cannot afford to have most of its wall clock be certificate
    parsing.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(timeout=120.0)
    return _CLIENT


def _stream_completion(cp: str, model_id: str, prompt: str, max_tokens: int,
                       on_token=None) -> str:
    """Stream a completion, calling ``on_token(i)`` after each chunk."""
    return _stream_timed(cp, model_id, prompt, max_tokens, on_token)["text"]


def _stream_timed(cp: str, model_id: str, prompt: str, max_tokens: int,
                  on_token=None) -> dict:
    """Stream a completion and record when each delta actually landed.

    The arrival timestamps are what make a mid-stream kill provable. Without them
    a gate can only assert "the text came out right", which is also what happens
    when the server finished generating before the kill ever landed.
    """
    text, i = [], 0
    at: list[float] = []
    with _http().stream("POST", f"{cp}/v1/chat/completions", json={
            "model": model_id, "stream": True, "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            piece = chunk["choices"][0].get("delta", {}).get("content", "")
            if piece:
                text.append(piece)
                at.append(time.time())
                i += 1
                if on_token is not None:
                    on_token(i)
    return {"text": "".join(text), "at": at}


def _deploy(cp: str) -> dict:
    r = _http().post(f"{cp}/api/v1/models", json={"model": "nano-gpt"}, timeout=120)
    r.raise_for_status()
    return r.json()


def _view(cp: str, model_id: str) -> dict:
    return _http().get(f"{cp}/api/v1/models/{model_id}", timeout=30).json()


def main() -> int:
    ok = True
    # 4 agents: three carry the pipeline, the fourth exists to be a spare.
    h = Harness(n_agents=4, cp_port=8175,
                data_dir=str(Path(__file__).resolve().parent / "_data_midstream"))
    try:
        print("== starting cluster (4 agents: 3 stages + standby capacity) ==")
        h.start()
        cp = h.cp_url()
        h.wait_nodes_online(4)

        # ---- test 1: warm spares are actually placed ---------------------- #
        print("\n== test 1: the pool places warm standby nodes ==")
        view = _deploy(cp)
        model_id = view["id"]
        view = _view(cp, model_id)
        stages = view["stages"]
        protected = view.get("protected_stages", 0)
        print(f"  {len(stages)} stage(s): "
              + ", ".join(f"{s['node_name']}[{s['first']}:{s['last']}]" for s in stages))
        print(f"  spares: {view.get('spares')}")
        if protected < 1:
            ok = False
            print(f"  FAIL: no stage got a warm spare (protected_stages={protected}); "
                  f"nothing below can be tested")
            raise SystemExit
        print(f"  PASS: {protected}/{len(stages)} stage(s) have a warm standby")

        # ---- test 2: baseline, nothing killed ----------------------------- #
        print("\n== test 2: baseline completion (nothing is killed) ==")
        PROMPT = "the "
        MAXTOK = 48
        baseline = _stream_completion(cp, model_id, PROMPT, MAXTOK)
        print(f"  {len(baseline)} chars: {baseline[:60]!r}...")
        if not baseline:
            ok = False
            print("  FAIL: baseline produced no text")
            raise SystemExit
        print(f"  PASS: baseline is {len(baseline)} chars")

        # ---- test 3: kill a protected node MID-STREAM --------------------- #
        print("\n== test 3: kill a protected stage node while tokens stream ==")
        spares = view.get("spares", {})
        # Prefer the FIRST stage when it has a spare. That is the hardest case: the
        # client-side activation cache deliberately does not store boundary 0 (raw
        # tokens), so stage 0 is the one stage no activation replay can rebuild --
        # it is recovered from the token journal instead. Killing it exercises the
        # path the other stages do not.
        first_stage_node = stages[0]["node_name"]
        victim_name = (first_stage_node if first_stage_node in spares
                       else sorted(spares)[0])
        victim_stage = next((i for i, s in enumerate(stages)
                             if s["node_name"] == victim_name), -1)
        # sim-node-N is agent index N-1 (harness.start_agent names idx as idx+1).
        victim_idx = int(victim_name.rsplit("-", 1)[1]) - 1
        print(f"  will kill {victim_name} (stage {victim_stage} of {len(stages)}, "
              f"agent idx {victim_idx}, spare: {spares[victim_name]}) "
              f"after ~1/3 of the stream")
        if victim_stage == 0:
            print("  (this is the first stage — recovery must come from the token "
                  "journal, not the activation cache)")

        killed = threading.Event()
        kill_done_at: list[float] = []
        fire_at = max(2, MAXTOK // 6)

        def kill_once(i: int) -> None:
            # The kill runs on its own thread. Calling taskkill inline would stop
            # the client reading the socket for the ~100-300 ms the kill takes,
            # during which the server happily fills the TCP buffer with the rest
            # of a 44-token answer -- and the gate would then "survive a failure"
            # that landed after the work was already done.
            if i != fire_at or killed.is_set():
                return
            killed.set()

            def do_kill() -> None:
                h.stop_agent(victim_idx, fast=True)
                kill_done_at.append(time.time())
                print(f"  >> killed {victim_name} at token {i}")

            threading.Thread(target=do_kill, daemon=True).start()

        t0 = time.time()
        try:
            res = _stream_timed(cp, model_id, PROMPT, MAXTOK, on_token=kill_once)
            after = res["text"]
        except Exception as exc:  # noqa: BLE001
            ok = False
            after = ""
            res = {"at": []}
            print(f"  FAIL: the stream died with {type(exc).__name__}: {exc}")
        elapsed = time.time() - t0

        # How many tokens were produced after the victim was provably gone? This is
        # the number that decides whether anything was tested at all.
        after_kill = 0
        if kill_done_at:
            after_kill = sum(1 for ts in res["at"] if ts > kill_done_at[0])

        if not killed.is_set():
            ok = False
            print(f"  FAIL: the stream produced fewer than {fire_at} chunks, so the "
                  f"node was never killed — this gate proved nothing")
        elif after_kill < 3:
            ok = False
            print(f"  FAIL: only {after_kill} token(s) arrived after {victim_name} "
                  f"was confirmed dead. The answer was essentially already generated "
                  f"when the kill landed, so surviving it proves nothing. Make the "
                  f"generation longer or slower than the kill.")
        elif after == baseline:
            print(f"  PASS: {after_kill} of {len(res['at'])} tokens were generated "
                  f"after {victim_name} was confirmed dead, and the text is "
                  f"IDENTICAL to baseline ({len(after)} chars, {elapsed:.2f}s)")
        else:
            ok = False
            print(f"  FAIL: text changed after failover\n"
                  f"    baseline={baseline!r}\n"
                  f"    after   ={after!r}")

        # ---- test 4: non-vacuity — the failover really happened ----------- #
        print("\n== test 4: the deployment shows the spare took over ==")
        # Counts are persisted across batcher retirement on purpose, so this stays
        # a valid oracle even if the control plane noticed the dead node and
        # re-partitioned between the stream ending and this read.
        live = _view(cp, model_id)
        failovers = live.get("failovers")
        serving = live.get("serving_nodes", [])
        retired = live.get("retired_nodes", [])
        print(f"  failovers={failovers} hydrations={live.get('hydrations')} "
              f"serving={serving} retired={retired}")
        if not failovers:
            ok = False
            print(f"  FAIL: the deployment reports {failovers} failovers — the kill "
                  f"did not actually force a rotation, so test 3's PASS is "
                  f"meaningless (it would pass with hot spares removed entirely)")
        elif victim_name in serving:
            ok = False
            print(f"  FAIL: {victim_name} was killed but is still listed as serving")
        elif victim_name not in retired:
            ok = False
            print(f"  FAIL: {victim_name} was killed but is not listed as retired")
        else:
            print(f"  PASS: {victim_name} retired, {spares[victim_name]} serving, "
                  f"{failovers} failover(s), {live.get('hydrations')} rebuild(s)")

    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            h.stop()
        finally:
            if _CLIENT is not None:
                _CLIENT.close()

    print("\nALL MIDSTREAM STAGE-FAILURE CHECKS PASSED" if ok
          else "\nSOME MIDSTREAM STAGE-FAILURE CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
