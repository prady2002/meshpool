"""Weight Swarm: a machine that has never seen a model must be able to serve it.

Plan section 3.9. Replacing a node that holds 69 GB of weights by re-downloading them
from an origin at 1 Gbps is ~9 minutes of downtime, and it is the reason Petals swarms
were so often empty: the answer to "is anyone online with the model you want?" was
usually no. MeshPool's answer is content-addressed weight groups fetched **peer to
peer**, so the coordinator moves ids and peer URLs and never the weights themselves,
and any node that already holds a group serves it to the next.

That design only holds up if the integrity story is airtight, because peer-to-peer
weight distribution is the single most attractive attack surface in the product. A
node that serves subtly wrong weight bytes does not crash anything. The model loads,
answers fluently, and is quietly wrong -- the hardest possible failure to notice and
the easiest to blame on the model. Content addressing is what is supposed to make
that impossible, so this gate attacks it directly rather than taking it on trust.

What this proves, in order:

1. a pool of machines with no weights on disk goes from an uploaded checkpoint to
   serving, within a budget, and serves *those* weights (byte-identical to a
   single-process decode of the uploaded checkpoint, and different from the built-in
   one -- the gap is the proof the bytes really travelled);
2. every weight group reached its node by a direct peer pull, with the control plane
   carrying none of the bytes;
3. a node that joins later -- cold, with an empty blob store -- can be handed a stage
   and serves the same answer;
4. a dishonest peer is detected and routed around: with one holder serving wrong
   bytes, the pull still succeeds from an honest one and what lands on disk verifies;
5. **when every holder is dishonest the pull fails closed** -- nothing is stored, no
   half-verified bytes are kept, and the node does not end up hosting a poisoned
   shard. This is the one that matters. A system that falls back to "well, somebody
   gave me *something*" is worse than one with no replication at all;
6. re-fetching a group a node already holds moves zero bytes.

Check 5 is paired with an explicit non-vacuity probe: the gate first calls the
dishonest peer directly and proves it really is returning bytes of the right length
that hash to the wrong id. Without that, "the pull failed" would be indistinguishable
from "the peer was down", and the gate would pass on a mesh where nothing lied.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.inference import model as M  # noqa: E402

CP_PORT = 8182
DATA_DIR = str(Path(__file__).resolve().parent / "_data_weightswarm")
N_AGENTS = 3
PROMPT = "MeshPool turns idle machines into "
N_TOK = 24

# Budget for "an empty pool becomes a serving pool". Generous on purpose: this runs
# three agent processes plus a control plane on one desktop, so it is a regression
# tripwire, not a performance claim. The real number is printed either way.
COLDSTART_BUDGET_S = 90.0

# One persistent client: a one-shot httpx call pays 400-600 ms building the Windows
# trust store, which would dominate every duration measured here.
CLIENT = httpx.Client(timeout=300)

_failures: list[str] = []


def _fail(msg: str) -> None:
    _failures.append(msg)
    print(f"  FAIL: {msg}")


def _ok(msg: str) -> None:
    print(f"  PASS: {msg}")


def greedy(P: dict, cfg, itos, prompt: str, n: int) -> str:
    """Single-process greedy decode -- the oracle the distributed run must match."""
    tok = M.CharTokenizer(itos=itos, vocab_size=cfg.vocab_size)
    ids = tok.encode(prompt) or [tok.encode(" ")[0]]
    out = M.generate_with(lambda ctx: M.forward_full(P, cfg, ctx), cfg, ids,
                          max_new_tokens=n, temperature=0.0)
    return tok.decode(out)


def chat(cp: str, mid: str) -> str:
    r = CLIENT.post(f"{cp}/api/v1/inference/chat",
                    json={"prompt": PROMPT, "max_new_tokens": N_TOK,
                          "temperature": 0.0, "model_id": mid}, timeout=180)
    r.raise_for_status()
    return r.json()["text"]


def summary(cp: str) -> dict:
    return CLIENT.get(f"{cp}/api/v1/cluster", timeout=30).json()["summary"]


def nodes(cp: str) -> list[dict]:
    return CLIENT.get(f"{cp}/api/v1/nodes", timeout=30).json()["nodes"]


def model_view(cp: str, mid: str) -> dict:
    return CLIENT.get(f"{cp}/api/v1/models/{mid}", timeout=30).json()


def agent_url(h: Harness, idx: int) -> str:
    return f"http://{h.host}:{h.agent_ports[idx]}"


def blob_has(h: Harness, idx: int, bid: str) -> bool:
    r = CLIENT.post(f"{agent_url(h, idx)}/blob_has", json={"blob_id": bid}, timeout=30)
    r.raise_for_status()
    return bool(r.json().get("has"))


def blob_get_raw(h: Harness, idx: int, bid: str) -> bytes:
    """Exactly what a peer would hand over, corruption included."""
    import base64
    r = CLIENT.post(f"{agent_url(h, idx)}/blob_get", json={"blob_id": bid}, timeout=60)
    r.raise_for_status()
    body = r.json()
    if not body.get("found"):
        return b""
    return base64.b64decode(body.get("data", ""), validate=True)


def main() -> int:  # noqa: C901 - a gate reads best as one linear narrative
    h = Harness(n_agents=N_AGENTS, cp_port=CP_PORT, data_dir=DATA_DIR, scale="home")
    try:
        print(f"== starting a cold pool ({N_AGENTS} agents, no weights anywhere) ==")
        h.start()
        cp = h.cp_url()
        h.wait_nodes_online(N_AGENTS)
        cfg = M.REGISTRY["nano-gpt"]

        # A checkpoint deliberately different from the one every node ships with, so
        # matching it proves the served weights travelled through the pool rather than
        # being read off local disk.
        pooled_cfg = dataclasses.replace(cfg, seed=90210)
        pooled_params = M.random_params(pooled_cfg)
        itos = M.default_charset(cfg.vocab_size)
        raw = M.dumps_npz(pooled_params, itos=itos)
        oracle = greedy(pooled_params, cfg, itos, PROMPT, N_TOK)
        builtin_P, builtin_meta = M.load_params(cfg)
        builtin_oracle = greedy(builtin_P, cfg, (builtin_meta or {}).get("itos"),
                                PROMPT, N_TOK)
        if oracle == builtin_oracle:
            _fail("test integrity: the uploaded checkpoint decodes to the same text as "
                  "the built-in one, so matching it would prove nothing")
            raise SystemExit
        print(f"  checkpoint: {len(pooled_params)} tensors, {len(raw)} bytes")

        # ---- 1: an empty pool becomes a serving pool ---------------------- #
        print("\n== 1: cold start — upload to serving ==")
        t0 = time.perf_counter()
        r = CLIENT.post(f"{cp}/api/v1/models/upload?model=nano-gpt",
                        content=raw, timeout=300)
        r.raise_for_status()
        view = r.json()
        cold_s = time.perf_counter() - t0
        mid = view["id"]
        print(f"  {cold_s:.1f}s to ready, {len(view['stages'])} stage(s): "
              + ", ".join(f"{s['node_name']}[{s['first']}:{s['last']}]"
                          for s in view["stages"]))
        if not (view.get("deployed") and view.get("status") == "ready"):
            _fail(f"model did not reach ready: {view.get('status')!r}")
            raise SystemExit
        if view.get("weight_groups") != cfg.n_layer + 3:
            _fail(f"checkpoint was not split per layer "
                  f"(groups={view.get('weight_groups')}, want {cfg.n_layer + 3})")
        if cold_s > COLDSTART_BUDGET_S:
            _fail(f"cold start took {cold_s:.1f}s, over the {COLDSTART_BUDGET_S:.0f}s "
                  f"budget — a node join is supposed to be cheap enough that churn is "
                  f"survivable rather than an outage")
        else:
            _ok(f"a pool with no weights on disk was serving in {cold_s:.1f}s")

        served = chat(cp, mid)
        if served != oracle:
            _fail(f"served text does not match a single-process decode of the uploaded "
                  f"checkpoint\n    want={oracle!r}\n    got ={served!r}")
        elif served == builtin_oracle:
            _fail("served text matches the BUILT-IN checkpoint; the uploaded weights "
                  "never actually reached the nodes")
        else:
            _ok(f"the pool serves the uploaded weights, not its own "
                f"({len(served)} chars, byte-identical to the oracle)")

        # ---- 2: the coordinator carried no weight bytes ------------------- #
        print("\n== 2: the weights moved peer-to-peer, not through the coordinator ==")
        s = summary(cp)
        pulls = s.get("weight_group_pulls", 0)
        pushes = s.get("weight_group_pushes", 0)
        cached = s.get("weight_group_cached", 0)
        print(f"  {pulls} peer pull(s), {pushes} coordinator relay(s), "
              f"{cached} already-resident")
        if pulls + pushes < 1:
            _fail("no weight group was distributed at all — nothing here was measured")
        elif pushes:
            _fail(f"the coordinator relayed {pushes} weight group(s); on a LAN pool "
                  f"every group should arrive by direct peer pull")
        else:
            _ok(f"all {pulls} weight group placements were direct peer pulls")

        # ---- 3: a machine that has never seen this model joins ------------ #
        print("\n== 3: a cold machine joins an already-serving pool ==")
        cold_idx = N_AGENTS
        h.start_agent(cold_idx)
        h.wait_nodes_online(N_AGENTS + 1)
        newcomer = next((n for n in nodes(cp)
                         if n.get("name") == f"sim-node-{cold_idx + 1}"), None)
        if newcomer is None:
            _fail("the newly started agent never appeared in the registry")
            raise SystemExit
        print(f"  {newcomer['name']} joined ({newcomer['id']})")

        victim = view["stages"][0]["node_name"]
        victim_idx = int(victim.rsplit("-", 1)[1]) - 1
        before = summary(cp)
        t0 = time.perf_counter()
        h.stop_agent(victim_idx, fast=True)
        print(f"  killed {victim}; waiting for the pool to re-form around the newcomer")

        deadline = time.time() + 180
        live = model_view(cp, mid)
        while time.time() < deadline:
            live = model_view(cp, mid)
            serving = [st["node_name"] for st in live.get("stages", [])]
            if live.get("status") == "ready" and victim not in serving:
                break
            time.sleep(2.0)
        heal_s = time.perf_counter() - t0
        serving = [st["node_name"] for st in live.get("stages", [])]
        after = summary(cp)
        print(f"  {heal_s:.1f}s to re-form; serving={serving}")
        if victim in serving or live.get("status") != "ready":
            _fail(f"the pool never re-formed without {victim} "
                  f"(status={live.get('status')!r}, serving={serving})")
        else:
            _ok(f"re-partitioned onto the survivors in {heal_s:.1f}s")

        moved = ((after.get("weight_group_pulls", 0) - before.get("weight_group_pulls", 0))
                 + (after.get("weight_group_pushes", 0)
                    - before.get("weight_group_pushes", 0)))
        if f"sim-node-{cold_idx + 1}" in serving and moved < 1:
            _fail("the newcomer is serving a stage but no weight group was transferred "
                  "to it — it cannot have the layers it claims to host")
        healed = chat(cp, mid)
        if healed != oracle:
            _fail(f"after re-forming, the answer changed\n    want={oracle!r}\n"
                  f"    got ={healed!r}")
        else:
            _ok(f"the re-formed pool still serves the uploaded weights, "
                f"byte-identical ({moved} group(s) moved to do it)")

        # ---- 4/5/6: content addressing under a dishonest peer ------------- #
        # Driven through a known pool object rather than a weight group so the gate
        # can name the exact id, the exact holders and the exact bytes. The endpoint
        # under test (/blob_pull) and the verification path are the same ones the
        # weight distributor uses.
        print("\n== 4: a dishonest peer is detected and routed around ==")
        payload = b"weight-swarm integrity probe\n" + bytes(range(256)) * 64
        want_id = hashlib.sha256(payload).hexdigest()
        pr = CLIENT.post(f"{cp}/api/v1/blobs?name=probe.bin&replication=2",
                         content=payload, timeout=120)
        pr.raise_for_status()
        meta = pr.json()
        if meta["id"] != want_id:
            _fail(f"the pool stored the probe under {meta['id'][:12]} but its sha256 "
                  f"is {want_id[:12]} — content addressing is already broken")
            raise SystemExit
        by_id = {n["id"]: n for n in nodes(cp)}
        holders = [by_id[nid]["name"] for nid in meta.get("replicas", []) if nid in by_id]
        holder_idx = [int(n.rsplit("-", 1)[1]) - 1 for n in holders]
        print(f"  probe {want_id[:12]}… ({len(payload)} bytes) held by {holders}")
        if len(holder_idx) < 2:
            _fail(f"probe landed on {len(holder_idx)} node(s); the routed-around check "
                  f"needs one dishonest holder and one honest one")
            raise SystemExit
        if blob_has(h, cold_idx, want_id):
            _fail("the cold node already holds the probe; it is not a cold fetch")

        liar, honest = holder_idx[0], holder_idx[1]
        h.set_corrupt_blobs(liar, True)

        # Non-vacuity: prove the liar really is lying, in the specific way that only a
        # hash can catch -- right length, valid base64, wrong content.
        bad = blob_get_raw(h, liar, want_id)
        if len(bad) != len(payload):
            _fail(f"the dishonest peer returned {len(bad)} bytes instead of "
                  f"{len(payload)}; a length check would catch that, so this would not "
                  f"be testing content addressing at all")
        elif hashlib.sha256(bad).hexdigest() == want_id:
            _fail("the dishonest peer served correct bytes — the fault injection did "
                  "not take effect and everything below would pass vacuously")
        else:
            _ok(f"sim-node-{liar + 1} is serving {len(bad)} well-formed bytes that hash "
                f"to the wrong id")

        plan = {"blob_id": want_id,
                "sources": [agent_url(h, liar), agent_url(h, honest)]}
        rr = CLIENT.post(f"{agent_url(h, cold_idx)}/blob_pull", json=plan, timeout=180)
        if rr.status_code != 200:
            _fail(f"the pull failed ({rr.status_code}) even though an honest source was "
                  f"listed; one bad peer must not deny service: {rr.text[:200]}")
        else:
            got = blob_get_raw(h, cold_idx, want_id)
            if hashlib.sha256(got).hexdigest() != want_id:
                _fail("the node stored bytes that do not match the id it stored them "
                      "under — a poisoned shard on disk")
            else:
                _ok("skipped the liar, pulled from the honest peer, and what landed on "
                    "disk verifies against its id")

        print("\n== 5: when every holder lies, the pull fails closed ==")
        CLIENT.post(f"{agent_url(h, cold_idx)}/blob_delete",
                    json={"blob_id": want_id}, timeout=30).raise_for_status()
        if blob_has(h, cold_idx, want_id):
            _fail("could not clear the probe from the cold node; check 5 would be "
                  "testing a cache hit")
            raise SystemExit
        for i in holder_idx:
            h.set_corrupt_blobs(i, True)
        rr = CLIENT.post(f"{agent_url(h, cold_idx)}/blob_pull",
                         json={"blob_id": want_id,
                               "sources": [agent_url(h, i) for i in holder_idx]},
                         timeout=180)
        stored = blob_has(h, cold_idx, want_id)
        print(f"  pull returned {rr.status_code}; node holds it afterwards: {stored}")
        if rr.status_code == 200:
            _fail("the pull reported success with no honest source available — the "
                  "node accepted bytes nobody could vouch for")
        elif stored:
            _fail("the pull failed but the node kept the bytes anyway; a shard loaded "
                  "from these would be silently, plausibly wrong")
        else:
            _ok(f"refused every copy and stored nothing ({rr.status_code}); a poisoned "
                f"mesh cannot hand you a poisoned model")
        for i in holder_idx:
            h.set_corrupt_blobs(i, None)

        # A dishonest peer must not have disturbed the running model either.
        after_liars = chat(cp, mid)
        if after_liars != oracle:
            _fail("the served answer changed while dishonest peers were active")
        else:
            _ok("the live model kept answering byte-identically throughout")

        print("\n== 6: re-fetching something already held moves no bytes ==")
        CLIENT.post(f"{agent_url(h, cold_idx)}/blob_pull",
                    json={"blob_id": want_id,
                          "sources": [agent_url(h, i) for i in holder_idx]},
                    timeout=180).raise_for_status()
        again = CLIENT.post(f"{agent_url(h, cold_idx)}/blob_pull",
                            json={"blob_id": want_id,
                                  "sources": [agent_url(h, i) for i in holder_idx]},
                            timeout=180)
        again.raise_for_status()
        body = again.json()
        print(f"  second pull: cached={body.get('cached')} bytes={body.get('bytes')}")
        if not body.get("cached") or body.get("bytes"):
            _fail(f"a node re-fetched {body.get('bytes')} bytes it already had; with "
                  f"real weights that is gigabytes of pointless transfer on every "
                  f"re-placement, which is exactly the cold-start cost this is meant "
                  f"to remove")
        else:
            _ok("already-resident content is recognised and nothing is transferred")

    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _fail(f"unexpected {type(exc).__name__}: {exc}")
    finally:
        h.stop()

    print()
    if _failures:
        print(f"SOME WEIGHT-SWARM CHECKS FAILED ({len(_failures)}):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS: cold machines join, fetch weights from peers, and no dishonest peer "
          "can change what the model says")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
