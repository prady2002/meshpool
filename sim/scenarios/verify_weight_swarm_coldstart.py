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


def durability(cp: str, mid: str) -> dict:
    r = CLIENT.get(f"{cp}/api/v1/models/{mid}/durability", timeout=30)
    r.raise_for_status()
    return r.json()


def wait_reform(cp: str, mid: str, gone: str, timeout: float = 180) -> tuple[dict, float]:
    """Block until the deployment is ready again without ``gone`` in it."""
    t0 = time.perf_counter()
    deadline = time.time() + timeout
    live = model_view(cp, mid)
    while time.time() < deadline:
        live = model_view(cp, mid)
        serving = [st["node_name"] for st in live.get("stages", [])]
        if live.get("status") == "ready" and gone not in serving:
            break
        time.sleep(2.0)
    return live, time.perf_counter() - t0


def _await_redundancy(cp: str, mid: str, timeout: float = 150.0) -> dict:
    """Poll until residency restoration settles, then return the durability snapshot.

    ``wait_reform`` returns as soon as the deployment can SERVE again -- status
    ``ready`` with the departed machine out of the stage list. Putting the *second*
    copy of each layer group back is separate background work that continues past
    that point, so sampling durability at the instant of readiness races it.

    That race had a real cost. Inside the mutation meta-gate, which runs dozens of
    mutation trials and saturates the machine, the background restoration lost the
    race and check 9 failed on UNMUTATED code. A guard that fails on working code
    would have failed whatever the mutation did, so it silently converted two
    mutations' "caught" verdicts into evidence of nothing -- the meta-gate reported
    them as vacuous rather than caught, which is the only reason this was noticed.

    Polling to convergence is the honest form of the claim: redundancy is restored,
    not restored within one round trip. If it never converges the check below still
    fails, and reports the final state.
    """
    deadline = time.time() + timeout
    dur = durability(cp, mid)
    while time.time() < deadline:
        dur = durability(cp, mid)
        if all(len(g["nodes"]) >= 2 for g in dur.get("per_group", [])):
            break
        time.sleep(2.0)
    return dur


def agent_url(h: Harness, idx: int) -> str:
    return f"http://{h.host}:{h.agent_ports[idx]}"


def live_indices(cp: str) -> list[int]:
    """Agent indices of the nodes the coordinator currently considers online."""
    out = []
    for n in nodes(cp):
        if n.get("status") == "online" and str(n.get("name", "")).startswith("sim-node-"):
            out.append(int(n["name"].rsplit("-", 1)[1]) - 1)
    return sorted(out)


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

        dur0 = durability(cp, mid)
        print(f"  durability at upload: {dur0['groups']} groups, "
              f"min {dur0['min_holders']} holder(s) each, "
              f"{dur0['total_placements']} placements total")

        # Placement alone puts a group on the node that *serves* it, which for a
        # 3-stage pipeline is exactly one machine per group. That model reads as
        # healthy and answers correctly while being one departure away from losing a
        # layer permanently -- and a layer cannot be recovered from the other layers.
        # Warm residency exists to close that window at deploy time rather than
        # waiting for the pool to happen to get used.
        live_at_deploy = len(live_indices(cp))
        if live_at_deploy < 2:
            _fail(f"only {live_at_deploy} machine(s) online, so a second holder was "
                  f"never possible and this check proves nothing")
        elif dur0["min_holders"] < 2:
            thin = [g["blob"][:12] for g in dur0["per_group"] if len(g["nodes"]) < 2]
            _fail(f"freshly deployed and {len(thin)} layer group(s) sit on a single "
                  f"machine ({thin[:4]}); the model is one departure away from being "
                  f"permanently unservable and nothing is trying to fix that")
        else:
            _ok(f"every one of {dur0['groups']} layer group(s) is already on >=2 of "
                f"the {live_at_deploy} machines at deploy time, before any node has "
                f"left — the model can survive a departure it has not suffered yet")

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
        h.stop_agent(victim_idx, fast=True)
        print(f"  killed {victim}; waiting for the pool to re-form around the newcomer")

        live, heal_s = wait_reform(cp, mid, victim)
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
        # The puller has to be a node that does *not* already hold the probe. Which
        # nodes the pool replicated it to is its own decision, so pick the puller from
        # what actually happened rather than assuming the newcomer was passed over.
        spare = [i for i in live_indices(cp) if i not in holder_idx]
        if not spare:
            _fail("every online node already holds the probe, so there is nobody left "
                  "to perform a cold fetch")
            raise SystemExit
        puller = spare[0]
        print(f"  puller: sim-node-{puller + 1} (holds nothing yet)")
        if blob_has(h, puller, want_id):
            _fail(f"sim-node-{puller + 1} already holds the probe; not a cold fetch")

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
        rr = CLIENT.post(f"{agent_url(h, puller)}/blob_pull", json=plan, timeout=180)
        if rr.status_code != 200:
            _fail(f"the pull failed ({rr.status_code}) even though an honest source was "
                  f"listed; one bad peer must not deny service: {rr.text[:200]}")
        else:
            got = blob_get_raw(h, puller, want_id)
            if hashlib.sha256(got).hexdigest() != want_id:
                _fail("the node stored bytes that do not match the id it stored them "
                      "under — a poisoned shard on disk")
            else:
                _ok("skipped the liar, pulled from the honest peer, and what landed on "
                    "disk verifies against its id")

        print("\n== 5: when every holder lies, the pull fails closed ==")
        CLIENT.post(f"{agent_url(h, puller)}/blob_delete",
                    json={"blob_id": want_id}, timeout=30).raise_for_status()
        if blob_has(h, puller, want_id):
            _fail("could not clear the probe from the cold node; check 5 would be "
                  "testing a cache hit")
            raise SystemExit
        for i in holder_idx:
            h.set_corrupt_blobs(i, True)
        rr = CLIENT.post(f"{agent_url(h, puller)}/blob_pull",
                         json={"blob_id": want_id,
                               "sources": [agent_url(h, i) for i in holder_idx]},
                         timeout=180)
        stored = blob_has(h, puller, want_id)
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
        CLIENT.post(f"{agent_url(h, puller)}/blob_pull",
                    json={"blob_id": want_id,
                          "sources": [agent_url(h, i) for i in holder_idx]},
                    timeout=180).raise_for_status()
        again = CLIENT.post(f"{agent_url(h, puller)}/blob_pull",
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

        # ---- 7: serving a layer makes you a source for it ----------------- #
        print("\n== 7: the swarm grows — a node that served a layer now seeds it ==")
        dur1 = durability(cp, mid)
        newcomer_id = newcomer["id"]
        seeded = [g["group"] for g in dur1["per_group"] if newcomer_id in g["nodes"]]
        # Deliberately NOT asserted on the placement total: nodes have left since the
        # upload, and a departing node is correctly struck from every group it held,
        # so the total is not monotonic and comparing the two snapshots would measure
        # churn rather than growth. The churn-proof form of the claim is per-node --
        # this specific machine held none of these groups and now seeds several.
        at_upload = {n for g in dur0["per_group"] for n in g["nodes"]}
        print(f"  placements {dur0['total_placements']} -> {dur1['total_placements']} "
              f"(net of departures); {newcomer['name']} now seeds {len(seeded)} "
              f"group(s); survives_machine_losses={dur1['survives_machine_losses']}")
        if newcomer_id in at_upload:
            _fail(f"{newcomer['name']} was already a holder at upload time, so it "
                  f"seeding groups now proves nothing")
        elif not seeded:
            _fail(f"{newcomer['name']} pulled and is serving layers it demonstrably "
                  f"holds, but is offered as a source for none of them. Every future "
                  f"joiner is therefore routed back to the machines the upload "
                  f"replicated to, and the swarm never grows past them")
        else:
            _ok(f"{newcomer['name']} joined holding nothing and now seeds "
                f"{len(seeded)} of {dur1['groups']} group(s) — the pool became more "
                f"durable by being used")

        # ---- 8: outliving every machine it was uploaded to ---------------- #
        # The strongest form of the claim, and the one a user actually cares about:
        # the machines that received the upload are borrowed, and they will leave.
        print("\n== 8: every machine the model was uploaded to leaves ==")
        # Two more newcomers, not one: the model needs 8 GB of donated RAM to place at
        # all, and the survivors of three departures cannot supply it otherwise. The
        # claim under test is about *where the weights come from*, so it must not be
        # allowed to fail for want of somewhere to put them.
        late_idx = [N_AGENTS + 1, N_AGENTS + 2]
        for i in late_idx:
            h.start_agent(i)
        h.wait_nodes_online(N_AGENTS + len(late_idx), timeout=120)
        late_names = {f"sim-node-{cold_idx + 1}"} | {f"sim-node-{i + 1}" for i in late_idx}
        print(f"  machines that joined after the upload: {sorted(late_names)}")
        originals = [f"sim-node-{i + 1}" for i in range(N_AGENTS)]
        push_before = summary(cp).get("weight_group_pushes", 0)
        for name in originals[1:]:            # sim-node-1 was already killed in 3
            idx = int(name.rsplit("-", 1)[1]) - 1
            h.stop_agent(idx, fast=True)
            live, secs = wait_reform(cp, mid, name)
            serving = [st["node_name"] for st in live.get("stages", [])]
            print(f"  {name} gone -> re-formed in {secs:.1f}s on {serving}")
            if live.get("status") != "ready" or name in serving:
                _fail(f"the pool could not re-form after {name} left "
                      f"(status={live.get('status')!r}, serving={serving}); the layers "
                      f"it held were nowhere else")
                break
        else:
            final = chat(cp, mid)
            dur2 = durability(cp, mid)
            push_after = summary(cp).get("weight_group_pushes", 0)
            late_ids = {n["id"] for n in nodes(cp) if n.get("name") in late_names}
            stranded = sorted({n for g in dur2["per_group"] for n in g["nodes"]}
                              - late_ids)
            print(f"  every original machine is gone; {dur2['groups']} group(s) still "
                  f"held, min {dur2['min_holders']} holder(s) each")
            if final != oracle:
                _fail(f"the answer changed after the original machines left\n"
                      f"    want={oracle!r}\n    got ={final!r}")
            elif dur2["min_holders"] < 1:
                _fail(f"a layer group has no live holder at all: {dur2['per_group']}")
            elif stranded:
                _fail(f"the model is still counted as held by machines that are gone: "
                      f"{stranded}")
            elif push_after != push_before:
                _fail(f"the coordinator relayed {push_after - push_before} weight "
                      f"group(s) to keep the model alive; it should have been able to "
                      f"source every one of them from a peer")
            else:
                _ok(f"all {N_AGENTS} machines the checkpoint was uploaded to have "
                    f"left, every layer group is now held only by machines that "
                    f"joined afterwards, and the model still answers byte-identically")

        # ---- 9: redundancy is restored after churn, not just at deploy ---- #
        # Surviving the departures in 8 proves the weights were somewhere else at the
        # moment they were needed. It does not prove the pool is ready for the *next*
        # departure -- re-forming onto survivors can quietly consume the last spare
        # copy of a layer and leave the model as fragile as it was before check 2
        # fixed it. The property has to hold again after the dust settles, which is
        # why the re-partition path re-runs residency rather than trusting deploy.
        print("\n== 9: after all that churn, the model is still not one machine away "
              "from being lost ==")
        dur3 = _await_redundancy(cp, mid)
        survivors = len(live_indices(cp))
        thin = [(g["blob"][:12], len(g["nodes"])) for g in dur3["per_group"]
                if len(g["nodes"]) < 2]
        print(f"  {survivors} machine(s) left standing; {dur3['groups']} group(s), "
              f"min {dur3['min_holders']} holder(s), "
              f"survives_machine_losses={dur3['survives_machine_losses']}")
        if survivors < 2:
            _fail(f"only {survivors} machine(s) survived, so a second holder was never "
                  f"possible and this check proves nothing")
        elif thin:
            _fail(f"the pool re-formed but {len(thin)} layer group(s) came out of it on "
                  f"a single machine ({thin[:4]}); it survived the departures it has "
                  f"had and would not survive the next one")
        elif dur3["survives_machine_losses"] < 1:
            _fail(f"durability reports the model cannot survive any further loss "
                  f"despite every group having >=2 holders — the metric and the "
                  f"placement disagree: {dur3['per_group']}")
        else:
            _ok(f"every layer group is back on >=2 of the {survivors} surviving "
                f"machines after {N_AGENTS} departures and 3 arrivals; redundancy is "
                f"maintained continuously, not just at deploy")

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
