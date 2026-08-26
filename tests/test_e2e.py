"""End-to-end acceptance tests — spin up a real multi-process cluster via the sim
harness and run the MVP acceptance scenarios as subprocesses.

These are the MVP definition-of-done gates (spec §9 a-e). They are SLOW (each spins
up a control plane + 3 agent processes), so they're marked ``slow`` and skipped by
default. Run them explicitly with:

    pytest -m slow            # just the e2e gates
    pytest -m "slow or not slow"   # everything (unit + slow), portable across shells
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_scenario(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(REPO_ROOT), capture_output=True, text=True, errors="replace", timeout=600,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{module} failed (exit {proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )


@pytest.mark.slow
def test_dashboard_shows_live_nodes():
    """(b) installed machines appear live in the dashboard: the node API lists them
    online with real capacity, the summary aggregates the pool, and the dashboard
    page + static assets + SSE stream are served."""
    _run_scenario("sim.scenarios.verify_dashboard")


@pytest.mark.slow
def test_batch_engine_distributes_and_retries():
    """(d) a batch job uses all three machines; killing one retries its units."""
    _run_scenario("sim.scenarios.verify_batch")


@pytest.mark.slow
def test_hosting_engine_survives_node_loss():
    """(c) a web app deployed across nodes survives one going to sleep."""
    _run_scenario("sim.scenarios.verify_hosting")


@pytest.mark.slow
def test_host_your_own_static_site_across_the_mesh():
    """A user-supplied static website (index.html + nested asset) ships in the
    deploy request, is served through the load balancer with correct content and
    types, 404s missing paths, and keeps serving after a hosting node is killed
    (the durable deployment re-ships the site to a survivor on self-heal)."""
    _run_scenario("sim.scenarios.verify_site_hosting")


@pytest.mark.slow
def test_host_your_own_multipage_app_folder_across_the_mesh():
    """A real multi-page app folder (nested routes + JS/CSS/JSON/binary assets) is
    packed with hosting.site.pack_dir, deployed across nodes, and served through the
    load balancer with correct Content-Types (notably .js -> text/javascript) and
    intact binary bytes -- and every route keeps serving after a node is killed."""
    _run_scenario("sim.scenarios.verify_realapp")


@pytest.mark.slow
def test_host_your_own_dynamic_server_app_across_the_mesh():
    """A real dynamic app WITH ITS OWN BACKEND (a Python server, not static files) is
    launched per replica on the injected $PORT, load-balanced across distinct worker
    processes (proven by distinct live pids), answers its own routes/404, and is
    relaunched on a survivor after a hosting node is killed (failover + self-heal)."""
    _run_scenario("sim.scenarios.verify_procapp")


@pytest.mark.slow
def test_hosted_app_keeps_each_users_session_pinned_to_one_replica():
    """A stateful multi-page app (per-session in-memory counter keyed by its own
    cookie) stays usable with >1 replica: the proxy relays the app's Set-Cookie and the
    client's Cookie both ways, pins each client to one worker (session survives), still
    spreads different clients across workers, and re-pins to a survivor with no dropped
    request when the pinned worker is killed."""
    _run_scenario("sim.scenarios.verify_sticky")


@pytest.mark.slow
def test_hosted_app_streams_responses_incrementally_not_buffered():
    """A hosted app that streams a long response (Server-Sent Events, one chunk every
    0.3s) is relayed to the client CHUNK BY CHUNK through the proxy -- the client sees
    the first byte early and each chunk arrives spread over time (proven by per-chunk
    arrival timestamps), not buffered to the end -- and the app keeps serving after a
    node is killed."""
    _run_scenario("sim.scenarios.verify_stream")


@pytest.mark.slow
def test_hosted_app_receives_deployer_env_config():
    """A dynamic app deployed with ``env={...}`` actually receives that config on every
    replica (read back through the proxy from its own /env route); the control plane
    rejects an invalid env var name at the boundary; reserved platform vars (PORT) win
    over user-supplied values; and the config survives a node being killed."""
    _run_scenario("sim.scenarios.verify_env")


@pytest.mark.slow
def test_hosted_app_logs_are_captured_and_retrievable():
    """A hosted process app's stdout/stderr is captured on each replica's node and
    retrievable from the control plane: the startup banner is captured on every replica,
    live request activity shows up (proving streaming capture, not a one-shot snapshot),
    an unknown deployment id returns 404, and logs stay retrievable after a node dies."""
    _run_scenario("sim.scenarios.verify_logs")


@pytest.mark.slow
def test_pooled_storage_replicates_and_survives_node_loss():
    """A pooled object store (spec §P3): an object POSTed to the control plane is
    content-addressed (id == SHA-256 of the bytes) and replicated across two distinct
    nodes' donated disk, downloaded back byte-for-byte, deduped when re-uploaded, and --
    crucially -- stays downloadable after a holding node is killed while the manager
    self-heals the missing copy onto a third node; deletion then clears it everywhere."""
    _run_scenario("sim.scenarios.verify_storage")


@pytest.mark.slow
def test_host_large_app_that_does_not_fit_inline_via_the_pool():
    """The headline "host ANY large app" gate: an app bigger than the inline control
    frame is staged whole in the pooled, content-addressed blob store (chunked +
    replicated across the mesh, hidden from the user's Files), hydrated back onto each
    replica so every page/asset serves through the proxy, works through both the JSON
    and browser .zip entry points, and is re-hydrated FROM THE POOL onto a survivor
    after a hosting node is killed."""
    _run_scenario("sim.scenarios.verify_bigapp")


@pytest.mark.slow
def test_inference_shards_model_too_big_for_one_node():
    """(e) a model too big for any single node is sharded and served; distributed
    output is bit-identical to single-machine and survives a node dying."""
    _run_scenario("sim.scenarios.verify_inference")


@pytest.mark.slow
def test_serve_a_model_whose_weights_live_in_the_pool():
    """"Run a model one machine can't hold" for BYO weights: a user uploads a trained
    checkpoint, it is staged in the pooled blob store (chunked + replicated, hidden
    from Files), and each pipeline stage is shipped only the slice its layers need,
    hydrated from the pool. Distributed greedy output is bit-identical to the pooled
    checkpoint and differs from the node-local built-in one (proving the served weights
    came from the pool), and re-hydrates from the pool after a node is killed."""
    _run_scenario("sim.scenarios.verify_pooled_model")


@pytest.mark.slow
def test_secret_brokering_is_scoped_and_single_use():
    """Scoped secret brokering (spec §5): a unit signs with a credential never in
    its job spec; the value never leaks; unrequested units don't get it; leases
    are single-use."""
    _run_scenario("sim.scenarios.verify_secrets")


@pytest.mark.slow
def test_control_plane_reaches_nodes_at_their_advertised_address():
    """Real multi-machine path (spec §9 a): over DISTINCT network addresses with
    real hardware probes, the control plane reaches every node at the address it
    advertised — for batch dispatch, hosted-app proxying, and inference activation
    hops — not a hardcoded localhost. Self-skips where extra loopback aliases can't
    bind (e.g. stock macOS)."""
    _run_scenario("sim.scenarios.verify_lan")


@pytest.mark.slow
def test_global_tunnel_only_path_carries_all_workloads():
    """Global path (Phase 1 flagship): agents are reachable ONLY via an outbound
    reverse tunnel — each advertises an unreachable sentinel URL — yet batch
    fan-out+retry, app hosting+failover, and sharded inference+re-partition all
    succeed end-to-end with node kills. If nothing can reach an agent directly, any
    success proves the tunnel carried it (dispatch, proxying, activation hops)."""
    _run_scenario("sim.scenarios.verify_global")


@pytest.mark.slow
def test_public_sandbox_protects_donor():
    """Public (Tier-2) safety: every stranger's curated unit runs in the locked-down
    sandbox. A legitimate job is correct end-to-end (sandbox transparent on success);
    an abusive over-cap unit is rejected by in-sandbox validation and fails fast
    instead of burning the donor's CPU."""
    _run_scenario("sim.scenarios.verify_sandbox")


@pytest.mark.slow
def test_public_trust_and_credits_economy():
    """Public (Tier-2) economy: a new device gets a starter grant; consuming without
    a session is denied; an authenticated fan-out completes with every accepted
    result correct while a deliberately dishonest node is caught on every unit,
    paid nothing, and down-scored; the credit ledger conserves (donated == consumed);
    and once a consumer spends its balance, further submits are refused (402)."""
    _run_scenario("sim.scenarios.verify_trust_credits")


@pytest.mark.slow
def test_public_ux_contribute_and_use_flow():
    """Public (Tier-2) product path: the dashboard + assets are served; onboarding
    advertises a one-command join per OS; zero-signup local sign-in mints a session
    and the starter grant (no keys in the browser); the Wallet reflects it; the
    'Use' flow is accepted with the session and blocked (401) without one; and the
    ledger exposes both the per-owner and per-node shapes from a single route."""
    _run_scenario("sim.scenarios.verify_ux")


@pytest.mark.slow
def test_onboarding_unzip_and_run_works_clean_room():
    """Phase-3 onboarding: the exact file set the build zip ships (allowlist copy, no
    tests/ or sim/ or prior state) runs with one command. Extracts clean and runs
    ``python -m meshpool.launcher self-test`` — control plane + local agent + a real
    batch job — proving the shipped artifact is self-contained and 'unzip and run'
    actually works, with a persisted device identity for repeat launches."""
    _run_scenario("sim.scenarios.verify_onboarding")


@pytest.mark.slow
def test_coordinator_restart_self_heals():
    """Phase-4 chaos: the coordinator itself is hard-killed and restarted while
    tunnel-only agents are connected. On its own — no re-onboarding — each agent
    re-registers and re-attaches its reverse tunnel; the durable credit ledger
    survives the crash (the starter grant is not re-issued); and a brand-new job
    submitted after the restart completes correctly with the ledger still conserving.
    Proves the mesh recovers from losing its brain, not just from losing a worker."""
    _run_scenario("sim.scenarios.verify_chaos")


@pytest.mark.slow
def test_live_job_progress_streams_to_dashboard():
    """UX smoothness: a running batch job emits per-unit ``job_progress`` events over
    the dashboard SSE stream (not just submit/finish). Opens the live event stream,
    submits a multi-unit job, and proves the done-count climbs monotonically as units
    land and a terminal ``job_finished: completed`` arrives — the signal the dashboard
    progress bars render, so users watch work advance in real time."""
    _run_scenario("sim.scenarios.verify_progress")


@pytest.mark.slow
def test_scale_many_nodes_many_concurrent_jobs():
    """Scale/concurrency gate: five nodes with several batch jobs submitted at once all
    complete with exact accounting — every unit runs exactly once, zero failures,
    results correct — while work fans out across the whole fleet (every node executes
    units, none lost or double-counted). Proves the scheduler stays correct under
    concurrent multi-job load, not just one job at a time."""
    _run_scenario("sim.scenarios.verify_scale")
