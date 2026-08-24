"""``meshpool`` — a friendly command-line client for the control plane.

Talks to the control-plane REST API over HTTP (the same API the dashboard uses), so
it works against the local sim harness or a real deployment identically. Point it at a
control plane with ``--control-plane URL`` or the ``MESHPOOL_CONTROL_PLANE_URL``
environment variable (flag takes precedence over the env var).
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from meshpool.storage.crypto import ClientKeyring, decrypt_object, encrypt_object, is_encrypted_package

app = typer.Typer(add_completion=False, help="MeshPool — pool your machines into one computer.")
console = Console()

# Module-level override set by the --control-plane global flag; None means "not set".
_override_url: Optional[str] = None


@app.callback()
def _global_options(
    control_plane: Optional[str] = typer.Option(
        None,
        "--control-plane",
        help=(
            "Control-plane URL to connect to. "
            "Overrides the MESHPOOL_CONTROL_PLANE_URL environment variable "
            "and the built-in default (http://127.0.0.1:8080). "
            "Must be an http or https URL. "
            "Place this flag before the subcommand name, "
            "e.g. meshpool --control-plane http://host:9090 nodes."
        ),
    )
) -> None:
    global _override_url
    if control_plane is not None:
        parsed = urlparse(control_plane)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            console.print(
                f"[bold red]error:[/] --control-plane must be a valid http or https URL, "
                f"got {control_plane!r}"
            )
            raise typer.Exit(1)
        _override_url = control_plane


def base_url() -> str:
    """Return the control-plane base URL.

    Precedence: ``--control-plane`` flag > ``MESHPOOL_CONTROL_PLANE_URL`` env var >
    built-in default (``http://127.0.0.1:8080``).
    """
    if _override_url is not None:
        return _override_url.rstrip("/")
    return os.environ.get("MESHPOOL_CONTROL_PLANE_URL", "http://127.0.0.1:8080").rstrip("/")


def _get(path: str) -> dict:
    r = httpx.get(f"{base_url()}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: Optional[dict] = None) -> dict:
    r = httpx.post(f"{base_url()}{path}", json=body or {}, timeout=120)
    r.raise_for_status()
    return r.json()


def _delete(path: str) -> dict:
    r = httpx.request("DELETE", f"{base_url()}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _post_bytes(path: str, data: bytes, content_type: str = "application/octet-stream") -> dict:
    r = httpx.post(f"{base_url()}{path}", content=data,
                   headers={"Content-Type": content_type}, timeout=300)
    r.raise_for_status()
    return r.json()


def _get_bytes(path: str) -> bytes:
    r = httpx.get(f"{base_url()}{path}", timeout=300)
    r.raise_for_status()
    return r.content


def _fail(exc: Exception) -> None:
    detail = str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            detail = exc.response.text
    console.print(f"[bold red]error:[/] {detail}")
    raise typer.Exit(1)


# --------------------------------------------------------------------------- #
@app.command()
def nodes() -> None:
    """List all machines in the pool and their live status."""
    try:
        data = _get("/api/v1/cluster")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    s = data["summary"]
    t = Table(title="MeshPool nodes")
    for c in ("name", "status", "cores", "ram (GB)", "cpu free", "trust", "owner"):
        t.add_column(c)
    for n in data["nodes"]:
        cap = n["capability"]
        status = n["status"]
        color = {"online": "green", "draining": "yellow"}.get(status, "red")
        t.add_row(n["name"], f"[{color}]{status}[/]", str(cap["cpu_cores"]),
                  f"{cap['ram_mb'] / 1024:.0f}", f"{n['headroom_cpu_pct']:.0f}%",
                  f"{n['trust_score']:.2f}", n["owner"])
    console.print(t)
    console.print(f"[dim]{s['online']}/{s['total_nodes']} online · "
                  f"{s['total_cores']} cores · {s['total_ram_mb'] / 1024:.0f} GB pooled[/]")


@app.command()
def onboarding() -> None:
    """Show the pairing code / join command for adding a machine."""
    try:
        d = _get("/api/v1/onboarding")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(Panel.fit(
        f"[bold]Add a machine to this pool[/]\n\n"
        f"Control plane : {d['control_plane_url']}\n"
        f"Pairing token : [bold cyan]{d['invite_token']}[/]\n\n"
        f"Run in the [bold]meshpool[/] folder on the new machine:\n"
        f"  [dim]Windows[/]     [green]{d.get('join_windows', d['join_command'])}[/]\n"
        f"  [dim]macOS/Linux[/] [green]{d.get('join_unix', d['join_command'])}[/]",
        title="onboarding"))


@app.command()
def join(url: str = typer.Argument(..., help="control-plane URL, e.g. http://192.168.1.20:8080"),
         token: str = typer.Option("", "--token", help="pairing token")) -> None:
    """Print the one-command way to join this machine to a pool."""
    console.print(Panel.fit(
        f"Run in the [bold]meshpool[/] folder on this machine:\n\n"
        f"  [dim]Windows[/]     [green]powershell -ExecutionPolicy Bypass -File "
        f"deploy\\join.ps1 -Server {url} -Token {token}[/]\n"
        f"  [dim]macOS/Linux[/] [green]bash deploy/join.sh --server {url} --token {token}[/]\n\n"
        f"The agent auto-detects this machine's LAN IP and self-registers.",
        title="join"))


# ---- batch --------------------------------------------------------------- #
jobs_app = typer.Typer(help="Run distributed batch jobs.")
app.add_typer(jobs_app, name="jobs")


@jobs_app.command("submit")
def jobs_submit(func: str = typer.Argument("sha_pow", help="work function name"),
                count: int = typer.Option(12, "--count", "-n", help="number of work units"),
                cpu: int = typer.Option(15, help="cpu %% per unit"),
                ram: int = typer.Option(128, help="ram MB per unit")) -> None:
    """Submit a batch job of independent work units across the pool."""
    payloads = [{"seed": i, "rounds": 200000} for i in range(count)]
    try:
        r = _post("/api/v1/jobs", {"name": func, "func": func, "payloads": payloads,
                                   "cpu_pct": cpu, "ram_mb": ram})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(f"[green]submitted[/] job [bold]{r['job_id']}[/] with {r['units']} units")


@jobs_app.command("list")
def jobs_list() -> None:
    """List batch jobs and their progress."""
    try:
        data = _get("/api/v1/jobs")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    t = Table(title="Batch jobs")
    for c in ("id", "name", "status", "progress"):
        t.add_column(c)
    for j in data["jobs"]:
        p = j["progress"]
        total = p.get("total", 0) or 0
        pct = (100 * p.get("done", 0) / total) if total else 0.0
        t.add_row(j["id"], j["name"], j["status"],
                  f"{p.get('done', 0)}/{total} ({pct:.0f}%)")
    console.print(t)


# ---- DAG workflows ------------------------------------------------------- #
dag_app = typer.Typer(help="Run durable batch DAG workflows.")
app.add_typer(dag_app, name="dag")


@dag_app.command("submit")
def dag_submit(spec: str = typer.Argument(..., help="path to DAG JSON file"),
               name: str = typer.Option("", "--name", help="workflow name"),
               wait: bool = typer.Option(False, "--wait/--no-wait",
                                         help="poll until the DAG finishes")) -> None:
    """Submit a JSON DAG with stages, dependencies, and optional reduce stages."""
    import json
    import time
    p = Path(spec).expanduser()
    if not p.is_file():
        console.print(f"[bold red]error:[/] no such file: {p}")
        raise typer.Exit(1)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]error:[/] invalid DAG JSON: {exc}")
        raise typer.Exit(1)
    body = raw if isinstance(raw, dict) else {"stages": raw}
    if name:
        body["name"] = name
    try:
        r = _post("/api/v1/dags", body)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    dag_id = r["dag_id"]
    console.print(f"[green]submitted[/] DAG [bold]{dag_id}[/] with {r['stages']} stages")
    if not wait:
        return
    deadline = time.time() + 900
    with console.status("running DAG on the pool ..."):
        while time.time() < deadline:
            try:
                d = _get(f"/api/v1/dags/{dag_id}")
            except Exception as exc:  # noqa: BLE001
                _fail(exc)
            if str(d["status"]).startswith("completed"):
                break
            time.sleep(1.0)
        else:
            console.print("[yellow]timed out waiting for the DAG[/]")
            raise typer.Exit(1)
    _print_dag(d)
    if d["status"] != "completed":
        raise typer.Exit(1)


@dag_app.command("list")
def dag_list() -> None:
    """List DAG workflows and their progress."""
    try:
        data = _get("/api/v1/dags")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    t = Table(title="DAG workflows")
    for c in ("id", "name", "status", "progress"):
        t.add_column(c)
    for d in data["dags"]:
        p = d["progress"]
        total = p.get("total", 0) or 0
        t.add_row(d["id"], d["name"], d["status"],
                  f"{p.get('succeeded', 0)}/{total} succeeded")
    console.print(t)


@dag_app.command("show")
def dag_show(dag_id: str = typer.Argument(..., help="DAG id")) -> None:
    """Show one DAG's stage state and execution counts."""
    try:
        d = _get(f"/api/v1/dags/{dag_id}")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    _print_dag(d)


def _print_dag(d: dict) -> None:
    t = Table(title=f"DAG {d['id']} — {d['status']}")
    for c in ("stage", "status", "job", "executions", "error"):
        t.add_column(c)
    for s in d.get("stages", []):
        t.add_row(s["id"], s["status"], str(s.get("job_id") or "-"),
                  str(s.get("execution_count", 0)), str(s.get("error") or ""))
    console.print(t)


# ---- functions (bring your own code) ------------------------------------- #
fn_app = typer.Typer(help="Run your own code on the pool (WebAssembly).")
app.add_typer(fn_app, name="fn")


@fn_app.command("deploy")
def fn_deploy(module: str = typer.Argument(..., help="path to a .wasm module "
                                                     "(wasm32-wasip1 command module)"),
              name: str = typer.Option("", "--name", help="friendly label for the module")) -> None:
    """Upload your own WebAssembly module and get a function name back.

    Compile any language that targets ``wasm32-wasip1`` (Rust, Go, Zig, C, ...).
    Your module reads its JSON payload from stdin and writes its JSON result to
    stdout. It runs sandboxed on donor machines: no filesystem, no sockets, no host
    calls, with fuel/memory/wall-clock ceilings.
    """
    p = Path(module).expanduser()
    if not p.is_file():
        console.print(f"[bold red]error:[/] no such file: {p}")
        raise typer.Exit(1)
    data = p.read_bytes()
    q = f"?name={quote(name)}" if name else ""
    try:
        r = _post_bytes(f"/api/v1/functions{q}", data, "application/wasm")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(Panel.fit(
        f"[bold]{r['function']}[/]\n\n"
        f"size    : {_fmt_bytes(r['size_bytes'])}\n"
        f"abi     : {r['abi']}\n"
        f"exports : {', '.join(r['exports']) or '-'}\n\n"
        f"Run it:\n"
        f"  [green]meshpool fn run {r['function']} --payload '{{\"x\": 1}}'[/]",
        title="function deployed"))


@fn_app.command("run")
def fn_run(function: str = typer.Argument(..., help="function name, e.g. wasm:<sha256>"),
           payload: list[str] = typer.Option([], "--payload", "-p",
                                             help="JSON payload; repeat for multiple units"),
           count: int = typer.Option(0, "--count", "-n",
                                     help="fan the single payload out into N units, "
                                          "each stamped with a distinct 'index'"),
           cpu: int = typer.Option(15, help="cpu %% per unit"),
           ram: int = typer.Option(256, help="ram MB per unit"),
           wait: bool = typer.Option(True, "--wait/--no-wait",
                                     help="poll until the job finishes")) -> None:
    """Run a deployed function across the pool, one work unit per payload."""
    import json
    import time
    try:
        parsed = [json.loads(x) for x in payload] or [{}]
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]error:[/] --payload must be JSON: {exc}")
        raise typer.Exit(1)
    if count > 0:
        base = parsed[0]
        parsed = [{**base, "index": i} for i in range(count)]
    try:
        r = _post("/api/v1/jobs", {"name": function[:40], "func": function,
                                   "payloads": parsed, "cpu_pct": cpu, "ram_mb": ram})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    job_id = r["job_id"]
    console.print(f"[green]submitted[/] [bold]{job_id}[/] with {r['units']} units")
    if not wait:
        return
    deadline = time.time() + 600
    with console.status("running on the pool ..."):
        while time.time() < deadline:
            try:
                d = _get(f"/api/v1/jobs/{job_id}")
            except Exception as exc:  # noqa: BLE001
                _fail(exc)
            if str(d["status"]).startswith("completed"):
                break
            time.sleep(1.0)
        else:
            console.print("[yellow]timed out waiting for the job[/]")
            raise typer.Exit(1)
    t = Table(title=f"{function[:28]} — {d['status']}")
    for c in ("unit", "node", "result"):
        t.add_column(c)
    for u in d.get("units", [])[:50]:
        res = u.get("result")
        text = "[red]" + str(u.get("error") or "failed")[:60] + "[/]" if res is None \
            else str(res)[:60]
        t.add_row(str(u.get("id", "")), str(u.get("assigned_node") or "-"), text)
    console.print(t)
    p = d["progress"]
    console.print(f"[dim]{p.get('done', 0)} done · {p.get('failed', 0)} failed "
                  f"of {p.get('total', 0)}[/]")
    if p.get("failed", 0):
        raise typer.Exit(1)


@fn_app.command("builtin")
def fn_builtin() -> None:
    """List the built-in functions that ship with MeshPool."""
    from ..batch.functions import REGISTRY
    t = Table(title="Built-in functions")
    t.add_column("name")
    t.add_column("what it does")
    for n, f in sorted(REGISTRY.items()):
        doc = (f.__doc__ or "").strip().splitlines()
        t.add_row(n, doc[0] if doc else "-")
    console.print(t)
    console.print("[dim]bring your own with [bold]meshpool fn deploy ./module.wasm[/][/]")


# ---- hosting ------------------------------------------------------------- #
def _detect_start_cmd(folder: Path) -> str:
    """Auto-discover a dynamic app's launch command from a bundle: a ``Procfile``
    ``web:`` line or a ``meshpool.json`` ``start`` field. Returns "" for a static site."""
    proc = folder / "Procfile"
    if proc.is_file():
        for line in proc.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, _, val = line.partition(":")
            if key.strip().lower() == "web" and val.strip():
                return val.strip()
    manifest = folder / "meshpool.json"
    if manifest.is_file():
        import json
        try:
            start = json.loads(manifest.read_text(encoding="utf-8")).get("start")
        except (ValueError, OSError):
            start = None
        if isinstance(start, str) and start.strip():
            return start.strip()
    return ""


@app.command()
def deploy(target: str = typer.Argument("hello",
               help="built-in app name, or a path to a static-site folder or .zip"),
           replicas: int = typer.Option(2, "--replicas", "-r"),
           cmd: str = typer.Option("", "--cmd",
               help="start command for a dynamic app; its server must listen on $PORT "
                    "(e.g. \"python server.py\"). Auto-detected from Procfile/meshpool.json."),
           env: Optional[list[str]] = typer.Option(None, "--env", "-e",
               help="set an env var for a dynamic app as KEY=VALUE (repeatable), e.g. "
                    "-e DATABASE_URL=postgres://... -e DEBUG=1"),
           name: str = typer.Option("", "--name", help="deployment name (defaults from target)")) -> None:
    """Deploy across the pool with load-balancing + failover.

    ``target`` may be a built-in app name (e.g. ``hello``), a path to a folder
    holding your static multi-page site (with an ``index.html``), or a ``.zip`` of
    that site. Folders/zips are packed and shipped to every replica, so
    ``meshpool deploy ./dist --replicas 2`` hosts your own app across the pool.

    Pass ``--cmd`` (or ship a ``Procfile``/``meshpool.json``) to run a **dynamic**
    app that has its own server — MeshPool launches it on each replica with ``$PORT``
    set and load-balances across them, e.g.
    ``meshpool deploy ./api --cmd "python server.py" -r 2``.
    """
    from ..hosting import site as sitelib
    body: dict = {"replicas": replicas}
    env_map: dict[str, str] = {}
    for item in (env or []):
        key, sep, val = item.partition("=")
        if not sep or not key.strip():
            _fail(f"bad --env {item!r}; expected KEY=VALUE")
        env_map[key.strip()] = val
    p = Path(target)
    if p.is_dir():
        start_cmd = cmd or _detect_start_cmd(p)
        try:
            site = sitelib.pack_dir(str(p), require_index=not start_cmd,
                                    max_bytes=sitelib.MAX_BUNDLE_BYTES)
        except ValueError as exc:
            _fail(exc)
        body["site"] = site
        dep_name = name or p.name or "site"
        if start_cmd:
            body["app"], body["start_cmd"] = "proc", start_cmd
            summary = f"your app '{dep_name}' ({len(site)} files, runs: {start_cmd})"
        else:
            body["app"] = "site"
            summary = f"your site '{dep_name}' ({len(site)} files)"
    elif p.is_file() and p.suffix.lower() == ".zip":
        try:
            site = sitelib.pack_zip(p.read_bytes(), require_index=not cmd,
                                    max_bytes=sitelib.MAX_BUNDLE_BYTES)
        except (ValueError, OSError) as exc:
            _fail(exc)
        body["site"] = site
        dep_name = name or p.stem or "site"
        if cmd:
            body["app"], body["start_cmd"] = "proc", cmd
            summary = f"your app '{dep_name}' ({len(site)} files from {p.name}, runs: {cmd})"
        else:
            body["app"] = "site"
            summary = f"your site '{dep_name}' ({len(site)} files from {p.name})"
    else:
        body["app"] = target
        dep_name = name or f"{target}-app"
        summary = f"built-in app '{target}'"
    body["name"] = dep_name
    if env_map:
        body["env"] = env_map
    try:
        r = _post("/api/v1/deployments", body)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(f"[green]deploying[/] {summary} -> [bold]{r['id']}[/] "
                  f"({r['replicas']} replicas, status {r['status']})")
    console.print(f"[dim]serve at {base_url()}/app/{r['id']}/[/]")


@app.command()
def deployments() -> None:
    """List hosted app deployments."""
    try:
        data = _get("/api/v1/deployments")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    t = Table(title="Deployments")
    for c in ("id", "name", "status", "replicas", "nodes"):
        t.add_column(c)
    for d in data["deployments"]:
        reps = d["replicas"]
        nodes = ", ".join(sorted({r["node_name"] for r in reps if r.get("healthy")}))
        t.add_row(d["id"], d["name"], d["status"],
                  f"{sum(1 for r in reps if r.get('healthy'))}/{d['replicas_desired']}", nodes)
    console.print(t)


@app.command()
def undeploy(dep_id: str = typer.Argument(...,
                 help="deployment id to stop (from `meshpool deployments`)")) -> None:
    """Stop a hosted app and remove all its replicas across the pool."""
    try:
        _delete(f"/api/v1/deployments/{dep_id}")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(f"[green]removed[/] deployment [bold]{dep_id}[/]")


@app.command()
def logs(dep_id: str = typer.Argument(...,
             help="deployment id (from `meshpool deployments`)"),
         tail: int = typer.Option(200, "--tail", "-n",
             help="max lines to show per replica")) -> None:
    """Show recent stdout/stderr from each replica of a hosted process app."""
    try:
        data = _get(f"/api/v1/deployments/{dep_id}/logs?tail={max(1, tail)}")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    reps = data.get("replicas") or []
    if not reps:
        console.print("[dim]no replicas for this deployment[/]")
        return
    for rep in reps:
        state_c = "green" if rep.get("healthy") else "red"
        badge = "healthy" if rep.get("healthy") else "unhealthy"
        console.print(f"\n[bold]{rep['replica_id']}[/] @ "
                      f"{rep.get('node_name') or '?'} [{state_c}]{badge}[/]")
        lines = rep.get("lines") or []
        if not lines:
            console.print("  [dim](no output captured — static site, or nothing logged yet)[/]")
        for ln in lines:
            console.print("  " + ln, markup=False, highlight=False)


# ---- pooled storage ------------------------------------------------------ #
storage_app = typer.Typer(help="Store files across the pool (replicated, fault-tolerant).")
app.add_typer(storage_app, name="storage")


def _fmt_bytes(n: int) -> str:
    v = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024:
            return f"{int(v)} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"


def _storage_index_path() -> Path:
    key_path = Path(os.environ.get("MESHPOOL_CLIENT_KEY_PATH", "") or
                    (Path(os.environ.get("MESHPOOL_DATA_DIR", str(Path.home() / ".meshpool")))
                     / "client-root-key.json"))
    return key_path.with_name("client-object-index.json")


def _load_storage_index() -> dict:
    path = _storage_index_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_storage_index(index: dict) -> None:
    path = _storage_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


@storage_app.command("put")
def storage_put(path: str = typer.Argument(..., help="file to upload into the pool"),
                name: str = typer.Option("", "--name",
                    help="display name (defaults to the filename)"),
                bucket: str = typer.Option("default", "--bucket",
                    help="client-side encryption bucket/key namespace"),
                encrypt: bool = typer.Option(True, "--encrypt/--no-encrypt",
                    help="encrypt locally before upload (default: on)"),
                replication: int = typer.Option(2, "--replication", "-r",
                    help="how many machines to keep a copy on (1-5)")) -> None:
    """Upload a file into the pool — encrypted locally, then replicated.

    Encryption happens before bytes leave this process. Nodes and the control plane
    store only ciphertext packages unless ``--no-encrypt`` is explicitly selected.
    """
    p = Path(path)
    if not p.is_file():
        _fail(f"no such file: {path}")
    disp = name or p.name
    try:
        data = p.read_bytes()
    except OSError as exc:
        _fail(exc)
    upload = data
    plaintext_id = hashlib.sha256(data).hexdigest()
    if encrypt:
        upload, version_id = encrypt_object(data, bucket=bucket, key=disp,
                                            keyring=ClientKeyring())
    else:
        version_id = ""
    q = f"/api/v1/blobs?name={quote(disp)}&replication={max(1, min(replication, 5))}"
    try:
        meta = _post_bytes(q, upload)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    if encrypt:
        idx = _load_storage_index()
        idx[plaintext_id] = {"blob_id": meta["id"], "bucket": bucket, "key": disp,
                             "version_id": version_id}
        _save_storage_index(idx)
        console.print(f"[green]stored encrypted[/] [bold]{disp}[/] -> {plaintext_id}")
        console.print(f"[dim]ciphertext blob {meta['id']} · bucket {bucket} · version {version_id}[/]")
    else:
        console.print(f"[green]stored[/] [bold]{disp}[/] -> {meta['id']}")
    if meta.get("chunks"):
        dur = f"{len(meta['chunks'])} chunks across the pool"
    else:
        dur = f"{len(meta.get('replicas') or [])} copies"
    console.print(f"[dim]{_fmt_bytes(meta['size'])} · {dur} · status {meta['status']}[/]")


@storage_app.command("ls")
def storage_ls() -> None:
    """List files stored in the pool."""
    try:
        data = _get("/api/v1/blobs")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    blobs = data.get("blobs") or []
    t = Table(title="Pooled storage")
    for c in ("id", "name", "size", "copies", "status"):
        t.add_column(c)
    for b in blobs:
        status = b.get("status", "stored")
        color = {"stored": "green", "degraded": "yellow"}.get(status, "red")
        chunks = b.get("chunks") or []
        copies = f"{len(chunks)} chunks" if chunks else str(len(b.get("replicas") or []))
        t.add_row(b["id"][:16] + "…", b.get("name") or "—", _fmt_bytes(b.get("size", 0)),
                  copies, f"[{color}]{status}[/]")
    console.print(t)
    if not blobs:
        console.print("[dim]no files stored yet — `meshpool storage put <file>`[/]")


@storage_app.command("get")
def storage_get(blob_id: str = typer.Argument(...,
                    help="blob id (from `meshpool storage ls`)"),
                out: str = typer.Argument(..., help="path to write the downloaded file to")) -> None:
    """Download a file from the pool and decrypt client-side when needed."""
    idx = _load_storage_index()
    remote_id = idx.get(blob_id, {}).get("blob_id", blob_id)
    try:
        raw = _get_bytes(f"/api/v1/blobs/{remote_id}")
        if is_encrypted_package(raw):
            raw = decrypt_object(raw, keyring=ClientKeyring())
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    try:
        Path(out).write_bytes(raw)
    except OSError as exc:
        _fail(exc)
    console.print(f"[green]downloaded[/] {_fmt_bytes(len(raw))} -> [bold]{out}[/]")


@storage_app.command("rm")
def storage_rm(blob_id: str = typer.Argument(...,
                   help="blob id to remove from the pool (from `meshpool storage ls`)")) -> None:
    """Remove a file from the pool (deletes every replica)."""
    idx = _load_storage_index()
    remote_id = idx.get(blob_id, {}).get("blob_id", blob_id)
    try:
        _delete(f"/api/v1/blobs/{remote_id}")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    if blob_id in idx:
        idx.pop(blob_id, None)
        _save_storage_index(idx)
    console.print(f"[green]removed[/] [bold]{blob_id}[/]")


# ---- inference ----------------------------------------------------------- #
models_app = typer.Typer(help="Deploy and inspect sharded models.")
app.add_typer(models_app, name="models")


@models_app.command("deploy")
def models_deploy(model: str = typer.Argument("nano-gpt")) -> None:
    """Run a model across the pool. Real models (served by pooled Ollama) are
    replicated one-per-node for throughput + failover; the built-in net is sharded
    pipeline-parallel across nodes."""
    try:
        v = _post("/api/v1/models", {"model": model})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    if v.get("backend", "meshshard") != "meshshard":
        console.print(f"[green]deployed[/] {v['model']} · {v['backend']} · "
                      f"{v['replicas']} replica(s)")
        for s in v["stages"]:
            console.print(f"  [cyan]{s['node_name']}[/] full model")
    else:
        console.print(f"[green]deployed[/] {v['model']} · {v['n_layers']} layers · "
                      f"{len(v['stages'])} stages")
        for s in v["stages"]:
            console.print(f"  [cyan]{s['node_name']}[/] layers {s['first']}–{s['last']}")


@models_app.command("available")
def models_available() -> None:
    """List models the pool can serve right now: the built-in nets plus every Ollama
    model advertised by an online node (and which nodes hold each)."""
    try:
        d = _get("/api/v1/models/available")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    t = Table(title="Deployable models")
    for c in ("model", "backend", "on nodes"):
        t.add_column(c)
    for m in d.get("ollama", []):
        t.add_row(m["model"], "ollama", ", ".join(m["nodes"]))
    for m in d.get("builtin", []):
        t.add_row(m, "meshshard", "sharded across the pool")
    console.print(t)


@models_app.command("list")
def models_list() -> None:
    """Show the current model deployment."""
    try:
        v = _get("/api/v1/models")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    if not v.get("deployed"):
        console.print("[yellow]no model deployed[/] — run [bold]meshpool models deploy[/]")
        return
    if v.get("backend", "meshshard") != "meshshard":
        console.print(f"[bold]{v['model']}[/] · {v['backend']} · {v['replicas']} replica(s) · "
                      f"status {v['status']} · tokens/sec {v.get('tokens_per_sec')}")
        for s in v["stages"]:
            console.print(f"  [cyan]{s['node_name']}[/] full model")
    else:
        console.print(f"[bold]{v['model']}[/] · {v['n_layers']} layers · status {v['status']} · "
                      f"tokens/sec {v.get('tokens_per_sec')}")
        for s in v["stages"]:
            console.print(f"  [cyan]{s['node_name']}[/] layers {s['first']}–{s['last']}")


@models_app.command("upload")
def models_upload(checkpoint: str = typer.Argument(...,
                      help="trained checkpoint (.npz) to serve from the pool"),
                  model: str = typer.Option("nano-gpt", "--model",
                      help="model architecture the checkpoint was trained for")) -> None:
    """Bring your own weights: stage a trained checkpoint in the pool and serve it.

    The checkpoint is split locally into per-layer groups and each group is streamed to
    the pool on its own, so neither this client nor the control plane ever holds the
    whole checkpoint -- a model whose weights dwarf any single machine still uploads and
    runs across the pool. Each pipeline stage is then shipped only the layers it runs.
    """
    p = Path(checkpoint)
    if not p.is_file():
        _fail(f"no such file: {checkpoint}")
    from ..inference import model as M
    cfg = M.REGISTRY.get(model)
    if cfg is None:
        _fail(f"unknown model architecture {model!r}; "
              f"known: {', '.join(sorted(M.REGISTRY))}")
    import numpy as np
    groups: dict[str, str] = {}
    meta_id = None
    total = 0
    gurl = f"/api/v1/models/weights/group?model={quote(model)}"
    try:
        with np.load(p, allow_pickle=False) as data:
            names = set(data.files)
            missing = [k for _n, ks in M.group_specs(cfg) for k in ks if k not in names]
            if missing:
                raise ValueError(f"checkpoint is missing {len(missing)} parameter(s) for "
                                 f"{model} (e.g. {sorted(missing)[:3]})")
            for gname, keys in M.group_specs(cfg):
                blob = M.dumps_npz({k: data[k] for k in keys})
                total += len(blob)
                groups[gname] = _post_bytes(gurl, blob)["blob_id"]
            if "__itos__" in names:
                meta_id = _post_bytes(
                    gurl, M.dumps_npz({}, itos=list(data["__itos__"])))["blob_id"]
    except (OSError, ValueError) as exc:
        _fail(exc)
    except httpx.HTTPStatusError as exc:
        _fail(exc)
    try:
        v = _post(f"/api/v1/models/weights/register?model={quote(model)}",
                  {"groups": groups, "meta": meta_id})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(f"[green]deployed[/] {v['model']} · your weights "
                  f"(pooled, streamed in {len(groups)} groups) · {_fmt_bytes(total)} · "
                  f"{v['n_layers']} layers · {len(v['stages'])} stages")
    for s in v["stages"]:
        console.print(f"  [cyan]{s['node_name']}[/] layers {s['first']}–{s['last']}")


@app.command()
def chat(prompt: str = typer.Argument(..., help="prompt text"),
         tokens: int = typer.Option(48, "--tokens", "-n"),
         temperature: float = typer.Option(0.8, "--temp")) -> None:
    """Chat with the model running across the pool."""
    try:
        r = _post("/api/v1/inference/chat", {"prompt": prompt, "max_new_tokens": tokens,
                                             "temperature": temperature})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(Panel(r["text"] or "[dim](empty)[/]", title=f"{r['model']}"))
    where = r.get("served_by") or f"{len(r['stages'])} pipeline stages"
    tail = f"[dim]{r['tokens']} tokens · {r['tokens_per_sec']} tok/s · {where}"
    if r.get("backend"):
        tail += f" · {r['backend']}"
    console.print(tail + "[/]")


@app.command()
def bench(model: str = typer.Option("qwen2.5-coder:7b", "--model", "-m"),
         concurrency: int = typer.Option(8, "--concurrency", "-c"),
         requests: int = typer.Option(0, "--requests"),
         tokens: int = typer.Option(64, "--tokens", "-n"),
         ollama: str = typer.Option("http://127.0.0.1:11434", "--ollama"),
         mode: str = typer.Option("both", "--mode", help="ollama | pool | both"),
         deploy_first: bool = typer.Option(False, "--deploy",
                                           help="deploy the model on the pool first")) -> None:
    """Honest throughput benchmark: single-box Ollama vs the pooled mesh (aggregate
    tok/s under concurrency). Run on the head machine of a real multi-box pool."""
    from ..inference import bench as B
    n = requests or (2 * concurrency)
    prompts = B.build_prompts(n)
    cp = base_url()
    console.print(f"[dim]model={model} requests={n} concurrency={concurrency} "
                  f"tokens/req={tokens}[/]")
    baseline = pooled = None
    if mode in ("ollama", "both"):
        console.print(f"[dim]baseline: single Ollama at {ollama} ...[/]")
        baseline = B.run_load("single-box Ollama",
                              lambda p: B.gen_ollama(ollama, model, p, tokens, 0.7),
                              prompts, concurrency)
    if mode in ("pool", "both"):
        if deploy_first:
            try:
                _post("/api/v1/models", {"model": model})
            except Exception as exc:  # noqa: BLE001
                _fail(exc)
        console.print(f"[dim]pooled: coordinator at {cp} ...[/]")
        pooled = B.run_load("MeshPool (pooled)",
                            lambda p: B.gen_pool(cp, p, tokens, 0.7),
                            prompts, concurrency)
    B.print_report(baseline, pooled, concurrency)


# ---- ledger -------------------------------------------------------------- #
@app.command()
def ledger() -> None:
    """Show the contribution ledger (compute donated vs consumed)."""
    try:
        d = _get("/api/v1/ledger")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    t = Table(title="Contribution ledger")
    for c in ("node", "owner", "donated", "consumed"):
        t.add_column(c)
    for row in d["by_node"]:
        t.add_row(row["name"], row["owner"], f"{row['donated']:.1f}", f"{row['consumed']:.1f}")
    console.print(t)


# ---- secrets ------------------------------------------------------------- #
secrets_app = typer.Typer(help="Manage scoped, single-use secrets for workloads.")
app.add_typer(secrets_app, name="secrets")


@secrets_app.command("put")
def secrets_put(name: str = typer.Argument(..., help="secret name, e.g. API_TOKEN"),
                value: str = typer.Argument(..., help="secret value (kept in memory only)")) -> None:
    """Store a secret. The value is held in memory by the control plane and only
    ever handed to a workload via a single-use, node-scoped lease."""
    try:
        _post("/api/v1/secrets", {"name": name, "value": value})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(f"[green]stored[/] secret [bold]{name}[/] (value hidden)")


@secrets_app.command("list")
def secrets_list() -> None:
    """List secret names (values are never shown)."""
    try:
        d = _get("/api/v1/secrets")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    t = Table(title="Secrets (names only)")
    t.add_column("name")
    for n in d["names"]:
        t.add_row(n)
    console.print(t)
    console.print(f"[dim]active leases: {d['active_leases']}[/]")


@secrets_app.command("rm")
def secrets_rm(name: str = typer.Argument(..., help="secret name to delete")) -> None:
    """Delete a stored secret."""
    try:
        d = _delete(f"/api/v1/secrets/{name}")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    msg = "deleted" if d.get("deleted") else "not found"
    console.print(f"[yellow]{msg}[/] {name}")


# ---- node control -------------------------------------------------------- #
@app.command()
def reclaim(node_id: str) -> None:
    """Instantly reclaim a machine's donated resources (kill-switch)."""
    try:
        _post(f"/api/v1/nodes/{node_id}/reclaim")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(f"[yellow]reclaimed[/] {node_id} — resources freed")


@app.command()
def resume(node_id: str) -> None:
    """Resume donating on a previously reclaimed machine."""
    try:
        _post(f"/api/v1/nodes/{node_id}/resume")
    except Exception as exc:  # noqa: BLE001
        _fail(exc)
    console.print(f"[green]resumed[/] {node_id}")


@app.command()
def up(agents: int = typer.Option(3, "--agents", "-n"),
       port: int = typer.Option(8080, "--port")) -> None:
    """Launch a local simulation cluster (control plane + N agents)."""
    from sim.harness import Harness
    import time
    h = Harness(n_agents=agents, cp_port=port)
    console.print(f"[green]starting[/] control plane + {agents} agents ...")
    h.start()
    console.print(Panel.fit(f"Dashboard: [bold]{h.cp_url()}[/]\nCtrl-C to stop.",
                            title="MeshPool up"))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("stopping ...")
    finally:
        h.stop()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
