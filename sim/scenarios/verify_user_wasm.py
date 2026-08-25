"""Acceptance gate: a user's OWN code runs on the mesh, sandboxed.

Everything else the pool can do is something *we* wrote: six curated Python
functions shipped on every node. That is a hard ceiling — a user cannot run their
own logic on the mesh at all. This gate proves that ceiling is gone.

What it proves end to end, against a real multi-process cluster:
  1. A WASM module uploaded to the control plane is validated at upload time and
     returned as a content-addressed ``wasm:<sha256>`` function name.
  2. A batch job naming that function fans out across MULTIPLE nodes and every
     unit is computed BY THE USER'S CODE (results are checked against the
     function's actual semantics, not just "not an error").
  3. The module is distributed by the pooled, content-addressed blob store — the
     same machinery that ships model weights — and pushed at most once per node,
     not once per unit.
  4. The sandbox holds: a module that tries to touch the host filesystem cannot,
     an infinite loop is killed rather than hanging a donor's machine, and a
     module importing host functions is refused at upload time.
  5. Garbage uploads are rejected with 4xx, never a 500.

Non-vacuity is asserted throughout: the job must actually have been spread over
more than one node, and the results must be ones only the user's code could
produce (a stubbed-out runtime returning ``None`` or echoing the input fails).

Skips cleanly (exit 0) if wasmtime is not installed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from sim.harness import Harness  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "sim" / "scenarios" / "_data_userwasm"
CP_PORT = 8177

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)
    return ok


# --------------------------------------------------------------------------- #
# The user's "code". Built from WAT so the gate needs no Rust/Go toolchain, but
# it is real compiled WebAssembly running under real WASI.
# --------------------------------------------------------------------------- #

def _wat(src: str) -> bytes:
    import wasmtime
    # wat2wasm returns a bytearray; httpx iterates a bytearray into ints, so the
    # upload must be real bytes.
    return bytes(wasmtime.wat2wasm(src))


def _triple_module() -> bytes:
    """Reads {"x": N} on stdin, writes {"tripled": N*3} on stdout.

    Parses the integer itself, in the guest, so a correct answer can only come
    from the user's code having actually executed.
    """
    return _wat(r"""
(module
  (import "wasi_snapshot_preview1" "fd_read"
    (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 4)
  (data (i32.const 8192) "{\"tripled\":")
  (data (i32.const 8300) "}")

  ;; scan the input buffer for ascii digits and fold them into an integer
  (func $parse (param $ptr i32) (param $len i32) (result i32)
    (local $i i32) (local $c i32) (local $acc i32) (local $seen i32)
    (block $out
      (loop $scan
        (br_if $out (i32.ge_u (local.get $i) (local.get $len)))
        (local.set $c (i32.load8_u (i32.add (local.get $ptr) (local.get $i))))
        (if (i32.and (i32.ge_u (local.get $c) (i32.const 48))
                     (i32.le_u (local.get $c) (i32.const 57)))
          (then
            (local.set $acc (i32.add (i32.mul (local.get $acc) (i32.const 10))
                                     (i32.sub (local.get $c) (i32.const 48))))
            (local.set $seen (i32.const 1)))
          (else
            ;; digits are contiguous: stop at the first non-digit after any digit
            (br_if $out (local.get $seen))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $scan)
      )
    )
    (local.get $acc)
  )

  ;; write a non-negative integer as ascii at $ptr, return the byte count
  (func $itoa (param $n i32) (param $ptr i32) (result i32)
    (local $len i32) (local $i i32) (local $j i32) (local $t i32)
    (if (i32.eqz (local.get $n))
      (then
        (i32.store8 (local.get $ptr) (i32.const 48))
        (return (i32.const 1))))
    (local.set $t (local.get $n))
    (block $done
      (loop $count
        (br_if $done (i32.eqz (local.get $t)))
        (local.set $len (i32.add (local.get $len) (i32.const 1)))
        (local.set $t (i32.div_u (local.get $t) (i32.const 10)))
        (br $count)))
    (local.set $t (local.get $n))
    (local.set $i (local.get $len))
    (block $wdone
      (loop $write
        (br_if $wdone (i32.eqz (local.get $i)))
        (local.set $i (i32.sub (local.get $i) (i32.const 1)))
        (i32.store8 (i32.add (local.get $ptr) (local.get $i))
                    (i32.add (i32.const 48) (i32.rem_u (local.get $t) (i32.const 10))))
        (local.set $t (i32.div_u (local.get $t) (i32.const 10)))
        (br $write)))
    (local.get $len)
  )

  (func (export "_start")
    (local $n i32) (local $digits i32) (local $total i32)
    (i32.store (i32.const 0) (i32.const 1024))
    (i32.store (i32.const 4) (i32.const 4096))
    (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 16)))
    (local.set $n (call $parse (i32.const 1024) (i32.load (i32.const 16))))
    ;; build {"tripled":<n*3>}
    (local.set $digits
      (call $itoa (i32.mul (local.get $n) (i32.const 3)) (i32.const 8203)))
    ;; prefix is 11 bytes at 8192; move it to sit right before the digits
    (memory.copy (i32.const 8192) (i32.const 8192) (i32.const 11))
    (i32.store8 (i32.add (i32.const 8203) (local.get $digits)) (i32.const 125))
    (local.set $total (i32.add (i32.const 12) (local.get $digits)))
    (i32.store (i32.const 0) (i32.const 8192))
    (i32.store (i32.const 4) (local.get $total))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 20)))
  )
)
""")


def _spin_module() -> bytes:
    return _wat(r"""
(module (memory (export "memory") 1)
  (func (export "_start") (local $i i64)
    (loop $f (local.set $i (i64.add (local.get $i) (i64.const 1))) (br $f))))
""")


def _fs_module() -> bytes:
    path = "/etc/passwd"
    esc = "".join(f"\\{b:02x}" for b in path.encode())
    return _wat(f"""
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 200) "{esc}")
  (func (export "_start")
    (local $e i32)
    (local.set $e (call $path_open (i32.const 3) (i32.const 0)
                        (i32.const 200) (i32.const {len(path)}) (i32.const 0)
                        (i64.const 0) (i64.const 0) (i32.const 0) (i32.const 400)))
    (i32.store (i32.const 500) (i32.add (i32.const 48) (local.get $e)))
    (i32.store (i32.const 0) (i32.const 500))
    (i32.store (i32.const 4) (i32.const 1))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 16)))))
""")


def _host_import_module() -> bytes:
    return _wat(r"""
(module
  (import "env" "read_host_memory" (func $x (param i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start") (drop (call $x (i32.const 0)))))
""")


def _wait_job(cp: str, job_id: str, timeout: float = 120.0) -> dict:
    end = time.time() + timeout
    last = {}
    while time.time() < end:
        last = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()
        if last.get("status") in ("completed", "failed"):
            return last
        time.sleep(0.4)
    return last


def main() -> int:  # noqa: C901
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        print("SKIP: wasmtime is not installed (pip install wasmtime)")
        return 0

    print("== verifying: a user's own WASM code runs across the mesh ==", flush=True)

    h = Harness(n_agents=3, cp_port=CP_PORT, scale="home", data_dir=str(DATA))
    try:
        h.start()
        cp = h.cp_url()
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
        check("cluster is up", len(nodes) == 3, f"{len(nodes)} nodes")

        # ---- 1) upload + preflight ---------------------------------------- #
        module = _triple_module()
        r = httpx.post(f"{cp}/api/v1/functions", content=module,
                       headers={"Content-Type": "application/octet-stream"},
                       timeout=60)
        ok = check("user module accepted and content-addressed",
                   r.status_code == 200, f"HTTP {r.status_code} {r.text[:200]}")
        if not ok:
            return 1
        info = r.json()
        func = info["function"]
        check("returned id is the real sha256 of the bytes",
              info["sha256"] == hashlib.sha256(module).hexdigest(),
              info["sha256"][:16])
        check("module is declared to use WASI preview 1 only",
              all(i.startswith("wasi_snapshot_preview1::") for i in info["imports"]),
              ", ".join(info["imports"]))

        # ---- 2) the user's code actually computes -------------------------- #
        payloads = [{"x": i} for i in range(1, 25)]
        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "user-wasm", "func": func, "payloads": payloads,
            "cpu_pct": 20, "ram_mb": 128}, timeout=30)
        ok = check("job naming a wasm function was accepted",
                   r.status_code == 200, f"HTTP {r.status_code} {r.text[:200]}")
        if not ok:
            return 1
        job = _wait_job(cp, r.json()["job_id"])
        check("every unit completed", job.get("status") == "completed",
              f"status={job.get('status')} "
              f"errors={[u.get('error') for u in job.get('units', []) if u.get('error')][:3]}")

        units = job.get("units", [])
        wrong = [(u["payload"]["x"], u.get("result")) for u in units
                 if u.get("result") != {"tripled": u["payload"]["x"] * 3}]
        check("results are what the USER'S code computes (x -> x*3)",
              not wrong and len(units) == len(payloads),
              f"{len(units) - len(wrong)}/{len(payloads)} correct"
              + (f"; first wrong: {wrong[0]}" if wrong else ""))

        used = {u.get("assigned_node") for u in units if u.get("assigned_node")}
        check("the work was really spread across the pool", len(used) >= 2,
              f"{len(used)} distinct node(s) ran units")

        # ---- 3) distribution cost --------------------------------------- #
        summary = httpx.get(f"{cp}/api/v1/cluster", timeout=10).json()["summary"]
        pushes = summary.get("wasm_module_pushes", -1)
        check("module shipped once per node, not once per unit",
              1 <= pushes <= len(nodes),
              f"{pushes} push(es) for {len(units)} units across {len(used)} node(s)")

        # ---- 4) the sandbox holds ---------------------------------------- #
        r = httpx.post(f"{cp}/api/v1/functions", content=_fs_module(),
                       headers={"Content-Type": "application/octet-stream"}, timeout=60)
        if check("a module that pokes at the filesystem still uploads",
                 r.status_code == 200, f"HTTP {r.status_code}"):
            fsf = r.json()["function"]
            r = httpx.post(f"{cp}/api/v1/jobs", json={
                "name": "fs-escape", "func": fsf, "payloads": [{}],
                "cpu_pct": 20, "ram_mb": 128}, timeout=30)
            j = _wait_job(cp, r.json()["job_id"])
            res = (j.get("units") or [{}])[0].get("result")
            check("guest CANNOT open a host file (WASI denies it: EBADF)",
                  res == 8, f"path_open returned errno {res!r} (8 = EBADF)")

        r = httpx.post(f"{cp}/api/v1/functions", content=_spin_module(),
                       headers={"Content-Type": "application/octet-stream"}, timeout=60)
        if check("an infinite-loop module uploads (it is not statically detectable)",
                 r.status_code == 200, f"HTTP {r.status_code}"):
            spinf = r.json()["function"]
            r = httpx.post(f"{cp}/api/v1/jobs", json={
                "name": "spin", "func": spinf, "payloads": [{}],
                "cpu_pct": 20, "ram_mb": 128}, timeout=30)
            t0 = time.time()
            j = _wait_job(cp, r.json()["job_id"], timeout=180)
            elapsed = time.time() - t0
            u0 = (j.get("units") or [{}])[0]
            check("a runaway function is KILLED, not left burning a donor's CPU",
                  j.get("status") == "failed" or u0.get("error"),
                  f"status={j.get('status')} after {elapsed:.1f}s "
                  f"err={str(u0.get('error'))[:120]}")

        # ---- 5) bad uploads are refused cleanly -------------------------- #
        r = httpx.post(f"{cp}/api/v1/functions", content=_host_import_module(),
                       headers={"Content-Type": "application/octet-stream"}, timeout=60)
        check("a module wanting host FFI is refused at UPLOAD time",
              r.status_code == 400, f"HTTP {r.status_code} {r.text[:160]}")

        r = httpx.post(f"{cp}/api/v1/functions", content=b"definitely not wasm",
                       headers={"Content-Type": "application/octet-stream"}, timeout=60)
        check("garbage upload is a clean 400, never a 500",
              r.status_code == 400, f"HTTP {r.status_code} {r.text[:160]}")

        r = httpx.post(f"{cp}/api/v1/jobs", json={
            "name": "bad-name", "func": "wasm:not-a-hash", "payloads": [{}],
            "cpu_pct": 10, "ram_mb": 64}, timeout=30)
        if r.status_code == 200:
            j = _wait_job(cp, r.json()["job_id"], timeout=60)
            u0 = (j.get("units") or [{}])[0]
            check("a job naming a malformed module id fails, never hangs",
                  j.get("status") == "failed" or u0.get("error"),
                  f"status={j.get('status')} err={str(u0.get('error'))[:100]}")
        else:
            check("a job naming a malformed module id is rejected",
                  400 <= r.status_code < 500, f"HTTP {r.status_code}")

    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        fails.append(f"exception: {type(exc).__name__}: {exc}")
    finally:
        h.stop()

    if fails:
        print(f"\n== FAILURES: {', '.join(fails)} ==")
        return 1
    print("\nPASS: users can run their own code on the mesh, sandboxed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
