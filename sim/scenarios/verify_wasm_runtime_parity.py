"""Acceptance gate: the Python (wasmtime) and Go (wazero) function runtimes agree.

Why this gate has to exist
--------------------------
A job's units are fanned out across whichever agents happen to be online, and the mesh
deliberately mixes Python agents and single-binary Go agents. So the property that
actually matters is not "each runtime is self-consistent" -- it is "both runtimes give
the *same answer for the same bytes*". If they diverge, two things break at once:

  * results become non-deterministic across the pool, and
  * the coordinator's spot-check oracle (which re-runs a sampled unit against its own
    copy of the module) starts flagging honest nodes as fraudulent and slashing them.

Testing each runtime separately -- which is what tests/test_wasm_runtime.py does for
Python and internal/sandbox for Go -- cannot see that class of bug at all.

What it does
------------
Builds each module *once* from WAT, then feeds the identical bytes to both runtimes and
compares the outcome: the returned value for successes, the error ``kind`` for failures.
The kind taxonomy is what the coordinator branches on to decide "retry elsewhere" versus
"this job is broken", so a mismatch there is as damaging as a wrong value.

NON-VACUITY
-----------
Three independent guards, because a cross-runtime comparison is unusually easy to make
vacuous -- if either side silently fails to run, "both agree" degenerates into "both
errored the same way":

 1. A minimum number of cases must actually execute (``MIN_CASES``).
 2. At least one case must return a real *computed value* through both runtimes, so a
    build where every module failed to load cannot pass.
 3. A deliberate control: the same module is run with two different payloads and the
    two runtimes must both track the payload. If a runtime ignored stdin, its answers
    would be identical across payloads and the control fails.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MIN_CASES = 8


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def _build_probe(workdir: Path) -> Path | None:
    """Compile cmd/wasmfn-probe. Returns None if the Go toolchain is unavailable."""
    if shutil.which("go") is None:
        return None
    out = workdir / ("wasmfn-probe.exe" if sys.platform == "win32" else "wasmfn-probe")
    proc = subprocess.run(
        ["go", "build", "-o", str(out), "./cmd/wasmfn-probe"],
        cwd=str(ROOT / "agent-go"), capture_output=True, text=True, errors="replace", timeout=600,
    )
    if proc.returncode != 0:
        print(f"  FAIL: go build failed:\n{proc.stdout}\n{proc.stderr}")
        return None
    return out


def _run_go(probe: Path, module_path: Path, payload, timeout_s: float = 10.0) -> dict:
    proc = subprocess.run(
        [str(probe), "--module", str(module_path),
         "--payload", json.dumps(payload),
         "--timeout", f"{timeout_s}s"],
        capture_output=True, text=True, errors="replace", timeout=timeout_s + 60,
    )
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return {"ok": False, "kind": "probe",
                "message": f"probe printed nothing (rc={proc.returncode}): {proc.stderr[:400]}"}
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "kind": "probe", "message": f"probe output not JSON: {exc}"}


def _run_py(module_bytes: bytes, payload, timeout_s: float = 10.0) -> dict:
    from meshpool.functions import wasm
    limits = wasm.Limits(wall_timeout_s=timeout_s)
    try:
        res = wasm.run(module_bytes, payload, limits=limits)
        return {"ok": True, "value": res.value}
    except wasm.FunctionError as exc:
        return {"ok": False, "kind": exc.kind, "message": str(exc)}


def _wat_bytes(s: str) -> str:
    """Hex-escape a string for a WAT data segment (WAT has no \\" escape)."""
    return "".join(f"\\{b:02x}" for b in s.encode("utf-8"))


def _fs_boolean_module() -> bytes:
    """path_open a host file and report *whether it worked* as a JSON boolean.

    Reporting a boolean rather than the errno is what makes this usable across two
    different WASM runtimes: wasmtime and wazero disagree on which errno a refused
    open produces, but they must never disagree on whether the open succeeded.
    """
    import wasmtime

    path = "/etc/passwd"
    yes, no = '{"opened":true}', '{"opened":false}'
    wat = f"""
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 200) "{_wat_bytes(path)}")
  (data (i32.const 600) "{_wat_bytes(yes)}")
  (data (i32.const 640) "{_wat_bytes(no)}")
  (func (export "_start")
    (local $errno i32)
    ;; fd 3 is the first preopen slot; with no preopens it does not exist.
    ;; Rights are passed as 0 -- an all-ones mask overflows wasmtime's Rights
    ;; conversion before the sandbox check is even reached.
    (local.set $errno
      (call $path_open (i32.const 3) (i32.const 0)
                       (i32.const 200) (i32.const {len(path)})
                       (i32.const 0) (i64.const 0) (i64.const 0)
                       (i32.const 0) (i32.const 400)))
    (if (i32.eqz (local.get $errno))
      (then
        (i32.store (i32.const 0) (i32.const 600))
        (i32.store (i32.const 4) (i32.const {len(yes)})))
      (else
        (i32.store (i32.const 0) (i32.const 640))
        (i32.store (i32.const 4) (i32.const {len(no)}))))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 16)))
  )
)
"""
    return bytes(wasmtime.wat2wasm(wat))


def main() -> int:  # noqa: C901
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        return _skip("wasmtime is not installed; the Python runtime cannot be exercised")

    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_wasm_runtime import (  # type: ignore  # noqa: E402
            _const_output_module,
            _echo_module,
            _exit_module,
            _foreign_import_module,
            _silent_module,
            _spin_module,
            _trap_module,
        )
    except ImportError as exc:
        return _skip(f"could not import the WAT module builders: {exc}")

    print("== building the Go probe (cmd/wasmfn-probe) ==")
    tmp = Path(tempfile.mkdtemp(prefix="wasmparity-"))
    try:
        probe = _build_probe(tmp)
        if probe is None:
            return _skip("the Go toolchain is unavailable, so meshd cannot be exercised")
        print(f"  built {probe.name}")

        # (label, module bytes, payload, expectation)
        # expectation is "value" (both must succeed and agree) or "kind" (both must fail
        # with the same kind).
        cases: list[tuple[str, bytes, object, str]] = [
            ("echo dict",       _echo_module(), {"x": 1, "y": "two"}, "value"),
            ("echo list",       _echo_module(), [1, 2, 3],            "value"),
            ("echo scalar",     _echo_module(), 42,                   "value"),
            ("echo nested",     _echo_module(), {"a": {"b": [1, {"c": True}]}}, "value"),
            ("const output",    _const_output_module('{"ok":true,"n":7}'), None, "value"),
            ("non-JSON stdout", _const_output_module("not json at all"), None, "kind"),
            ("exit non-zero",   _exit_module(3),  None, "kind"),
            ("exit zero+silent", _exit_module(0), None, "kind"),
            ("trap",            _trap_module(),   None, "kind"),
            ("silent",          _silent_module(), None, "kind"),
            ("foreign import",  _foreign_import_module(), None, "kind"),
        ]

        ok = True
        executed = 0
        value_agreements = 0
        results: dict[str, tuple[dict, dict]] = {}

        print("\n== test 1: both runtimes agree on the same module bytes ==")
        for label, module_bytes, payload, expect in cases:
            mod_path = tmp / f"{abs(hash(label)):x}.wasm"
            mod_path.write_bytes(module_bytes)
            py = _run_py(module_bytes, payload)
            go = _run_go(probe, mod_path, payload)
            results[label] = (py, go)
            executed += 1

            if go.get("kind") == "probe":
                ok = False
                print(f"  FAIL: {label}: the Go probe itself failed: {go.get('message')}")
                continue

            if py["ok"] != go["ok"]:
                ok = False
                print(f"  FAIL: {label}: python ok={py['ok']} but go ok={go['ok']}")
                print(f"        python: {py}")
                print(f"        go:     {go}")
                continue

            if py["ok"]:
                if expect != "value":
                    ok = False
                    print(f"  FAIL: {label}: expected both to fail, both succeeded")
                    continue
                if py["value"] != go["value"]:
                    ok = False
                    print(f"  FAIL: {label}: value mismatch\n"
                          f"        python: {py['value']!r}\n"
                          f"        go:     {go['value']!r}")
                    continue
                value_agreements += 1
                print(f"  ok   {label}: both -> {json.dumps(py['value'])[:60]}")
            else:
                if expect != "kind":
                    ok = False
                    print(f"  FAIL: {label}: expected both to succeed, both failed "
                          f"(py={py.get('kind')}, go={go.get('kind')})")
                    continue
                if py["kind"] != go["kind"]:
                    ok = False
                    print(f"  FAIL: {label}: error-kind mismatch — the coordinator "
                          f"branches on this to decide retry-vs-abort\n"
                          f"        python: {py['kind']}: {py.get('message', '')[:120]}\n"
                          f"        go:     {go['kind']}: {go.get('message', '')[:120]}")
                    continue
                print(f"  ok   {label}: both -> {py['kind']}")

        # ---- non-vacuity 1: enough cases actually ran --------------------- #
        print("\n== test 2: non-vacuity — enough cases executed ==")
        if executed < MIN_CASES:
            ok = False
            print(f"  FAIL: only {executed} case(s) ran, need >= {MIN_CASES}")
        else:
            print(f"  PASS: {executed} cases executed")

        # ---- non-vacuity 2: real values crossed both runtimes ------------- #
        print("\n== test 3: non-vacuity — real values agreed, not just errors ==")
        if value_agreements < 3:
            ok = False
            print(f"  FAIL: only {value_agreements} case(s) produced an agreeing *value*. "
                  f"A build where every module failed to load would otherwise 'agree'.")
        else:
            print(f"  PASS: {value_agreements} cases agreed on a computed value")

        # ---- non-vacuity 3: both runtimes actually read stdin ------------- #
        print("\n== test 4: control — both runtimes are actually reading the payload ==")
        echo = _echo_module()
        echo_path = tmp / "echo_control.wasm"
        echo_path.write_bytes(echo)
        a, b = {"probe": "alpha"}, {"probe": "beta"}
        py_a, py_b = _run_py(echo, a), _run_py(echo, b)
        go_a, go_b = _run_go(probe, echo_path, a), _run_go(probe, echo_path, b)
        if not (py_a["ok"] and py_b["ok"] and go_a["ok"] and go_b["ok"]):
            ok = False
            print(f"  FAIL: control runs did not all succeed: "
                  f"py={py_a['ok']}/{py_b['ok']} go={go_a['ok']}/{go_b['ok']}")
        elif py_a["value"] == py_b["value"] or go_a["value"] == go_b["value"]:
            ok = False
            print(f"  FAIL: a runtime returned the same value for different payloads, "
                  f"so it is not reading stdin — every 'agreement' above is vacuous. "
                  f"py: {py_a['value']!r} vs {py_b['value']!r}; "
                  f"go: {go_a['value']!r} vs {go_b['value']!r}")
        elif py_a["value"] != go_a["value"] or py_b["value"] != go_b["value"]:
            ok = False
            print("  FAIL: control payloads did not round-trip identically")
        else:
            print(f"  PASS: both runtimes tracked the payload "
                  f"({py_a['value']} vs {py_b['value']}) and agreed on each")

        # ---- test 5: neither runtime grants a filesystem ------------------ #
        # The module below asks path_open to open /etc/passwd against fd 3 (the first
        # preopen slot) and reports *whether it succeeded* as a JSON boolean.
        #
        # It deliberately does NOT report the errno. tests/test_wasm_runtime.py's
        # variant emits `48 + errno` as one ASCII character, which is fine against a
        # single runtime but useless across two: wasmtime answers 8 (EBADF) and wazero
        # answers 63, so the byte is '8' on one side and 'o' on the other -- and 'o' is
        # not valid JSON, so the Go run came back as a `bad_output` failure. An earlier
        # version of this check read that failure as "no errno to compare" and passed.
        # That is vacuous: a sandbox escape that also crashed would have passed too.
        # Reporting a boolean makes the assertion independent of errno numbering, and
        # makes a failed run distinguishable from a refused open.
        print("\n== test 5: neither runtime grants the guest a filesystem ==")
        fs_probe = _fs_boolean_module()
        fs_path = tmp / "fs_boolean.wasm"
        fs_path.write_bytes(fs_probe)
        go_fs = _run_go(probe, fs_path, None)
        py_fs = _run_py(fs_probe, None)

        if not go_fs.get("ok") or not py_fs.get("ok"):
            ok = False
            print(f"  FAIL: the probe module did not run on both runtimes, so this "
                  f"check proves nothing about isolation.")
            print(f"        python: {py_fs}")
            print(f"        go:     {go_fs}")
        else:
            py_opened = (py_fs["value"] or {}).get("opened")
            go_opened = (go_fs["value"] or {}).get("opened")
            if py_opened is not False or go_opened is not False:
                ok = False
                print(f"  FAIL: SANDBOX ESCAPE — a guest opened a host file "
                      f"(python opened={py_opened!r}, go opened={go_opened!r})")
            else:
                print("  PASS: path_open refused on both runtimes "
                      "(no preopened directory exists to resolve against)")

        if ok:
            print("\nALL WASM RUNTIME PARITY CHECKS PASSED")
        else:
            print("\nSOME WASM RUNTIME PARITY CHECKS FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
