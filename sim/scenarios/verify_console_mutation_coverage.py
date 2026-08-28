#!/usr/bin/env python
"""Meta-gate: prove the browser end-to-end tests can actually FAIL.

`sim/scenarios/verify_mutation_coverage.py` runs its mutations inside a git
worktree built from HEAD. That cannot work for the console, because `web/dist`
and `web/node_modules` are both gitignored, so a fresh worktree has neither a
built bundle nor a toolchain to build one. These mutations therefore run
IN PLACE, with the original bytes restored in a finally block and the working
tree re-verified against git afterwards.

Each mutation below is a defect that leaves every one of the 119 jsdom tests in
`web/tests/` green, because jsdom implements neither canvas painting, nor
EventSource retry, nor real HTTP. If a mutation SURVIVES, the corresponding
browser test is asserting on a symptom that occurs either way, and the test is
wrong -- not the code.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

MUTATIONS = [
    {
        "id": "console-sse-handlers-back-in-effect-deps",
        "path": "web/src/hooks/useSSE.ts",
        "old": b"  }, [url])",
        "new": b"  }, [handlers, url])",
        "test": "e2e/sse.spec.ts",
        "why": (
            "Restores the original reconnect loop. Callers pass a fresh handlers "
            "object literal every render, so the effect re-runs every render; "
            "onMessage sets state, so every delivered message causes a render, "
            "which reopens the stream. Measured at 729 connections in 10s from a "
            "single tab. Nothing errors, every screen still renders and all 119 "
            "jsdom tests still pass -- the damage is connection churn against the "
            "control plane, which is only visible by counting connections over "
            "time in a real browser."
        ),
    },
    {
        "id": "console-meshmap-canvas-collapses-to-zero-width",
        "path": "web/src/components/MeshMap.tsx",
        "old": b"const W = 1000",
        "new": b"const W = 0",
        "test": "e2e/console.spec.ts -g \"canvas actually paints\"",
        "why": (
            "The Mesh Map is the product's signature screen. Collapsing its "
            "drawing surface leaves the component mounting normally, still "
            "exposing role='img' and its aria-label, still handling clicks and "
            "keyboard events, and still passing all 26 jsdom tests for the "
            "screen -- because jsdom's getContext() is an unimplemented stub, so "
            "those tests never observed a pixel in the first place. The user "
            "simply sees nothing. Only reading back the canvas can tell."
        ),
    },
    {
        "id": "console-command-palette-hotkey-dead",
        "path": "web/src/components/CommandPalette.tsx",
        "old": b"      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {",
        "new": b"      if (false) {",
        "test": "e2e/console.spec.ts -g \"command palette\"",
        "why": (
            "The palette is the keyboard-first entry point to every screen. A "
            "jsdom test can dispatch a synthetic KeyboardEvent and assert the "
            "component opens, which passes as long as the handler exists -- it "
            "cannot verify the browser's real key event reaches it through the "
            "real listener. This mutation leaves the palette fully functional "
            "when opened by any other means."
        ),
    },
    {
        "id": "console-deeplink-returns-404",
        "path": "meshpool/controlplane/app.py",
        "old": b'    @app.get("/console/{path:path}")',
        "new": b'    @app.get("/console/__disabled/{path:path}")',
        "test": "e2e/console.spec.ts -g \"deep-linking\"",
        "why": (
            "The console is client-side routed but server-served: every "
            "/console/* path must return index.html or a browser refresh on a "
            "sub-route 404s. In-app navigation -- which is all a component test "
            "ever exercises -- keeps working perfectly under this mutation, "
            "because React Router handles it without touching the server. Only a "
            "real navigation to a deep link can observe it."
        ),
    },
    {
        "id": "console-muted-text-fails-contrast",
        "path": "web/src/index.css",
        "old": b"  --color-muted: #475569;",
        "new": b"  --color-muted: #64748b;",
        "test": "e2e/a11y.spec.ts",
        "why": (
            "Restores the original muted colour: slate-500 on the slate-100 "
            "surface is 4.47:1, just under the 4.5 AA floor, across 9 elements "
            "including the command-palette hint on every screen. jsdom "
            "implements no CSS cascade and no layout, so not one of the 119 "
            "component tests can compute a contrast ratio -- they all stayed "
            "green through the far worse version of this bug, where the invite "
            "token rendered slate-100 on white. Only a real browser can see it."
        ),
    },
    {
        "id": "console-slider-loses-its-accessible-name",
        "path": "web/src/pages/Playground.tsx",
        "old": b'<Slider.Thumb aria-label="Temperature"',
        "new": b"<Slider.Thumb",
        "test": "e2e/a11y.spec.ts",
        "why": (
            "Radix renders the thumb as role='slider'; without a name a screen "
            "reader announces only a number, so the control is unusable. The "
            "slider still works with a mouse and still fires onValueChange, so "
            "every behavioural test passes either way."
        ),
    },
    {
        "id": "console-prompt-box-loses-its-label",
        "path": "web/src/pages/Playground.tsx",
        "old": b'<label htmlFor="playground-prompt"',
        "new": b"<label",
        "test": "e2e/a11y.spec.ts",
        "why": (
            "Detaches the label from the flagship's prompt box, leaving it an "
            "unnamed form control -- a WCAG critical. The visible text still "
            "reads 'Prompt', so the screen looks correct and typing still "
            "works; only the programmatic association is gone, which is exactly "
            "what a rendering-only assertion cannot distinguish."
        ),
    },
]


def _exe(name: str) -> str | None:
    """Resolve a node tool. On Windows these are .cmd shims, which CreateProcess
    will not find from a bare name, so a missing resolution must be reported as
    unrunnable rather than crashing the whole gate."""
    return shutil.which(name) or shutil.which(f"{name}.cmd")


NPM = _exe("npm")
NPX = _exe("npx")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _apply(path: Path, old: bytes, new: bytes) -> bytes | None:
    """Swap `old` for `new`, returning the original bytes, or None on a miss."""
    raw = path.read_bytes()
    if raw.count(old) != 1:
        return None
    path.write_bytes(raw.replace(old, new))
    return raw


def main() -> int:
    only = set(sys.argv[1:])
    selected = [m for m in MUTATIONS if not only or m["id"] in only]
    if only:
        unknown = only - {m["id"] for m in MUTATIONS}
        if unknown:
            print(f"unknown mutation id(s): {', '.join(sorted(unknown))}")
            return 2

    if NPM is None or NPX is None:
        print("UNRUNNABLE: npm/npx not on PATH. An unevaluated claim is not a passed one.")
        return 1
    if not (WEB / "node_modules").exists():
        print("UNRUNNABLE: web/node_modules is absent -- run `npm --prefix web install`.")
        print("An unevaluated claim is not a passed one.")
        return 1

    # Baseline. If the suite is not green to begin with, every 'CAUGHT' below
    # would be meaningless: the tests would be failing for reasons of their own.
    print("building console and establishing a green baseline...")
    build = _run([NPM, "run", "build"], WEB)
    if build.returncode != 0:
        print("UNRUNNABLE: baseline build failed\n" + build.stdout[-2000:] + build.stderr[-2000:])
        return 1
    base = _run([NPX, "playwright", "test"], WEB)
    if base.returncode != 0:
        print("UNRUNNABLE: baseline e2e run is not green, so mutation results "
              "would be uninterpretable.\n" + base.stdout[-3000:])
        return 1
    print("baseline green.\n")

    caught, survived, unrunnable = [], [], []

    for mut in selected:
        target = ROOT / mut["path"]
        original = _apply(target, mut["old"], mut["new"])
        if original is None:
            n = target.read_bytes().count(mut["old"])
            unrunnable.append(f"{mut['id']}: ANCHOR MISS ({n} matches) in {mut['path']}")
            print(f"ANCHOR MISS  {mut['id']}  ({n} matches)")
            continue

        try:
            is_web = mut["path"].startswith("web/")
            if is_web:
                b = _run([NPM, "run", "build"], WEB)
                if b.returncode != 0:
                    # A mutation that does not compile is not evidence about the
                    # test; it is evidence about the mutation.
                    unrunnable.append(f"{mut['id']}: build failed under mutation")
                    print(f"UNRUNNABLE   {mut['id']}  (build failed)")
                    continue
            cmd = [NPX, "playwright", "test", *mut["test"].split(" ", 1)[0].split()]
            if " -g " in mut["test"]:
                cmd += ["-g", mut["test"].split(" -g ", 1)[1].strip('"')]
            res = _run(cmd, WEB)
            if res.returncode != 0:
                caught.append(mut["id"])
                print(f"CAUGHT       {mut['id']}")
            else:
                survived.append(mut["id"])
                print(f"SURVIVED     {mut['id']}  <-- the test cannot see this defect")
        finally:
            target.write_bytes(original)

    # Restore the built bundle so the tree is left exactly as found.
    _run([NPM, "run", "build"], WEB)

    dirty = _run(["git", "status", "--porcelain", "--", "web/src", "meshpool"], ROOT)
    if dirty.stdout.strip():
        print("\nERROR: mutations were not fully restored:\n" + dirty.stdout)
        return 1

    print(f"\n{len(caught)} caught, {len(survived)} survived, "
          f"{len(unrunnable)} unrunnable, of {len(selected)} selected")
    for u in unrunnable:
        print(f"  UNRUNNABLE  {u}")
    for s in survived:
        print(f"  SURVIVED    {s}")

    if survived or unrunnable:
        print("\nA surviving mutation means the TEST is wrong, not that the code was fine.")
        return 1
    print(f"\nALL {len(caught)} CONSOLE MUTATIONS CAUGHT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
