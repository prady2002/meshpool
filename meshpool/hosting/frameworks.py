"""Framework auto-detection for ``meshpool deploy``.

The CLI uses this module to make the Vercel/Fly-style happy path automatic:
inspect the project, write a small ``meshpool.json`` that records the inferred
build/run plan, and deploy the real output rather than asking the user to choose
between hosting modes.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppPlan:
    root: Path
    kind: str                    # "site" | "proc"
    framework: str
    deploy_dir: Path
    start_cmd: str = ""
    build_cmd: str = ""
    wrote_config: bool = False


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _first_existing(root: Path, names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    return None


def infer_app(root: Path, explicit_cmd: str = "") -> AppPlan:
    """Infer how to deploy *root* without prompting.

    Explicit Procfile/``meshpool.json`` wins. Otherwise common JavaScript and
    Python layouts are detected. Unknown folders fall back to static-site hosting
    if they already contain ``index.html``.
    """
    root = root.resolve()
    procfile = root / "Procfile"
    manifest = root / "meshpool.json"
    manifest_data = _read_json(manifest) if manifest.is_file() else {}
    if explicit_cmd:
        return AppPlan(root, "proc", "custom", root, start_cmd=explicit_cmd)
    if procfile.is_file():
        for line in procfile.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, _, val = line.partition(":")
            if key.strip().lower() == "web" and val.strip():
                return AppPlan(root, "proc", "procfile", root, start_cmd=val.strip())
    start = manifest_data.get("start")
    if isinstance(start, str) and start.strip():
        return AppPlan(root, "proc", str(manifest_data.get("framework") or "meshpool"),
                       root, start_cmd=start.strip())

    package = root / "package.json"
    if package.is_file():
        pkg = _read_json(package)
        scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        build_cmd = "npm run build" if "build" in scripts else ""
        output = _first_existing(root, ("dist", "build", "out", ".next"))
        if "vite" in deps or (root / "vite.config.ts").is_file() or (root / "vite.config.js").is_file():
            return AppPlan(root, "site", "vite", output or root / "dist", build_cmd=build_cmd)
        if "react-scripts" in deps:
            return AppPlan(root, "site", "create-react-app", output or root / "build",
                           build_cmd=build_cmd)
        if "next" in deps:
            if "start" in scripts:
                return AppPlan(root, "proc", "next", root, start_cmd="npm run start",
                               build_cmd=build_cmd)
            return AppPlan(root, "site", "next-static", output or root / "out",
                           build_cmd=build_cmd)
        if "start" in scripts:
            return AppPlan(root, "proc", "node", root, start_cmd="npm run start",
                           build_cmd=build_cmd)
        if build_cmd:
            return AppPlan(root, "site", "node-static", output or root / "dist",
                           build_cmd=build_cmd)

    if (root / "pyproject.toml").is_file() or (root / "requirements.txt").is_file():
        for module in ("app", "main"):
            p = root / f"{module}.py"
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "FastAPI(" in text:
                return AppPlan(root, "proc", "fastapi", root,
                               start_cmd=f"python -m uvicorn {module}:app "
                                         "--host 0.0.0.0 --port {port}")
            if "Flask(" in text:
                return AppPlan(root, "proc", "flask", root,
                               start_cmd=f"python -m flask --app {module} run "
                                         "--host 0.0.0.0 --port {port}")

    if (root / "index.html").is_file():
        return AppPlan(root, "site", "static", root)
    return AppPlan(root, "site", "unknown-static", root)


def write_config(plan: AppPlan) -> AppPlan:
    """Persist the inferred plan so future deploys are explainable and repeatable."""
    path = plan.root / "meshpool.json"
    data = _read_json(path) if path.is_file() else {}
    changed = False
    for key, value in {
        "framework": plan.framework,
        "type": plan.kind,
        "build": plan.build_cmd,
        "start": plan.start_cmd,
        "output": str(plan.deploy_dir.relative_to(plan.root))
        if plan.deploy_dir != plan.root and plan.deploy_dir.is_relative_to(plan.root) else "",
    }.items():
        if value and data.get(key) != value:
            data[key] = value
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return AppPlan(plan.root, plan.kind, plan.framework, plan.deploy_dir,
                   plan.start_cmd, plan.build_cmd, wrote_config=changed)


def prepare_app(root: Path, explicit_cmd: str = "", run_build: bool = True) -> AppPlan:
    """Infer, persist config, and run the detected build step when needed."""
    plan = write_config(infer_app(root, explicit_cmd))
    if run_build and plan.build_cmd and not plan.start_cmd:
        subprocess.run(plan.build_cmd, cwd=str(plan.root), shell=True, check=True, timeout=600)
    return plan
