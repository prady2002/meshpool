"""Static-site hosting helpers: validate, write, and serve a user-supplied
static website (the "host your own app / website" capability).

A *site* is a JSON-friendly mapping of ``relative/path -> base64(bytes)`` so it
travels intact inside the existing ``start_replica`` control frame — over the LAN
*or* the reverse tunnel, unchanged. These helpers are the **single trust
boundary** for hosted content: they reject path traversal, cap the file count and
total size, require an ``index.html`` entry point, and resolve every request back
to a file strictly inside the replica's own scratch directory. A hosted site can
therefore never read or write outside its sandbox dir, and only *static bytes*
are ever served (no user code executes — arbitrary containers remain a separate,
later opt-in).
"""
from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import os
import re
import zipfile
from typing import Optional

# Real multi-page apps (SPAs, static-site-generator output) are far larger than a
# toy demo, so raise the ceilings well above the old 3 MB / 60 files and let a
# power user push them further via env with no code change. The site still ships
# inside one control frame, so the default stays comfortable over the LAN/tunnel.
MAX_FILES = int(os.environ.get("MESHPOOL_SITE_MAX_FILES", "1000"))
MAX_TOTAL_BYTES = int(os.environ.get("MESHPOOL_SITE_MAX_BYTES", str(25_000_000)))  # 25 MB
# An app larger than the inline ceiling above is not rejected: the deploy path stages
# it in the pooled, content-addressed blob store (split into replicated chunks spread
# across the mesh) and hydrates it back on each replica start, so a genuinely large
# multi-page app can be hosted even though it never fits in a single control frame.
# This is the hard cap for that pooled path (also bounds a replica's scratch disk).
MAX_BUNDLE_BYTES = int(os.environ.get("MESHPOOL_BUNDLE_MAX_BYTES", str(500_000_000)))  # 500 MB
MAX_PATH_LEN = 200


class SiteTooLarge(ValueError):
    """A site/bundle exceeded the given byte ceiling. Subclasses ``ValueError`` so
    every existing ``except ValueError`` handler still catches it, while letting the
    deploy path detect *specifically* an oversize bundle and route it through the
    pooled blob store instead of rejecting it outright."""

# A hosted **dynamic** app is configured the 12-factor way, through environment
# variables (DATABASE_URL, API keys, feature flags, ...). User-supplied env is part
# of the same hosting trust boundary as the bundle, so validate it here: names must
# be shell-safe identifiers and the whole set is capped so a deployment frame can't
# be abused to ship an unbounded blob. Values may be secrets, so they only ride the
# (LAN or encrypted-tunnel) start_replica frame and are applied on a trusted node.
MAX_ENV_VARS = int(os.environ.get("MESHPOOL_MAX_ENV_VARS", "200"))
MAX_ENV_BYTES = int(os.environ.get("MESHPOOL_MAX_ENV_BYTES", str(64 * 1024)))  # 64 KB
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def clean_env(env: Optional[dict]) -> dict[str, str]:
    """Validate and normalise a user-supplied environment map for a hosted app.

    Returns a plain ``{str: str}`` dict; raises ``ValueError`` if a key is not a valid
    shell identifier or the set exceeds the count/byte caps. Reserved platform vars
    (``PORT`` and the ``MESHPOOL_*`` injected at launch) always win over user values,
    so setting them here is harmless -- the launcher re-applies them last.
    """
    if not env:
        return {}
    if len(env) > MAX_ENV_VARS:
        raise ValueError(f"too many env vars ({len(env)} > {MAX_ENV_VARS})")
    out: dict[str, str] = {}
    total = 0
    for k, v in env.items():
        key = str(k)
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"invalid env var name: {key!r}")
        val = "" if v is None else str(v)
        total += len(key) + len(val)
        if total > MAX_ENV_BYTES:
            raise ValueError(f"env exceeds the {MAX_ENV_BYTES}-byte limit")
        out[key] = val
    return out

# Serving an asset with the wrong Content-Type breaks real apps in the browser
# (ES modules and stylesheets are rejected under strict MIME checking). The stdlib
# mimetypes registry is OS-dependent -- notably on Windows ``.js`` can resolve to
# ``text/plain`` -- so pin the web-critical types explicitly at import time.
for _ext, _ct in {
    ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
    ".json": "application/json", ".map": "application/json", ".xml": "application/xml",
    ".svg": "image/svg+xml", ".webp": "image/webp", ".avif": "image/avif",
    ".ico": "image/x-icon", ".png": "image/png", ".wasm": "application/wasm",
    ".webmanifest": "application/manifest+json",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".otf": "font/otf",
}.items():
    mimetypes.add_type(_ct, _ext)

_CHARSET_TYPES = {
    "text/html", "text/css", "text/plain", "text/javascript",
    "application/javascript", "application/json", "image/svg+xml",
    "application/manifest+json", "application/xml", "text/xml", "text/markdown",
}


def _clean_relpath(raw: str) -> str:
    """Return a normalized forward-slash relative path, or raise ``ValueError``."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("site path must be a non-empty string")
    if len(raw) > MAX_PATH_LEN:
        raise ValueError(f"site path too long: {raw[:40]!r}...")
    p = raw.replace("\\", "/").strip()
    if p.startswith("/") or p.startswith("~"):
        raise ValueError(f"site path must be relative: {raw!r}")
    if ":" in p:                                  # drive letters / url schemes
        raise ValueError(f"illegal character in site path: {raw!r}")
    parts = [seg for seg in p.split("/") if seg not in ("", ".")]
    if any(seg == ".." for seg in parts):
        raise ValueError(f"path traversal is not allowed: {raw!r}")
    if not parts:
        raise ValueError(f"empty site path: {raw!r}")
    return "/".join(parts)


def normalize_site(site: dict, require_index: bool = True,
                   max_bytes: Optional[int] = None) -> dict[str, str]:
    """Validate a ``{path: base64}`` site map (path-safety, count, size, base64).

    Returns a new map with cleaned keys and the original base64 values. Raises
    ``ValueError`` on any violation, so a bad upload is rejected before it is ever
    written to disk or shipped to a donor machine. ``require_index=False`` skips the
    ``index.html`` requirement -- used for dynamic *process* apps, whose bundle is a
    server (e.g. ``server.py``) rather than a static site with an entry page.
    ``max_bytes`` is the total-size ceiling (defaults to ``MAX_TOTAL_BYTES``); the
    deploy path raises it to ``MAX_BUNDLE_BYTES`` for pooled bundles (an oversize
    site raises ``SiteTooLarge``).
    """
    if max_bytes is None:
        max_bytes = MAX_TOTAL_BYTES
    if not isinstance(site, dict) or not site:
        raise ValueError("site must be a non-empty object of path -> base64")
    if len(site) > MAX_FILES:
        raise ValueError(f"too many files ({len(site)} > {MAX_FILES})")
    out: dict[str, str] = {}
    total = 0
    for raw_path, b64 in site.items():
        rel = _clean_relpath(raw_path)
        if not isinstance(b64, str):
            raise ValueError(f"file {raw_path!r} content must be base64 text")
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError(f"file {raw_path!r} is not valid base64")
        total += len(data)
        if total > max_bytes:
            raise SiteTooLarge(f"site exceeds the {max_bytes}-byte limit")
        out[rel] = b64
    if require_index and "index.html" not in out:
        raise ValueError("site must contain an 'index.html' at its root")
    return out


def safe_join(root: str, rel: str) -> Optional[str]:
    """Join ``rel`` under ``root``; return the absolute path only if it stays
    inside ``root`` (defense-in-depth at both write and serve time). ``None`` on
    any escape."""
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, rel))
    if target == root_abs or target.startswith(root_abs + os.sep):
        return target
    return None


def write_site(dest_dir: str, site: dict, require_index: bool = True,
               max_bytes: Optional[int] = None) -> dict[str, str]:
    """Validate then decode+write a site under ``dest_dir`` (created if needed).

    Returns the normalized map. Raises ``ValueError`` for an invalid/unsafe site."""
    clean = normalize_site(site, require_index=require_index, max_bytes=max_bytes)
    os.makedirs(dest_dir, exist_ok=True)
    for rel, b64 in clean.items():
        target = safe_join(dest_dir, rel)
        if target is None:
            raise ValueError(f"unsafe site path: {rel!r}")
        os.makedirs(os.path.dirname(target) or dest_dir, exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(base64.b64decode(b64, validate=True))
    return clean


def content_type(path: str) -> str:
    ctype, _ = mimetypes.guess_type(path)
    ctype = ctype or "application/octet-stream"
    if ctype in _CHARSET_TYPES:
        ctype += "; charset=utf-8"
    return ctype


def resolve_request(site_dir: str, url_path: str) -> Optional[tuple[bytes, str]]:
    """Resolve an HTTP path to ``(bytes, content_type)`` from a site dir, or
    ``None`` if it does not map to a file strictly inside the dir.

    Strips the query/fragment, maps a trailing ``/`` (or the root) to
    ``index.html``, and refuses to serve anything outside ``site_dir``.
    """
    path = url_path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
    if path in ("", "."):
        path = "index.html"
    target = safe_join(site_dir, path)
    if target is None:
        return None
    if os.path.isdir(target):
        target = os.path.join(target, "index.html")
    if not os.path.isfile(target):
        return None
    with open(target, "rb") as fh:
        data = fh.read()
    return data, content_type(target)


# --- packing local content into a validated site map ----------------------- #
_SKIP = {".git", ".svn", ".hg", "node_modules", "__pycache__", ".DS_Store", "Thumbs.db"}


def pack_dir(root: str, require_index: bool = True,
             max_bytes: Optional[int] = None) -> dict[str, str]:
    """Walk a local directory into a validated ``{relpath: base64}`` site map.

    Skips VCS/build junk (``.git``, ``node_modules``, ...) and enforces the same
    count/size/path rules as an uploaded site (via :func:`normalize_site`), failing
    fast on the size cap so a huge tree is never fully read into memory. This is how
    ``meshpool deploy ./mysite`` turns a real multi-page app folder into something
    the hosting engine can ship to every replica.
    """
    if max_bytes is None:
        max_bytes = MAX_TOTAL_BYTES
    root_abs = os.path.abspath(root)
    if not os.path.isdir(root_abs):
        raise ValueError(f"not a directory: {root!r}")
    raw: dict[str, str] = {}
    total = 0
    for dirpath, dirnames, filenames in os.walk(root_abs):
        dirnames[:] = [d for d in dirnames if d not in _SKIP]
        for fn in filenames:
            if fn in _SKIP:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root_abs).replace(os.sep, "/")
            try:
                with open(full, "rb") as fh:
                    blob = fh.read()
            except OSError as exc:
                raise ValueError(f"cannot read {rel!r}: {exc}")
            total += len(blob)
            if total > max_bytes:
                raise SiteTooLarge(f"site exceeds the {max_bytes}-byte limit")
            raw[rel] = base64.b64encode(blob).decode("ascii")
    if not raw:
        raise ValueError(f"no files found under {root!r}")
    return normalize_site(raw, require_index=require_index, max_bytes=max_bytes)


def pack_zip(data: bytes, require_index: bool = True,
             max_bytes: Optional[int] = None) -> dict[str, str]:
    """Turn a ``.zip`` archive's bytes into a validated ``{relpath: base64}`` site
    map. If every entry sits under a single common top-level folder (i.e. you
    zipped ``mysite/`` rather than its contents), that folder is stripped so the
    site's ``index.html`` lands at the root. Path-safety (zip-slip) is enforced by
    :func:`normalize_site`."""
    if max_bytes is None:
        max_bytes = MAX_TOTAL_BYTES
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid zip archive: {exc}")
    raw: dict[str, str] = {}
    total = 0
    with zf:
        infos = [zi for zi in zf.infolist() if not zi.is_dir()]
        names = [zi.filename for zi in infos]
        tops = {n.split("/", 1)[0] for n in names}
        strip = next(iter(tops)) + "/" if names and len(tops) == 1 and all("/" in n for n in names) else ""
        for zi in infos:
            name = zi.filename[len(strip):] if strip and zi.filename.startswith(strip) else zi.filename
            if not name:
                continue
            total += zi.file_size
            if total > max_bytes:
                raise SiteTooLarge(f"site exceeds the {max_bytes}-byte limit")
            raw[name] = base64.b64encode(zf.read(zi)).decode("ascii")
    if not raw:
        raise ValueError("zip archive contains no files")
    return normalize_site(raw, require_index=require_index, max_bytes=max_bytes)


def detect_start_cmd_from_bundle(site: dict) -> str:
    """Derive a dynamic app's launch command from an already-packed ``{path: base64}``
    bundle: a ``Procfile`` ``web:`` line or a ``meshpool.json`` ``start`` field. Returns
    "" for a plain static site. This is the server-side equivalent of the CLI's
    folder-based detection, used when an app is uploaded through the dashboard so a
    browser user can host a real server app (not just static files)."""
    proc = site.get("Procfile")
    if isinstance(proc, str):
        try:
            text = base64.b64decode(proc, validate=True).decode("utf-8", "ignore")
        except (binascii.Error, ValueError):
            text = ""
        for line in text.splitlines():
            key, _, val = line.partition(":")
            if key.strip().lower() == "web" and val.strip():
                return val.strip()
    manifest = site.get("meshpool.json")
    if isinstance(manifest, str):
        import json
        try:
            data = base64.b64decode(manifest, validate=True).decode("utf-8", "ignore")
            start = json.loads(data).get("start")
        except (binascii.Error, ValueError):
            start = None
        if isinstance(start, str) and start.strip():
            return start.strip()
    return ""
