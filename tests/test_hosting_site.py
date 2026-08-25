"""Tests for the static-site hosting trust boundary (meshpool.hosting.site).

These lock down the "host your own website" feature's safety-critical helpers:
validation (path traversal, size/count caps, base64, index.html requirement),
safe file writing, and request resolution that never escapes the site dir.
"""
from __future__ import annotations

import base64

import pytest

from meshpool.hosting import site


def b64(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode()
    return base64.b64encode(s).decode()


# --------------------------- normalize_site -------------------------------- #
def test_normalize_accepts_a_valid_site():
    src = {"index.html": b64("<h1>hi</h1>"), "assets/app.css": b64("body{}")}
    out = site.normalize_site(src)
    assert out == src


def test_normalize_requires_index_html_at_root():
    with pytest.raises(ValueError, match="index.html"):
        site.normalize_site({"pages/home.html": b64("<h1>hi</h1>")})


def test_normalize_normalizes_backslashes_and_dot_segments():
    out = site.normalize_site({"index.html": b64("x"), ".\\assets\\a.js": b64("y")})
    assert "assets/a.js" in out


@pytest.mark.parametrize("bad", [
    "../secret.txt",
    "a/../../etc/passwd",
    "/etc/passwd",
    "~/.ssh/id_rsa",
    "C:\\Windows\\win.ini",
    "..\\..\\x",
])
def test_normalize_rejects_path_traversal(bad):
    with pytest.raises(ValueError):
        site.normalize_site({"index.html": b64("ok"), bad: b64("evil")})


def test_normalize_rejects_non_base64_content():
    with pytest.raises(ValueError, match="base64"):
        site.normalize_site({"index.html": "not*base64*!!"})


def test_normalize_enforces_file_count():
    many = {f"f{i}.txt": b64("x") for i in range(site.MAX_FILES + 1)}
    many["index.html"] = b64("x")
    with pytest.raises(ValueError, match="too many files"):
        site.normalize_site(many)


def test_normalize_enforces_total_size():
    big = b64("A" * (site.MAX_TOTAL_BYTES + 10))
    with pytest.raises(ValueError, match="exceeds"):
        site.normalize_site({"index.html": big})


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        site.normalize_site({})


# ------------------------------ safe_join ---------------------------------- #
def test_safe_join_allows_inside(tmp_path):
    got = site.safe_join(str(tmp_path), "a/b.txt")
    assert got is not None and got.startswith(str(tmp_path))


def test_safe_join_blocks_escape(tmp_path):
    assert site.safe_join(str(tmp_path), "../../etc/passwd") is None


# ------------------------------ write_site --------------------------------- #
def test_write_site_writes_files_and_subdirs(tmp_path):
    dest = tmp_path / "site"
    site.write_site(str(dest), {
        "index.html": b64("<h1>root</h1>"),
        "assets/style.css": b64("body{color:red}"),
    })
    assert (dest / "index.html").read_text() == "<h1>root</h1>"
    assert (dest / "assets" / "style.css").read_text() == "body{color:red}"


def test_write_site_rejects_invalid(tmp_path):
    with pytest.raises(ValueError):
        site.write_site(str(tmp_path / "s"), {"index.html": b64("x"), "../evil": b64("y")})


# --------------------------- resolve_request ------------------------------- #
def test_resolve_serves_index_for_root(tmp_path):
    site.write_site(str(tmp_path), {"index.html": b64("<h1>home</h1>")})
    for path in ("/", "", "/index.html", "/?x=1", "/#frag"):
        hit = site.resolve_request(str(tmp_path), path)
        assert hit is not None, path
        assert b"home" in hit[0]
        assert hit[1].startswith("text/html")


def test_resolve_serves_nested_asset_with_type(tmp_path):
    site.write_site(str(tmp_path), {"index.html": b64("x"), "a/app.js": b64("console.log(1)")})
    hit = site.resolve_request(str(tmp_path), "/a/app.js")
    assert hit is not None
    assert b"console.log" in hit[0]
    assert "javascript" in hit[1]


def test_resolve_404_for_missing(tmp_path):
    site.write_site(str(tmp_path), {"index.html": b64("x")})
    assert site.resolve_request(str(tmp_path), "/nope.html") is None


def test_resolve_blocks_traversal(tmp_path):
    (tmp_path / "secret.txt").write_text("TOPSECRET")
    docroot = tmp_path / "site"
    site.write_site(str(docroot), {"index.html": b64("x")})
    assert site.resolve_request(str(docroot), "/../secret.txt") is None
    assert site.resolve_request(str(docroot), "/..%2fsecret.txt") is None


# ------------------------------ pack_dir ----------------------------------- #
def _write_tree(root):
    (root / "assets" / "js").mkdir(parents=True)
    (root / "sub").mkdir()
    (root / "index.html").write_text("<h1>home</h1>")
    (root / "sub" / "index.html").write_text("<h1>sub</h1>")
    (root / "assets" / "js" / "app.js").write_text("export const x=1;")
    (root / "favicon.png").write_bytes(b"\x89PNG\r\n\x1a\n")


def test_pack_dir_walks_tree_with_forward_slash_keys(tmp_path):
    _write_tree(tmp_path)
    out = site.pack_dir(str(tmp_path))
    assert set(out) == {"index.html", "sub/index.html", "assets/js/app.js", "favicon.png"}
    # values are valid base64 that round-trips to the original bytes
    assert base64.b64decode(out["favicon.png"]) == b"\x89PNG\r\n\x1a\n"
    assert base64.b64decode(out["assets/js/app.js"]).decode() == "export const x=1;"


def test_pack_dir_skips_junk_dirs(tmp_path):
    _write_tree(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "left-pad.js").write_text("module.exports=1;")
    out = site.pack_dir(str(tmp_path))
    assert not any(k.startswith((".git/", "node_modules/")) for k in out)


def test_pack_dir_requires_index_html(tmp_path):
    (tmp_path / "page.html").write_text("<h1>no index</h1>")
    with pytest.raises(ValueError, match="index.html"):
        site.pack_dir(str(tmp_path))


def test_pack_dir_rejects_non_directory(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        site.pack_dir(str(tmp_path / "does-not-exist"))


def test_pack_dir_enforces_size_cap(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    (tmp_path / "big.bin").write_bytes(b"x" * 5000)
    monkeypatch.setattr(site, "MAX_TOTAL_BYTES", 1000)
    with pytest.raises(ValueError, match="limit"):
        site.pack_dir(str(tmp_path))


# ------------------------------ pack_zip ----------------------------------- #
def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_pack_zip_reads_entries(tmp_path):
    data = _zip_bytes({
        "index.html": b"<h1>home</h1>",
        "assets/app.js": b"export const x=1;",
    })
    out = site.pack_zip(data)
    assert set(out) == {"index.html", "assets/app.js"}
    assert base64.b64decode(out["assets/app.js"]).decode() == "export const x=1;"


def test_pack_zip_strips_single_top_level_folder(tmp_path):
    data = _zip_bytes({
        "mysite/index.html": b"<h1>home</h1>",
        "mysite/assets/app.js": b"1;",
    })
    out = site.pack_zip(data)
    assert set(out) == {"index.html", "assets/app.js"}


def test_pack_zip_rejects_zip_slip(tmp_path):
    data = _zip_bytes({
        "index.html": b"<h1>home</h1>",
        "../evil.txt": b"pwn",
    })
    with pytest.raises(ValueError):
        site.pack_zip(data)


def test_pack_zip_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        site.pack_zip(b"not a zip file at all")


# ------------------- detect_start_cmd_from_bundle -------------------------- #
def test_detect_start_cmd_from_bundle_reads_procfile_web_line():
    bundle = site.pack_zip(_zip_bytes({
        "server.py": b"print('hi')",
        "Procfile": b"web: python -u server.py\n",
    }), require_index=False)
    assert site.detect_start_cmd_from_bundle(bundle) == "python -u server.py"


def test_detect_start_cmd_from_bundle_reads_manifest_start():
    bundle = site.pack_zip(_zip_bytes({
        "server.js": b"//",
        "meshpool.json": b'{"start": "node server.js"}',
    }), require_index=False)
    assert site.detect_start_cmd_from_bundle(bundle) == "node server.js"


def test_detect_start_cmd_from_bundle_procfile_wins_over_manifest():
    bundle = site.pack_zip(_zip_bytes({
        "server.py": b"#",
        "Procfile": b"web: python server.py",
        "meshpool.json": b'{"start": "node server.js"}',
    }), require_index=False)
    assert site.detect_start_cmd_from_bundle(bundle) == "python server.py"


def test_detect_start_cmd_from_bundle_static_site_returns_empty():
    bundle = site.pack_zip(_zip_bytes({"index.html": b"<h1>home</h1>"}))
    assert site.detect_start_cmd_from_bundle(bundle) == ""


def test_detect_start_cmd_from_bundle_ignores_undecodable_procfile():
    # A corrupt (non-base64) Procfile value must not raise — treat as "no command".
    assert site.detect_start_cmd_from_bundle({"Procfile": "!!!not base64!!!"}) == ""


# ----------------- SiteTooLarge: the large-app pooled path ----------------- #
# An app over the inline ceiling is NOT rejected outright: normalize/pack raise
# SiteTooLarge (a ValueError subclass) so the deploy path can catch it and stage
# the bundle in the pooled blob store, then retry under the higher MAX_BUNDLE_BYTES.
def test_site_too_large_is_a_value_error():
    assert issubclass(site.SiteTooLarge, ValueError)


def test_normalize_raises_site_too_large_over_cap(monkeypatch):
    monkeypatch.setattr(site, "MAX_TOTAL_BYTES", 100)
    big = b64("A" * 500)
    with pytest.raises(site.SiteTooLarge, match="limit"):
        site.normalize_site({"index.html": big})


def test_normalize_accepts_large_site_under_higher_cap(monkeypatch):
    # Too big for the inline cap, but fits the pooled ceiling passed explicitly.
    monkeypatch.setattr(site, "MAX_TOTAL_BYTES", 100)
    big = b64("A" * 500)
    out = site.normalize_site({"index.html": big}, max_bytes=site.MAX_BUNDLE_BYTES)
    assert out == {"index.html": big}


def test_pack_dir_raises_site_too_large_but_passes_under_higher_cap(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    (tmp_path / "big.bin").write_bytes(b"x" * 5000)
    monkeypatch.setattr(site, "MAX_TOTAL_BYTES", 1000)
    with pytest.raises(site.SiteTooLarge):
        site.pack_dir(str(tmp_path))
    out = site.pack_dir(str(tmp_path), max_bytes=site.MAX_BUNDLE_BYTES)
    assert "big.bin" in out and "index.html" in out


def test_pack_zip_raises_site_too_large_but_passes_under_higher_cap(monkeypatch):
    data = _zip_bytes({"index.html": b"<h1>home</h1>", "big.bin": b"y" * 5000})
    monkeypatch.setattr(site, "MAX_TOTAL_BYTES", 1000)
    with pytest.raises(site.SiteTooLarge):
        site.pack_zip(data)
    out = site.pack_zip(data, max_bytes=site.MAX_BUNDLE_BYTES)
    assert "big.bin" in out and "index.html" in out


def test_write_site_accepts_large_bundle_under_higher_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "MAX_TOTAL_BYTES", 100)
    big = b64("A" * 500)
    dest = tmp_path / "s"
    with pytest.raises(site.SiteTooLarge):
        site.write_site(str(dest), {"index.html": big})
    site.write_site(str(dest), {"index.html": big}, max_bytes=site.MAX_BUNDLE_BYTES)
    assert (dest / "index.html").read_bytes() == b"A" * 500


