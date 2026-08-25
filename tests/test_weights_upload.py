"""Streaming "bring your own weights" upload: the control plane assembles a model from
per-layer group blobs streamed one at a time, so a checkpoint too big to buffer whole
still uploads. These in-process tests lock the register endpoint's trust boundary -- a
client cannot register a mismatched or partial manifest. The full stream -> deploy ->
serve path (with live nodes) is covered by ``sim.scenarios.verify_pooled_model``.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from meshpool.common.config import Settings
from meshpool.controlplane.app import build_app
from meshpool.inference import model as M


def _home_app(tmp_path):
    settings = Settings.load(scale="home", data_dir=str(tmp_path),
                             host="127.0.0.1", port=8097)
    return build_app(settings)


def test_register_rejects_unknown_model(tmp_path):
    with TestClient(_home_app(tmp_path)) as client:
        r = client.post("/api/v1/models/weights/register?model=does-not-exist",
                        json={"groups": {"wte": "00"}, "meta": None})
        assert r.status_code == 404, r.text


def test_register_rejects_empty_groups(tmp_path):
    with TestClient(_home_app(tmp_path)) as client:
        r = client.post("/api/v1/models/weights/register?model=nano-gpt",
                        json={"groups": {}, "meta": None})
        assert r.status_code == 400, r.text


def test_register_rejects_incomplete_manifest(tmp_path):
    """A manifest that names only some of the required layer groups is refused before a
    single blob is fetched -- so a partial upload can never deploy as a whole model."""
    cfg = M.REGISTRY["nano-gpt"]
    group_names = {name for name, _ in M.group_specs(cfg)}
    assert len(group_names) == cfg.n_layer + 3          # wte + wpe + lnf + one per layer
    with TestClient(_home_app(tmp_path)) as client:
        r = client.post("/api/v1/models/weights/register?model=nano-gpt",
                        json={"groups": {"wte": "00"}, "meta": None})
        assert r.status_code == 400, r.text
        assert "missing" in r.json()["detail"].lower()
