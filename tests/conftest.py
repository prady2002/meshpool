"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def tmp_store(tmp_path):
    from meshpool.common.db import Store
    s = Store(str(tmp_path / "test.db"))
    yield s
    s.close()
