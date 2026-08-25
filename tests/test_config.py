"""Config: scale 1->3 is a config change, not a code change (spec §1, §13.5)."""
from __future__ import annotations

from meshpool.common.config import SCALE_PRESETS, Settings


def test_home_preset_is_tier0_no_enroll():
    s = Settings.load(scale="home")
    assert s.trust_tier == 0
    assert s.require_enroll_token is False
    assert s.ha is False


def test_office_preset_enables_tier1_and_enroll():
    s = Settings.load(scale="office")
    assert s.trust_tier == 1
    assert s.require_enroll_token is True
    assert s.benchmark_probes is True


def test_enterprise_preset_enables_ha():
    s = Settings.load(scale="enterprise")
    assert s.ha is True
    assert s.trust_tier == 1


def test_all_presets_have_consistent_keys():
    keys = [set(v.keys()) for v in SCALE_PRESETS.values()]
    assert all(k == keys[0] for k in keys)  # same knobs across every scale
