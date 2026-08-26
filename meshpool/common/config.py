"""Configuration — scale 1->3 is a config change, not a code change (spec §1).

A single ``Settings`` object is loaded from (in priority order) explicit kwargs,
environment variables (``MESHPOOL_*``), an optional YAML file, then defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from .netutil import lan_ip


# Scale presets. The same binary behaves differently purely via these knobs.
SCALE_PRESETS: dict[str, dict[str, Any]] = {
    "home": {           # 2-5 machines, single owner, LAN, Tier 0
        "trust_tier": 0,
        "require_enroll_token": False,
        "node_timeout_s": 12,
        "benchmark_probes": False,
        "ha": False,
    },
    "office": {         # 5-50 machines, admin-enrolled, Tier 1
        "trust_tier": 1,
        "require_enroll_token": True,
        "node_timeout_s": 20,
        "benchmark_probes": True,
        "ha": False,
    },
    "enterprise": {     # 50+ machines, multi-site, Tier 1 + HA
        "trust_tier": 1,
        "require_enroll_token": True,
        "node_timeout_s": 30,
        "benchmark_probes": True,
        "ha": True,
    },
    "public": {         # global mesh: anyone on the internet, adversarial, Tier 2
        # No admin enroll token — membership is by cryptographic device identity
        # (see common/identity.py), not a shared secret. Reputation + credits +
        # verification (spec §7) are ACTIVE here, so strangers can donate/consume
        # safely. This is the posture the hosted "MeshPool Cloud" coordinator runs.
        "trust_tier": 2,
        "require_enroll_token": False,
        "node_timeout_s": 20,
        "benchmark_probes": True,
        "ha": False,
    },
}


def _default_data_dir() -> str:
    base = os.environ.get("MESHPOOL_DATA_DIR")
    if base:
        return base
    return str(Path.home() / ".meshpool")


@dataclass
class Settings:
    scale: str = "home"
    control_plane_url: str = "http://127.0.0.1:8080"
    host: str = "0.0.0.0"            # bind address (all interfaces → LAN reachable)
    advertise_host: str = ""         # address other machines use; "" → auto-detect LAN IP
    port: int = 8080
    data_dir: str = field(default_factory=_default_data_dir)
    heartbeat_interval_s: int = 3
    node_timeout_s: int = 12
    trust_tier: int = 0
    require_enroll_token: bool = False
    enroll_token: str = ""
    benchmark_probes: bool = False
    ha: bool = False
    owner: str = "default"
    # Authentication mode.
    # ``open``     — existing zero-friction LAN behaviour preserved; callers are
    #                implicitly granted an anonymous full-access Principal. This is
    #                the default so home/office installs need no configuration change.
    # ``enforced`` — every non-public endpoint requires a valid capability token,
    #                session token, or node token. Suitable for public/cloud deployments.
    auth_mode: str = "open"
    # Transport fabric behind AgentTransport. "http" keeps the original LAN/direct
    # path only, "tunnel" uses the HTTPS reverse tunnel, and "quic"/"auto" also
    # accepts agent-initiated QUIC sessions with transparent tunnel fallback.
    transport: str = "tunnel"
    quic_port: int = 0

    # --- construction ---------------------------------------------------- #
    @classmethod
    def load(cls, path: Optional[str] = None, **overrides: Any) -> "Settings":
        data: dict[str, Any] = {}
        cfg_path = path or os.environ.get("MESHPOOL_CONFIG")
        if cfg_path and Path(cfg_path).exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data.update(yaml.safe_load(fh) or {})

        # environment overrides (MESHPOOL_PORT, MESHPOOL_SCALE, ...)
        for f in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            env = os.environ.get(f"MESHPOOL_{f.upper()}")
            if env is not None:
                data[f] = env

        data.update({k: v for k, v in overrides.items() if v is not None})

        # Did the operator pin the advertised control-plane URL explicitly? If not,
        # we derive it from the detected LAN IP below so the onboarding/join command
        # points other machines at a reachable address instead of loopback.
        explicit_cp_url = "control_plane_url" in data

        settings = cls(**{k: v for k, v in data.items()
                          if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]
        settings._coerce()
        if not settings.advertise_host:
            settings.advertise_host = lan_ip()
        if not explicit_cp_url:
            settings.control_plane_url = f"http://{settings.advertise_host}:{settings.port}"
        if settings.quic_port <= 0:
            settings.quic_port = settings.port + 100
        settings._apply_scale_preset()
        return settings

    def _coerce(self) -> None:
        self.port = int(self.port)
        self.heartbeat_interval_s = int(self.heartbeat_interval_s)
        self.node_timeout_s = int(self.node_timeout_s)
        self.trust_tier = int(self.trust_tier)
        self.quic_port = int(self.quic_port)
        self.ha = str(self.ha).lower() in ("1", "true", "yes") if not isinstance(self.ha, bool) else self.ha
        self.require_enroll_token = (
            str(self.require_enroll_token).lower() in ("1", "true", "yes")
            if not isinstance(self.require_enroll_token, bool) else self.require_enroll_token
        )
        self.benchmark_probes = (
            str(self.benchmark_probes).lower() in ("1", "true", "yes")
            if not isinstance(self.benchmark_probes, bool) else self.benchmark_probes
        )

    def _apply_scale_preset(self) -> None:
        preset = SCALE_PRESETS.get(self.scale)
        if not preset:
            return
        # Preset provides defaults; only fill values the user didn't explicitly set
        # via env/yaml. For MVP simplicity we apply the preset wholesale except for
        # fields commonly overridden (port/host/url/data_dir/enroll_token/owner).
        for k, v in preset.items():
            setattr(self, k, v)

    def db_path(self) -> str:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.data_dir) / "meshpool.db")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
