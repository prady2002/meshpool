"""Donation policy, idle detection, live load sampling and the kill-switch (spec §4.1).

The owner sets a cap + schedule; the agent measures real system load and idle time
so the control plane's headroom math and schedule gating are based on measured
values, not guesses. Enforcement of the CPU cap itself lives in the runtime
(worker-slot limiting); this module decides *whether* work is allowed right now.
"""
from __future__ import annotations

import ctypes
import platform
import time
from typing import Optional

from ..common.logging import get_logger
from ..common.models import DonationPolicy, NodeLoad
from .probe import _avail_ram_mb, _total_ram_mb

log = get_logger("donation")


class _WinCpuSampler:
    def __init__(self) -> None:
        self._prev: Optional[tuple[int, int, int]] = None

    @staticmethod
    def _filetime_to_int(ft) -> int:
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    def sample(self) -> float:
        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]
        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        i = self._filetime_to_int(idle)
        k = self._filetime_to_int(kernel)
        u = self._filetime_to_int(user)
        if self._prev is None:
            self._prev = (i, k, u)
            return 0.0
        di, dk, du = i - self._prev[0], k - self._prev[1], u - self._prev[2]
        self._prev = (i, k, u)
        total = dk + du  # kernel already includes idle on Windows
        if total <= 0:
            return 0.0
        return round(max(0.0, min(100.0, (1.0 - di / total) * 100.0)), 1)


class _LinuxCpuSampler:
    def __init__(self) -> None:
        self._prev: Optional[tuple[int, int]] = None

    def sample(self) -> float:
        try:
            with open("/proc/stat", "r") as fh:
                parts = fh.readline().split()[1:]
            vals = list(map(int, parts))
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            if self._prev is None:
                self._prev = (idle, total)
                return 0.0
            di, dt = idle - self._prev[0], total - self._prev[1]
            self._prev = (idle, total)
            return round(max(0.0, min(100.0, (1.0 - di / dt) * 100.0)), 1) if dt > 0 else 0.0
        except Exception:  # noqa: BLE001
            return 0.0


def _make_cpu_sampler():
    if platform.system() == "Windows":
        return _WinCpuSampler()
    if platform.system() == "Linux":
        return _LinuxCpuSampler()
    return None  # macOS: best-effort, reported as 0 (documented limitation)


def _idle_seconds() -> float:
    """Seconds since the last local user input (keyboard/mouse)."""
    try:
        if platform.system() == "Windows":
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]
            info = LASTINPUTINFO()
            info.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
                millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
                return round(max(0.0, millis / 1000.0), 1)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


class DonationController:
    def __init__(self, policy: DonationPolicy, sim_idle_seconds: Optional[float] = None) -> None:
        self.policy = policy
        self._sampler = _make_cpu_sampler()
        self._sim_idle = sim_idle_seconds        # harness can pin idle time
        self._reclaimed = False
        self._total_ram = _total_ram_mb()

    def set_policy(self, policy: DonationPolicy) -> None:
        self.policy = policy
        log.info("donation policy updated", cpu_pct=policy.cpu_pct,
                 ram_mb=policy.ram_mb, schedule=policy.schedule.value)

    def reclaim(self) -> None:
        self._reclaimed = True

    def resume(self) -> None:
        self._reclaimed = False

    @property
    def reclaimed(self) -> bool:
        return self._reclaimed

    def idle_seconds(self) -> float:
        return self._sim_idle if self._sim_idle is not None else _idle_seconds()

    def current_load(self, running_units: int) -> NodeLoad:
        cpu = self._sampler.sample() if self._sampler else 0.0
        ram_used = max(0, self._total_ram - _avail_ram_mb())
        return NodeLoad(cpu_load_pct=cpu, ram_used_mb=ram_used,
                        running_units=running_units, idle_seconds=self.idle_seconds())
