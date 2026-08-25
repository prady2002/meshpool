"""Hardware capability probe (spec §4.1).

Measures — never guesses — CPU, RAM, GPU, disk, a rough FLOPS score via a real
micro-benchmark, and link latency/bandwidth to the control plane. Uses only the
stdlib + numpy (psutil isn't available in this environment), degrading gracefully.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common.logging import get_logger
from ..common.models import Capability, LinkStats

log = get_logger("probe")


_UNKNOWN_INT = 0


@dataclass(frozen=True)
class HardwareProbe:
    """Internal hardware details that do not all fit in Capability yet.

    ``None`` means unknown / not measured.  Integer ``0`` is used only at the
    existing wire boundary for backwards compatibility with ``Capability``.
    """
    cpu_cores: int
    cpu_arch: str
    cpu_model: Optional[str]
    ram_mb: Optional[int]
    gpu: bool
    vram_mb: Optional[int]
    gpu_names: list[str] = field(default_factory=list)
    disk_free_mb: Optional[int] = None
    disk_total_mb: Optional[int] = None
    thermal_state: str = "unknown"
    battery_powered: Optional[bool] = None
    battery_percent: Optional[float] = None
    network_class: str = "unknown"


def _positive_int_or_none(value: object) -> Optional[int]:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _parse_windows_total_physical_memory(stdout: str) -> Optional[int]:
    """Parse Win32_ComputerSystem.TotalPhysicalMemory output (bytes -> MiB)."""
    for line in stdout.splitlines():
        n = _positive_int_or_none(line)
        if n:
            return n // (1024 * 1024)
    return None


def _parse_proc_meminfo(stdout: str) -> Optional[int]:
    for line in stdout.splitlines():
        if not line.startswith("MemTotal:"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            kb = _positive_int_or_none(parts[1])
            return None if kb is None else kb // 1024
    return None


def _parse_sysctl_memsize(stdout: str) -> Optional[int]:
    n = _positive_int_or_none(stdout)
    return None if n is None else n // (1024 * 1024)


def _parse_lscpu_model(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "model name":
            return value.strip() or None
    return None


def _detect_cpu_model() -> Optional[str]:
    sysname = platform.system()
    try:
        if sysname == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
                 "-ExpandProperty Name)"],
                capture_output=True, text=True, timeout=4,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        if sysname == "Linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                    parsed = _parse_lscpu_model(fh.read())
                if parsed:
                    return parsed
            except OSError:
                pass
            out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=4)
            if out.returncode == 0:
                return _parse_lscpu_model(out.stdout)
        if sysname == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=4,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or None


def _total_ram_mb() -> int:
    try:
        if platform.system() == "Windows":
            class _MemStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = _MemStatusEx()
            stat.dwLength = ctypes.sizeof(_MemStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            measured = int(stat.ullTotalPhys // (1024 * 1024))
            return measured if measured > 0 else _UNKNOWN_INT
        # POSIX
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        measured = int(pages * page_size // (1024 * 1024))
        return measured if measured > 0 else _UNKNOWN_INT
    except Exception:  # noqa: BLE001
        return _UNKNOWN_INT


def _avail_ram_mb() -> int:
    try:
        if platform.system() == "Windows":
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = _MS(); s.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return int(s.ullAvailPhys // (1024 * 1024))
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return _UNKNOWN_INT


_INTEGRATED_GPU_HINTS = (
    "intel", "uhd", "iris", "vega", "radeon graphics", "apple", "microsoft basic",
)


def _looks_integrated_gpu(name: str) -> bool:
    low = name.lower()
    return (
        any(h in low for h in _INTEGRATED_GPU_HINTS)
        or ("ryzen" in low and "radeon" in low)
        or re.search(r"radeon.*graphics", low) is not None
        or re.search(r"radeon.*\b[5678]\d{2}m\b", low) is not None
    )


def _parse_nvidia_smi_vram(stdout: str) -> tuple[bool, int]:
    values = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split(",", 1)[0].strip()
        n = _positive_int_or_none(first)
        if n:
            values.append(n)
    return (bool(values), max(values) if values else 0)


def _parse_rocm_smi_vram(stdout: str) -> tuple[bool, int]:
    values = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        for part in parts[1:]:
            n = _positive_int_or_none(part)
            if n:
                # rocm-smi CSV is commonly bytes for VRAM total.
                values.append(n // (1024 * 1024) if n > 1024 * 1024 else n)
                break
    return (bool(values), max(values) if values else 0)


def _parse_macos_displays_json(stdout: str) -> tuple[bool, int, list[str]]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return False, 0, []
    values: list[int] = []
    names: list[str] = []
    for d in data.get("SPDisplaysDataType", []) or []:
        name = str(d.get("sppci_model") or d.get("_name") or d.get("spdisplays_device_name") or "").strip()
        if name:
            names.append(name)
        vram_str = str(d.get("sppci_vram") or d.get("spdisplays_vram") or "").strip()
        if not vram_str:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)", vram_str, flags=re.I)
        if m:
            val = float(m.group(1))
            values.append(int(val * 1024) if m.group(2).upper() == "GB" else int(val))
    # Apple Silicon reports shared unified memory, not dedicated VRAM; return GPU
    # present with unknown VRAM rather than shrinking a large-RAM machine.
    if names and all(_looks_integrated_gpu(n) for n in names):
        return True, 0, names
    return bool(names or values), max(values) if values else 0, names


def _parse_windows_video_json(stdout: str) -> tuple[bool, int, list[str]]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return False, 0, []
    rows = data if isinstance(data, list) else [data]
    values: list[int] = []
    names: list[str] = []
    saw_gpu = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()
        if name:
            names.append(name)
            saw_gpu = True
        adapter_ram = _positive_int_or_none(row.get("AdapterRAM"))
        if adapter_ram is None:
            continue
        if name and _looks_integrated_gpu(name):
            saw_gpu = True
            continue
        values.append(adapter_ram // (1024 * 1024))
    return saw_gpu or bool(values), max(values) if values else 0, names


def _parse_linux_gpu_lspci(stdout: str) -> tuple[bool, list[str]]:
    names = []
    for line in stdout.splitlines():
        if re.search(r"\b(VGA|3D|Display)\b", line, flags=re.I):
            names.append(line.strip())
    return bool(names), names


def _read_first_float(paths: list[str]) -> Optional[float]:
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
            val = float(raw)
            # Linux thermal zones usually report millidegrees Celsius.
            if val > 1000:
                val /= 1000.0
            return val
        except Exception:  # noqa: BLE001
            continue
    return None


def _thermal_state_from_celsius(c: Optional[float]) -> str:
    if c is None:
        return "unknown"
    if c >= 90:
        return "critical"
    if c >= 80:
        return "hot"
    if c >= 65:
        return "warm"
    return "nominal"


def _detect_thermal_state() -> str:
    sysname = platform.system()
    if sysname == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance -Namespace root/wmi MSAcpi_ThermalZoneTemperature | "
                 "Select-Object -ExpandProperty CurrentTemperature"],
                capture_output=True, text=True, timeout=4,
            )
            if out.returncode == 0:
                vals = []
                for line in out.stdout.splitlines():
                    n = _positive_int_or_none(line)
                    if n:
                        vals.append((n / 10.0) - 273.15)
                if vals:
                    return _thermal_state_from_celsius(max(vals))
        except Exception:  # noqa: BLE001
            pass
    elif sysname == "Linux":
        try:
            temps = []
            for root, _, files in os.walk("/sys/class/thermal"):
                if "temp" in files:
                    t = _read_first_float([os.path.join(root, "temp")])
                    if t is not None:
                        temps.append(t)
            if temps:
                return _thermal_state_from_celsius(max(temps))
        except Exception:  # noqa: BLE001
            pass
    elif sysname == "Darwin":
        # macOS exposes this via powermetrics, which normally requires sudo.
        return "unknown"
    return "unknown"


def _detect_battery() -> tuple[Optional[bool], Optional[float]]:
    """Return (on_battery_power, percent) or (None, None) when unavailable."""
    sysname = platform.system()
    if sysname == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Battery | "
                 "Select-Object BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json"],
                capture_output=True, text=True, timeout=4,
            )
            if out.returncode == 0 and out.stdout.strip():
                data = json.loads(out.stdout)
                row = data[0] if isinstance(data, list) else data
                status = int(row.get("BatteryStatus", 0) or 0)
                pct = row.get("EstimatedChargeRemaining")
                # WMI: 1=discharging, 2=AC, 6=charging, 7=charging/high, etc.
                on_battery = status == 1
                return on_battery, float(pct) if pct is not None else None
        except Exception:  # noqa: BLE001
            pass
    elif sysname == "Linux":
        try:
            for name in os.listdir("/sys/class/power_supply"):
                base = os.path.join("/sys/class/power_supply", name)
                typ = open(os.path.join(base, "type"), encoding="utf-8").read().strip().lower()
                if typ != "battery":
                    continue
                status = open(os.path.join(base, "status"), encoding="utf-8").read().strip().lower()
                pct_raw = open(os.path.join(base, "capacity"), encoding="utf-8").read().strip()
                return status == "discharging", float(pct_raw)
        except Exception:  # noqa: BLE001
            pass
    elif sysname == "Darwin":
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=4)
            if out.returncode == 0:
                text = out.stdout
                pct_match = re.search(r"(\d+)%", text)
                pct = float(pct_match.group(1)) if pct_match else None
                return "'Battery Power'" in text, pct
        except Exception:  # noqa: BLE001
            pass
    return None, None


def _network_class_from_link(latency_ms: float, bandwidth_mbps: float) -> str:
    if latency_ms == 0.0 and bandwidth_mbps == 0.0:
        return "unknown"
    if latency_ms < 2.0 and (bandwidth_mbps == 0.0 or bandwidth_mbps >= 100.0):
        return "loopback_or_lan"
    if latency_ms < 20.0 and (bandwidth_mbps == 0.0 or bandwidth_mbps >= 25.0):
        return "lan_or_metro"
    if bandwidth_mbps and bandwidth_mbps < 5.0:
        return "constrained_wan"
    return "wan"


def _detect_gpu() -> tuple[bool, int, list[str]]:
    """Best-effort GPU + VRAM detection. Tries nvidia-smi, then AMD rocm-smi, then
    platform-specific fallbacks (Apple Metal via system_profiler, Windows WMI via
    Get-CimInstance). Returns (has_gpu, vram_mb, names); never raises, never guesses
    VRAM. Names are best-effort: not knowing the model must never make the GPU
    itself disappear, so an unnamed adapter still reports has_gpu=True."""

    # ── NVIDIA ──────────────────────────────────────────────────────────────
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode == 0 and out.stdout.strip():
            has_gpu, vram = _parse_nvidia_smi_vram(out.stdout)
            return has_gpu, vram, _nvidia_gpu_names()
    except Exception:  # noqa: BLE001
        pass

    # ── AMD ROCm ─────────────────────────────────────────────────────────────
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode == 0:
            has_gpu, vram = _parse_rocm_smi_vram(out.stdout)
            if has_gpu:
                return has_gpu, vram, []
    except Exception:  # noqa: BLE001
        pass

    # ── Apple Metal (macOS) ──────────────────────────────────────────────────
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                has_gpu, vram, names = _parse_macos_displays_json(out.stdout)
                if has_gpu:
                    return has_gpu, vram, names
        except Exception:  # noqa: BLE001
            pass

    # ── Windows WMI fallback (covers integrated + AMD on Windows) ──────────
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | "
                 "Select-Object Name,AdapterRAM | ConvertTo-Json"],
                capture_output=True, text=True, timeout=6,
            )
            if out.returncode == 0 and out.stdout.strip():
                has_gpu, vram, names = _parse_windows_video_json(out.stdout)
                if has_gpu:
                    return has_gpu, vram, names
        except Exception:  # noqa: BLE001
            pass

    return False, 0, []


def _nvidia_gpu_names() -> list[str]:
    """Adapter names from nvidia-smi. Best-effort by design: see _detect_gpu."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode == 0:
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        pass
    return []


def probe_hardware_detail(link: Optional[_LinkMeasurement] = None) -> HardwareProbe:
    """Measure detailed hardware locally; safe to call even when probes are absent."""
    gpu, vram, gpu_names = _detect_gpu()
    disk = shutil.disk_usage(os.getcwd())
    batt_power, batt_pct = _detect_battery()
    return HardwareProbe(
        cpu_cores=os.cpu_count() or 1,
        cpu_arch=platform.machine() or "unknown",
        cpu_model=_detect_cpu_model(),
        ram_mb=_total_ram_mb() or None,
        gpu=gpu,
        vram_mb=vram or None,
        gpu_names=gpu_names,
        disk_free_mb=int(disk.free // (1024 * 1024)),
        disk_total_mb=int(disk.total // (1024 * 1024)),
        thermal_state=_detect_thermal_state(),
        battery_powered=batt_power,
        battery_percent=batt_pct,
        network_class=(
            _network_class_from_link(link.latency_p50_ms, link.bandwidth_mbps or 0.0)
            if link is not None else "unknown"
        ),
    )


def _flops_estimate() -> float:
    """Rough compute score: sustained GFLOP/s on an adaptive matmul micro-benchmark.

    Grows the matrix until measured wall-time exceeds 100 ms (stable territory above
    timer noise and CPU turbo one-shot effects), then runs several trials and returns
    the best (peak sustained) score.  Total wall time stays well under 2 s.
    """
    try:
        # Find a matrix size where a single trial takes ≥ 100 ms
        n = 256
        while True:
            a = np.random.rand(n, n).astype(np.float32)
            b = np.random.rand(n, n).astype(np.float32)
            _ = a @ b  # warmup / JIT
            t0 = time.perf_counter()
            _ = a @ b
            if (time.perf_counter() - t0) >= 0.10 or n >= 2048:
                break
            n = min(int(n * 1.5), 2048)

        # Run 3 timed trials with the stabilised size, take the best throughput
        best_gflops = 0.0
        flops_per_call = 2.0 * n ** 3
        for _ in range(3):
            t0 = time.perf_counter()
            _ = a @ b
            dt = time.perf_counter() - t0
            if dt > 0:
                best_gflops = max(best_gflops, flops_per_call / dt / 1e9)
        return round(best_gflops, 2)
    except Exception:  # noqa: BLE001
        return float(os.cpu_count() or 1)


# ── Richer link measurement result (kept internal to probe.py) ────────────

@dataclass
class _LinkMeasurement:
    """Richer result than the wire-format LinkStats; callers that only need the
    Pydantic model call ``to_link_stats()``.  ``bandwidth_mbps`` is None when no
    bandwidth-probe endpoint was available — never fabricated."""
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_jitter_ms: float = 0.0   # standard deviation across samples
    first_connect_ms: float = 0.0    # time including connection establishment
    bandwidth_mbps: Optional[float] = None   # None = not measured / unknown
    upload_mbps: Optional[float] = None      # None = not measured / unknown
    measured: bool = False

    def to_link_stats(self) -> LinkStats:
        # Keep the wire model's existing fields; bandwidth_mbps=0.0 as the default
        # is ambiguous (could be "not measured"), but we cannot change models.py.
        # Callers that need to distinguish should use _LinkMeasurement directly.
        return LinkStats(
            latency_ms=self.latency_p50_ms,
            bandwidth_mbps=self.bandwidth_mbps if self.bandwidth_mbps is not None else 0.0,
        )


def _measure_latency(client, url: str, n: int = 7) -> tuple[float, float, float]:
    """Return (p50_ms, p90_ms, stdev_ms) from ``n`` back-to-back GETs.
    The first sample is excluded from the percentile calculation because it includes
    connection setup / TLS; it is measured separately and returned via the caller."""
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.get(url, timeout=5.0)
        r.raise_for_status()
        samples.append((time.perf_counter() - t0) * 1000.0)
    # sample[0] = first-connect; remainder = steady-state RTTs
    rtt = sorted(samples[1:])
    p50 = rtt[len(rtt) // 2]
    p90 = rtt[int(len(rtt) * 0.9)]
    stdev = statistics.stdev(rtt) if len(rtt) >= 2 else 0.0
    return p50, p90, stdev


def measure_link(control_plane_url: str) -> LinkStats:
    """Measure latency to the control plane.  Bandwidth is reported only when a
    dedicated ``/api/v1/probe/download`` endpoint exists on the control plane;
    if absent we report bandwidth_mbps=0.0 (unknown) rather than fabricating a
    number from tiny health-check payloads.

    Returning LinkStats(latency_ms=0.0, bandwidth_mbps=0.0) on total failure is
    honest for the latency side only when running the simulation harness on localhost
    (~0 ms is the real value there).  Callers that need to distinguish "failed
    measurement" from "0 ms localhost" should use ``_measure_link_detail`` directly.
    """
    return _measure_link_detail(control_plane_url).to_link_stats()


def _measure_link_detail(control_plane_url: str) -> _LinkMeasurement:
    """Full measurement returning the richer _LinkMeasurement dataclass."""
    try:
        import httpx

        base = control_plane_url.rstrip("/")
        health_url = f"{base}/api/v1/health"
        probe_dl_url = f"{base}/api/v1/probe/download"

        with httpx.Client() as client:
            # ── Latency: first-connect + steady-state RTTs ─────────────────
            t_connect = time.perf_counter()
            r = client.get(health_url, timeout=5.0)
            r.raise_for_status()
            first_connect_ms = (time.perf_counter() - t_connect) * 1000.0

            p50, p90, stdev = _measure_latency(client, health_url)

            # ── Bandwidth: use a dedicated probe endpoint if available ──────
            bw: Optional[float] = None
            up: Optional[float] = None
            try:
                # HEAD first — cheap way to confirm the endpoint exists
                head = client.head(probe_dl_url, timeout=2.0)
                if head.status_code == 200:
                    bw = _measure_download_bw(client, probe_dl_url)
                    # Uplink is measured separately and is the number that actually
                    # bounds a pipeline stage: it *sends* activations onward, and
                    # residential links are typically ~25x slower up than down.
                    up = _measure_upload_bw(client, f"{base}/api/v1/probe/upload")
            except Exception:  # noqa: BLE001
                # Endpoint absent or timed out — leave bw=None (honestly unknown)
                pass

        return _LinkMeasurement(
            latency_p50_ms=round(p50, 3),
            latency_p90_ms=round(p90, 3),
            latency_jitter_ms=round(stdev, 3),
            first_connect_ms=round(first_connect_ms, 3),
            bandwidth_mbps=round(bw, 2) if bw is not None else None,
            upload_mbps=round(up, 2) if up is not None else None,
            measured=True,
        )
    except Exception:  # noqa: BLE001
        # Return a clearly-unmeasured result; latency 0.0 is accurate for localhost
        # (the sim harness), so this is not a fabrication in that context.
        return _LinkMeasurement(measured=False)


def _measure_download_bw(client, url: str) -> float:
    """Ramped download bandwidth measurement.

    Starts with a small request and doubles the requested size until either the
    transfer takes ≥ 200 ms or the size cap (8 MB) is hit, then computes throughput
    excluding the warm-up transfer.  Returns Mbps.
    """
    size_bytes = 64 * 1024   # start at 64 KB
    cap_bytes = 8 * 1024 * 1024
    time_target = 0.2        # seconds; aim for a transfer long enough to be stable

    # Warm-up transfer (not measured)
    client.get(url, params={"size": size_bytes}, timeout=10.0)

    best_mbps = 0.0
    while size_bytes <= cap_bytes:
        t0 = time.perf_counter()
        r = client.get(url, params={"size": size_bytes}, timeout=30.0)
        if r.status_code == 429:
            # The control plane's per-IP probe budget is shared by every node
            # behind the same public address. Returning what we already measured
            # beats aborting the whole measurement (and, before this, also
            # skipping the *upload* probe, which is the direction that actually
            # bounds a pipeline stage).
            break
        r.raise_for_status()
        dt = time.perf_counter() - t0
        actual = len(r.content)
        if dt > 0 and actual > 0:
            mbps = (actual * 8) / dt / 1e6
            best_mbps = max(best_mbps, mbps)
        if dt >= time_target:
            break
        size_bytes = min(size_bytes * 2, cap_bytes)
        if size_bytes == cap_bytes and best_mbps > 0 and dt < time_target:
            # Already at the cap and still faster than the time target. Take one
            # final sample and stop.
            #
            # Without this break the loop is unbounded: ``min(size*2, cap)`` pins
            # size_bytes at the cap forever, so on a fast link (localhost, LAN,
            # gigabit fibre) the condition ``size_bytes <= cap_bytes`` stays true
            # and the agent downloads 8 MB over and over until the *server's*
            # per-IP budget cuts it off with a 429. Measured: one agent on
            # loopback consumed the entire 128 MB budget at startup, so the next
            # node to join from the same public IP could not measure its link at
            # all and advertised bandwidth_mbps=0 permanently. Every node behind
            # one NAT — and every node behind CGNAT, which the plan budgets at
            # 25-40% of the fleet — was affected.
            #
            # The upload ramp below always had this guard; the download ramp did
            # not, which is the only reason the two behaved differently.
            t0 = time.perf_counter()
            r = client.get(url, params={"size": size_bytes}, timeout=30.0)
            r.raise_for_status()
            dt = time.perf_counter() - t0
            actual = len(r.content)
            if dt > 0 and actual > 0:
                best_mbps = max(best_mbps, (actual * 8) / dt / 1e6)
            break

    return best_mbps


def _measure_upload_bw(client, url: str) -> float:
    """Ramped upload bandwidth measurement — the mirror of ``_measure_download_bw``.

    This is the direction that actually bounds a pipeline stage: a stage receives one
    activation tensor and *sends* one onward, and residential uplinks are commonly
    20-40x slower than the downlink. Placing a stage on a node chosen by its download
    speed is how you build a pipeline that stalls.

    The server reports how long it spent receiving, so a slow local serialisation does
    not get charged to the network.
    """
    size_bytes = 64 * 1024
    cap_bytes = 4 * 1024 * 1024   # half the download cap: uplinks are slower
    time_target = 0.2

    payload = os.urandom(size_bytes)
    try:
        client.post(url, content=payload, timeout=10.0)   # warm-up, not measured
    except Exception:  # noqa: BLE001
        return 0.0

    best_mbps = 0.0
    while size_bytes <= cap_bytes:
        payload = os.urandom(size_bytes)
        t0 = time.perf_counter()
        r = client.post(url, content=payload, timeout=30.0)
        dt = time.perf_counter() - t0
        if r.status_code == 429:
            break
        r.raise_for_status()
        # Trust the server's byte count over our own: a proxy may have truncated.
        try:
            acked = int(r.json().get("bytes", 0))
        except Exception:  # noqa: BLE001
            acked = 0
        if dt > 0 and acked > 0:
            best_mbps = max(best_mbps, (acked * 8) / dt / 1e6)
        if dt >= time_target:
            break
        size_bytes = min(size_bytes * 2, cap_bytes)
        if size_bytes == cap_bytes and best_mbps > 0 and dt < time_target:
            # Already at the cap and still fast — one final sample at the cap.
            payload = os.urandom(size_bytes)
            t0 = time.perf_counter()
            r = client.post(url, content=payload, timeout=30.0)
            dt = time.perf_counter() - t0
            r.raise_for_status()
            try:
                acked = int(r.json().get("bytes", 0))
            except Exception:  # noqa: BLE001
                acked = 0
            if dt > 0 and acked > 0:
                best_mbps = max(best_mbps, (acked * 8) / dt / 1e6)
            break

    return best_mbps


def network_class_from_link(link: LinkStats) -> str:
    """Public wrapper: classify a node from its *measured* link.

    The capability probe runs before the control plane is reachable, so it
    cannot know the link yet. Registration measures the link and refines the
    class here, which is why network_class is not frozen at probe time.
    """
    return _network_class_from_link(link.latency_ms, link.bandwidth_mbps or 0.0)


def probe_capability(sim_overrides: Optional[dict] = None) -> Capability:
    """Full capability probe. ``sim_overrides`` lets the simulation harness pin
    heterogeneous fake capacities (cores/ram/gpu) onto processes on one machine."""
    detail = probe_hardware_detail()
    cap = Capability(
        cpu_cores=detail.cpu_cores,
        cpu_arch=detail.cpu_arch,
        ram_mb=detail.ram_mb or _UNKNOWN_INT,
        gpu=detail.gpu,
        vram_mb=detail.vram_mb or _UNKNOWN_INT,
        disk_mb=detail.disk_free_mb or _UNKNOWN_INT,
        flops_estimate=_flops_estimate(),
        cpu_model=detail.cpu_model or "unknown",
        gpu_names=list(detail.gpu_names),
        disk_total_mb=detail.disk_total_mb or 0,
        thermal_state=detail.thermal_state,
        battery_powered=detail.battery_powered,
        battery_percent=detail.battery_percent,
        network_class=detail.network_class,
    )
    if sim_overrides:
        for k, v in sim_overrides.items():
            if hasattr(cap, k):
                setattr(cap, k, v)
    log.info("probed capability", cores=cap.cpu_cores, ram=cap.ram_mb,
             gpu=cap.gpu, vram=cap.vram_mb, disk=cap.disk_mb,
             thermal=detail.thermal_state, battery_powered=detail.battery_powered,
             network_class=detail.network_class, gflops=cap.flops_estimate)
    return cap
