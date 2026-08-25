"""Cross-platform hardware probe parser tests.

These are fixture tests for platforms this Windows host cannot execute.  They
guard the real defect class: plausible-looking guessed hardware is worse than
an explicit unknown because placement trusts these numbers.
"""
from __future__ import annotations

from meshpool.agent import probe as P


def test_linux_meminfo_parser_reports_real_mib_not_guess():
    sample = "MemTotal:       32800664 kB\nMemFree:         123456 kB\n"
    assert P._parse_proc_meminfo(sample) == 32031
    assert P._parse_proc_meminfo("MemFree: 1 kB\n") is None


def test_windows_memory_parser_reports_real_mib_not_guess():
    assert P._parse_windows_total_physical_memory("33585074176\r\n") == 32029
    assert P._parse_windows_total_physical_memory("\r\n") is None


def test_macos_sysctl_memory_parser_reports_real_mib_not_guess():
    assert P._parse_sysctl_memsize("34359738368\n") == 32768
    assert P._parse_sysctl_memsize("not-a-number\n") is None


def test_linux_cpu_model_parser():
    assert P._parse_lscpu_model(
        "vendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7-8650U CPU\n"
    ) == "Intel(R) Core(TM) i7-8650U CPU"


def test_nvidia_smi_parser_uses_largest_real_dedicated_vram():
    has_gpu, vram = P._parse_nvidia_smi_vram("8192\n24576\n")
    assert has_gpu is True
    assert vram == 24576


def test_windows_integrated_gpu_vram_is_unknown_not_fake_512mb():
    sample = """
[
  {"Name":"Intel(R) Iris(R) Xe Graphics","AdapterRAM":536870912},
  {"Name":"AMD Radeon(TM) Graphics","AdapterRAM":536870912},
  {"Name":"AMD Ryzen 5 PRO 8540U w/ Radeon 740M Graphics","AdapterRAM":536870912}
]
"""
    has_gpu, vram, names = P._parse_windows_video_json(sample)
    assert has_gpu is True
    assert "Intel" in names[0]
    assert vram == 0, "shared-memory iGPU VRAM must be unknown, not 512 MB"


def test_windows_dedicated_gpu_vram_parser():
    sample = '{"Name":"NVIDIA GeForce RTX 4090","AdapterRAM":25757220864}'
    has_gpu, vram, names = P._parse_windows_video_json(sample)
    assert has_gpu is True
    assert names == ["NVIDIA GeForce RTX 4090"]
    assert 24000 <= vram <= 25000


def test_macos_apple_silicon_gpu_reports_unknown_vram():
    sample = """
{"SPDisplaysDataType":[{"sppci_model":"Apple M2","spdisplays_vram":"1536 MB"}]}
"""
    has_gpu, vram, names = P._parse_macos_displays_json(sample)
    assert has_gpu is True
    assert names == ["Apple M2"]
    assert vram == 0


def test_macos_discrete_gpu_vram_parser():
    sample = """
{"SPDisplaysDataType":[{"sppci_model":"AMD Radeon Pro 5600M","sppci_vram":"8 GB"}]}
"""
    has_gpu, vram, _ = P._parse_macos_displays_json(sample)
    assert has_gpu is True
    assert vram == 8192


def test_linux_lspci_gpu_presence_without_vram_guess():
    has_gpu, names = P._parse_linux_gpu_lspci(
        "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620\n"
    )
    assert has_gpu is True
    assert names


def test_network_class_unknown_when_link_unmeasured():
    assert P._network_class_from_link(0.0, 0.0) == "unknown"
    assert P._network_class_from_link(1.0, 500.0) == "loopback_or_lan"
    assert P._network_class_from_link(80.0, 2.0) == "constrained_wan"


def test_thermal_thresholds_are_ordered():
    assert P._thermal_state_from_celsius(None) == "unknown"
    assert P._thermal_state_from_celsius(45.0) == "nominal"
    assert P._thermal_state_from_celsius(70.0) == "warm"
    assert P._thermal_state_from_celsius(85.0) == "hot"
    assert P._thermal_state_from_celsius(95.0) == "critical"
