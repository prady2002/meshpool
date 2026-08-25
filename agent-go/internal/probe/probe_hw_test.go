package probe

import "testing"

func TestParseCrossPlatformRAMFixtures(t *testing.T) {
	if got := parseProcMeminfoMB("MemTotal:       32800664 kB\n"); got != 32031 {
		t.Fatalf("linux meminfo parsed %d MB, want 32031", got)
	}
	if got := parseBytesAsMB("33585074176\r\n"); got != 32029 {
		t.Fatalf("windows bytes parsed %d MB, want 32029", got)
	}
	if got := parseBytesAsMB("34359738368\n"); got != 32768 {
		t.Fatalf("darwin sysctl bytes parsed %d MB, want 32768", got)
	}
	if got := parseProcMeminfoMB("MemFree: 1 kB\n"); got != 0 {
		t.Fatalf("missing MemTotal parsed %d; unknown must stay 0", got)
	}
}

func TestParseNVIDIASMIFixture(t *testing.T) {
	has, vram := parseNVIDIASMIVRAM("8192\n24576\n")
	if !has || vram != 24576 {
		t.Fatalf("nvidia-smi parsed has=%v vram=%d, want true/24576", has, vram)
	}
}

func TestWindowsIntegratedGPUDoesNotInventVRAM(t *testing.T) {
	sample := `[
	  {"Name":"Intel(R) Iris(R) Xe Graphics","AdapterRAM":536870912},
	  {"Name":"AMD Radeon(TM) Graphics","AdapterRAM":536870912},
	  {"Name":"AMD Ryzen 5 PRO 8540U w/ Radeon 740M Graphics","AdapterRAM":536870912}
	]`
	has, vram, names := parseWindowsVideoJSON(sample)
	if !has {
		t.Fatal("expected GPU presence from Windows video fixture")
	}
	if vram != 0 {
		t.Fatalf("iGPU shared memory parsed as %d MB; should be unknown/0", vram)
	}
	if len(names) != 3 {
		t.Fatalf("expected all 3 adapter names to be reported, got %v", names)
	}
}

func TestWindowsDiscreteGPUVRAMFixture(t *testing.T) {
	has, vram, names := parseWindowsVideoJSON(`{"Name":"NVIDIA GeForce RTX 4090","AdapterRAM":25757220864}`)
	if !has || vram < 24000 || vram > 25000 {
		t.Fatalf("discrete GPU parsed has=%v vram=%d, want about 24576 MB", has, vram)
	}
	if len(names) != 1 || names[0] != "NVIDIA GeForce RTX 4090" {
		t.Fatalf("expected the adapter name to be reported, got %v", names)
	}
}

func TestMacAppleSiliconDoesNotInventDedicatedVRAM(t *testing.T) {
	has, vram, names := parseMacDisplaysJSON(
		`{"SPDisplaysDataType":[{"sppci_model":"Apple M2","spdisplays_vram":"1536 MB"}]}`,
	)
	if !has || len(names) != 1 {
		t.Fatalf("mac fixture parsed has=%v names=%v", has, names)
	}
	if vram != 0 {
		t.Fatalf("Apple unified-memory GPU parsed as %d MB dedicated VRAM; want unknown/0", vram)
	}
}

func TestMacDiscreteGPUVRAMFixture(t *testing.T) {
	has, vram, _ := parseMacDisplaysJSON(
		`{"SPDisplaysDataType":[{"sppci_model":"AMD Radeon Pro 5600M","sppci_vram":"8 GB"}]}`,
	)
	if !has || vram != 8192 {
		t.Fatalf("mac discrete GPU parsed has=%v vram=%d, want true/8192", has, vram)
	}
}

func TestLinuxLspciDetectsGPUWithoutVRAMGuess(t *testing.T) {
	has, names := parseLinuxLspciGPU("00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620\n")
	if !has || len(names) != 1 {
		t.Fatalf("lspci parsed has=%v names=%v", has, names)
	}
}

func TestBatteryAndNetworkClassParsers(t *testing.T) {
	onBatt, pct, ok := parseWindowsBatteryJSON(`{"BatteryStatus":1,"EstimatedChargeRemaining":77}`)
	if !ok || !onBatt || pct != 77 {
		t.Fatalf("windows battery parsed on=%v pct=%.1f ok=%v", onBatt, pct, ok)
	}
	onBatt, pct, ok = parseDarwinBattery("Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged;")
	if !ok || onBatt || pct != 100 {
		t.Fatalf("darwin battery parsed on=%v pct=%.1f ok=%v", onBatt, pct, ok)
	}
	if got := networkClass(Link{}); got != "unknown" {
		t.Fatalf("zero link class = %q, want unknown", got)
	}
	if got := networkClass(Link{LatencyMS: 1, BandwidthMbps: 500}); got != "loopback_or_lan" {
		t.Fatalf("fast low-latency link class = %q", got)
	}
	if got := thermalStateFromC(85); got != "hot" {
		t.Fatalf("thermal 85C = %q, want hot", got)
	}
}

func TestParseDFAvailableMB(t *testing.T) {
	sample := "Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/sda1 1000000 250000 750000 25% /\n"
	if got := parseDFAvailableMB(sample); got != 732 {
		t.Fatalf("df available parsed %d MB, want 732", got)
	}
}

func TestFLOPSEstimateIsMeasuredPositive(t *testing.T) {
	got := detectFLOPSEstimate()
	if got <= 0 {
		t.Fatalf("FLOPS estimate = %.3f, want a measured positive value", got)
	}
}
