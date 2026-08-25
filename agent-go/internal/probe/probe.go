package probe

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"meshd/internal/config"
)

type Capability struct {
	CPUCores       int      `json:"cpu_cores"`
	CPUArch        string   `json:"cpu_arch"`
	CPUModel       string   `json:"cpu_model,omitempty"`
	RAMMB          int      `json:"ram_mb"`
	GPU            bool     `json:"gpu"`
	VRAMMB         int      `json:"vram_mb"`
	DiskMB         int      `json:"disk_mb"`
	DiskTotalMB    int      `json:"disk_total_mb,omitempty"`
	// GPUNames is deliberately NOT omitempty and is always initialised to a
	// non-nil slice. Go marshals a nil slice as JSON null, and the control
	// plane rejects null for a list field, which does not degrade the node —
	// it makes registration fail permanently. Keeping it non-nil at the source
	// is the cheap half of that defence.
	GPUNames []string `json:"gpu_names"`
	FLOPSEstimate  float64  `json:"flops_estimate"`
	ThermalState   string   `json:"thermal_state,omitempty"`
	BatteryPowered *bool    `json:"battery_powered,omitempty"`
	BatteryPercent *float64 `json:"battery_percent,omitempty"`
	NetworkClass   string   `json:"network_class,omitempty"`
	OllamaModels   []string `json:"ollama_models"`
	// Runtimes this agent can actually serve. meshd runs wasm modules but
	// deliberately does not reimplement the control plane's curated Python
	// function registry, so it advertises "wasm" only and the scheduler will
	// never hand it a builtin unit it would have to reject.
	Runtimes []string `json:"runtimes"`
	// Workload classes meshd can serve at all. It runs work units and holds
	// blobs; it does not run app replicas or model shards, and those handlers
	// return 501. Advertising that keeps the hosting and inference planners
	// from committing to a plan that includes this node — which matters more
	// than the batch case, because those planners do not retry elsewhere and
	// they count an unservable node's RAM toward pooled capacity.
	Services []string `json:"services"`
}

// Runtimes is what meshd advertises to the control plane.
var Runtimes = []string{"wasm"}

// Services is the set of workload classes meshd actually implements. Keep this
// in sync with the non-501 routes in internal/agent.
var Services = []string{"batch", "storage"}

var flopsSink float64

type Link struct {
	LatencyMS     float64 `json:"latency_ms"`
	BandwidthMbps float64 `json:"bandwidth_mbps"`
}

// Probe gathers local hardware capability and the link to the control plane.
//
// Every detector below blocks on a child process, and on Windows most of them
// are a separate `powershell -NoProfile -Command Get-CimInstance ...` that costs
// seconds just to start the shell. Run one after another the seven of them took
// ~45s (measured: `go test ./internal/probe` at 45.4s), and the agent does not
// bind its HTTP port until Probe returns -- so a node stayed unreachable and
// unregistered for the whole time. That is what made verify_hw_probe fail with
// "service not ready at 127.0.0.1:8362" against a 30s budget: the agent was
// healthy, just still shelling out.
//
// The work is I/O-bound, so running the detectors concurrently collapses the
// cost to roughly the slowest one instead of the sum. Two further savings come
// free: a detector whose result a MESHD_SIM_* override would immediately
// discard is not run at all, which is exactly equivalent to running it and
// throwing the answer away, and the link measurement overlaps the hardware
// probes since it shares nothing with them.
func Probe(cfg config.Config) (Capability, Link) {
	var (
		wg          sync.WaitGroup
		cpuModel    string
		diskFree    int
		diskTotal   int
		flops       float64
		thermal     string
		battPowered *bool
		battPct     *float64
		ramMB       int
		gpu         bool
		vramMB      int
		gpuNames    []string
		link        Link
	)

	run := func(fn func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn()
		}()
	}

	run(func() { cpuModel = detectCPUModel() })
	run(func() { diskFree, diskTotal = detectDiskMB() })
	run(func() { thermal = detectThermalState() })
	run(func() { battPowered, battPct = detectBattery() })
	run(func() { link = measureLink(cfg.ControlPlaneURL) })

	// Skipped when overridden: the value would be overwritten below anyway, and
	// the FLOPS estimate is a real CPU benchmark that the sim harness always
	// discards.
	if cfg.SimFLOPS <= 0 {
		run(func() { flops = detectFLOPSEstimate() })
	}
	if cfg.SimRAMMB <= 0 {
		run(func() { ramMB = detectRAMMB() })
	}
	if !cfg.SimGPU {
		run(func() { gpu, vramMB, gpuNames = detectGPU() })
	}

	wg.Wait()

	cap := Capability{
		CPUCores:       runtime.NumCPU(),
		CPUArch:        runtime.GOARCH,
		CPUModel:       cpuModel,
		RAMMB:          ramMB,
		GPU:            gpu,
		VRAMMB:         vramMB,
		DiskMB:         diskFree,
		DiskTotalMB:    diskTotal,
		FLOPSEstimate:  flops,
		ThermalState:   thermal,
		NetworkClass:   "unknown",
		OllamaModels:   []string{},
		GPUNames:       gpuNames,
		Runtimes:       Runtimes,
		Services:       Services,
		BatteryPowered: battPowered,
		BatteryPercent: battPct,
	}
	if cap.CPUCores < 1 {
		cap.CPUCores = 1
	}
	if cap.DiskMB <= 0 {
		cap.DiskMB = cfg.DonateDiskMB
	}
	if cfg.SimCores > 0 {
		cap.CPUCores = cfg.SimCores
	}
	if cfg.SimRAMMB > 0 {
		cap.RAMMB = cfg.SimRAMMB
	}
	if cfg.SimVRAMMB > 0 {
		cap.VRAMMB = cfg.SimVRAMMB
	}
	if cfg.SimFLOPS > 0 {
		cap.FLOPSEstimate = cfg.SimFLOPS
	}
	if cfg.SimGPU {
		cap.GPU = true
	}
	// Never nil: Go marshals a nil slice as JSON null, and the control plane
	// rejects null for this field.
	if cap.GPUNames == nil {
		cap.GPUNames = []string{}
	}
	cap.NetworkClass = networkClass(link)
	return cap, link
}

// detectRAMMB reports installed physical memory in MB, 0 if it cannot be
// determined. Previously meshd advertised ram_mb=0 unless MESHD_SIM_RAM_MB was
// set, which made every real Go node score zero usable layer memory and be
// silently excluded from model placement.
func detectRAMMB() int {
	switch runtime.GOOS {
	case "windows":
		out, err := exec.Command("powershell", "-NoProfile", "-Command",
			"(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory").Output()
		if err == nil {
			return parseBytesAsMB(string(out))
		}
	case "linux":
		data, err := os.ReadFile("/proc/meminfo")
		if err == nil {
			return parseProcMeminfoMB(string(data))
		}
	case "darwin":
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err == nil {
			return parseBytesAsMB(string(out))
		}
	}
	return 0
}

func parseProcMeminfoMB(text string) int {
	for _, line := range strings.Split(text, "\n") {
		if !strings.HasPrefix(line, "MemTotal:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			if kb, err := strconv.ParseInt(fields[1], 10, 64); err == nil && kb > 0 {
				return int(kb / 1024)
			}
		}
	}
	return 0
}

func parseBytesAsMB(text string) int {
	v, err := strconv.ParseInt(strings.TrimSpace(text), 10, 64)
	if err != nil || v <= 0 {
		return 0
	}
	return int(v / 1024 / 1024)
}

func detectCPUModel() string {
	switch runtime.GOOS {
	case "windows":
		out, err := exec.Command("powershell", "-NoProfile", "-Command",
			"(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)").Output()
		if err == nil {
			return strings.TrimSpace(string(out))
		}
	case "linux":
		data, err := os.ReadFile("/proc/cpuinfo")
		if err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(strings.ToLower(line), "model name") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						return strings.TrimSpace(parts[1])
					}
				}
			}
		}
	case "darwin":
		out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
		if err == nil {
			return strings.TrimSpace(string(out))
		}
	}
	return ""
}

func detectGPU() (bool, int, []string) {
	if has, vram := detectNVIDIAVRAM(); has {
		return true, vram, detectNVIDIANames()
	}
	switch runtime.GOOS {
	case "windows":
		cmd := exec.Command("powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json")
		if out, err := cmd.Output(); err == nil {
			return parseWindowsVideoJSON(string(out))
		}
	case "linux":
		if out, err := exec.Command("lspci").Output(); err == nil {
			has, names := parseLinuxLspciGPU(string(out))
			return has, 0, names
		}
	case "darwin":
		if out, err := exec.Command("system_profiler", "SPDisplaysDataType", "-json").Output(); err == nil {
			has, vram, names := parseMacDisplaysJSON(string(out))
			return has, vram, names
		}
	}
	return false, 0, []string{}
}

// detectNVIDIANames is best-effort: a missing name must never make the GPU
// itself disappear, so failures return an empty (never nil) slice.
func detectNVIDIANames() []string {
	names := []string{}
	out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return names
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if s := strings.TrimSpace(line); s != "" {
			names = append(names, s)
		}
	}
	return names
}

func detectNVIDIAVRAM() (bool, int) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return false, 0
	}
	return parseNVIDIASMIVRAM(string(out))
}

func parseNVIDIASMIVRAM(text string) (bool, int) {
	best := 0
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(strings.Split(line, ",")[0])
		v, err := strconv.Atoi(line)
		if err == nil && v > best {
			best = v
		}
	}
	return best > 0, best
}

func looksIntegratedGPU(name string) bool {
	n := strings.ToLower(name)
	for _, hint := range []string{"intel", "uhd", "iris", "vega", "radeon graphics", "apple", "microsoft basic"} {
		if strings.Contains(n, hint) {
			return true
		}
	}
	if strings.Contains(n, "ryzen") && strings.Contains(n, "radeon") {
		return true
	}
	if regexp.MustCompile(`radeon.*graphics`).MatchString(n) {
		return true
	}
	if regexp.MustCompile(`radeon.*\b[5678]\d{2}m\b`).MatchString(n) {
		return true
	}
	return false
}

func parseWindowsVideoJSON(text string) (bool, int, []string) {
	var one map[string]any
	var many []map[string]any
	rows := []map[string]any{}
	if err := json.Unmarshal([]byte(text), &many); err == nil {
		rows = many
	} else if err := json.Unmarshal([]byte(text), &one); err == nil {
		rows = []map[string]any{one}
	}
	hasGPU := false
	best := 0
	names := []string{}
	for _, row := range rows {
		name, _ := row["Name"].(string)
		if strings.TrimSpace(name) != "" {
			hasGPU = true
			names = append(names, strings.TrimSpace(name))
		}
		if looksIntegratedGPU(name) {
			continue
		}
		var bytesVal int64
		switch v := row["AdapterRAM"].(type) {
		case float64:
			bytesVal = int64(v)
		case string:
			bytesVal, _ = strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		}
		if bytesVal > 0 {
			mb := int(bytesVal / 1024 / 1024)
			if mb > best {
				best = mb
			}
		}
	}
	return hasGPU || best > 0, best, names
}

func parseLinuxLspciGPU(text string) (bool, []string) {
	names := []string{}
	re := regexp.MustCompile(`(?i)\b(VGA|3D|Display)\b`)
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if re.MatchString(line) {
			names = append(names, line)
		}
	}
	return len(names) > 0, names
}

func parseMacDisplaysJSON(text string) (bool, int, []string) {
	var root map[string][]map[string]any
	if err := json.Unmarshal([]byte(text), &root); err != nil {
		return false, 0, nil
	}
	names := []string{}
	best := 0
	re := regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(GB|MB)`)
	for _, row := range root["SPDisplaysDataType"] {
		name := ""
		for _, key := range []string{"sppci_model", "_name", "spdisplays_device_name"} {
			if v, ok := row[key].(string); ok && strings.TrimSpace(v) != "" {
				name = strings.TrimSpace(v)
				break
			}
		}
		if name != "" {
			names = append(names, name)
		}
		vram := ""
		for _, key := range []string{"sppci_vram", "spdisplays_vram"} {
			if v, ok := row[key].(string); ok && strings.TrimSpace(v) != "" {
				vram = strings.TrimSpace(v)
				break
			}
		}
		if m := re.FindStringSubmatch(vram); len(m) == 3 {
			f, _ := strconv.ParseFloat(m[1], 64)
			mb := int(f)
			if strings.EqualFold(m[2], "GB") {
				mb = int(f * 1024)
			}
			if mb > best {
				best = mb
			}
		}
	}
	allIntegrated := len(names) > 0
	for _, name := range names {
		if !looksIntegratedGPU(name) {
			allIntegrated = false
		}
	}
	if allIntegrated {
		best = 0
	}
	return len(names) > 0 || best > 0, best, names
}

func detectThermalState() string {
	switch runtime.GOOS {
	case "windows":
		out, err := exec.Command("powershell", "-NoProfile", "-Command",
			"Get-CimInstance -Namespace root/wmi MSAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature").Output()
		if err == nil {
			vals := []float64{}
			for _, line := range strings.Split(string(out), "\n") {
				v, err := strconv.ParseFloat(strings.TrimSpace(line), 64)
				if err == nil && v > 0 {
					vals = append(vals, (v/10.0)-273.15)
				}
			}
			if len(vals) > 0 {
				sort.Float64s(vals)
				return thermalStateFromC(vals[len(vals)-1])
			}
		}
	case "linux":
		entries, err := os.ReadDir("/sys/class/thermal")
		if err == nil {
			vals := []float64{}
			for _, e := range entries {
				b, err := os.ReadFile(filepath.Join("/sys/class/thermal", e.Name(), "temp"))
				if err == nil {
					v, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
					if err == nil && v > 0 {
						if v > 1000 {
							v = v / 1000.0
						}
						vals = append(vals, v)
					}
				}
			}
			if len(vals) > 0 {
				sort.Float64s(vals)
				return thermalStateFromC(vals[len(vals)-1])
			}
		}
	}
	return "unknown"
}

func thermalStateFromC(c float64) string {
	switch {
	case c >= 90:
		return "critical"
	case c >= 80:
		return "hot"
	case c >= 65:
		return "warm"
	default:
		return "nominal"
	}
}

func detectBattery() (*bool, *float64) {
	switch runtime.GOOS {
	case "windows":
		out, err := exec.Command("powershell", "-NoProfile", "-Command",
			"Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json").Output()
		if err == nil && strings.TrimSpace(string(out)) != "" {
			onBatt, pct, ok := parseWindowsBatteryJSON(string(out))
			if ok {
				return &onBatt, &pct
			}
		}
	case "darwin":
		out, err := exec.Command("pmset", "-g", "batt").Output()
		if err == nil {
			onBatt, pct, ok := parseDarwinBattery(string(out))
			if ok {
				return &onBatt, &pct
			}
		}
	case "linux":
		entries, err := os.ReadDir("/sys/class/power_supply")
		if err == nil {
			for _, e := range entries {
				base := filepath.Join("/sys/class/power_supply", e.Name())
				typ, err := os.ReadFile(filepath.Join(base, "type"))
				if err != nil || strings.TrimSpace(strings.ToLower(string(typ))) != "battery" {
					continue
				}
				status, _ := os.ReadFile(filepath.Join(base, "status"))
				capacity, _ := os.ReadFile(filepath.Join(base, "capacity"))
				onBatt := strings.EqualFold(strings.TrimSpace(string(status)), "discharging")
				if pct, err := strconv.ParseFloat(strings.TrimSpace(string(capacity)), 64); err == nil {
					return &onBatt, &pct
				}
				return &onBatt, nil
			}
		}
	}
	return nil, nil
}

func parseWindowsBatteryJSON(text string) (bool, float64, bool) {
	var one map[string]any
	var many []map[string]any
	rows := []map[string]any{}
	if err := json.Unmarshal([]byte(text), &many); err == nil {
		rows = many
	} else if err := json.Unmarshal([]byte(text), &one); err == nil {
		rows = []map[string]any{one}
	}
	for _, row := range rows {
		status := 0
		switch v := row["BatteryStatus"].(type) {
		case float64:
			status = int(v)
		case string:
			status, _ = strconv.Atoi(strings.TrimSpace(v))
		}
		pct := 0.0
		switch v := row["EstimatedChargeRemaining"].(type) {
		case float64:
			pct = v
		case string:
			pct, _ = strconv.ParseFloat(strings.TrimSpace(v), 64)
		}
		return status == 1, pct, true
	}
	return false, 0, false
}

func parseDarwinBattery(text string) (bool, float64, bool) {
	re := regexp.MustCompile(`(\d+)%`)
	m := re.FindStringSubmatch(text)
	if len(m) != 2 {
		return strings.Contains(text, "'Battery Power'"), 0, strings.Contains(text, "Now drawing from")
	}
	pct, _ := strconv.ParseFloat(m[1], 64)
	return strings.Contains(text, "'Battery Power'"), pct, true
}

func networkClass(link Link) string {
	if link.LatencyMS == 0 && link.BandwidthMbps == 0 {
		return "unknown"
	}
	if link.LatencyMS < 2 && (link.BandwidthMbps == 0 || link.BandwidthMbps >= 100) {
		return "loopback_or_lan"
	}
	if link.LatencyMS < 20 && (link.BandwidthMbps == 0 || link.BandwidthMbps >= 25) {
		return "lan_or_metro"
	}
	if link.BandwidthMbps > 0 && link.BandwidthMbps < 5 {
		return "constrained_wan"
	}
	return "wan"
}

// detectDiskMB returns (freeMB, totalMB) for the volume holding the working
// directory, sampled in a single call.
//
// These used to be two independent detectors, and the total was computed as
// Used+Free from Get-PSDrive. Both of those numbers move while the machine is
// running and each rounds independently, so the "total" -- a static property of
// the volume -- jittered by a megabyte between probes. That made every
// heartbeat look like a capability change to the control plane, and it also
// allowed free > total if the two spawns happened to straddle a large write.
//
// Reading Size and FreeSpace from one Win32_LogicalDisk row fixes all of that:
// Size is the volume capacity and does not move, the two values are mutually
// consistent because they come from the same sample, and it costs one child
// process instead of two.
func detectDiskMB() (free int, total int) {
	switch runtime.GOOS {
	case "windows":
		cwd, err := os.Getwd()
		if err != nil || len(cwd) < 2 || cwd[1] != ':' {
			return 0, 0
		}
		drive := strings.ToUpper(string(cwd[0]))
		out, err := exec.Command("powershell", "-NoProfile", "-Command",
			fmt.Sprintf(`$d = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='%s:'"; `+
				`"$($d.FreeSpace)`+"`n"+`$($d.Size)"`, drive)).Output()
		if err != nil {
			return 0, 0
		}
		return parseFreeAndTotalMB(string(out))
	default:
		out, err := exec.Command("df", "-k", ".").Output()
		if err != nil {
			return 0, 0
		}
		return parseDFAvailableMB(string(out)), parseDFTotalMB(string(out))
	}
}

// parseFreeAndTotalMB reads two byte counts on separate lines.
func parseFreeAndTotalMB(text string) (free int, total int) {
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) < 2 {
		return 0, 0
	}
	return parseBytesAsMB(lines[0]), parseBytesAsMB(lines[1])
}

// parseDFTotalMB reads the 1K-blocks (total size) column, where
// parseDFAvailableMB reads the available column.
func parseDFTotalMB(text string) int {
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) < 2 {
		return 0
	}
	fields := strings.Fields(lines[len(lines)-1])
	if len(fields) < 2 {
		return 0
	}
	kb, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil || kb < 0 {
		return 0
	}
	return int(kb / 1024)
}

func parseDFAvailableMB(text string) int {
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) < 2 {
		return 0
	}
	fields := strings.Fields(lines[len(lines)-1])
	if len(fields) < 4 {
		return 0
	}
	kb, err := strconv.ParseInt(fields[3], 10, 64)
	if err != nil || kb <= 0 {
		return 0
	}
	return int(kb / 1024)
}

func detectFLOPSEstimate() float64 {
	const n = 1 << 18
	a := make([]float64, n)
	b := make([]float64, n)
	for i := range a {
		a[i] = float64((i%97)+1) / 97.0
		b[i] = float64((i%89)+1) / 89.0
	}
	best := 0.0
	for trial := 0; trial < 4; trial++ {
		start := time.Now()
		sum := 0.0
		iters := 32
		for k := 0; k < iters; k++ {
			for i := 0; i < n; i++ {
				sum += a[i] * b[i]
			}
		}
		flopsSink = sum
		dt := time.Since(start).Seconds()
		if dt > 0 {
			gflops := float64(n*iters*2) / dt / 1e9
			if gflops > best {
				best = gflops
			}
		}
	}
	return best
}

// measureLink measures latency and *real* download bandwidth against the control
// plane's dedicated probe endpoints — the same ones meshpool/agent/probe.py uses.
//
// Previously this returned BandwidthMbps: 0 unconditionally, so every meshd node
// advertised "unknown bandwidth" forever. Placement uses that number, so a Go node
// was permanently indistinguishable from a node on a dial-up link.
//
// Bandwidth stays 0 when the endpoint is absent, which is honest ("not measured"),
// never a fabricated guess derived from health-check payload sizes.
func measureLink(baseURL string) Link {
	base := strings.TrimRight(baseURL, "/")
	client := &http.Client{Timeout: 30 * time.Second}
	latency := measureLatencyMS(client, base+"/api/v1/health")
	if latency < 0 {
		return Link{}
	}
	return Link{LatencyMS: latency, BandwidthMbps: measureDownloadMbps(client, base+"/api/v1/probe/download")}
}

// measureLatencyMS returns the median steady-state RTT in ms, or -1 if the control
// plane never answered. The first sample is discarded: it includes connection setup.
func measureLatencyMS(client *http.Client, url string) float64 {
	vals := make([]float64, 0, 8)
	for i := 0; i < 8; i++ {
		start := time.Now()
		resp, err := client.Get(url)
		if err != nil {
			continue
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if i == 0 {
			continue // connection establishment, not steady-state RTT
		}
		vals = append(vals, float64(time.Since(start).Microseconds())/1000.0)
	}
	if len(vals) == 0 {
		return -1
	}
	sort.Float64s(vals)
	return vals[len(vals)/2]
}

// measureDownloadMbps ramps the request size until one transfer takes at least
// 200 ms (or the 8 MB cap is reached), mirroring the Python agent's algorithm so
// the two implementations report comparable numbers. Returns 0 when unmeasurable.
func measureDownloadMbps(client *http.Client, url string) float64 {
	const capBytes = 8 * 1024 * 1024
	const timeTargetS = 0.2

	fetch := func(size int) (int, float64, error) {
		start := time.Now()
		resp, err := client.Get(fmt.Sprintf("%s?size=%d", url, size))
		if err != nil {
			return 0, 0, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return 0, 0, fmt.Errorf("probe download HTTP %d", resp.StatusCode)
		}
		n, err := io.Copy(io.Discard, resp.Body)
		return int(n), time.Since(start).Seconds(), err
	}

	head, err := client.Head(url)
	if err != nil {
		log.Printf("bandwidth probe: HEAD %s failed: %v (advertising unknown bandwidth)", url, err)
		return 0
	}
	head.Body.Close()
	if head.StatusCode != http.StatusOK {
		log.Printf("bandwidth probe: HEAD %s returned %d (advertising unknown bandwidth)", url, head.StatusCode)
		return 0 // no probe endpoint: unknown, not zero-as-a-guess
	}

	size := 64 * 1024
	if _, _, err := fetch(size); err != nil { // warm-up, not measured
		log.Printf("bandwidth probe: warm-up fetch failed: %v (advertising unknown bandwidth)", err)
		return 0
	}
	best := 0.0
	for size <= capBytes {
		n, dt, err := fetch(size)
		if err != nil {
			break
		}
		if dt > 0 && n > 0 {
			if mbps := float64(n*8) / dt / 1e6; mbps > best {
				best = mbps
			}
		}
		if dt >= timeTargetS {
			break
		}
		size *= 2
		if size > capBytes {
			size = capBytes
		}
		if size == capBytes && dt < timeTargetS && best > 0 {
			// one final measurement at the cap, then stop
			if n, dt, err := fetch(size); err == nil && dt > 0 && n > 0 {
				if mbps := float64(n*8) / dt / 1e6; mbps > best {
					best = mbps
				}
			}
			break
		}
	}
	return best
}

func BlobRoot(dataDir string) string {
	return filepath.Join(dataDir, "blobs")
}
