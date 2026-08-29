package config

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Config struct {
	ControlPlaneURL string
	AgentPort       int
	AgentHost       string
	AgentBind       string
	Name            string
	Owner           string
	EnrollToken     string
	DataDir         string
	DonateCPUPct    int
	DonateRAMMB     int
	DonateDiskMB    int
	Schedule        string
	SimCores        int
	SimRAMMB        int
	SimVRAMMB       int
	SimFLOPS        float64
	SimGPU          bool
	AdvertiseURL    string
	QuicAddr        string
	Lie             bool
}

func Load() (Config, error) {
	hostname, _ := os.Hostname()
	dataDir := os.Getenv("MESHPOOL_DATA_DIR")
	if dataDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return Config{}, fmt.Errorf("user home dir: %w", err)
		}
		dataDir = filepath.Join(home, ".meshpool")
	}
	totalRAM := totalRAMMB()
	if totalRAM <= 0 {
		totalRAM = 4096
	}
	freeDisk := freeDiskMB(dataDir)
	if freeDisk <= 0 {
		freeDisk = 1024
	}
	cfg := Config{
		ControlPlaneURL: envOr("MESHPOOL_CONTROL_PLANE_URL", "http://127.0.0.1:8080"),
		AgentPort:       envInt("MESHPOOL_AGENT_PORT", 9001),
		AgentHost:       os.Getenv("MESHPOOL_AGENT_HOST"),
		AgentBind:       envOr("MESHPOOL_AGENT_BIND", "0.0.0.0"),
		Name:            envOr("MESHPOOL_NAME", hostname),
		Owner:           envOr("MESHPOOL_OWNER", "default"),
		EnrollToken:     os.Getenv("MESHPOOL_ENROLL_TOKEN"),
		DataDir:         dataDir,
		DonateCPUPct:    envInt("MESHPOOL_DONATE_CPU_PCT", 50),
		DonateRAMMB:     envInt("MESHPOOL_DONATE_RAM_MB", maxInt(totalRAM/2, 2048)),
		DonateDiskMB:    envInt("MESHPOOL_DONATE_DISK_MB", maxInt(freeDisk/4, 256)),
		Schedule:        envOr("MESHPOOL_SCHEDULE", "always"),
		SimCores:        envInt("MESHPOOL_SIM_CORES", 0),
		SimRAMMB:        envInt("MESHPOOL_SIM_RAM_MB", 0),
		SimVRAMMB:       envInt("MESHPOOL_SIM_VRAM_MB", 0),
		SimFLOPS:        envFloat("MESHPOOL_SIM_FLOPS", 0),
		SimGPU:          envBool("MESHPOOL_SIM_GPU", false),
		AdvertiseURL:    os.Getenv("MESHPOOL_ADVERTISE_URL"),
		QuicAddr:        os.Getenv("MESHPOOL_QUIC_ADDR"),
		Lie:             envBool("MESHPOOL_LIE", false),
	}
	if cfg.AgentHost == "" {
		cfg.AgentHost = detectLANIP()
	}
	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		return Config{}, fmt.Errorf("create data dir: %w", err)
	}
	return cfg, nil
}

func (c Config) AgentURL() string {
	if c.AdvertiseURL != "" {
		return c.AdvertiseURL
	}
	return fmt.Sprintf("http://%s:%d", c.AgentHost, c.AgentPort)
}

// LocalURL is where this agent's own HTTP server is reachable from this
// machine. It is deliberately not AgentURL(): the advertised address may be a
// hostname this node cannot resolve, or may be unreachable from the internet
// entirely (which is the normal case behind CGNAT, and precisely when the
// tunnel is carrying the traffic). Forwarding tunnel frames must target the
// real socket, not the advertisement.
func (c Config) LocalURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", c.AgentPort)
}

// LoadEnvFile loads KEY=VALUE pairs from a config file into the process
// environment. Blank lines and '#' comments are ignored, an optional leading
// "export " is tolerated, and matching surrounding quotes are stripped. The
// format is deliberately the same one systemd's EnvironmentFile= accepts, so
// the file the installer writes works both for systemd and for a manual
// `meshd --config` run.
//
// Values already present in the environment are NOT overwritten. An operator
// who sets a variable explicitly when launching the process is being more
// specific than a file on disk, and silently discarding that override is the
// kind of surprise that costs an afternoon of debugging.
func LoadEnvFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(strings.TrimSuffix(raw, "\r"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return fmt.Errorf("config: %s line %d: expected KEY=VALUE, got %q", path, i+1, line)
		}
		key = strings.TrimSpace(key)
		if key == "" {
			return fmt.Errorf("config: %s line %d: empty key", path, i+1)
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 && (value[0] == '"' || value[0] == '\'') && value[len(value)-1] == value[0] {
			value = value[1 : len(value)-1]
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			return err
		}
	}
	return nil
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func envInt(name string, fallback int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func envFloat(name string, fallback float64) float64 {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil {
			return n
		}
	}
	return fallback
}

func envBool(name string, fallback bool) bool {
	if v := os.Getenv(name); v != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			return true
		case "0", "false", "no", "off":
			return false
		}
	}
	return fallback
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func detectLANIP() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "127.0.0.1"
	}
	for _, addr := range addrs {
		ipNet, ok := addr.(*net.IPNet)
		if !ok || ipNet.IP.IsLoopback() {
			continue
		}
		ip := ipNet.IP.To4()
		if ip != nil {
			return ip.String()
		}
	}
	return "127.0.0.1"
}
