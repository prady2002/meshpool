//go:build !windows

package config

import (
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// totalRAMMB reports installed physical memory in MiB, or 0 when it cannot be
// determined. Returning 0 (and letting the caller fall back) is deliberate: an
// invented number would flow straight into placement decisions.
func totalRAMMB() int {
	switch runtime.GOOS {
	case "linux":
		return memTotalFromProc("/proc/meminfo")
	case "darwin":
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			return 0
		}
		n, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		if err != nil {
			return 0
		}
		return int(n / 1024 / 1024)
	default:
		return 0
	}
}

// memTotalFromProc parses the "MemTotal:  N kB" line of a meminfo-format file.
func memTotalFromProc(path string) int {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "MemTotal:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return int(kb / 1024)
	}
	return 0
}
