package distribution

import (
	"fmt"
	"strings"
	"testing"
)

// An unquoted service binary path containing spaces is read by the Windows SCM
// as the executable "C:\Program" followed by stray arguments. That is both a
// broken install and a well-known privilege-escalation vector, so the quoting
// is worth asserting directly.
func TestQuoteCommandQuotesPathsContainingSpaces(t *testing.T) {
	got := quoteCommand(`C:\Program Files\MeshPool\meshd.exe`,
		[]string{"--config", `C:\ProgramData\MeshPool\meshd.env`})
	want := `"C:\Program Files\MeshPool\meshd.exe" --config C:\ProgramData\MeshPool\meshd.env`
	if got != want {
		t.Errorf("quoteCommand =\n  %s\nwant\n  %s", got, want)
	}
}

func TestQuoteCommandLeavesSimplePathsAlone(t *testing.T) {
	got := quoteCommand("/usr/local/bin/meshd", []string{"--config", "/etc/meshpool/meshd.env"})
	want := "/usr/local/bin/meshd --config /etc/meshpool/meshd.env"
	if got != want {
		t.Errorf("quoteCommand = %q, want %q", got, want)
	}
}

func TestSystemServiceSelectsPerPlatformImplementation(t *testing.T) {
	for platform, want := range map[Platform]string{
		Windows: "*distribution.windowsService",
		Darwin:  "*distribution.launchdService",
		Linux:   "*distribution.systemdService",
	} {
		got := typeName(SystemService(platform))
		if got != want {
			t.Errorf("SystemService(%q) = %s, want %s", platform, got, want)
		}
	}
}

func TestWindowsRegisterBuildsScCreateCommand(t *testing.T) {
	var calls [][]string
	svc := &windowsService{run: func(name string, args ...string) ([]byte, error) {
		calls = append(calls, append([]string{name}, args...))
		return nil, nil
	}}
	err := svc.Register(ServiceSpec{
		Name:       ServiceName,
		BinaryPath: `C:\Program Files\MeshPool\meshd.exe`,
		ConfigPath: `C:\ProgramData\MeshPool\meshd.env`,
		Args:       []string{"--config", `C:\ProgramData\MeshPool\meshd.env`},
	})
	if err != nil {
		t.Fatalf("Register: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected exactly one command, got %d: %v", len(calls), calls)
	}
	joined := strings.Join(calls[0], " ")
	for _, want := range []string{
		"sc.exe", "create", ServiceName,
		// sc's parser requires the space after "binPath=" and one quoted string.
		`binPath= "C:\Program Files\MeshPool\meshd.exe" --config`,
		"start= auto",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("sc.exe command missing %q\n  got: %s", want, joined)
		}
	}
}

func TestSystemdRegisterReloadsBeforeEnabling(t *testing.T) {
	var calls [][]string
	svc := &systemdService{run: func(name string, args ...string) ([]byte, error) {
		calls = append(calls, append([]string{name}, args...))
		return nil, nil
	}}
	if err := svc.Register(ServiceSpec{Name: ServiceName}); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if len(calls) != 2 {
		t.Fatalf("expected daemon-reload then enable, got %v", calls)
	}
	// Enabling a unit systemd has not re-read yet fails, so the order matters.
	if calls[0][1] != "daemon-reload" || calls[1][1] != "enable" {
		t.Errorf("expected [daemon-reload, enable], got %v", calls)
	}
}

func TestStatusReportsNotInstalledRatherThanErroring(t *testing.T) {
	svc := &systemdService{run: func(name string, args ...string) ([]byte, error) {
		return []byte("unknown"), nil
	}}
	state, found, err := svc.Status(ServiceName)
	if err != nil {
		t.Fatalf("Status returned an error for an absent service: %v", err)
	}
	if found {
		t.Errorf("an absent service must report found=false, got %+v", state)
	}
}

func typeName(v any) string {
	return fmt.Sprintf("%T", v)
}
