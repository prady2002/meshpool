package distribution

import (
	"fmt"
	"os/exec"
	"strings"
)

// commandRunner executes a service-manager command and returns its combined
// output. It is a field rather than a direct exec.Command call so tests can
// verify the *construction* of these commands without registering anything with
// the host's real service manager -- which needs elevation, differs per OS, and
// would leave residue on a developer machine. Command construction is the part
// that is easy to get wrong and cheap to check; actual registration is verified
// on the target platform during release testing.
type commandRunner func(name string, args ...string) ([]byte, error)

func execRunner(name string, args ...string) ([]byte, error) {
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		return out, fmt.Errorf("distribution: %s %s: %w: %s",
			name, strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

// SystemService returns the Service implementation that drives the native
// service manager for the given platform: SCM on Windows, launchd on macOS and
// systemd everywhere else.
func SystemService(platform Platform) Service {
	switch platform {
	case Windows:
		return &windowsService{run: execRunner}
	case Darwin:
		return &launchdService{run: execRunner}
	default:
		return &systemdService{run: execRunner}
	}
}

// --------------------------------------------------------------------------- //
// Windows: Service Control Manager via sc.exe
// --------------------------------------------------------------------------- //
type windowsService struct{ run commandRunner }

func (s *windowsService) Register(spec ServiceSpec) error {
	// sc.exe binPath= takes ONE string containing the executable and its
	// arguments, and the space after "binPath=" is required by sc's parser.
	bin := quoteCommand(spec.BinaryPath, spec.Args)
	_, err := s.run("sc.exe", "create", spec.Name, "binPath= "+bin, "start= auto")
	return err
}

func (s *windowsService) Start(name string) error {
	_, err := s.run("sc.exe", "start", name)
	return err
}

func (s *windowsService) Stop(name string) error {
	_, err := s.run("sc.exe", "stop", name)
	return err
}

func (s *windowsService) Unregister(name string) error {
	_, err := s.run("sc.exe", "delete", name)
	return err
}

func (s *windowsService) Status(name string) (ServiceState, bool, error) {
	out, err := s.run("sc.exe", "query", name)
	if err != nil {
		// A service that is not installed is a legitimate answer, not a failure.
		return ServiceState{}, false, nil
	}
	text := string(out)
	return ServiceState{
		Spec:    ServiceSpec{Name: name},
		Started: strings.Contains(text, "RUNNING"),
	}, true, nil
}

// --------------------------------------------------------------------------- //
// macOS: launchd via launchctl
// --------------------------------------------------------------------------- //
type launchdService struct{ run commandRunner }

func (s *launchdService) Register(spec ServiceSpec) error {
	// The plist itself is written by writeServiceDescriptor(); launchctl only
	// needs to be pointed at it.
	_, err := s.run("launchctl", "load", "-w", spec.ConfigPath)
	return err
}

func (s *launchdService) Start(name string) error {
	_, err := s.run("launchctl", "start", name)
	return err
}

func (s *launchdService) Stop(name string) error {
	_, err := s.run("launchctl", "stop", name)
	return err
}

func (s *launchdService) Unregister(name string) error {
	_, err := s.run("launchctl", "remove", name)
	return err
}

func (s *launchdService) Status(name string) (ServiceState, bool, error) {
	out, err := s.run("launchctl", "list", name)
	if err != nil {
		return ServiceState{}, false, nil
	}
	return ServiceState{
		Spec:    ServiceSpec{Name: name},
		Started: strings.Contains(string(out), "PID"),
	}, true, nil
}

// --------------------------------------------------------------------------- //
// Linux and everything else: systemd via systemctl
// --------------------------------------------------------------------------- //
type systemdService struct{ run commandRunner }

func (s *systemdService) Register(spec ServiceSpec) error {
	// The unit file is written by writeServiceDescriptor(); systemd has to be
	// told to re-read it before the unit can be enabled.
	if _, err := s.run("systemctl", "daemon-reload"); err != nil {
		return err
	}
	_, err := s.run("systemctl", "enable", spec.Name)
	return err
}

func (s *systemdService) Start(name string) error {
	_, err := s.run("systemctl", "start", name)
	return err
}

func (s *systemdService) Stop(name string) error {
	_, err := s.run("systemctl", "stop", name)
	return err
}

func (s *systemdService) Unregister(name string) error {
	if _, err := s.run("systemctl", "disable", name); err != nil {
		return err
	}
	_, err := s.run("systemctl", "daemon-reload")
	return err
}

func (s *systemdService) Status(name string) (ServiceState, bool, error) {
	out, err := s.run("systemctl", "is-active", name)
	state := strings.TrimSpace(string(out))
	if err != nil && state == "" {
		return ServiceState{}, false, nil
	}
	// "inactive" and "failed" are still installed services, so presence is not
	// the same question as running -- report both separately.
	if state == "unknown" || strings.Contains(state, "could not be found") {
		return ServiceState{}, false, nil
	}
	return ServiceState{
		Spec:    ServiceSpec{Name: name},
		Started: state == "active",
	}, true, nil
}

// quoteCommand renders an executable plus arguments as a single command string
// for service managers that take one. Any element containing whitespace is
// double-quoted, because an unquoted "C:\Program Files\..." path is read by the
// SCM as the executable "C:\Program" followed by stray arguments -- a classic
// unquoted-service-path defect that is also a privilege-escalation vector.
func quoteCommand(binary string, args []string) string {
	parts := make([]string, 0, len(args)+1)
	for _, p := range append([]string{binary}, args...) {
		if strings.ContainsAny(p, " \t") {
			p = `"` + p + `"`
		}
		parts = append(parts, p)
	}
	return strings.Join(parts, " ")
}
