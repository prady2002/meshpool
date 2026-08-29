// Package distribution contains the install, service-registration and release
// packaging primitives for shipping meshd as a normal background service.
package distribution

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

const ServiceName = "meshpool-meshd"

type Platform string

const (
	Windows Platform = "windows"
	Darwin  Platform = "darwin"
	Linux   Platform = "linux"
)

type Plan struct {
	Platform       Platform
	Root           string
	BinaryPath     string
	ConfigPath     string
	DataDir        string
	IdentityPath   string
	WeightCacheDir string
	ServicePath    string
}

type planOptions struct {
	root string
}

type PlanOption func(*planOptions)

func WithRoot(root string) PlanOption {
	return func(o *planOptions) { o.root = root }
}

func NewPlan(platform Platform, opts ...PlanOption) Plan {
	var po planOptions
	for _, opt := range opts {
		opt(&po)
	}
	root := po.root
	if root == "" {
		root = string(os.PathSeparator)
	}
	join := func(elem ...string) string {
		return filepath.Join(append([]string{root}, elem...)...)
	}
	switch platform {
	case Windows:
		return Plan{
			Platform:       platform,
			Root:           root,
			BinaryPath:     join("Program Files", "MeshPool", "meshd.exe"),
			ConfigPath:     join("ProgramData", "MeshPool", "meshd.env"),
			DataDir:        join("ProgramData", "MeshPool"),
			IdentityPath:   join("ProgramData", "MeshPool", "identity.json"),
			WeightCacheDir: join("ProgramData", "MeshPool", "weights"),
			ServicePath:    join("ProgramData", "MeshPool", "meshpool-meshd.service.json"),
		}
	case Darwin:
		return Plan{
			Platform:       platform,
			Root:           root,
			BinaryPath:     join("usr", "local", "bin", "meshd"),
			ConfigPath:     join("Library", "Application Support", "MeshPool", "meshd.env"),
			DataDir:        join("Library", "Application Support", "MeshPool"),
			IdentityPath:   join("Library", "Application Support", "MeshPool", "identity.json"),
			WeightCacheDir: join("Library", "Caches", "MeshPool", "weights"),
			ServicePath:    join("Library", "LaunchDaemons", "com.meshpool.meshd.plist"),
		}
	default:
		return Plan{
			Platform:       Linux,
			Root:           root,
			BinaryPath:     join("usr", "local", "bin", "meshd"),
			ConfigPath:     join("etc", "meshpool", "meshd.env"),
			DataDir:        join("var", "lib", "meshpool"),
			IdentityPath:   join("var", "lib", "meshpool", "identity.json"),
			WeightCacheDir: join("var", "cache", "meshpool", "weights"),
			ServicePath:    join("etc", "systemd", "system", "meshpool-meshd.service"),
		}
	}
}

func PlanForCurrentOS(opts ...PlanOption) Plan {
	switch runtime.GOOS {
	case "windows":
		return NewPlan(Windows, opts...)
	case "darwin":
		return NewPlan(Darwin, opts...)
	default:
		return NewPlan(Linux, opts...)
	}
}

func (p Plan) AllRemovalPaths() []string {
	return []string{
		p.BinaryPath,
		p.ConfigPath,
		p.IdentityPath,
		p.WeightCacheDir,
		p.ServicePath,
	}
}

type ServiceSpec struct {
	Name       string
	BinaryPath string
	ConfigPath string
	Args       []string
}

type ServiceState struct {
	Spec       ServiceSpec
	BinaryPath string
	ConfigPath string
	Started    bool
}

type Service interface {
	Register(ServiceSpec) error
	Start(name string) error
	Stop(name string) error
	Unregister(name string) error
	Status(name string) (ServiceState, bool, error)
}

type InstallRequest struct {
	Plan            Plan
	SourceBinary    string
	ControlPlaneURL string
	EnrollToken     string
	NodeName        string
	DonateCPUPct    int
	Service         Service
}

func Install(req InstallRequest) error {
	if req.Plan.BinaryPath == "" {
		req.Plan = PlanForCurrentOS()
	}
	if req.Service == nil {
		req.Service = SystemService(req.Plan.Platform)
	}
	if req.SourceBinary == "" {
		return errors.New("distribution: source binary is required")
	}
	for _, dir := range []string{filepath.Dir(req.Plan.BinaryPath), filepath.Dir(req.Plan.ConfigPath), req.Plan.DataDir, req.Plan.WeightCacheDir, filepath.Dir(req.Plan.ServicePath)} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	if err := copyFile(req.SourceBinary, req.Plan.BinaryPath, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(req.Plan.ConfigPath, []byte(envConfig(req)), 0o600); err != nil {
		return err
	}
	spec := ServiceSpec{
		Name:       ServiceName,
		BinaryPath: req.Plan.BinaryPath,
		ConfigPath: req.Plan.ConfigPath,
		Args:       []string{"--config", req.Plan.ConfigPath},
	}
	if err := writeServiceDescriptor(req.Plan, spec); err != nil {
		return err
	}
	if err := req.Service.Register(spec); err != nil {
		return err
	}
	if err := req.Service.Start(ServiceName); err != nil {
		return err
	}
	return nil
}

type UninstallRequest struct {
	Plan    Plan
	Service Service
}

func Uninstall(req UninstallRequest) error {
	if req.Plan.BinaryPath == "" {
		req.Plan = PlanForCurrentOS()
	}
	if req.Service == nil {
		req.Service = SystemService(req.Plan.Platform)
	}
	var errs []string
	if err := req.Service.Stop(ServiceName); err != nil {
		errs = append(errs, "stop service: "+err.Error())
	}
	if err := req.Service.Unregister(ServiceName); err != nil {
		errs = append(errs, "unregister service: "+err.Error())
	}
	for _, p := range req.Plan.AllRemovalPaths() {
		if err := os.RemoveAll(p); err != nil {
			errs = append(errs, "remove "+p+": "+err.Error())
		}
	}
	if len(errs) > 0 {
		return errors.New("distribution: uninstall incomplete: " + strings.Join(errs, "; "))
	}
	return nil
}

func envConfig(req InstallRequest) string {
	return fmt.Sprintf(
		"MESHPOOL_CONTROL_PLANE_URL=%s\nMESHPOOL_ENROLL_TOKEN=%s\nMESHPOOL_NAME=%s\nMESHPOOL_DONATE_CPU_PCT=%d\nMESHPOOL_DATA_DIR=%s\n",
		req.ControlPlaneURL, req.EnrollToken, req.NodeName, req.DonateCPUPct, req.Plan.DataDir,
	)
}

func writeServiceDescriptor(plan Plan, spec ServiceSpec) error {
	var body string
	switch plan.Platform {
	case Windows:
		body = fmt.Sprintf(`{"name":%q,"binary":%q,"config":%q,"args":["--config",%q]}`+"\n", spec.Name, spec.BinaryPath, spec.ConfigPath, spec.ConfigPath)
	case Darwin:
		body = fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
<key>Label</key><string>com.meshpool.meshd</string>
<key>ProgramArguments</key><array><string>%s</string><string>--config</string><string>%s</string></array>
<key>RunAtLoad</key><true/>
</dict></plist>
`, spec.BinaryPath, spec.ConfigPath)
	default:
		body = fmt.Sprintf(`[Unit]
Description=MeshPool meshd node agent
After=network-online.target

[Service]
EnvironmentFile=%s
ExecStart=%s
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
`, spec.ConfigPath, spec.BinaryPath)
	}
	return os.WriteFile(plan.ServicePath, []byte(body), 0o644)
}

func copyFile(src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Chmod(dst, mode)
}

type memoryService struct {
	mu            sync.Mutex
	services      map[string]ServiceState
	stopErr       error
	registerErr   error
	startErr      error
	unregisterErr error
}

func newMemoryService() *memoryService {
	return &memoryService{services: map[string]ServiceState{}}
}

func (m *memoryService) Register(spec ServiceSpec) error {
	if m.registerErr != nil {
		return m.registerErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.services[spec.Name] = ServiceState{Spec: spec, BinaryPath: spec.BinaryPath, ConfigPath: spec.ConfigPath}
	return nil
}

func (m *memoryService) Start(name string) error {
	if m.startErr != nil {
		return m.startErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	st, ok := m.services[name]
	if !ok {
		return fmt.Errorf("service %s is not registered", name)
	}
	st.Started = true
	m.services[name] = st
	return nil
}

func (m *memoryService) Stop(name string) error {
	if m.stopErr != nil {
		return m.stopErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	st, ok := m.services[name]
	if !ok {
		return nil
	}
	st.Started = false
	m.services[name] = st
	return nil
}

func (m *memoryService) Unregister(name string) error {
	if m.unregisterErr != nil {
		return m.unregisterErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.services, name)
	return nil
}

func (m *memoryService) Status(name string) (ServiceState, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	st, ok := m.services[name]
	return st, ok, nil
}

func (m *memoryService) snapshot() map[string]ServiceState {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]ServiceState, len(m.services))
	for k, v := range m.services {
		out[k] = v
	}
	return out
}
