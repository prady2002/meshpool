package distribution

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInstallCreatesConcreteServiceStateAndStartsIt(t *testing.T) {
	root := t.TempDir()
	binary := filepath.Join(root, "build", "meshd.exe")
	if err := os.MkdirAll(filepath.Dir(binary), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(binary, []byte("real meshd binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	svc := newMemoryService()
	plan := NewPlan(Windows, WithRoot(root))

	if err := Install(InstallRequest{
		Plan:            plan,
		SourceBinary:    binary,
		ControlPlaneURL: "https://cp.example",
		EnrollToken:     "invite-123",
		NodeName:        "studio-pc",
		DonateCPUPct:    40,
		Service:         svc,
	}); err != nil {
		t.Fatalf("install: %v", err)
	}

	assertFileEquals(t, plan.BinaryPath, "real meshd binary")
	assertFileContains(t, plan.ConfigPath, "MESHPOOL_CONTROL_PLANE_URL=https://cp.example")
	assertFileContains(t, plan.ConfigPath, "MESHPOOL_ENROLL_TOKEN=invite-123")
	assertFileContains(t, plan.ConfigPath, "MESHPOOL_NAME=studio-pc")
	assertFileContains(t, plan.ConfigPath, "MESHPOOL_DONATE_CPU_PCT=40")

	got := svc.snapshot()
	st, ok := got[ServiceName]
	if !ok {
		t.Fatalf("service %q was not registered; state = %#v", ServiceName, got)
	}
	if !st.Started {
		t.Fatalf("service %q registered but not started: %#v", ServiceName, st)
	}
	if st.BinaryPath != plan.BinaryPath {
		t.Fatalf("registered binary = %q, want %q", st.BinaryPath, plan.BinaryPath)
	}
	if st.ConfigPath != plan.ConfigPath {
		t.Fatalf("registered config = %q, want %q", st.ConfigPath, plan.ConfigPath)
	}
}

func TestUninstallPurgesEveryInstallerAndRuntimeArtifact(t *testing.T) {
	root := t.TempDir()
	plan := NewPlan(Linux, WithRoot(root))
	svc := newMemoryService()
	if err := seedInstalledState(plan, svc); err != nil {
		t.Fatal(err)
	}

	paths := plan.AllRemovalPaths()
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("non-vacuity: expected %s to exist before uninstall: %v", p, err)
		}
	}
	for _, p := range []string{plan.BinaryPath, plan.ConfigPath, plan.IdentityPath, plan.WeightCacheDir, plan.ServicePath} {
		if !contains(paths, p) {
			t.Fatalf("removal plan does not enumerate required artifact %s", p)
		}
	}
	if !svc.snapshot()[ServiceName].Started {
		t.Fatal("non-vacuity: service was not started before uninstall")
	}

	if err := Uninstall(UninstallRequest{Plan: plan, Service: svc}); err != nil {
		t.Fatalf("uninstall: %v", err)
	}

	for _, p := range paths {
		if _, err := os.Stat(p); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s survived uninstall; err=%v", p, err)
		}
	}
	if _, ok := svc.snapshot()[ServiceName]; ok {
		t.Fatalf("service %q survived uninstall: %#v", ServiceName, svc.snapshot())
	}
}

func TestUninstallReportsServiceAndDeleteFailures(t *testing.T) {
	root := t.TempDir()
	plan := NewPlan(Darwin, WithRoot(root))
	svc := newMemoryService()
	if err := seedInstalledState(plan, svc); err != nil {
		t.Fatal(err)
	}
	svc.stopErr = errors.New("launchd refused stop")

	err := Uninstall(UninstallRequest{Plan: plan, Service: svc})
	if err == nil {
		t.Fatal("uninstall reported success while stop failed")
	}
	if !strings.Contains(err.Error(), "launchd refused stop") {
		t.Fatalf("uninstall hid the failed step: %v", err)
	}
}

func TestPackagingPlanCoversAllSupportedPlatformsWithPinnedBuildFlags(t *testing.T) {
	plan := ReleasePlan("1.2.3", "abc123", "https://releases.example/meshpool")
	if len(plan.Artifacts) != 6 {
		t.Fatalf("artifact count = %d, want 6: %#v", len(plan.Artifacts), plan.Artifacts)
	}
	want := map[string]bool{
		"windows/amd64": false, "windows/arm64": false,
		"darwin/amd64": false, "darwin/arm64": false,
		"linux/amd64": false, "linux/arm64": false,
	}
	for _, a := range plan.Artifacts {
		key := a.OS + "/" + a.Arch
		if _, ok := want[key]; !ok {
			t.Fatalf("unexpected platform %s in %#v", key, plan.Artifacts)
		}
		want[key] = true
		if !contains(a.BuildArgs, "-trimpath") {
			t.Fatalf("%s build args miss -trimpath: %#v", key, a.BuildArgs)
		}
		joined := strings.Join(a.BuildArgs, " ")
		if !strings.Contains(joined, "-buildid=") ||
			!strings.Contains(joined, "-X meshd/internal/update.Version=1.2.3") ||
			!strings.Contains(joined, "-X main.commit=abc123") {
			t.Fatalf("%s build args do not pin reproducible stamps: %s", key, joined)
		}
		if !strings.HasPrefix(a.URL, "https://releases.example/meshpool/") {
			t.Fatalf("%s URL is not release-hosted: %s", key, a.URL)
		}
	}
	for key, seen := range want {
		if !seen {
			t.Fatalf("missing platform %s", key)
		}
	}
}

func TestInstallerSigningFailsClosedWithoutACertificate(t *testing.T) {
	root := t.TempDir()
	unsigned := filepath.Join(root, "meshd-windows-amd64.msi")
	signed := filepath.Join(root, "meshd-windows-amd64.signed.msi")
	if err := os.WriteFile(unsigned, []byte("unsigned package"), 0o644); err != nil {
		t.Fatal(err)
	}

	err := SignInstaller(SignRequest{Platform: Windows, UnsignedPath: unsigned, SignedPath: signed})
	if !errors.Is(err, ErrSigningUnavailable) {
		t.Fatalf("err = %v, want ErrSigningUnavailable", err)
	}
	if _, statErr := os.Stat(signed); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("unsigned package was emitted as signed at %s", signed)
	}
}

func seedInstalledState(plan Plan, svc *memoryService) error {
	for _, dir := range []string{filepath.Dir(plan.BinaryPath), filepath.Dir(plan.ConfigPath), plan.DataDir, plan.WeightCacheDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	if err := os.WriteFile(plan.BinaryPath, []byte("binary"), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(plan.ConfigPath, []byte("config"), 0o600); err != nil {
		return err
	}
	if err := os.WriteFile(plan.IdentityPath, []byte("identity-key"), 0o600); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(plan.WeightCacheDir, "model.bin"), []byte("weights"), 0o600); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(plan.ServicePath), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(plan.ServicePath, []byte("service registration"), 0o644); err != nil {
		return err
	}
	if err := svc.Register(ServiceSpec{Name: ServiceName, BinaryPath: plan.BinaryPath, ConfigPath: plan.ConfigPath}); err != nil {
		return err
	}
	return svc.Start(ServiceName)
}

func assertFileEquals(t *testing.T, path, want string) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if string(got) != want {
		t.Fatalf("%s = %q, want %q", path, got, want)
	}
}

func assertFileContains(t *testing.T, path, want string) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if !strings.Contains(string(got), want) {
		t.Fatalf("%s does not contain %q:\n%s", path, want, got)
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
