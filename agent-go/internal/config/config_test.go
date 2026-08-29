package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadEnvFileParsesAndPreservesExplicitEnv(t *testing.T) {
	path := filepath.Join(t.TempDir(), "meshd.env")
	body := strings.Join([]string{
		"# written by the installer",
		"",
		"export MESHPOOL_TEST_A=alpha",
		`MESHPOOL_TEST_B="quoted value"`,
		"MESHPOOL_TEST_C='single'",
		"MESHPOOL_TEST_D=from-file",
	}, "\n")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"MESHPOOL_TEST_A", "MESHPOOL_TEST_B", "MESHPOOL_TEST_C"} {
		key := key
		os.Unsetenv(key)
		t.Cleanup(func() { os.Unsetenv(key) })
	}
	t.Setenv("MESHPOOL_TEST_D", "from-environment")

	if err := LoadEnvFile(path); err != nil {
		t.Fatalf("LoadEnvFile: %v", err)
	}
	for key, want := range map[string]string{
		"MESHPOOL_TEST_A": "alpha",
		"MESHPOOL_TEST_B": "quoted value",
		"MESHPOOL_TEST_C": "single",
		// An explicit environment value is more specific than a file on disk
		// and must survive the load.
		"MESHPOOL_TEST_D": "from-environment",
	} {
		if got := os.Getenv(key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}
}

func TestLoadEnvFileRejectsMalformedLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.env")
	if err := os.WriteFile(path, []byte("NOT_AN_ASSIGNMENT\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := LoadEnvFile(path); err == nil {
		t.Fatal("a line with no '=' must be an error, not silently skipped")
	}
}

func TestLoadEnvFileMissingFileIsAnError(t *testing.T) {
	if err := LoadEnvFile(filepath.Join(t.TempDir(), "absent.env")); err == nil {
		t.Fatal("a missing --config file must fail loudly rather than start with defaults")
	}
}
