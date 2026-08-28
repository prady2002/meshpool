package update

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// The property running through this file: an update check runs unattended on
// someone else's computer. Every failure -- a dead host, a truncated body, a
// hostile manifest -- must end with the node still working exactly as before.

type releaseServer struct {
	*httptest.Server
	manifest atomic.Value // []byte
	artifact atomic.Value // []byte
	hits     atomic.Int64
}

func newReleaseServer(t *testing.T) *releaseServer {
	t.Helper()
	rs := &releaseServer{}
	mux := http.NewServeMux()
	mux.HandleFunc("/manifest.json", func(w http.ResponseWriter, r *http.Request) {
		rs.hits.Add(1)
		b, _ := rs.manifest.Load().([]byte)
		if b == nil {
			http.Error(w, "no manifest", http.StatusNotFound)
			return
		}
		w.Write(b)
	})
	mux.HandleFunc("/artifact.bin", func(w http.ResponseWriter, r *http.Request) {
		b, _ := rs.artifact.Load().([]byte)
		w.Write(b)
	})
	rs.Server = httptest.NewServer(mux)
	t.Cleanup(rs.Close)
	return rs
}

// testArtifact pins a synthetic platform so one manifest works on every machine
// this suite runs on. It is deliberately a separate helper rather than a global
// that init() mutates: a package-level switch flipped by one test file's init
// would silently change what the other files assert, and a test that quietly
// tests something else is worse than one that fails.
func testArtifact(body []byte) *Artifact {
	a := stagedArtifact(body)
	a.OS, a.Arch = "testos", "testarch"
	return a
}

func (rs *releaseServer) publish(t *testing.T, version string, body []byte, priv ed25519.PrivateKey) {
	t.Helper()
	a := testArtifact(body)
	a.URL = rs.URL + "/artifact.bin"
	sm, err := Sign(Manifest{Version: version, Artifacts: []Artifact{*a}}, priv)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	blob, err := json.Marshal(sm)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	rs.manifest.Store(blob)
	rs.artifact.Store(body)
}

func checkerFor(t *testing.T, rs *releaseServer, pub ed25519.PublicKey, current string) (*Checker, string) {
	t.Helper()
	dir := t.TempDir()
	target := filepath.Join(dir, "meshd.exe")
	if err := os.WriteFile(target, []byte("running binary"), 0o755); err != nil {
		t.Fatalf("seed: %v", err)
	}
	return &Checker{
		ManifestURL:    rs.URL + "/manifest.json",
		Trusted:        TrustedKeys{KeyIDFor(pub): pub},
		CurrentVersion: current,
		Target:         target,
		GOOS:           "testos",
		GOARCH:         "testarch",
	}, target
}

// stagedArtifact in install_test.go uses the real platform; these tests pin a
// synthetic one so the same manifest works on every machine.
func TestCheckAndStageFetchesVerifiesAndStages(t *testing.T) {
	pub, priv := newKey(t)
	rs := newReleaseServer(t)
	body := []byte("meshd 2.0.0 payload")
	rs.publish(t, "2.0.0", body, priv)

	c, target := checkerFor(t, rs, pub, "1.0.0")
	got, err := c.CheckAndStage(context.Background(), true)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if got != "2.0.0" {
		t.Fatalf("staged version = %q, want 2.0.0", got)
	}

	// Staging must not disturb the running binary; that is what makes it safe to
	// run unattended.
	if b, _ := os.ReadFile(target); string(b) != "running binary" {
		t.Fatalf("running binary changed during a staging-only check: %q", b)
	}

	in := &Installer{Target: target, Artifact: stagedArtifact(body)}
	if err := in.Install(); err != nil {
		t.Fatalf("install: %v", err)
	}
	if b, _ := os.ReadFile(target); string(b) != string(body) {
		t.Fatalf("target = %q, want %q", b, body)
	}
}

// "You are up to date" is the overwhelmingly common outcome and is not a fault.
// If it shared an error value with real failures, a genuinely broken release
// channel would be invisible among thousands of routine no-op checks.
func TestUpToDateIsItsOwnOutcomeNotAnError(t *testing.T) {
	pub, priv := newKey(t)
	rs := newReleaseServer(t)
	rs.publish(t, "1.0.0", []byte("same version"), priv)

	c, target := checkerFor(t, rs, pub, "1.0.0")
	_, err := c.CheckAndStage(context.Background(), true)
	if !errors.Is(err, ErrNoUpdate) {
		t.Fatalf("err = %v, want ErrNoUpdate", err)
	}
	if errors.Is(err, ErrDowngrade) {
		t.Fatal("a routine up-to-date check is being reported as a downgrade attack")
	}
	if _, statErr := os.Stat(target + ".new"); !os.IsNotExist(statErr) {
		t.Fatal("nothing should have been staged")
	}
}

func TestManifestSignedByAnUntrustedKeyStagesNothing(t *testing.T) {
	pub, _ := newKey(t)
	_, attacker := newKey(t)
	rs := newReleaseServer(t)
	rs.publish(t, "9.9.9", []byte("malware"), attacker)

	c, target := checkerFor(t, rs, pub, "1.0.0")
	if _, err := c.CheckAndStage(context.Background(), true); !errors.Is(err, ErrBadSignature) {
		t.Fatalf("err = %v, want ErrBadSignature", err)
	}
	if _, statErr := os.Stat(target + ".new"); !os.IsNotExist(statErr) {
		t.Fatal("an unsigned build was staged where a later Install would take it")
	}
	if b, _ := os.ReadFile(target); string(b) != "running binary" {
		t.Fatalf("running binary was disturbed: %q", b)
	}
}

// A signed manifest whose artifact the host then substitutes: the signature is
// genuine, so only the hash stands between the node and the attacker's bytes.
func TestSignedManifestWithSubstitutedArtifactStagesNothing(t *testing.T) {
	pub, priv := newKey(t)
	rs := newReleaseServer(t)
	rs.publish(t, "2.0.0", []byte("the real build"), priv)
	rs.artifact.Store([]byte("the fake build")) // same length, different bytes

	c, target := checkerFor(t, rs, pub, "1.0.0")
	if _, err := c.CheckAndStage(context.Background(), true); !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("err = %v, want ErrHashMismatch", err)
	}
	if _, statErr := os.Stat(target + ".new"); !os.IsNotExist(statErr) {
		t.Fatal("a substituted artifact was staged")
	}
}

// countingTransport measures what the client actually pulls off the wire.
//
// Server-side write counts cannot answer this: a small body vanishes into the
// socket buffer whether the client reads one byte of it or all of it. The only
// place the question "did we stop reading?" has an honest answer is the client's
// own response body.
type countingTransport struct {
	inner http.RoundTripper
	read  atomic.Int64
}

type countingBody struct {
	inner io.ReadCloser
	n     *atomic.Int64
}

func (b *countingBody) Read(p []byte) (int, error) {
	n, err := b.inner.Read(p)
	b.n.Add(int64(n))
	return n, err
}
func (b *countingBody) Close() error { return b.inner.Close() }

func (t *countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	inner := t.inner
	if inner == nil {
		inner = http.DefaultTransport
	}
	resp, err := inner.RoundTrip(r)
	if err != nil || resp.Body == nil {
		return resp, err
	}
	resp.Body = &countingBody{inner: resp.Body, n: &t.read}
	return resp, nil
}

// The manifest is read before anything about it is trusted, so it needs a bound
// of its own -- the very first byte of an update check otherwise comes from an
// unauthenticated host with no ceiling on how many follow.
//
// The first version of this test asserted on the error and its message. It
// survived deleting the limit outright, because an unbounded read still ends at
// the same length check, just after pulling the whole body. That is the second
// time the same blind spot appeared in this package, which says the fix is
// structural rather than a matter of care: the property is consumption, so the
// assertion has to be a byte count.
func TestOversizedManifestIsBoundedNotJustRejected(t *testing.T) {
	pub, _ := newKey(t)
	rs := newReleaseServer(t)

	const huge = 8 << 20
	rs.manifest.Store(bytes.Repeat([]byte("A"), huge))

	ct := &countingTransport{}
	c, _ := checkerFor(t, rs, pub, "1.0.0")
	c.MaxManifestBytes = 4096
	c.Client = &http.Client{Transport: ct}

	_, err := c.CheckAndStage(context.Background(), true)
	if !errors.Is(err, ErrMalformed) {
		t.Fatalf("err = %v, want ErrMalformed", err)
	}

	// Reading one byte past the ceiling is how we know the body is too long.
	// Anything more is memory an unauthenticated host made us spend. Allow a
	// single buffer's slack for the transport's own read granularity.
	limit := c.MaxManifestBytes + 1 + 32<<10
	if got := ct.read.Load(); got > limit {
		t.Fatalf("pulled %d bytes for a manifest capped at %d: the fetch is unbounded",
			got, c.MaxManifestBytes)
	}
	if ct.read.Load() == 0 {
		t.Fatal("nothing was read; this test would pass with no bound at all")
	}
}

func TestPlaintextURLsAreRefused(t *testing.T) {
	pub, priv := newKey(t)
	rs := newReleaseServer(t)
	rs.publish(t, "2.0.0", []byte("payload"), priv)

	c, _ := checkerFor(t, rs, pub, "1.0.0") // httptest is plain http
	if _, err := c.CheckAndStage(context.Background(), false); !errors.Is(err, ErrInsecureURL) {
		t.Fatalf("err = %v, want ErrInsecureURL", err)
	}
}

// A dead or broken release host is the most likely failure by far, and it must
// be indistinguishable in effect from never having checked.
func TestADeadReleaseHostLeavesTheNodeUntouched(t *testing.T) {
	pub, _ := newKey(t)
	rs := newReleaseServer(t)
	// No manifest published: the handler 404s.

	c, target := checkerFor(t, rs, pub, "1.0.0")
	if _, err := c.CheckAndStage(context.Background(), true); err == nil {
		t.Fatal("a 404 was reported as success")
	}
	if b, _ := os.ReadFile(target); string(b) != "running binary" {
		t.Fatalf("running binary changed: %q", b)
	}
	if _, statErr := os.Stat(target + ".new"); !os.IsNotExist(statErr) {
		t.Fatal("something was staged from a failed fetch")
	}
}

func TestCancelledContextStopsTheCheck(t *testing.T) {
	pub, priv := newKey(t)
	rs := newReleaseServer(t)
	rs.publish(t, "2.0.0", []byte("payload"), priv)

	c, target := checkerFor(t, rs, pub, "1.0.0")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := c.CheckAndStage(ctx, true); err == nil {
		t.Fatal("a cancelled check reported success")
	}
	if _, statErr := os.Stat(target + ".new"); !os.IsNotExist(statErr) {
		t.Fatal("a cancelled check still staged a binary")
	}
}

// A manifest with no build for this platform is a normal state during a
// staggered rollout, not an attack and not a crash.
func TestNoArtifactForThisPlatformIsHandled(t *testing.T) {
	pub, priv := newKey(t)
	rs := newReleaseServer(t)
	rs.publish(t, "2.0.0", []byte("payload"), priv)

	c, _ := checkerFor(t, rs, pub, "1.0.0")
	c.GOOS, c.GOARCH = "plan9", "sparc64"

	if _, err := c.CheckAndStage(context.Background(), true); !errors.Is(err, ErrNoArtifact) {
		t.Fatalf("err = %v, want ErrNoArtifact", err)
	}
}
