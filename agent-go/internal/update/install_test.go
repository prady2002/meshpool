package update

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// The property under test throughout: after any outcome, success or failure, a
// complete and correct binary exists at the target path. "The update failed" is
// acceptable. "There is no binary" never is.

func stagedArtifact(body []byte) *Artifact {
	sum := sha256.Sum256(body)
	return &Artifact{
		OS: runtime.GOOS, Arch: runtime.GOARCH,
		SHA256: hex.EncodeToString(sum[:]),
		Size:   int64(len(body)),
	}
}

func installerFor(t *testing.T, current []byte, next []byte) (*Installer, string) {
	t.Helper()
	dir := t.TempDir()
	target := filepath.Join(dir, "meshd.exe")
	if err := os.WriteFile(target, current, 0o755); err != nil {
		t.Fatalf("seed target: %v", err)
	}
	return &Installer{Target: target, Artifact: stagedArtifact(next)}, target
}

func mustRead(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read %s: %v", p, err)
	}
	return b
}

func TestStageThenInstallReplacesTheBinary(t *testing.T) {
	next := []byte("new binary v2")
	in, target := installerFor(t, []byte("old binary v1"), next)

	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}
	// Staging must not touch the running binary: an update that is downloaded
	// but never installed has to leave the node fully working.
	if got := string(mustRead(t, target)); got != "old binary v1" {
		t.Fatalf("target changed during staging: %q", got)
	}

	if err := in.Install(); err != nil {
		t.Fatalf("install: %v", err)
	}
	if got := string(mustRead(t, target)); got != string(next) {
		t.Fatalf("target = %q, want %q", got, next)
	}

	// The displaced binary survives until explicitly cleaned up, because on
	// Windows it is still locked by the process performing the update.
	if _, err := os.Stat(target + ".old"); err != nil {
		t.Fatalf("previous binary not preserved: %v", err)
	}
	if err := in.CleanupPrevious(); err != nil {
		t.Fatalf("cleanup: %v", err)
	}
	if _, err := os.Stat(target + ".old"); !os.IsNotExist(err) {
		t.Fatalf("previous binary still present after cleanup: %v", err)
	}
}

func TestStageRejectsBytesThatDoNotMatchTheSignedHash(t *testing.T) {
	promised := []byte("the signed binary")
	in, target := installerFor(t, []byte("old"), promised)

	// Same length so the size check cannot be what saves us.
	delivered := []byte("the s1gned binary")
	if len(delivered) != len(promised) {
		t.Fatalf("test bug: bodies must be the same length to exercise the hash")
	}

	if err := in.Stage(bytes.NewReader(delivered)); !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("err = %v, want ErrHashMismatch", err)
	}
	if got := string(mustRead(t, target)); got != "old" {
		t.Fatalf("running binary was disturbed by a rejected download: %q", got)
	}
	if _, err := os.Stat(in.stagedPath()); !os.IsNotExist(err) {
		t.Fatal("a rejected download was left staged, where a later Install could pick it up")
	}
}

func TestStageRefusesWithoutASignedArtifact(t *testing.T) {
	dir := t.TempDir()
	in := &Installer{Target: filepath.Join(dir, "meshd.exe")}
	if err := in.Stage(bytes.NewReader([]byte("anything"))); !errors.Is(err, ErrStagingFailed) {
		t.Fatalf("err = %v, want ErrStagingFailed", err)
	}
}

// countingReader reports how much was actually pulled from the source, which is
// the only way to observe a bound that exists to stop us reading.
type countingReader struct {
	n   int64
	cap int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	if c.n >= c.cap {
		// Stand in for an endless source without letting a broken bound turn a
		// failing test into a hanging one.
		return 0, errors.New("countingReader: source exhausted the safety cap")
	}
	for i := range p {
		p[i] = 'A'
	}
	c.n += int64(len(p))
	return len(p), nil
}

// A hostile or broken host that streams forever must not fill the contributor's
// disk, so staging has to stop reading at the signed size.
//
// The first version of this test asserted only that the result was
// ErrHashMismatch, and it survived deleting the bound entirely -- because an
// unbounded read still ends in a length mismatch, just after writing however
// many gigabytes the attacker felt like sending. The rejection was never the
// property. The property is that we stopped reading, and only a count can see it.
func TestStageStopsReadingAtTheSignedSize(t *testing.T) {
	body := []byte("small")
	in, _ := installerFor(t, []byte("old"), body)

	src := &countingReader{cap: 64 << 20}
	err := in.Stage(src)
	if !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("err = %v, want ErrHashMismatch", err)
	}

	// One byte past the signed size is enough to know the stream is too long.
	// Anything beyond that is disk we spent on an attacker's behalf.
	limit := in.Artifact.Size + 1
	if src.n > limit {
		t.Fatalf("read %d bytes from an endless source, signed size is %d: the download is unbounded",
			src.n, in.Artifact.Size)
	}
	if _, statErr := os.Stat(in.stagedPath()); !os.IsNotExist(statErr) {
		t.Fatal("oversized download was left on disk")
	}

	// Non-vacuity: the reader really would have kept going, so the small count
	// above is the bound working and not the source running dry.
	if src.n == 0 {
		t.Fatal("nothing was read at all; this test would pass even with no bound")
	}
}

// Staging and installing are separated in time, and anything with write access
// to the directory can swap the staged file in between. Install must re-verify.
func TestInstallRefusesAStagedFileTamperedWithAfterStaging(t *testing.T) {
	next := []byte("new binary v2")
	in, target := installerFor(t, []byte("old binary v1"), next)
	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}

	if err := os.WriteFile(in.stagedPath(), []byte("evil binary!!"), 0o755); err != nil {
		t.Fatalf("tamper: %v", err)
	}

	if err := in.Install(); !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("err = %v, want ErrHashMismatch", err)
	}
	if got := string(mustRead(t, target)); got != "old binary v1" {
		t.Fatalf("target = %q, want the original binary untouched", got)
	}
	if _, err := os.Stat(in.stagedPath()); !os.IsNotExist(err) {
		t.Fatal("a tampered staged file was left in place to be retried")
	}
}

func TestInstallWithoutStagingIsRefused(t *testing.T) {
	in, target := installerFor(t, []byte("old"), []byte("new"))
	if err := in.Install(); !errors.Is(err, ErrNotStaged) {
		t.Fatalf("err = %v, want ErrNotStaged", err)
	}
	if got := string(mustRead(t, target)); got != "old" {
		t.Fatalf("target = %q, want untouched", got)
	}
}

// The dangerous window: the current binary has been renamed aside and the new
// one cannot be moved into place. The old binary must come back, because the
// alternative is an install directory with no binary in it at all.
//
// The first version of this test tried to block the destination with a
// directory. It passed -- because a rename moves a directory aside just as
// happily as a file, so the swap simply succeeded and the rollback branch never
// ran. Reaching this branch requires failing the *second* rename specifically,
// which is why the syscall is injected rather than arranged.
func TestFailedSwapRestoresThePreviousBinary(t *testing.T) {
	next := []byte("new binary v2")
	in, target := installerFor(t, []byte("old binary v1"), next)
	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}

	calls := 0
	in.rename = func(oldpath, newpath string) error {
		calls++
		if calls == 2 { // staged -> target
			return errors.New("simulated: destination is locked")
		}
		return os.Rename(oldpath, newpath)
	}

	err := in.Install()
	if err == nil {
		t.Fatal("install reported success while the swap could not happen")
	}
	if errors.Is(err, ErrBricked) {
		t.Fatalf("rollback did not run: %v", err)
	}
	if !errors.Is(err, ErrRollback) {
		t.Fatalf("err = %v, want ErrRollback", err)
	}
	if calls != 3 {
		t.Fatalf("rename called %d times, want 3 (aside, failed swap, rollback)", calls)
	}

	// The only thing that actually matters: a working binary is back where the
	// service manager will look for it, with its original contents.
	if got := string(mustRead(t, target)); got != "old binary v1" {
		t.Fatalf("restored binary = %q, want the original contents", got)
	}
	if _, statErr := os.Stat(in.previousPath()); statErr == nil {
		t.Fatal("the previous binary is stranded at .old instead of being restored")
	}
}

// When even the rollback fails there is nothing left to do but say so
// unmistakably, and name the file an operator has to move back by hand. A
// generic "update failed" here would send someone hunting for a binary that is
// sitting right next to where they are looking.
func TestUnrecoverableSwapSaysSoAndNamesTheFile(t *testing.T) {
	next := []byte("new binary v2")
	in, _ := installerFor(t, []byte("old binary v1"), next)
	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}

	calls := 0
	in.rename = func(oldpath, newpath string) error {
		calls++
		if calls == 1 {
			return os.Rename(oldpath, newpath) // target -> .old succeeds
		}
		return errors.New("simulated: the whole directory is locked")
	}

	err := in.Install()
	if !errors.Is(err, ErrBricked) {
		t.Fatalf("err = %v, want ErrBricked", err)
	}
	if !strings.Contains(err.Error(), in.previousPath()) {
		t.Fatalf("error does not name the file to restore by hand: %v", err)
	}
	if !strings.Contains(err.Error(), in.Target) {
		t.Fatalf("error does not name the destination: %v", err)
	}

	// And the old binary really is where the message says it is.
	if got := string(mustRead(t, in.previousPath())); got != "old binary v1" {
		t.Fatalf(".old = %q, want the original binary", got)
	}
}

// The safe failure: if the current binary cannot even be moved aside, nothing
// has changed and the node keeps running exactly as it was.
func TestAFailureBeforeTheWindowLeavesTheBinaryUntouched(t *testing.T) {
	next := []byte("new binary v2")
	in, target := installerFor(t, []byte("old binary v1"), next)
	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}
	in.rename = func(oldpath, newpath string) error {
		return errors.New("simulated: cannot move the running binary aside")
	}

	err := in.Install()
	if err == nil {
		t.Fatal("install reported success")
	}
	if errors.Is(err, ErrBricked) || errors.Is(err, ErrRollback) {
		t.Fatalf("a failure before the window was reported as a swap failure: %v", err)
	}
	if got := string(mustRead(t, target)); got != "old binary v1" {
		t.Fatalf("target = %q, want untouched", got)
	}
	// The download survives so the next attempt need not re-fetch it.
	if _, statErr := os.Stat(in.stagedPath()); statErr != nil {
		t.Fatalf("staged artifact discarded after a recoverable failure: %v", statErr)
	}
}

func TestInstallOverNoExistingBinaryIsAFreshInstallNotAFailure(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "meshd.exe")
	next := []byte("first ever binary")
	in := &Installer{Target: target, Artifact: stagedArtifact(next)}

	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}
	if err := in.Install(); err != nil {
		t.Fatalf("fresh install: %v", err)
	}
	if got := string(mustRead(t, target)); got != string(next) {
		t.Fatalf("target = %q, want %q", got, next)
	}
}

// An .old left by an interrupted earlier update must not block the next one --
// otherwise a single crash pins the node on its current version permanently,
// which is the update-freeze outcome arrived at by accident.
func TestALeftoverPreviousBinaryDoesNotBlockTheNextUpdate(t *testing.T) {
	next := []byte("new binary v3")
	in, target := installerFor(t, []byte("old binary v2"), next)
	if err := os.WriteFile(in.previousPath(), []byte("stale v1"), 0o755); err != nil {
		t.Fatalf("seed stale: %v", err)
	}

	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}
	if err := in.Install(); err != nil {
		t.Fatalf("install blocked by a leftover .old: %v", err)
	}
	if got := string(mustRead(t, target)); got != string(next) {
		t.Fatalf("target = %q, want %q", got, next)
	}
}

// Likewise a staged file from an abandoned attempt must be replaced, not
// appended to or silently reused.
func TestALeftoverStagedFileIsReplaced(t *testing.T) {
	next := []byte("new binary v3")
	in, target := installerFor(t, []byte("old"), next)
	if err := os.WriteFile(in.stagedPath(), []byte("abandoned attempt"), 0o755); err != nil {
		t.Fatalf("seed stale staged: %v", err)
	}

	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}
	if err := in.Install(); err != nil {
		t.Fatalf("install: %v", err)
	}
	if got := string(mustRead(t, target)); got != string(next) {
		t.Fatalf("target = %q, want %q", got, next)
	}
}

func TestCleanupAndDiscardAreIdempotent(t *testing.T) {
	in, _ := installerFor(t, []byte("old"), []byte("new"))
	for i := 0; i < 2; i++ {
		if err := in.CleanupPrevious(); err != nil {
			t.Fatalf("CleanupPrevious #%d: %v", i, err)
		}
		if err := in.DiscardStaged(); err != nil {
			t.Fatalf("DiscardStaged #%d: %v", i, err)
		}
	}
}

func TestDiscardStagedRemovesADownloadWeDecidedNotToInstall(t *testing.T) {
	next := []byte("new binary")
	in, target := installerFor(t, []byte("old"), next)
	if err := in.Stage(bytes.NewReader(next)); err != nil {
		t.Fatalf("stage: %v", err)
	}
	if err := in.DiscardStaged(); err != nil {
		t.Fatalf("discard: %v", err)
	}
	if err := in.Install(); !errors.Is(err, ErrNotStaged) {
		t.Fatalf("a discarded download was still installable: %v", err)
	}
	if got := string(mustRead(t, target)); got != "old" {
		t.Fatalf("target = %q, want untouched", got)
	}
}

// End to end: verify a signed manifest, then apply it. This is the path the
// agent will actually take, and it is worth pinning as one flow because the two
// halves are only useful joined -- a verified manifest that never reaches the
// installer, or an installer fed an unverified artifact, are both silent
// failures of the whole feature.
func TestVerifiedManifestDrivesTheInstall(t *testing.T) {
	pub, priv := newKey(t)
	body := []byte("meshd v2.0.0 payload")

	a := stagedArtifact(body)
	a.URL = "https://releases.example/meshd"
	sm := signed(t, Manifest{Version: "2.0.0", Artifacts: []Artifact{*a}}, priv)

	_, artifact, err := Verify(sm, TrustedKeys{KeyIDFor(pub): pub}, runtime.GOOS, runtime.GOARCH, "1.9.0")
	if err != nil {
		t.Fatalf("verify: %v", err)
	}

	dir := t.TempDir()
	target := filepath.Join(dir, "meshd.exe")
	if err := os.WriteFile(target, []byte("meshd v1.9.0 payload"), 0o755); err != nil {
		t.Fatalf("seed: %v", err)
	}
	in := &Installer{Target: target, Artifact: artifact}

	if err := in.Stage(bytes.NewReader(body)); err != nil {
		t.Fatalf("stage: %v", err)
	}
	if err := in.Install(); err != nil {
		t.Fatalf("install: %v", err)
	}
	if got := string(mustRead(t, target)); got != string(body) {
		t.Fatalf("target = %q, want %q", got, body)
	}
}
