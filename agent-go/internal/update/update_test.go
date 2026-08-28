package update

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"testing"
)

// The threat model these tests encode: an attacker fully controls the network
// and the download host. They can substitute any bytes, replay any bytes they
// have ever seen, and sign anything they like with keys they own. The only thing
// they lack is the release private key.

func newKey(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	return pub, priv
}

func artifactFor(body []byte) Artifact {
	sum := sha256.Sum256(body)
	return Artifact{
		OS:     "windows",
		Arch:   "amd64",
		URL:    "https://releases.example/meshd-windows-amd64.exe",
		SHA256: hex.EncodeToString(sum[:]),
		Size:   int64(len(body)),
	}
}

func signed(t *testing.T, m Manifest, priv ed25519.PrivateKey) SignedManifest {
	t.Helper()
	sm, err := Sign(m, priv)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	return sm
}

func TestHappyPathAcceptsAGenuineNewerRelease(t *testing.T) {
	pub, priv := newKey(t)
	body := []byte("a plausible binary")
	sm := signed(t, Manifest{Version: "1.4.0", Artifacts: []Artifact{artifactFor(body)}}, priv)

	m, a, err := Verify(sm, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.3.9")
	if err != nil {
		t.Fatalf("genuine manifest rejected: %v", err)
	}
	if m.Version != "1.4.0" {
		t.Fatalf("version = %q, want 1.4.0", m.Version)
	}
	if err := CheckArtifact(a, body); err != nil {
		t.Fatalf("genuine artifact rejected: %v", err)
	}
}

// Attack 1: swap the binary, leave the signed manifest alone.
func TestTamperedArtifactIsRejected(t *testing.T) {
	pub, priv := newKey(t)
	body := []byte("a plausible binary")
	sm := signed(t, Manifest{Version: "1.4.0", Artifacts: []Artifact{artifactFor(body)}}, priv)

	_, a, err := Verify(sm, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.3.9")
	if err != nil {
		t.Fatalf("setup: %v", err)
	}

	// Same length, one byte different: the length check must not be what saves us.
	evil := []byte("a plausible binafy")
	if len(evil) != len(body) {
		t.Fatalf("test bug: substituted body must be the same length to exercise the hash")
	}
	if err := CheckArtifact(a, evil); !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("tampered artifact accepted, err = %v, want ErrHashMismatch", err)
	}
}

// Attack 2: edit the manifest so it points at the attacker's binary.
func TestTamperedManifestIsRejected(t *testing.T) {
	pub, priv := newKey(t)
	sm := signed(t, Manifest{Version: "1.4.0", Artifacts: []Artifact{artifactFor([]byte("real"))}}, priv)

	var m Manifest
	if err := json.Unmarshal(sm.Payload, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	m.Artifacts[0].URL = "https://evil.example/payload.exe"
	m.Artifacts[0].SHA256 = hex.EncodeToString(func() []byte { s := sha256.Sum256([]byte("evil")); return s[:] }())
	repacked, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	forged := SignedManifest{Payload: repacked, Signature: sm.Signature, KeyID: sm.KeyID}
	if _, _, err := Verify(forged, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.3.9"); !errors.Is(err, ErrBadSignature) {
		t.Fatalf("tampered manifest accepted, err = %v, want ErrBadSignature", err)
	}
}

// Attack 3: a perfectly valid signature -- from a key we do not trust. This is
// what a leaked test key, a rotated-out key, or the attacker's own key looks
// like, and it is why "the signature verified" is never the whole question.
func TestValidSignatureFromUntrustedKeyIsRejected(t *testing.T) {
	trustedPub, _ := newKey(t)
	_, attackerPriv := newKey(t)

	sm := signed(t, Manifest{Version: "9.9.9", Artifacts: []Artifact{artifactFor([]byte("evil"))}}, attackerPriv)

	// The attacker also claims the trusted key's id, so a client that believed
	// KeyID would look up the right key and still have to do the real work.
	sm.KeyID = KeyIDFor(trustedPub)

	if _, _, err := Verify(sm, TrustedKeys{KeyIDFor(trustedPub): trustedPub}, "windows", "amd64", "1.3.9"); !errors.Is(err, ErrBadSignature) {
		t.Fatalf("untrusted signer accepted, err = %v, want ErrBadSignature", err)
	}
}

// Attack 4: the downgrade replay. Every byte is genuine and correctly signed --
// it is simply an *old* release, served to a patched fleet to reintroduce a
// vulnerability. Signature and hash checks pass by construction, so only an
// explicit version rule stops it.
func TestDowngradeReplayIsRejected(t *testing.T) {
	pub, priv := newKey(t)
	old := signed(t, Manifest{Version: "1.2.0", Artifacts: []Artifact{artifactFor([]byte("old binary"))}}, priv)

	trusted := TrustedKeys{KeyIDFor(pub): pub}

	// Sanity: the replayed manifest really is authentic, so the rejection below
	// can only come from the version rule.
	if !ed25519.Verify(pub, old.Payload, old.Signature) {
		t.Fatal("test bug: replayed manifest should carry a genuine signature")
	}

	if _, _, err := Verify(old, trusted, "windows", "amd64", "1.5.0"); !errors.Is(err, ErrDowngrade) {
		t.Fatalf("downgrade accepted, err = %v, want ErrDowngrade", err)
	}
}

// A re-install of the running version is also a replay: it lets an attacker
// force an unnecessary binary swap, and there is no legitimate reason to do it.
func TestSameVersionIsRejected(t *testing.T) {
	pub, priv := newKey(t)
	sm := signed(t, Manifest{Version: "1.5.0", Artifacts: []Artifact{artifactFor([]byte("x"))}}, priv)
	if _, _, err := Verify(sm, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.5.0"); !errors.Is(err, ErrDowngrade) {
		t.Fatalf("same-version replay accepted, err = %v, want ErrDowngrade", err)
	}
}

// Attack 5: the update freeze. KeyID travels outside the signature, so an
// attacker can rewrite it for free. If a client treated the hint as a filter,
// naming some *other* trusted key would make every genuine update fail -- no
// forged signature required -- pinning the fleet on whatever build it is running.
// That is a downgrade attack reached by denial rather than by replay, and it is
// invisible to every signature and hash test.
func TestRewritingTheKeyIDHintCannotBlockAGenuineUpdate(t *testing.T) {
	realPub, realPriv := newKey(t)
	decoyPub, _ := newKey(t)
	trusted := TrustedKeys{KeyIDFor(realPub): realPub, KeyIDFor(decoyPub): decoyPub}

	sm := signed(t, Manifest{Version: "2.0.0", Artifacts: []Artifact{artifactFor([]byte("x"))}}, realPriv)
	sm.KeyID = KeyIDFor(decoyPub) // the only thing the attacker touched

	m, _, err := Verify(sm, trusted, "windows", "amd64", "1.0.0")
	if err != nil {
		t.Fatalf("a genuine update was blocked by a rewritten KeyID hint: %v", err)
	}
	if m.Version != "2.0.0" {
		t.Fatalf("version = %q, want 2.0.0", m.Version)
	}

	// Non-vacuity: the hint must not have become a way *in* either. A garbage
	// signature is still rejected no matter which trusted key the hint names.
	sm.Signature = make([]byte, ed25519.SignatureSize)
	if _, _, err := Verify(sm, trusted, "windows", "amd64", "1.0.0"); !errors.Is(err, ErrBadSignature) {
		t.Fatalf("forgery accepted, err = %v, want ErrBadSignature", err)
	}
}

func TestEmptyTrustStoreRejectsEverything(t *testing.T) {
	_, priv := newKey(t)
	sm := signed(t, Manifest{Version: "9.0.0", Artifacts: []Artifact{artifactFor([]byte("x"))}}, priv)
	if _, _, err := Verify(sm, TrustedKeys{}, "windows", "amd64", "1.0.0"); !errors.Is(err, ErrUnknownKey) {
		t.Fatalf("empty trust store accepted an update, err = %v, want ErrUnknownKey", err)
	}
}

// Rotation: a build that trusts both the old and the new key must accept either,
// even when KeyID is wrong or missing, because KeyID is only ever a hint.
func TestRotationAcceptsEitherTrustedKeyRegardlessOfKeyIDHint(t *testing.T) {
	oldPub, _ := newKey(t)
	newPub, newPriv := newKey(t)
	trusted := TrustedKeys{KeyIDFor(oldPub): oldPub, KeyIDFor(newPub): newPub}

	sm := signed(t, Manifest{Version: "2.0.0", Artifacts: []Artifact{artifactFor([]byte("x"))}}, newPriv)
	sm.KeyID = "" // hint absent

	if _, _, err := Verify(sm, trusted, "windows", "amd64", "1.0.0"); err != nil {
		t.Fatalf("rotated key rejected with an absent hint: %v", err)
	}

	sm.KeyID = KeyIDFor(oldPub) // hint actively wrong
	if _, _, err := Verify(sm, trusted, "windows", "amd64", "1.0.0"); err != nil {
		t.Fatalf("rotated key rejected with a wrong hint: %v", err)
	}
}

func TestUnknownManifestFieldsAreRejected(t *testing.T) {
	pub, priv := newKey(t)
	payload := []byte(`{"version":"2.0.0","released_ms":0,"artifacts":[],"install_command":"calc.exe"}`)
	sm := SignedManifest{Payload: payload, Signature: ed25519.Sign(priv, payload), KeyID: KeyIDFor(pub)}

	if _, _, err := Verify(sm, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.0.0"); !errors.Is(err, ErrMalformed) {
		t.Fatalf("unknown field accepted, err = %v, want ErrMalformed", err)
	}
}

func TestMissingPlatformArtifactIsDistinctFromAnAttack(t *testing.T) {
	pub, priv := newKey(t)
	a := artifactFor([]byte("linux build"))
	a.OS, a.Arch = "linux", "arm64"
	sm := signed(t, Manifest{Version: "2.0.0", Artifacts: []Artifact{a}}, priv)

	if _, _, err := Verify(sm, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.0.0"); !errors.Is(err, ErrNoArtifact) {
		t.Fatalf("err = %v, want ErrNoArtifact", err)
	}
}

// Size is signed, so it is checked before the body is digested -- otherwise a
// hostile host can make a client hash an unbounded stream.
func TestOversizedBodyIsRejectedBeforeHashing(t *testing.T) {
	body := []byte("small")
	a := artifactFor(body)
	if err := CheckArtifact(&a, append(body, make([]byte, 4096)...)); !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("oversized body accepted, err = %v", err)
	}
}

func TestSignedManifestSurvivesAJSONRoundTripByteForByte(t *testing.T) {
	pub, priv := newKey(t)
	sm := signed(t, Manifest{Version: "3.1.4", Artifacts: []Artifact{artifactFor([]byte("x"))}}, priv)

	blob, err := json.Marshal(sm)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back SignedManifest
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(back.Payload) != string(sm.Payload) {
		t.Fatalf("payload changed across the wire:\n got %s\nwant %s", back.Payload, sm.Payload)
	}
	if _, _, err := Verify(back, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.0.0"); err != nil {
		t.Fatalf("round-tripped manifest failed verification: %v", err)
	}
}

func TestCompareVersions(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"1.0.0", "1.0.0", 0},
		{"1.0.1", "1.0.0", 1},
		{"1.0.0", "1.0.1", -1},
		{"1.10.0", "1.9.0", 1},    // not lexicographic
		{"2.0", "2.0.0", 0},       // missing components are zero
		{"v1.2.3", "1.2.3", 0},    // leading v is cosmetic
		{"1.2.3", "1.2.3-rc1", 1}, // a release outranks its own pre-release
		{"1.2.3-rc1", "1.2.3-rc2", -1},
		{"1.3.0-rc1", "1.2.9", 1},
	}
	for _, c := range cases {
		got, err := CompareVersions(c.a, c.b)
		if err != nil {
			t.Fatalf("CompareVersions(%q,%q): %v", c.a, c.b, err)
		}
		if got != c.want {
			t.Errorf("CompareVersions(%q,%q) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

func TestNonNumericVersionIsAnErrorNotAnUpgrade(t *testing.T) {
	pub, priv := newKey(t)
	sm := signed(t, Manifest{Version: "latest", Artifacts: []Artifact{artifactFor([]byte("x"))}}, priv)
	if _, _, err := Verify(sm, TrustedKeys{KeyIDFor(pub): pub}, "windows", "amd64", "1.0.0"); !errors.Is(err, ErrMalformed) {
		t.Fatalf("unparseable version accepted, err = %v, want ErrMalformed", err)
	}
}
