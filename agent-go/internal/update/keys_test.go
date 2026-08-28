package update

import (
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"testing"
)

// The compiled-in trust store is the root of trust for the whole update path.
// These tests exist so that changing it is a deliberate act with a visible cost,
// not something that happens in passing during a refactor.

func TestADefaultBuildCannotBeUpdatedByAnybody(t *testing.T) {
	trusted := TrustedRelease()
	if len(trusted) != 0 {
		t.Fatalf("the release trust store is no longer empty (%d keys). If a real "+
			"MeshPool signing key now exists this test must be rewritten to assert "+
			"the expected key ids -- but if this changed by accident, an update "+
			"channel just acquired a root of trust nobody reviewed.", len(trusted))
	}

	// The consequence that actually matters: with no key, no manifest is
	// installable, however well formed. Anyone can sign one; nobody is believed.
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	sm := signed(t, Manifest{Version: "99.0.0", Artifacts: []Artifact{artifactFor([]byte("x"))}}, priv)

	if _, _, err := Verify(sm, trusted, "windows", "amd64", Version); !errors.Is(err, ErrUnknownKey) {
		t.Fatalf("err = %v, want ErrUnknownKey -- a build with no release key "+
			"must refuse every update rather than fall through to accepting them", err)
	}
}

// Returning the live map would let anything in the process add a key at runtime,
// which defeats the entire reason for compiling the trust store in: it must not
// be reachable by anything the network can influence.
func TestTrustedReleaseCannotBeMutatedByItsCaller(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	got := TrustedRelease()
	got["attacker"] = pub

	if len(TrustedRelease()) != 0 {
		t.Fatal("a caller injected a key into the compiled-in root of trust")
	}
}

// A developer build must sort below every real release, so a locally built agent
// pointed at a live channel offers to update rather than declining because the
// default version looked newer than production.
func TestTheDevelopmentVersionIsOlderThanAnyRelease(t *testing.T) {
	for _, release := range []string{"0.0.1", "0.1.0", "1.0.0", "2.4.7"} {
		cmp, err := CompareVersions(Version, release)
		if err != nil {
			t.Fatalf("CompareVersions(%q, %q): %v", Version, release, err)
		}
		if cmp >= 0 {
			t.Errorf("default Version %q is not older than release %q", Version, release)
		}
	}
}
