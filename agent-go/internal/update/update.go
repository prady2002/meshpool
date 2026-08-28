// Package update verifies signed release manifests before an agent replaces its
// own binary.
//
// An auto-updating fleet is the highest-value attack surface this project has.
// Every contributor runs meshd on a machine they own, and an update channel that
// merely fetches over TLS is a fleet-wide remote code execution primitive the
// moment anyone controls the release host, a CDN edge, a mirror, or a stale DNS
// record. TLS authenticates the *connection*, not the artifact, so it cannot be
// the last line of defence.
//
// So the artifact is authenticated directly. The release manifest is signed with
// an Ed25519 release key whose public half is compiled into the binary, and the
// binary hash is bound by that same signature. An attacker who fully controls the
// network and the download host still cannot get code executed without the
// private key.
//
// Four rejections carry the weight, and three of them are the ones that get
// forgotten:
//
//   - a tampered artifact, caught by hash;
//   - a tampered manifest, caught by signature;
//   - a manifest validly signed by the *wrong* key, which is what happens when a
//     test key, a rotated-out key, or an attacker's key reaches production;
//   - a *downgrade* -- an older release, correctly signed, replayed by an
//     attacker to reintroduce a vulnerability that has since been patched. This
//     one passes every signature and hash check by construction, which is exactly
//     why it needs its own rule.
package update

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Errors are distinct values so callers, tests and operators can tell a hostile
// update apart from a merely absent one. Collapsing these into a single "update
// failed" is how a live attack gets logged as a network blip.
var (
	ErrBadSignature = errors.New("update: manifest signature is not valid for a trusted release key")
	ErrUnknownKey   = errors.New("update: no trusted release keys are compiled into this build")
	ErrHashMismatch = errors.New("update: artifact hash does not match the signed manifest")
	ErrDowngrade    = errors.New("update: refusing to install an older or equal version")
	ErrNoArtifact   = errors.New("update: manifest has no artifact for this platform")
	ErrMalformed    = errors.New("update: manifest is malformed")
)

// Artifact is one downloadable build. Size is signed alongside the hash so an
// oversized body is rejected before it is ever digested.
type Artifact struct {
	OS     string `json:"os"`
	Arch   string `json:"arch"`
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// Manifest is the signed payload. Everything a client acts on must live in here,
// because anything outside it is unauthenticated by definition.
type Manifest struct {
	Version   string     `json:"version"`
	ReleasedM int64      `json:"released_ms"`
	Artifacts []Artifact `json:"artifacts"`
}

// SignedManifest is the wire format: the exact payload bytes that were signed,
// plus the signature and the id of the key that made it.
//
// Payload is carried as raw bytes rather than as a decoded struct on purpose.
// Re-serialising a decoded struct to check a signature means verifying bytes the
// sender never sent, and any difference in key order, whitespace or unknown-field
// handling between signer and verifier silently becomes either a signature bypass
// or a spurious failure.
type SignedManifest struct {
	Payload   []byte
	Signature []byte
	KeyID     string
}

// TrustedKeys maps a key id to its Ed25519 public key, compiled into the binary.
// A map rather than a single key so the release key can be rotated: builds ship
// trusting both old and new for one cycle, then drop the old.
type TrustedKeys map[string]ed25519.PublicKey

// KeyIDFor derives a short, stable id from a public key so a manifest can say
// which key signed it. It is a performance hint and a diagnostic aid only: the
// field travels outside the signature, so it is never an authentication input
// and never a filter. See the reordering note in Verify.
func KeyIDFor(pub ed25519.PublicKey) string {
	sum := sha256.Sum256(pub)
	return hex.EncodeToString(sum[:8])
}

// Verify authenticates a manifest and selects the artifact for this platform.
//
// currentVersion is the running binary's version; an update must be strictly
// newer. Nothing inside the payload is read until the signature has been checked.
func Verify(sm SignedManifest, trusted TrustedKeys, goos, goarch, currentVersion string) (*Manifest, *Artifact, error) {
	if len(trusted) == 0 {
		return nil, nil, ErrUnknownKey
	}
	if len(sm.Signature) != ed25519.SignatureSize {
		return nil, nil, ErrBadSignature
	}

	// KeyID sits *outside* the signed payload, so it is unauthenticated: an
	// attacker can rewrite it at will without invalidating anything. It may
	// therefore reorder our work but must never change the outcome.
	//
	// The tempting version of this code trusts the hint and tries only the key it
	// names. That is not a signature bypass -- an unknown key still fails -- but
	// it is an availability attack, which here is a security attack: flipping one
	// unauthenticated byte to some *other* trusted key id makes every genuine
	// update fail verification, freezing the fleet on whatever version it is
	// running. Pinning a fleet to a known-vulnerable build is precisely the
	// outcome ErrDowngrade exists to prevent, reached by another road.
	//
	// So the hint only moves a key to the front of the queue.
	candidates := make([]ed25519.PublicKey, 0, len(trusted))
	if pub, ok := trusted[sm.KeyID]; ok {
		candidates = append(candidates, pub)
	}
	for id, pub := range trusted {
		if id != sm.KeyID {
			candidates = append(candidates, pub)
		}
	}

	verified := false
	for _, pub := range candidates {
		if ed25519.Verify(pub, sm.Payload, sm.Signature) {
			verified = true
			break
		}
	}
	if !verified {
		return nil, nil, ErrBadSignature
	}

	// Only now is it safe to look at the contents.
	dec := json.NewDecoder(strings.NewReader(string(sm.Payload)))
	dec.DisallowUnknownFields()
	var m Manifest
	if err := dec.Decode(&m); err != nil {
		return nil, nil, fmt.Errorf("%w: %v", ErrMalformed, err)
	}
	if m.Version == "" {
		return nil, nil, fmt.Errorf("%w: empty version", ErrMalformed)
	}

	cmp, err := CompareVersions(m.Version, currentVersion)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %v", ErrMalformed, err)
	}
	if cmp <= 0 {
		return nil, nil, fmt.Errorf("%w: manifest %s is not newer than running %s",
			ErrDowngrade, m.Version, currentVersion)
	}

	for i := range m.Artifacts {
		a := m.Artifacts[i]
		if a.OS == goos && a.Arch == goarch {
			if len(a.SHA256) != 64 {
				return nil, nil, fmt.Errorf("%w: artifact sha256 is not 32 hex-encoded bytes", ErrMalformed)
			}
			if a.Size <= 0 {
				return nil, nil, fmt.Errorf("%w: artifact size must be positive", ErrMalformed)
			}
			return &m, &a, nil
		}
	}
	return nil, nil, ErrNoArtifact
}

// CheckArtifact verifies downloaded bytes against the signed artifact entry.
func CheckArtifact(a *Artifact, data []byte) error {
	if a == nil {
		return ErrNoArtifact
	}
	if int64(len(data)) != a.Size {
		return fmt.Errorf("%w: got %d bytes, manifest says %d", ErrHashMismatch, len(data), a.Size)
	}
	sum := sha256.Sum256(data)
	got := hex.EncodeToString(sum[:])
	if !strings.EqualFold(got, a.SHA256) {
		return fmt.Errorf("%w: got %s, manifest says %s", ErrHashMismatch, got, a.SHA256)
	}
	return nil
}

// Sign produces a SignedManifest over the canonical encoding of m.
func Sign(m Manifest, priv ed25519.PrivateKey) (SignedManifest, error) {
	payload, err := json.Marshal(m)
	if err != nil {
		return SignedManifest{}, err
	}
	pub, ok := priv.Public().(ed25519.PublicKey)
	if !ok {
		return SignedManifest{}, errors.New("update: private key has no ed25519 public half")
	}
	return SignedManifest{
		Payload:   payload,
		Signature: ed25519.Sign(priv, payload),
		KeyID:     KeyIDFor(pub),
	}, nil
}

// wire is the JSON envelope actually transmitted.
type wire struct {
	Payload   string `json:"payload"`   // base64 of the exact signed bytes
	Signature string `json:"signature"` // base64
	KeyID     string `json:"key_id"`
}

func (sm SignedManifest) MarshalJSON() ([]byte, error) {
	return json.Marshal(wire{
		Payload:   base64.StdEncoding.EncodeToString(sm.Payload),
		Signature: base64.StdEncoding.EncodeToString(sm.Signature),
		KeyID:     sm.KeyID,
	})
}

func (sm *SignedManifest) UnmarshalJSON(b []byte) error {
	var w wire
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	payload, err := base64.StdEncoding.DecodeString(w.Payload)
	if err != nil {
		return fmt.Errorf("%w: payload is not base64", ErrMalformed)
	}
	sig, err := base64.StdEncoding.DecodeString(w.Signature)
	if err != nil {
		return fmt.Errorf("%w: signature is not base64", ErrMalformed)
	}
	sm.Payload, sm.Signature, sm.KeyID = payload, sig, w.KeyID
	return nil
}

// CompareVersions orders dotted numeric versions ("1.2.3"), tolerating a leading
// "v" and a pre-release suffix ("1.2.3-rc1").
//
// A pre-release sorts *before* its release, so 1.2.3-rc1 < 1.2.3. Without that
// rule an rc build would count as newer than the final release it precedes, and
// the fleet would be pinned to a candidate build forever.
func CompareVersions(a, b string) (int, error) {
	an, apre, err := parseVersion(a)
	if err != nil {
		return 0, err
	}
	bn, bpre, err := parseVersion(b)
	if err != nil {
		return 0, err
	}
	for i := 0; i < len(an) || i < len(bn); i++ {
		var x, y int
		if i < len(an) {
			x = an[i]
		}
		if i < len(bn) {
			y = bn[i]
		}
		if x != y {
			if x < y {
				return -1, nil
			}
			return 1, nil
		}
	}
	switch {
	case apre == "" && bpre == "":
		return 0, nil
	case apre == "":
		return 1, nil // a is the release, b is a pre-release of it
	case bpre == "":
		return -1, nil
	case apre < bpre:
		return -1, nil
	case apre > bpre:
		return 1, nil
	}
	return 0, nil
}

func parseVersion(v string) ([]int, string, error) {
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	pre := ""
	if i := strings.IndexAny(v, "-+"); i >= 0 {
		pre, v = v[i+1:], v[:i]
	}
	if v == "" {
		return nil, "", errors.New("empty version")
	}
	parts := strings.Split(v, ".")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, "", fmt.Errorf("version %q has non-numeric component %q", v, p)
		}
		if n < 0 {
			return nil, "", fmt.Errorf("version %q has negative component", v)
		}
		out = append(out, n)
	}
	return out, pre, nil
}

// SortArtifacts gives a manifest a stable artifact order, so two builds of the
// same release produce byte-identical signed payloads.
func SortArtifacts(as []Artifact) {
	sort.Slice(as, func(i, j int) bool {
		if as[i].OS != as[j].OS {
			return as[i].OS < as[j].OS
		}
		return as[i].Arch < as[j].Arch
	})
}
