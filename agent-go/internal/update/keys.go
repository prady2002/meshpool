package update

import "crypto/ed25519"

// Version is the release this binary claims to be, and it is the floor every
// manifest must beat. It is a var rather than a const so a release build can set
// it with -ldflags "-X meshd/internal/update.Version=1.2.3" instead of anyone
// having to remember to edit a file before tagging.
//
// The default is deliberately 0.0.0-dev: a developer build is older than every
// real release, so a locally built agent pointed at a live update channel offers
// to replace itself rather than silently declining because a hand-edited
// constant happened to look newer than production.
var Version = "0.0.0-dev"

// releaseKeys is the compiled-in root of trust for auto-updates.
//
// It is EMPTY, and that is the current, honest state of this feature: no
// MeshPool release signing key exists yet, so there is nothing legitimate to
// trust. An empty trust store makes Verify return ErrUnknownKey for every
// manifest, so a build produced today cannot be made to update itself by
// anybody -- including us.
//
// Failing closed is the only defensible default here. The alternative shapes are
// both worse than useless:
//
//   - a placeholder key nobody holds the private half of is a lie that looks
//     like security, and it survives review precisely because the code around it
//     is correct;
//   - "no keys configured, so skip verification" turns the absence of security
//     into the absence of a check, which is how an update channel becomes a
//     fleet-wide remote code execution primitive.
//
// Populating this map is a release-engineering task -- generate the key in an
// offline HSM, publish the public half, add it here -- not a code change to be
// done casually. TrustedRelease() is pinned by a test that asserts a default
// build refuses every update, so the day this stops being empty, that test has
// to be updated deliberately rather than by accident.
var releaseKeys = TrustedKeys{}

// TrustedRelease returns a copy of the compiled-in release trust store. A copy,
// so that nothing at runtime can add a key to the root of trust -- the whole
// point of compiling it in is that it is not reachable by anything the network
// can influence.
func TrustedRelease() TrustedKeys {
	out := make(TrustedKeys, len(releaseKeys))
	for id, pub := range releaseKeys {
		cp := make(ed25519.PublicKey, len(pub))
		copy(cp, pub)
		out[id] = cp
	}
	return out
}
