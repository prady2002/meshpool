package update

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"runtime"
	"time"
)

// Checker is the client half: fetch a manifest, authenticate it, download the
// artifact it names, and stage it for the next restart.
//
// It never installs on its own. Choosing the moment to swap a binary is a
// policy decision belonging to whoever knows what the node is doing -- a
// contributor mid-inference-stage should not have their process replaced
// underneath a customer's request -- so this type stops at "a verified build is
// staged and ready" and lets the caller pick the moment.
type Checker struct {
	ManifestURL    string
	Trusted        TrustedKeys
	CurrentVersion string
	Target         string
	Client         *http.Client

	// MaxManifestBytes bounds the metadata fetch. Zero means the default. The
	// artifact is bounded by its signed size, but the manifest is read *before*
	// anything about it is trusted, so it needs a bound of its own -- otherwise
	// the very first byte of an update check is an unbounded read from a host we
	// have not authenticated yet.
	MaxManifestBytes int64

	// GOOS and GOARCH select the artifact. Empty means this build's own, which
	// is what production always wants; they exist so tests do not have to build
	// a different manifest on every platform they run on.
	GOOS   string
	GOARCH string
}

func (c *Checker) platform() (string, string) {
	goos, goarch := c.GOOS, c.GOARCH
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	return goos, goarch
}

const defaultMaxManifestBytes = 1 << 20

// ErrNoUpdate means the check succeeded and there is nothing newer. It is a
// normal outcome, kept distinct from every failure so a routine "you are up to
// date" can never be logged as a problem, and so a real fault can never hide
// among thousands of no-op checks.
var ErrNoUpdate = errors.New("update: already running the newest release")

// ErrInsecureURL rejects plaintext transport. The artifact is hash-verified, so
// this is not what makes the update safe -- but shipping a build URL over http
// leaks which version every contributor runs, which is a map of who is still
// vulnerable to what.
var ErrInsecureURL = errors.New("update: refusing a non-https URL")

func (c *Checker) httpClient() *http.Client {
	if c.Client != nil {
		return c.Client
	}
	return &http.Client{Timeout: 60 * time.Second}
}

func requireHTTPS(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrMalformed, err)
	}
	if u.Scheme != "https" {
		return fmt.Errorf("%w: %s", ErrInsecureURL, raw)
	}
	return nil
}

// CheckAndStage runs the whole client flow and reports the version staged.
//
// Every failure leaves the running agent exactly as it was. This is the single
// most important property of the function: an update check runs unattended on
// someone else's computer, so a broken release host, a truncated download or a
// hostile manifest must all degrade to "nothing happened", never to "the node
// stopped working".
func (c *Checker) CheckAndStage(ctx context.Context, allowInsecure bool) (string, error) {
	if !allowInsecure {
		if err := requireHTTPS(c.ManifestURL); err != nil {
			return "", err
		}
	}

	sm, err := c.fetchManifest(ctx)
	if err != nil {
		return "", err
	}

	goos, goarch := c.platform()
	m, artifact, err := Verify(*sm, c.Trusted, goos, goarch, c.CurrentVersion)
	if err != nil {
		if errors.Is(err, ErrDowngrade) {
			// Not newer is the overwhelmingly common case, and it is not a fault.
			return "", ErrNoUpdate
		}
		return "", err
	}

	if !allowInsecure {
		if err := requireHTTPS(artifact.URL); err != nil {
			return "", err
		}
	}

	body, err := c.openArtifact(ctx, artifact.URL)
	if err != nil {
		return "", err
	}
	defer body.Close()

	in := &Installer{Target: c.Target, Artifact: artifact}
	if err := in.Stage(body); err != nil {
		return "", err
	}
	return m.Version, nil
}

func (c *Checker) fetchManifest(ctx context.Context) (*SignedManifest, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.ManifestURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("update: manifest fetch returned %s", resp.Status)
	}

	max := c.MaxManifestBytes
	if max <= 0 {
		max = defaultMaxManifestBytes
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, max+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > max {
		return nil, fmt.Errorf("%w: manifest exceeds %d bytes", ErrMalformed, max)
	}

	var sm SignedManifest
	if err := sm.UnmarshalJSON(raw); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrMalformed, err)
	}
	return &sm, nil
}

func (c *Checker) openArtifact(ctx context.Context, rawURL string) (io.ReadCloser, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("update: artifact fetch returned %s", resp.Status)
	}
	// The body is streamed rather than buffered and is bounded by Stage against
	// the signed size, so a lying Content-Length changes nothing.
	return resp.Body, nil
}
