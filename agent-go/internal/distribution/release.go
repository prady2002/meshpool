package distribution

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var ErrSigningUnavailable = errors.New("distribution: installer signing certificate is unavailable")

type ReleaseArtifact struct {
	OS        string
	Arch      string
	FileName  string
	URL       string
	BuildArgs []string
}

type Release struct {
	Version   string
	Commit    string
	Artifacts []ReleaseArtifact
}

func ReleasePlan(version, commit, baseURL string) Release {
	platforms := []struct{ goos, goarch string }{
		{"windows", "amd64"},
		{"windows", "arm64"},
		{"darwin", "amd64"},
		{"darwin", "arm64"},
		{"linux", "amd64"},
		{"linux", "arm64"},
	}
	arts := make([]ReleaseArtifact, 0, len(platforms))
	for _, p := range platforms {
		name := fmt.Sprintf("meshd-%s-%s", p.goos, p.goarch)
		if p.goos == "windows" {
			name += ".exe"
		}
		args := []string{
			"build",
			"-trimpath",
			"-ldflags",
			fmt.Sprintf("-s -w -buildid= -X meshd/internal/update.Version=%s -X main.commit=%s", version, commit),
			"-o",
			filepath.Join("dist", name),
			"./cmd/meshd",
		}
		arts = append(arts, ReleaseArtifact{
			OS:        p.goos,
			Arch:      p.goarch,
			FileName:  name,
			URL:       strings.TrimRight(baseURL, "/") + "/" + name,
			BuildArgs: args,
		})
	}
	return Release{Version: version, Commit: commit, Artifacts: arts}
}

type SignRequest struct {
	Platform     Platform
	UnsignedPath string
	SignedPath   string
	Certificate  string
	KeyPath      string
}

func SignInstaller(req SignRequest) error {
	if req.Certificate == "" && req.KeyPath == "" {
		return ErrSigningUnavailable
	}
	if req.SignedPath == "" {
		return errors.New("distribution: signed output path is required")
	}
	body, err := os.ReadFile(req.UnsignedPath)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(append([]byte(req.Certificate+req.KeyPath+string(req.Platform)), body...))
	signed := append([]byte{}, body...)
	signed = append(signed, []byte("\nMESHPOOL-INSTALLER-SIGNATURE-SHA256:"+hex.EncodeToString(sum[:])+"\n")...)
	return os.WriteFile(req.SignedPath, signed, 0o644)
}
