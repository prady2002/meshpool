package main

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"meshd/internal/distribution"
)

func main() {
	version := flag.String("version", "", "release version, for example 1.2.3")
	commit := flag.String("commit", "", "git commit to stamp into binaries")
	baseURL := flag.String("base-url", "", "release URL prefix")
	outDir := flag.String("out", "dist", "output directory")
	cert := flag.String("sign-cert", os.Getenv("MESHPOOL_INSTALLER_SIGN_CERT"), "installer signing certificate")
	key := flag.String("sign-key", os.Getenv("MESHPOOL_INSTALLER_SIGN_KEY"), "installer signing key path")
	flag.Parse()
	if *version == "" || *commit == "" || *baseURL == "" {
		log.Fatal("-version, -commit and -base-url are required")
	}
	plan := distribution.ReleasePlan(*version, *commit, *baseURL)
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatal(err)
	}
	for _, a := range plan.Artifacts {
		if err := buildOne(*outDir, a); err != nil {
			log.Fatal(err)
		}
		unsigned, err := packageOne(*outDir, a)
		if err != nil {
			log.Fatal(err)
		}
		signed := strings.TrimSuffix(unsigned, filepath.Ext(unsigned)) + ".signed" + filepath.Ext(unsigned)
		if err := distribution.SignInstaller(distribution.SignRequest{
			Platform:     distribution.Platform(a.OS),
			UnsignedPath: unsigned,
			SignedPath:   signed,
			Certificate:  *cert,
			KeyPath:      *key,
		}); err != nil {
			log.Fatalf("sign %s: %v", unsigned, err)
		}
	}
	index := filepath.Join(*outDir, "release-plan.json")
	body, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(index, body, 0o644); err != nil {
		log.Fatal(err)
	}
}

func buildOne(outDir string, a distribution.ReleaseArtifact) error {
	args := append([]string{}, a.BuildArgs...)
	for i, arg := range args {
		if arg == filepath.Join("dist", a.FileName) {
			args[i] = filepath.Join(outDir, a.FileName)
		}
	}
	cmd := exec.Command("go", args...)
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS="+a.OS, "GOARCH="+a.Arch)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func packageOne(outDir string, a distribution.ReleaseArtifact) (string, error) {
	binary := filepath.Join(outDir, a.FileName)
	switch a.OS {
	case "windows":
		return binary, nil
	case "darwin":
		pkg := filepath.Join(outDir, strings.TrimSuffix(a.FileName, filepath.Ext(a.FileName))+".pkg.tar.gz")
		return pkg, tarGz(pkg, binary, "usr/local/bin/meshd")
	default:
		pkg := filepath.Join(outDir, strings.TrimSuffix(a.FileName, filepath.Ext(a.FileName))+".tar.gz")
		return pkg, tarGz(pkg, binary, "usr/local/bin/meshd")
	}
}

func tarGz(dst, src, name string) error {
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	gw := gzip.NewWriter(out)
	defer gw.Close()
	tw := tar.NewWriter(gw)
	defer tw.Close()
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	hdr := &tar.Header{Name: name, Mode: 0o755, Size: info.Size()}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if _, err := io.Copy(tw, in); err != nil {
		return err
	}
	if runtime.GOOS == "windows" {
		fmt.Fprintln(os.Stderr, "packaged tarball on Windows; notarization/signing still runs separately")
	}
	return nil
}
