package update

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

// Applying an update is where a verified download still turns into a bricked
// install. The failure that matters is not "the update didn't happen" -- it is
// "the update half happened", leaving a contributor's machine with no working
// binary and no way to be told so, because the thing that would have told them
// is the binary. A node that fails to update is a nuisance; a node that cannot
// start is gone forever, and nobody debugs someone else's dead daemon.
//
// So the swap is built around one rule: at every instant between the first
// syscall and the last, some complete, verified binary must exist at a path the
// service manager will start. Never a truncated one, never none.
//
// Windows forces the shape of the solution. A running .exe cannot be deleted or
// overwritten -- but it CAN be renamed, because the lock is on the file's data,
// not on its directory entry. That gives the same three-rename dance that works
// on Unix, where renaming over a running binary is fine anyway since the old
// inode stays alive until the process exits:
//
//	 1. stage:    write bytes to <target>.new, fsync, verify hash
//	 2. displace: rename <target>      -> <target>.old
//	 3. install:  rename <target>.new  -> <target>
//	 4. later:    delete <target>.old on next start
//
// Steps 2 and 3 are the dangerous window. If 3 fails, step 2 has already moved
// the only working binary out of the way, so the recovery is mandatory rather
// than defensive: rename .old back and report the original error.

var (
	ErrNotStaged     = errors.New("update: no staged artifact to install")
	ErrStagingFailed = errors.New("update: could not stage the new binary")
	ErrRollback      = errors.New("update: install failed and the previous binary was restored")
	ErrBricked       = errors.New("update: install failed AND the previous binary could not be restored")
)

// Installer performs verified, recoverable binary replacement.
type Installer struct {
	// Target is the path being replaced, normally os.Executable().
	Target string
	// Artifact is the signed entry the staged bytes must match. Staging without
	// one is refused: verification after the swap is not verification.
	Artifact *Artifact

	// rename exists so the rollback path can be tested. That path is only ever
	// reached when a rename fails between the two swaps, which no portable test
	// can arrange -- the natural attempts either fail before the window opens or
	// get renamed aside themselves. Leaving it untested would mean the one branch
	// that decides whether a contributor's machine still has a binary is the one
	// branch that has never executed, which is how recovery code is always
	// discovered to be broken: at the only moment it was needed. Nil means
	// os.Rename.
	rename func(oldpath, newpath string) error
}

func (in *Installer) doRename(oldpath, newpath string) error {
	if in.rename != nil {
		return in.rename(oldpath, newpath)
	}
	return os.Rename(oldpath, newpath)
}

func (in *Installer) stagedPath() string   { return in.Target + ".new" }
func (in *Installer) previousPath() string { return in.Target + ".old" }

// Stage writes and verifies the downloaded bytes beside the target.
//
// It streams rather than buffering because a meshd build is tens of megabytes
// and a contributor's machine may be a 4 GB laptop already running the workload
// we sent it. The hash is computed over what actually reached the disk, not over
// what we believe we wrote.
func (in *Installer) Stage(r io.Reader) error {
	if in.Artifact == nil {
		return fmt.Errorf("%w: refusing to stage bytes with no signed artifact to check them against", ErrStagingFailed)
	}
	if in.Target == "" {
		return fmt.Errorf("%w: no target path", ErrStagingFailed)
	}

	dir := filepath.Dir(in.Target)
	// Stage in the target's own directory so the final step is a rename within
	// one filesystem. A cross-device rename is a copy, which is not atomic and
	// can be interrupted halfway -- exactly the state this design exists to
	// make impossible.
	tmp, err := os.CreateTemp(dir, ".meshd-update-*")
	if err != nil {
		return fmt.Errorf("%w: %v", ErrStagingFailed, err)
	}
	tmpName := tmp.Name()
	cleanup := func() {
		tmp.Close()
		os.Remove(tmpName)
	}

	// Bound the copy by the signed size. Without this a hostile or broken host
	// can stream forever and fill the contributor's disk, and a full disk on
	// someone else's machine is an outage we caused.
	limited := io.LimitReader(r, in.Artifact.Size+1)
	n, err := io.Copy(tmp, limited)
	if err != nil {
		cleanup()
		return fmt.Errorf("%w: %v", ErrStagingFailed, err)
	}
	if n != in.Artifact.Size {
		cleanup()
		return fmt.Errorf("%w: got %d bytes, manifest says %d", ErrHashMismatch, n, in.Artifact.Size)
	}
	if err := tmp.Sync(); err != nil {
		cleanup()
		return fmt.Errorf("%w: %v", ErrStagingFailed, err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("%w: %v", ErrStagingFailed, err)
	}

	// Re-read from disk. Hashing the buffer we just wrote would prove only that
	// our own memory is self-consistent; the claim that matters is about the
	// bytes that will actually be executed.
	written, err := os.ReadFile(tmpName)
	if err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("%w: %v", ErrStagingFailed, err)
	}
	if err := CheckArtifact(in.Artifact, written); err != nil {
		os.Remove(tmpName)
		return err
	}

	if runtime.GOOS != "windows" {
		if err := os.Chmod(tmpName, 0o755); err != nil {
			os.Remove(tmpName)
			return fmt.Errorf("%w: %v", ErrStagingFailed, err)
		}
	}

	staged := in.stagedPath()
	os.Remove(staged)
	if err := os.Rename(tmpName, staged); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("%w: %v", ErrStagingFailed, err)
	}
	return nil
}

// Install swaps the staged binary into place, restoring the previous one if the
// swap cannot be completed.
//
// It re-verifies the staged file first. Staging and installing are separated in
// time -- a download may finish minutes before a maintenance window -- and
// anything with write access to the directory can replace the staged file in
// between. Trusting the earlier check would make the gap a TOCTOU hole.
func (in *Installer) Install() error {
	staged := in.stagedPath()
	body, err := os.ReadFile(staged)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNotStaged, err)
	}
	if err := CheckArtifact(in.Artifact, body); err != nil {
		os.Remove(staged)
		return err
	}

	prev := in.previousPath()
	os.Remove(prev) // a leftover from an earlier update would block the rename

	targetExisted := true
	if _, err := os.Stat(in.Target); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		targetExisted = false
	}

	if targetExisted {
		if err := in.doRename(in.Target, prev); err != nil {
			// Nothing has moved yet, so the install is simply refused and the
			// running binary is untouched. This is the safe failure.
			return fmt.Errorf("update: could not move the current binary aside: %w", err)
		}
	}

	if err := in.doRename(staged, in.Target); err != nil {
		if targetExisted {
			if rbErr := in.doRename(prev, in.Target); rbErr != nil {
				// Both renames failed: the target directory is broken or something
				// is holding a lock. Say so in the loudest possible terms and name
				// the path the operator must restore by hand -- a generic error
				// here would send someone hunting for a binary that is sitting
				// right next to where they are looking.
				return fmt.Errorf("%w: %v (rollback also failed: %v). The previous binary is at %s and must be renamed back to %s by hand",
					ErrBricked, err, rbErr, prev, in.Target)
			}
			return fmt.Errorf("%w: %v", ErrRollback, err)
		}
		return fmt.Errorf("%w: %v", ErrRollback, err)
	}

	return nil
}

// CleanupPrevious removes the displaced binary. Call it at startup, not right
// after Install: on Windows the old file is still locked by the very process
// doing the update, so deleting it can only succeed once that process is gone.
// A failure here is genuinely harmless -- it leaves one stale file, and the next
// start tries again -- which is why it is a separate call and not part of the
// swap. Folding it into Install would let a cosmetic problem fail a good update.
func (in *Installer) CleanupPrevious() error {
	err := os.Remove(in.previousPath())
	if err != nil && os.IsNotExist(err) {
		return nil
	}
	return err
}

// DiscardStaged removes a staged artifact that will not be installed.
func (in *Installer) DiscardStaged() error {
	err := os.Remove(in.stagedPath())
	if err != nil && os.IsNotExist(err) {
		return nil
	}
	return err
}
