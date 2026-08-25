//go:build !windows

package config

import "syscall"

func freeDiskMB(path string) int {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0
	}
	return int((st.Bavail * uint64(st.Bsize)) / 1024 / 1024)
}
