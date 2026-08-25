//go:build windows

package config

import (
	"path/filepath"
	"syscall"
	"unsafe"
)

func freeDiskMB(path string) int {
	volume := filepath.VolumeName(path)
	if volume == "" {
		volume = `C:`
	}
	root := volume + `\`
	ptr, err := syscall.UTF16PtrFromString(root)
	if err != nil {
		return 0
	}
	k32 := syscall.NewLazyDLL("kernel32.dll")
	proc := k32.NewProc("GetDiskFreeSpaceExW")
	var freeBytes uint64
	r1, _, _ := proc.Call(uintptr(unsafe.Pointer(ptr)), uintptr(unsafe.Pointer(&freeBytes)), 0, 0)
	if r1 == 0 {
		return 0
	}
	return int(freeBytes / 1024 / 1024)
}
