//go:build windows

package config

import (
	"syscall"
	"unsafe"
)

type memoryStatusEx struct {
	cbSize                  uint32
	dwMemoryLoad            uint32
	ullTotalPhys            uint64
	ullAvailPhys            uint64
	ullTotalPageFile        uint64
	ullAvailPageFile        uint64
	ullTotalVirtual         uint64
	ullAvailVirtual         uint64
	ullAvailExtendedVirtual uint64
}

// totalRAMMB reports installed physical memory in MiB via GlobalMemoryStatusEx,
// or 0 when the call fails.
func totalRAMMB() int {
	k32 := syscall.NewLazyDLL("kernel32.dll")
	proc := k32.NewProc("GlobalMemoryStatusEx")
	var st memoryStatusEx
	st.cbSize = uint32(unsafe.Sizeof(st))
	r1, _, _ := proc.Call(uintptr(unsafe.Pointer(&st)))
	if r1 == 0 {
		return 0
	}
	return int(st.ullTotalPhys / 1024 / 1024)
}
