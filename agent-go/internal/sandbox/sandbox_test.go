package sandbox

import (
	"context"
	"strings"
	"testing"
	"time"
)

// The modules below are hand-assembled WASM binaries. They are written out byte by
// byte on purpose: the test must not depend on a WASM toolchain being installed, and
// hand-assembly makes it obvious that the runtime is genuinely executing bytecode
// rather than a Go switch statement pretending to.

// (module (func (export "add") (param i32 i32) (result i32)
//            local.get 0  local.get 1  i32.add))
var addModule = []byte{
	0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
	0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // type: (i32,i32)->i32
	0x03, 0x02, 0x01, 0x00, // func: one function, type 0
	0x07, 0x07, 0x01, 0x03, 'a', 'd', 'd', 0x00, 0x00, // export "add" -> func 0
	0x0a, 0x09, 0x01, 0x07, 0x00, // code: one body, 7 bytes, 0 locals
	0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b, // local.get 0; local.get 1; i32.add; end
}

// (module (func (export "spin") (loop br 0)))  -- never returns.
var spinModule = []byte{
	0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
	0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type: () -> ()
	0x03, 0x02, 0x01, 0x00,
	0x07, 0x08, 0x01, 0x04, 's', 'p', 'i', 'n', 0x00, 0x00,
	0x0a, 0x09, 0x01, 0x07, 0x00, // code: one body, 7 bytes, 0 locals
	0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b, // loop void; br 0; end loop; end func
}

// (module (import "wasi_snapshot_preview1" "proc_exit" (func (param i32))))
// A module that wants WASI at all must fail to instantiate: nothing is granted.
var wasiImportModule = []byte{
	0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
	0x01, 0x05, 0x01, 0x60, 0x01, 0x7f, 0x00, // type: (i32) -> ()
	0x02, 0x24, 0x01, // import section (36-byte payload), 1 import
	0x16, 'w', 'a', 's', 'i', '_', 's', 'n', 'a', 'p', 's', 'h', 'o', 't',
	'_', 'p', 'r', 'e', 'v', 'i', 'e', 'w', '1',
	0x09, 'p', 'r', 'o', 'c', '_', 'e', 'x', 'i', 't',
	0x00, 0x00, // kind func, type 0
}

// (module (func (export "boom") (result i32) unreachable))
var trapModule = []byte{
	0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
	0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f, // type: () -> i32
	0x03, 0x02, 0x01, 0x00,
	0x07, 0x08, 0x01, 0x04, 'b', 'o', 'o', 'm', 0x00, 0x00,
	0x0a, 0x05, 0x01, 0x03, 0x00, 0x00, 0x0b, // body: unreachable; end
}

func newRunner(t *testing.T, timeout time.Duration) *Runner {
	t.Helper()
	r, err := New(timeout)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = r.Close(context.Background()) })
	return r
}

func TestExecuteRunsRealWasm(t *testing.T) {
	r := newRunner(t, 5*time.Second)
	// Several cases so a hardcoded constant cannot pass: the answer must track input.
	for _, tc := range []struct{ a, b, want uint64 }{
		{2, 3, 5}, {0, 0, 0}, {7, 35, 42}, {1 << 20, 1, 1<<20 + 1},
	} {
		got, err := r.Execute(context.Background(), addModule, "add", tc.a, tc.b)
		if err != nil {
			t.Fatalf("Execute(%d,%d): %v", tc.a, tc.b, err)
		}
		if len(got.Values) != 1 || got.Values[0] != tc.want {
			t.Fatalf("add(%d,%d) = %v, want %d", tc.a, tc.b, got.Values, tc.want)
		}
	}
}

func TestWasiImportIsDenied(t *testing.T) {
	r := newRunner(t, 5*time.Second)
	_, err := r.Execute(context.Background(), wasiImportModule, "anything")
	if err == nil {
		t.Fatal("a module importing WASI was instantiated; the sandbox grants ambient authority")
	}
	if !strings.Contains(err.Error(), "instantiate") {
		t.Fatalf("expected an instantiation failure, got: %v", err)
	}
}

func TestInfiniteLoopIsInterrupted(t *testing.T) {
	// Non-vacuity: the module must genuinely compile and run, otherwise this test
	// would "pass" on a malformed binary that never executed at all.
	r := newRunner(t, 250*time.Millisecond)
	limit := 250 * time.Millisecond
	start := time.Now()
	_, err := r.Execute(context.Background(), spinModule, "spin")
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("an infinite loop returned successfully")
	}
	if !strings.Contains(err.Error(), "exceeded the") {
		t.Fatalf("expected a timeout, got a different failure (did the module even "+
			"compile?): %v", err)
	}
	if elapsed < limit/2 {
		t.Fatalf("returned after only %s — the module cannot have run", elapsed)
	}
	if elapsed > 5*time.Second {
		t.Fatalf("timeout did not fire promptly: took %s", elapsed)
	}
}

func TestTrapIsReportedNotSwallowed(t *testing.T) {
	r := newRunner(t, 5*time.Second)
	_, err := r.Execute(context.Background(), trapModule, "boom")
	if err == nil {
		t.Fatal("a trapping module reported success")
	}
	if !strings.Contains(err.Error(), "trapped") {
		t.Fatalf("expected a trap error, got: %v", err)
	}
}

func TestMissingExportAndBadArityAreRejected(t *testing.T) {
	r := newRunner(t, 5*time.Second)
	if _, err := r.Execute(context.Background(), addModule, "nope", 1, 2); err == nil {
		t.Fatal("calling a non-existent export succeeded")
	}
	if _, err := r.Execute(context.Background(), addModule, "add", 1); err == nil {
		t.Fatal("calling add/2 with one argument succeeded")
	}
}

func TestInstancesDoNotShareState(t *testing.T) {
	// Each Execute gets a fresh instance; running concurrently must not interleave.
	r := newRunner(t, 5*time.Second)
	done := make(chan error, 8)
	for i := 0; i < 8; i++ {
		go func(n uint64) {
			got, err := r.Execute(context.Background(), addModule, "add", n, n)
			if err == nil && got.Values[0] != 2*n {
				t.Errorf("add(%d,%d) = %d", n, n, got.Values[0])
			}
			done <- err
		}(uint64(i))
	}
	for i := 0; i < 8; i++ {
		if err := <-done; err != nil {
			t.Fatalf("concurrent Execute: %v", err)
		}
	}
}

func TestEmptyModuleRejected(t *testing.T) {
	r := newRunner(t, time.Second)
	if _, err := r.Execute(context.Background(), nil, "add"); err == nil {
		t.Fatal("an empty module was accepted")
	}
}
