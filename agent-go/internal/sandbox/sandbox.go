// Package sandbox executes untrusted WebAssembly under wazero with a deny-by-default
// capability set.
//
// This is the isolation boundary the Python agent never had: meshpool/agent/sandbox.py
// is a child process with an RLIMIT/Job-Object cap, which bounds *resource use* but is
// not an isolation boundary -- the code still runs with the agent's own filesystem and
// network access. A wazero module, by contrast, starts with no imports at all: no
// syscalls, no filesystem, no sockets, no clock. It can only read and write its own
// linear memory and return a value. Anything a module wants from the host must be
// explicitly granted, and nothing is granted here.
//
// wazero is pure Go: no cgo, no external runtime, so meshd stays a single static binary
// on every OS -- which is the whole point of rewriting the agent.
package sandbox

import (
	"context"
	"fmt"
	"time"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

// Runner compiles and executes WASM modules. It is safe for concurrent use.
type Runner struct {
	runtime wazero.Runtime
	timeout time.Duration
}

// New builds a runtime with a hard per-call wall-clock timeout. The timeout is
// enforced by wazero's context-based interruption, so a module stuck in an infinite
// loop is torn down rather than pinning a donor's CPU forever.
func New(timeout time.Duration) (*Runner, error) {
	if timeout <= 0 {
		return nil, fmt.Errorf("sandbox: timeout must be positive")
	}
	cfg := wazero.NewRuntimeConfig().
		// Interruptible execution: without this a `loop br 0` module ignores ctx.
		WithCloseOnContextDone(true)
	return &Runner{
		runtime: wazero.NewRuntimeWithConfig(context.Background(), cfg),
		timeout: timeout,
	}, nil
}

func (r *Runner) Close(ctx context.Context) error { return r.runtime.Close(ctx) }

// Result carries everything the coordinator needs to bill and verify one call.
type Result struct {
	Values   []uint64
	Duration time.Duration
}

// Execute instantiates `module` and calls the exported function `fn` with `args`.
//
// The module is instantiated with wazero.NewModuleConfig() and *no* host modules, so:
//   - a module importing wasi_snapshot_preview1 (i.e. wanting files/stdio/clock) fails
//     to instantiate rather than silently getting them;
//   - a module importing any host function at all fails the same way;
//   - there is no ambient authority to escape from.
//
// Every call gets a fresh instance, so one job cannot observe or corrupt another's
// memory. The instance is closed when the call returns.
func (r *Runner) Execute(parent context.Context, module []byte, fn string,
	args ...uint64) (Result, error) {
	if len(module) == 0 {
		return Result{}, fmt.Errorf("sandbox: empty module")
	}
	ctx, cancel := context.WithTimeout(parent, r.timeout)
	defer cancel()

	compiled, err := r.runtime.CompileModule(ctx, module)
	if err != nil {
		return Result{}, fmt.Errorf("sandbox: compile: %w", err)
	}
	defer compiled.Close(ctx) //nolint:errcheck

	// Deny by default: no WASI, no host imports, no env, no stdio, no filesystem.
	instance, err := r.runtime.InstantiateModule(
		ctx, compiled, wazero.NewModuleConfig().WithName(""))
	if err != nil {
		return Result{}, fmt.Errorf("sandbox: instantiate: %w", err)
	}
	defer instance.Close(ctx) //nolint:errcheck

	target := instance.ExportedFunction(fn)
	if target == nil {
		return Result{}, fmt.Errorf("sandbox: module exports no function %q", fn)
	}
	if err := checkArity(target, args); err != nil {
		return Result{}, err
	}

	start := time.Now()
	values, err := target.Call(ctx, args...)
	elapsed := time.Since(start)
	if err != nil {
		if ctx.Err() != nil {
			return Result{}, fmt.Errorf("sandbox: %q exceeded the %s limit", fn, r.timeout)
		}
		return Result{}, fmt.Errorf("sandbox: %q trapped: %w", fn, err)
	}
	return Result{Values: values, Duration: elapsed}, nil
}

func checkArity(f api.Function, args []uint64) error {
	want := len(f.Definition().ParamTypes())
	if want != len(args) {
		return fmt.Errorf("sandbox: %q takes %d arg(s), got %d",
			f.Definition().Name(), want, len(args))
	}
	return nil
}
