// Package wasmfn runs MeshPool user functions -- WASI *command* modules following the
// "wasi-command/json-stdio" ABI -- under wazero.
//
// This is the Go half of meshpool/functions/wasm.py. The two must agree byte for byte
// on the same module, because a job is fanned out across whichever agents happen to be
// online: a unit that returns 42 on a Python agent and 43 on a Go agent would make the
// coordinator's spot-check oracle flag honest nodes as fraudulent. verify_go_agent_parity
// feeds identical module bytes to both and compares.
//
// Why this exists at all: internal/sandbox is a *core* wasm runner with no imports
// whatsoever, which is the right boundary for arithmetic kernels but cannot run a WASI
// command -- a module importing wasi_snapshot_preview1 fails to instantiate there. Real
// user functions, produced by every language toolchain that targets wasm32-wasip1, are
// WASI commands. Without this package meshd answered /run_unit with 501 and any wasm:
// unit scheduled onto a Go agent burned its retry budget before landing on a Python one.
//
// The sandbox properties are the same ones the Python runtime documents:
//
//   - **No filesystem.** WithFSConfig is never called, so the guest has no preopened
//     directory and every path_open returns EBADF. The omission *is* the sandbox.
//   - **No network.** wazero's WASI p1 has no sockets to link, and no host module is
//     defined, so there is nothing to call.
//   - **No ambient host access.** stdin/stdout/stderr are in-memory buffers owned by
//     this process; the guest never receives a real file descriptor.
//   - **Bounded.** Wall-clock deadline (context cancellation tears down an infinite
//     loop), memory ceiling, and an output cap that stops a module from filling RAM
//     by printing.
//
// One honest divergence from the Python runtime, and it is deliberate: wasmtime meters
// *guest instructions* with fuel, and wazero has no equivalent. FuelUsed is therefore
// always 0 here and Metered reports false, so a caller can tell "used no fuel" apart
// from "this runtime cannot count fuel". Billing on a Go agent falls back to
// DurationMs and StdoutBytes, which the Python runtime's own docs already require --
// fuel counts guest instructions only, so an I/O-heavy function is under-metered by
// fuel on either runtime.
package wasmfn

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
	"github.com/tetratelabs/wazero/sys"
)

// ABI is the contract advertised to users: read JSON from stdin, write JSON to stdout.
const ABI = "wasi-command/json-stdio"

// wasiModule is the only import namespace a user function may reference.
const wasiModule = "wasi_snapshot_preview1"

const pageSize = 64 * 1024

// Limits mirrors meshpool.functions.wasm.Limits. Defaults are kept identical so a unit
// does not succeed on one runtime and hit a ceiling on the other.
type Limits struct {
	MemoryBytes    uint64
	WallTimeout    time.Duration
	MaxOutputBytes int
	MaxStderrBytes int
}

// DefaultLimits matches the Python runtime's defaults exactly.
func DefaultLimits() Limits {
	return Limits{
		MemoryBytes:    256 * 1024 * 1024,
		WallTimeout:    10 * time.Second,
		MaxOutputBytes: 8 * 1024 * 1024,
		MaxStderrBytes: 16 * 1024,
	}
}

// Result mirrors meshpool.functions.wasm.Result.
type Result struct {
	Value       any     `json:"value"`
	FuelUsed    int     `json:"fuel_used"`
	DurationMs  float64 `json:"duration_ms"`
	StdoutBytes int     `json:"stdout_bytes"`
	Stderr      string  `json:"stderr"`
	ColdStart   bool    `json:"cold_start"`
	// Metered is false because wazero cannot count guest instructions. It exists so a
	// biller can distinguish "no fuel consumed" from "fuel is unknown on this runtime"
	// rather than silently treating a Go agent's work as free.
	Metered bool `json:"metered"`
}

// Error carries the same ``kind`` taxonomy as the Python FunctionError, because the
// coordinator branches on it: a policy stop (timeout/memory/output) means "retry
// elsewhere", a user bug (exit/bad_output/trap/invalid_module) means "this job is
// broken, stop burning the retry budget".
type Error struct {
	Kind    string
	Message string
	Stderr  string
}

func (e *Error) Error() string { return e.Kind + ": " + e.Message }

func newErr(kind, msg, stderr string) *Error {
	return &Error{Kind: kind, Message: msg, Stderr: stderr}
}

// ModuleKey is the content address of a module: sha256, matching module_key() in Python
// so both runtimes name the same bytes the same way.
func ModuleKey(wasmBytes []byte) string {
	sum := sha256.Sum256(wasmBytes)
	return hex.EncodeToString(sum[:])
}

// Runner compiles and executes user functions. Safe for concurrent use.
type Runner struct {
	runtime wazero.Runtime
	limits  Limits

	mu    sync.Mutex
	cache map[string]wazero.CompiledModule
}

// NewRunner builds a runtime with WASI p1 linked and the given ceilings.
//
// The memory ceiling is applied at the *runtime* level via WithMemoryLimitPages, which
// caps what any module may grow to. A module declaring a larger minimum simply fails to
// instantiate, which is the correct outcome: better a clear rejection than an OOM on a
// donor's laptop.
func NewRunner(ctx context.Context, limits Limits) (*Runner, error) {
	d := DefaultLimits()
	if limits.MemoryBytes == 0 {
		limits.MemoryBytes = d.MemoryBytes
	}
	if limits.WallTimeout <= 0 {
		limits.WallTimeout = d.WallTimeout
	}
	if limits.MaxOutputBytes <= 0 {
		limits.MaxOutputBytes = d.MaxOutputBytes
	}
	if limits.MaxStderrBytes <= 0 {
		limits.MaxStderrBytes = d.MaxStderrBytes
	}

	pages := limits.MemoryBytes / pageSize
	if pages == 0 {
		pages = 1
	}
	if pages > 65536 {
		pages = 65536 // wasm32 address space ceiling
	}

	cfg := wazero.NewRuntimeConfig().
		// Without this, a module spinning in `loop br 0` ignores context cancellation
		// and pins a core until the process dies.
		WithCloseOnContextDone(true).
		WithMemoryLimitPages(uint32(pages))

	rt := wazero.NewRuntimeWithConfig(ctx, cfg)
	if _, err := wasi_snapshot_preview1.Instantiate(ctx, rt); err != nil {
		_ = rt.Close(ctx)
		return nil, fmt.Errorf("wasmfn: link wasi: %w", err)
	}
	return &Runner{runtime: rt, limits: limits, cache: map[string]wazero.CompiledModule{}}, nil
}

func (r *Runner) Close(ctx context.Context) error {
	r.mu.Lock()
	r.cache = map[string]wazero.CompiledModule{}
	r.mu.Unlock()
	return r.runtime.Close(ctx)
}

// Limits returns the ceilings this runner enforces.
func (r *Runner) Limits() Limits { return r.limits }

// CacheSize reports how many distinct modules are compiled and resident.
func (r *Runner) CacheSize() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.cache)
}

// Preflight compiles a module and checks it can only ever be a well-formed user
// function: WASI-only imports and a _start export.
//
// This runs before a module is accepted, so a broken upload is rejected while the
// author is watching instead of failing 200 times across the pool.
func (r *Runner) Preflight(ctx context.Context, wasmBytes []byte) (map[string]any, error) {
	compiled, _, err := r.compile(ctx, wasmBytes)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	names := []string{}
	for _, f := range compiled.ImportedFunctions() {
		mod, name, _ := f.Import()
		key := mod + "." + name
		if !seen[key] {
			seen[key] = true
			names = append(names, key)
		}
	}
	return map[string]any{
		"abi":          ABI,
		"key":          ModuleKey(wasmBytes),
		"size_bytes":   len(wasmBytes),
		"imports":      names,
		"import_count": len(names),
		"has_start":    hasStart(compiled),
	}, nil
}

func hasStart(compiled wazero.CompiledModule) bool {
	_, ok := compiled.ExportedFunctions()["_start"]
	return ok
}

// compile returns a cached CompiledModule, compiling on first sight. Modules are
// content-addressed and immutable, so the cache can never be stale -- this is what
// turns cold start into a once-per-module cost rather than a per-request one.
func (r *Runner) compile(ctx context.Context, wasmBytes []byte) (wazero.CompiledModule, bool, error) {
	if len(wasmBytes) == 0 {
		return nil, false, newErr("invalid_module", "module is empty", "")
	}
	key := ModuleKey(wasmBytes)

	r.mu.Lock()
	if c, ok := r.cache[key]; ok {
		r.mu.Unlock()
		return c, false, nil
	}
	r.mu.Unlock()

	compiled, err := r.runtime.CompileModule(ctx, wasmBytes)
	if err != nil {
		return nil, false, newErr("invalid_module",
			"module could not be compiled: "+err.Error(), "")
	}

	// Reject non-WASI imports at compile time. wazero would fail instantiation anyway,
	// but the error there is a link failure that reads like an internal fault; here we
	// can tell the author precisely which import is not available.
	for _, f := range compiled.ImportedFunctions() {
		mod, name, _ := f.Import()
		if mod != wasiModule {
			_ = compiled.Close(ctx)
			return nil, false, newErr("invalid_module", fmt.Sprintf(
				"module imports %q from %q; only %s is available to user functions",
				name, mod, wasiModule), "")
		}
	}
	if !hasStart(compiled) {
		_ = compiled.Close(ctx)
		return nil, false, newErr("invalid_module",
			"module has no _start export - build a WASI *command* "+
				"(a program with a main), not a library", "")
	}

	r.mu.Lock()
	if existing, ok := r.cache[key]; ok {
		// Lost a compile race; keep the winner so both callers share one instance.
		r.mu.Unlock()
		_ = compiled.Close(ctx)
		return existing, false, nil
	}
	r.cache[key] = compiled
	r.mu.Unlock()
	return compiled, true, nil
}

// cappedWriter stops accepting bytes past a ceiling and records the overrun, so a module
// that prints in a loop cannot exhaust the agent's memory before the deadline fires.
type cappedWriter struct {
	buf      bytes.Buffer
	cap      int
	overflow bool
	total    int
}

func (w *cappedWriter) Write(p []byte) (int, error) {
	w.total += len(p)
	if w.buf.Len() >= w.cap {
		w.overflow = true
		return len(p), nil // absorb silently; the caller reports the overrun
	}
	room := w.cap - w.buf.Len()
	if len(p) > room {
		w.overflow = true
		w.buf.Write(p[:room])
		return len(p), nil
	}
	return w.buf.Write(p)
}

func (w *cappedWriter) text() string {
	s := w.buf.String()
	if w.overflow {
		s += "\n...[truncated]"
	}
	return s
}

// Run executes one invocation: payload -> JSON on stdin, JSON on stdout -> value.
func (r *Runner) Run(parent context.Context, wasmBytes []byte, payload any) (Result, error) {
	request, err := json.Marshal(payload)
	if err != nil {
		return Result{}, newErr("bad_input",
			"payload is not JSON-serialisable: "+err.Error(), "")
	}

	compiled, cold, err := r.compile(parent, wasmBytes)
	if err != nil {
		return Result{}, err
	}

	ctx, cancel := context.WithTimeout(parent, r.limits.WallTimeout)
	defer cancel()

	stdout := &cappedWriter{cap: r.limits.MaxOutputBytes}
	stderrBuf := &cappedWriter{cap: r.limits.MaxStderrBytes}

	cfg := wazero.NewModuleConfig().
		WithName("").
		WithArgs("function").
		WithStdin(bytes.NewReader(request)).
		WithStdout(stdout).
		WithStderr(stderrBuf)
	// NOTE: no WithFSConfig call. That omission *is* the filesystem sandbox.

	started := time.Now()
	mod, runErr := r.runtime.InstantiateModule(ctx, compiled, cfg)
	elapsed := time.Since(started)
	if mod != nil {
		_ = mod.Close(ctx)
	}

	stderr := stderrBuf.text()

	exitCode := uint32(0)
	if runErr != nil {
		var exitErr *sys.ExitError
		switch {
		case errors.As(runErr, &exitErr) && ctx.Err() == nil:
			// WASI proc_exit: the normal way a command module finishes.
			exitCode = exitErr.ExitCode()
		case ctx.Err() != nil:
			// A module torn down by the deadline also surfaces as an ExitError, so the
			// deadline check must come first or a timeout is misreported as exit 0.
			return Result{}, newErr("timeout", fmt.Sprintf(
				"function exceeded its %.3gs wall-clock limit",
				r.limits.WallTimeout.Seconds()), stderr)
		case isMemoryErr(runErr):
			return Result{}, newErr("memory", fmt.Sprintf(
				"function exceeded its %d-byte memory limit", r.limits.MemoryBytes), stderr)
		default:
			return Result{}, newErr("trap",
				"function failed at runtime: "+runErr.Error(), stderr)
		}
	}

	// A context that expired concurrently with a clean return still means the module was
	// torn down mid-flight; its output cannot be trusted.
	if ctx.Err() != nil {
		return Result{}, newErr("timeout", fmt.Sprintf(
			"function exceeded its %.3gs wall-clock limit",
			r.limits.WallTimeout.Seconds()), stderr)
	}
	if exitCode != 0 {
		return Result{}, newErr("exit",
			fmt.Sprintf("function exited with status %d", exitCode), stderr)
	}
	if stdout.overflow {
		return Result{}, newErr("output", fmt.Sprintf(
			"function wrote %d bytes, over the %d-byte output limit",
			stdout.total, r.limits.MaxOutputBytes), stderr)
	}

	raw := stdout.buf.Bytes()
	if len(bytes.TrimSpace(raw)) == 0 {
		return Result{}, newErr("bad_output",
			"function produced no output - write the JSON result to stdout", stderr)
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		preview := raw
		if len(preview) > 200 {
			preview = preview[:200]
		}
		return Result{}, newErr("bad_output", fmt.Sprintf(
			"stdout is not valid JSON (%v); first bytes: %q", err, preview), stderr)
	}

	return Result{
		Value:       value,
		FuelUsed:    0,
		Metered:     false,
		DurationMs:  float64(elapsed.Nanoseconds()) / 1e6,
		StdoutBytes: len(raw),
		Stderr:      stderr,
		ColdStart:   cold,
	}, nil
}

func isMemoryErr(err error) bool {
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "out of memory") ||
		strings.Contains(s, "memory size") ||
		strings.Contains(s, "cannot grow") ||
		strings.Contains(s, "max memory")
}

var _ io.Writer = (*cappedWriter)(nil)
