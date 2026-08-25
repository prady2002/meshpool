// Command wasmfn-probe runs one user function through meshd's WASI runtime and prints
// the outcome as JSON on stdout.
//
// It exists so the cross-runtime parity gate can drive the *real* Go execution path
// from Python with the *same* module bytes it feeds to wasmtime. Testing the two
// runtimes independently would only prove each is self-consistent; a job is fanned out
// across whichever agents are online, so what actually has to hold is that they agree.
//
// Usage:
//
//	wasmfn-probe --module path/to/fn.wasm --payload '{"x":1}' [--timeout 10s]
//
// Exit code is always 0 when the probe itself ran; the JSON says whether the *function*
// succeeded. That keeps "the probe crashed" distinguishable from "the function trapped".
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"meshd/internal/wasmfn"
)

func main() {
	modulePath := flag.String("module", "", "path to the .wasm module")
	payload := flag.String("payload", "null", "JSON payload for stdin")
	timeout := flag.Duration("timeout", 10*time.Second, "wall-clock limit")
	maxOutput := flag.Int("max-output", 0, "output cap in bytes (0 = default)")
	memory := flag.Uint64("memory", 0, "memory cap in bytes (0 = default)")
	preflight := flag.Bool("preflight", false, "only inspect the module, do not run it")
	flag.Parse()

	if *modulePath == "" {
		emit(map[string]any{"ok": false, "kind": "probe", "message": "--module is required"})
		return
	}
	moduleBytes, err := os.ReadFile(*modulePath)
	if err != nil {
		emit(map[string]any{"ok": false, "kind": "probe", "message": err.Error()})
		return
	}

	var decoded any
	if err := json.Unmarshal([]byte(*payload), &decoded); err != nil {
		emit(map[string]any{"ok": false, "kind": "probe",
			"message": fmt.Sprintf("--payload is not JSON: %v", err)})
		return
	}

	limits := wasmfn.DefaultLimits()
	limits.WallTimeout = *timeout
	if *maxOutput > 0 {
		limits.MaxOutputBytes = *maxOutput
	}
	if *memory > 0 {
		limits.MemoryBytes = *memory
	}

	ctx := context.Background()
	runner, err := wasmfn.NewRunner(ctx, limits)
	if err != nil {
		emit(map[string]any{"ok": false, "kind": "probe", "message": err.Error()})
		return
	}
	defer runner.Close(ctx) //nolint:errcheck

	if *preflight {
		info, err := runner.Preflight(ctx, moduleBytes)
		if err != nil {
			emit(errPayload(err))
			return
		}
		info["ok"] = true
		emit(info)
		return
	}

	res, err := runner.Run(ctx, moduleBytes, decoded)
	if err != nil {
		emit(errPayload(err))
		return
	}
	emit(map[string]any{
		"ok":           true,
		"value":        res.Value,
		"fuel_used":    res.FuelUsed,
		"metered":      res.Metered,
		"duration_ms":  res.DurationMs,
		"stdout_bytes": res.StdoutBytes,
		"stderr":       res.Stderr,
		"cold_start":   res.ColdStart,
	})
}

func errPayload(err error) map[string]any {
	var fnErr *wasmfn.Error
	if errors.As(err, &fnErr) {
		return map[string]any{
			"ok":      false,
			"kind":    fnErr.Kind,
			"message": fnErr.Message,
			"stderr":  fnErr.Stderr,
		}
	}
	return map[string]any{"ok": false, "kind": "probe", "message": err.Error()}
}

func emit(v map[string]any) {
	enc := json.NewEncoder(os.Stdout)
	_ = enc.Encode(v)
}
