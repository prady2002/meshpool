package probe

import (
	"encoding/json"
	"testing"

	"meshd/internal/config"
)

// A real node must not advertise ram_mb=0: usable_layer_memory_mb() on the
// control plane takes max(vram, ram), so a zero here silently excludes the node
// from every model deployment.
func TestDetectRAMMBIsPositive(t *testing.T) {
	got := detectRAMMB()
	if got <= 0 {
		t.Fatalf("detectRAMMB() = %d, want > 0 on a real host", got)
	}
	if got < 256 {
		t.Fatalf("detectRAMMB() = %d MB, implausibly small", got)
	}
}

func TestProbeAdvertisesWasmRuntimeOnly(t *testing.T) {
	cap, _ := Probe(config.Config{ControlPlaneURL: "http://127.0.0.1:9"})
	if len(cap.Runtimes) != 1 || cap.Runtimes[0] != "wasm" {
		t.Fatalf("Runtimes = %v, want [wasm]", cap.Runtimes)
	}
	if cap.RAMMB <= 0 {
		t.Fatalf("RAMMB = %d, want > 0", cap.RAMMB)
	}
	blob, err := json.Marshal(cap)
	if err != nil {
		t.Fatal(err)
	}
	var round map[string]any
	if err := json.Unmarshal(blob, &round); err != nil {
		t.Fatal(err)
	}
	// The control plane reads this exact key; a rename here silently reverts the
	// scheduler to "this node can run anything".
	if _, ok := round["runtimes"]; !ok {
		t.Fatalf("serialized capability has no 'runtimes' key: %s", blob)
	}
	if _, ok := round["services"]; !ok {
		t.Fatalf("serialized capability has no 'services' key: %s", blob)
	}
}

// meshd returns 501 for /start_replica and /shard/load. It must say so up front:
// the hosting and inference planners commit to a placement plan and do not retry
// elsewhere, and topology_partition counts a node's RAM toward pooled capacity --
// so an over-broad advertisement here makes *adding* a meshd node break a model
// deployment that worked before it joined.
func TestProbeAdvertisesOnlyImplementedServices(t *testing.T) {
	cap, _ := Probe(config.Config{ControlPlaneURL: "http://127.0.0.1:9"})
	got := map[string]bool{}
	for _, s := range cap.Services {
		got[s] = true
	}
	for _, want := range []string{"batch", "storage"} {
		if !got[want] {
			t.Fatalf("Services = %v, missing implemented service %q", cap.Services, want)
		}
	}
	for _, unimplemented := range []string{"hosting", "shards"} {
		if got[unimplemented] {
			t.Fatalf("Services = %v advertises %q, but that handler returns 501",
				cap.Services, unimplemented)
		}
	}
}
