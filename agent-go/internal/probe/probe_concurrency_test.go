package probe

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"meshd/internal/config"
)

// Probe() blocks the agent from binding its HTTP port, so its wall time is dead
// time during which the node is unreachable and unregistered.
//
// Every detector shells out, and on Windows most are a separate
// `powershell -NoProfile -Command Get-CimInstance ...` that costs seconds just
// to start the shell. Run sequentially the seven of them took ~45s, which blew
// through verify_hw_probe's 30s readiness budget and surfaced as
// "service not ready at 127.0.0.1:8362" -- an error that reads like a crash, so
// it was written off as flakiness rather than diagnosed as a slow probe.
//
// The bound below is deliberately loose. The point is to catch a return to
// serial execution (which would roughly triple this), not to police seconds.
func TestProbeCompletesInsideTheReadinessBudget(t *testing.T) {
	start := time.Now()
	cap, _ := Probe(config.Config{ControlPlaneURL: "http://127.0.0.1:9"})
	elapsed := time.Since(start)

	if cap.CPUCores < 1 {
		t.Fatalf("CPUCores = %d, want >= 1", cap.CPUCores)
	}
	if elapsed > 25*time.Second {
		t.Fatalf("Probe() took %s; it blocks the agent from serving, so this "+
			"must stay well inside the readiness budget (detectors serial again?)",
			elapsed)
	}
}

// Running the detectors concurrently is only safe if they still all report.
// A goroutine whose result was dropped would leave a node advertising unknown
// RAM or no disk, and the control plane would size placements from that -- a
// silent capacity bug rather than a visible failure.
func TestConcurrentProbeStillPopulatesEveryField(t *testing.T) {
	cap, _ := Probe(config.Config{ControlPlaneURL: "http://127.0.0.1:9"})

	if cap.RAMMB <= 0 {
		t.Errorf("RAMMB = %d, want > 0", cap.RAMMB)
	}
	if cap.DiskMB <= 0 {
		t.Errorf("DiskMB = %d, want > 0", cap.DiskMB)
	}
	if cap.DiskTotalMB <= 0 {
		t.Errorf("DiskTotalMB = %d, want > 0", cap.DiskTotalMB)
	}
	// Free and total come from one sample of one volume, so free can never
	// exceed total. When they were two separate child processes a large write
	// landing between them could invert the pair, and a node advertising more
	// free space than it physically has is a placement bug waiting to happen.
	if cap.DiskMB > cap.DiskTotalMB {
		t.Errorf("DiskMB = %d exceeds DiskTotalMB = %d; free and total are "+
			"not from a consistent sample", cap.DiskMB, cap.DiskTotalMB)
	}
	if cap.FLOPSEstimate <= 0 {
		t.Errorf("FLOPSEstimate = %v, want > 0", cap.FLOPSEstimate)
	}
	if cap.CPUArch == "" {
		t.Error("CPUArch is empty")
	}
	if cap.ThermalState == "" {
		t.Error("ThermalState is empty")
	}
	if cap.GPUNames == nil {
		t.Error("GPUNames is nil; it marshals to JSON null and the control " +
			"plane rejects that")
	}
}

// Repeated probes must agree. Concurrency bugs of the "occasionally drops a
// result" kind do not reproduce on a single run, so hammer it.
func TestRepeatedProbesAreStable(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	first, _ := Probe(config.Config{ControlPlaneURL: "http://127.0.0.1:9"})
	for i := 0; i < 3; i++ {
		got, _ := Probe(config.Config{ControlPlaneURL: "http://127.0.0.1:9"})
		if got.RAMMB != first.RAMMB {
			t.Fatalf("run %d: RAMMB = %d, first run said %d", i, got.RAMMB, first.RAMMB)
		}
		if got.CPUCores != first.CPUCores {
			t.Fatalf("run %d: CPUCores = %d, first run said %d", i, got.CPUCores, first.CPUCores)
		}
		if got.CPUModel != first.CPUModel {
			t.Fatalf("run %d: CPUModel = %q, first run said %q", i, got.CPUModel, first.CPUModel)
		}
		if got.DiskTotalMB != first.DiskTotalMB {
			t.Fatalf("run %d: DiskTotalMB = %d, first run said %d", i, got.DiskTotalMB, first.DiskTotalMB)
		}
	}
}

// The total size of a volume does not change while the machine is running, so
// two probes must never disagree about it. This is a bug that was actually
// observed in a real run, not a hypothetical:
//
//	--- FAIL: TestRepeatedProbesAreStable
//	    run 0: DiskTotalMB = 487756, first run said 487755
//
// DiskTotalMB was derived as Used+Free from Get-PSDrive. Those are two live
// counters and their sum is not guaranteed to equal the volume capacity --
// Free is "bytes available to the caller", which reserved space and quotas can
// push below the true free count, so the sum drifts below Size by a varying
// amount. The volume's Size is the capacity and is invariant by definition.
//
// A capability field that changes between probes looks like a real capacity
// change to the control plane and churns anything that diffs capabilities or
// re-evaluates placement when they change.
//
// Honest scope: the drift is infrequent, so this is a regression detector
// rather than a proof -- it caught the original defect and will catch a
// systematic one, but a single clean run does not by itself establish that the
// value is invariant. The stronger guarantee comes from the free <= total
// invariant below, which a single consistent sample enforces structurally and
// two independently-sampled child processes cannot.
//
// This drives detectDiskMB directly rather than the full Probe: one child
// process per iteration instead of six, so it can afford real samples under
// real disk activity.
func TestDiskTotalDoesNotDriftUnderChurn(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}

	churn := filepath.Join(t.TempDir(), "churn.bin")
	blob := make([]byte, 8<<20)

	const samples = 8
	seen := map[int]int{}
	for i := 0; i < samples; i++ {
		// Move real bytes so the used/free counters actually change; without
		// this the old implementation looks stable on a quiet machine.
		if err := os.WriteFile(churn, blob, 0o600); err != nil {
			t.Fatalf("churn write: %v", err)
		}
		free, total := detectDiskMB()
		if total <= 0 {
			t.Fatalf("sample %d: total = %d, want > 0", i, total)
		}
		// Structural, not probabilistic: free and total come from one sample of
		// one volume, so this cannot fail by luck. Two separate child processes
		// could invert the pair if a large write landed between them, and a node
		// advertising more free space than the disk has is a placement bug.
		if free > total {
			t.Fatalf("sample %d: free = %d exceeds total = %d; free and total "+
				"are not from a consistent sample", i, free, total)
		}
		seen[total]++
		_ = os.Remove(churn)
	}

	if len(seen) != 1 {
		t.Fatalf("volume total drifted across %d samples: %v -- it must be a "+
			"static property of the volume, not derived from live counters",
			samples, seen)
	}
}

func TestSimOverridesWinAndSkipTheirDetectors(t *testing.T) {
	cfg := config.Config{
		ControlPlaneURL: "http://127.0.0.1:9",
		SimCores:        7,
		SimRAMMB:        12345,
		SimFLOPS:        42.5,
		SimVRAMMB:       2048,
		SimGPU:          true,
	}

	start := time.Now()
	cap, _ := Probe(cfg)
	elapsed := time.Since(start)

	if cap.CPUCores != 7 {
		t.Errorf("CPUCores = %d, want the override 7", cap.CPUCores)
	}
	if cap.RAMMB != 12345 {
		t.Errorf("RAMMB = %d, want the override 12345", cap.RAMMB)
	}
	if cap.FLOPSEstimate != 42.5 {
		t.Errorf("FLOPSEstimate = %v, want the override 42.5", cap.FLOPSEstimate)
	}
	if cap.VRAMMB != 2048 {
		t.Errorf("VRAMMB = %d, want the override 2048", cap.VRAMMB)
	}
	if !cap.GPU {
		t.Error("GPU = false, want the override true")
	}
	// Fields with no override must still be genuinely detected.
	if cap.DiskTotalMB <= 0 {
		t.Errorf("DiskTotalMB = %d; unrelated fields must still be probed", cap.DiskTotalMB)
	}
	if cap.GPUNames == nil {
		t.Error("GPUNames is nil even under SimGPU")
	}
	if elapsed > 25*time.Second {
		t.Fatalf("fully-overridden probe took %s", elapsed)
	}
}
