package probe

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"
)

// throttledProbeServer serves /api/v1/health and /api/v1/probe/download, shaping
// the download to approximately targetMbps so the measurement has a known answer.
func throttledProbeServer(targetMbps float64) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ok":true}`))
	})
	mux.HandleFunc("/api/v1/probe/download", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		size := 65536
		if v := r.URL.Query().Get("size"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				size = n
			}
		}
		const chunk = 32 * 1024
		perChunk := time.Duration(float64(chunk*8) / (targetMbps * 1e6) * float64(time.Second))
		buf := make([]byte, chunk)
		sent := 0
		for sent < size {
			n := chunk
			if size-sent < n {
				n = size - sent
			}
			time.Sleep(perChunk)
			w.Write(buf[:n])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			sent += n
		}
	})
	return httptest.NewServer(mux)
}

// meshd used to hardcode BandwidthMbps: 0, so every Go node advertised "unknown
// bandwidth" forever and looked identical to a node on a dial-up link to the
// placement algorithm. This proves the number is now really measured.
func TestMeasureDownloadTracksAKnownRate(t *testing.T) {
	for _, target := range []float64{20.0, 60.0} {
		t.Run(fmt.Sprintf("%.0fMbps", target), func(t *testing.T) {
			srv := throttledProbeServer(target)
			defer srv.Close()
			link := measureLink(srv.URL)
			if link.BandwidthMbps <= 0 {
				t.Fatalf("bandwidth not measured: %+v", link)
			}
			lo, hi := target*0.4, target*2.2
			if link.BandwidthMbps < lo || link.BandwidthMbps > hi {
				t.Fatalf("measured %.1f Mbps, want roughly %.0f (accepted %.1f..%.1f)",
					link.BandwidthMbps, target, lo, hi)
			}
			t.Logf("target %.0f Mbps -> measured %.1f Mbps, latency %.2f ms",
				target, link.BandwidthMbps, link.LatencyMS)
		})
	}
}

// A slower link must measure slower. This is the ordering property placement
// actually depends on, and it is immune to absolute-rate calibration drift.
func TestSlowerLinkMeasuresSlower(t *testing.T) {
	slow := throttledProbeServer(10.0)
	defer slow.Close()
	fast := throttledProbeServer(100.0)
	defer fast.Close()

	slowMbps := measureLink(slow.URL).BandwidthMbps
	fastMbps := measureLink(fast.URL).BandwidthMbps
	t.Logf("slow=%.1f Mbps fast=%.1f Mbps", slowMbps, fastMbps)
	if !(fastMbps > slowMbps*2) {
		t.Fatalf("a 10x faster link measured %.1f vs %.1f — placement cannot rank nodes",
			fastMbps, slowMbps)
	}
}

// Absent probe endpoint => 0 meaning "unknown", never a number invented from
// health-check payload sizes. Latency must still be measured.
func TestAbsentProbeEndpointReportsUnknownNotAGuess(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ok":true}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	link := measureLink(srv.URL)
	if link.BandwidthMbps != 0 {
		t.Fatalf("fabricated %.2f Mbps with no probe endpoint", link.BandwidthMbps)
	}
	if link.LatencyMS < 0 {
		t.Fatalf("latency should still be measured, got %.3f", link.LatencyMS)
	}
}

// An unreachable control plane must not report a plausible-looking zero-latency
// link; the whole struct is left empty.
func TestUnreachableControlPlaneReportsNothing(t *testing.T) {
	link := measureLink("http://127.0.0.1:9")
	if link.LatencyMS != 0 || link.BandwidthMbps != 0 {
		t.Fatalf("expected an empty Link for an unreachable CP, got %+v", link)
	}
}
