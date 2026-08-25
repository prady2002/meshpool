package agent

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"meshd/internal/probe"
)

// serviceForRoute maps an agent endpoint to the workload class the control-plane
// planners ask about before they place work.
var serviceForRoute = map[string]string{
	"/start_replica": "hosting",
	"/stop_replica":  "hosting",
	"/replica_logs":  "hosting",
	"/shard/load":    "shards",
	"/shard/forward": "shards",
	"/shard/drop":    "shards",
	"/shard/unload":  "shards",
	"/run_unit":      "batch",
	"/blob_put":      "storage",
	"/blob_get":      "storage",
}

// meshd must never advertise a service whose routes answer 501.
//
// This is the structural half of the guard: probe.Services is a hand-written
// list, and a hand-written list drifts. The moment someone implements
// /start_replica and removes it from unimplementedRoutes without adding
// "hosting" to probe.Services, meshd keeps being excluded from hosting forever
// and nothing else notices. The moment someone adds "shards" optimistically
// before implementing it, every model deploy that touches a meshd node breaks.
// Both directions are checked here.
func TestAdvertisedServicesMatchImplementedRoutes(t *testing.T) {
	advertised := map[string]bool{}
	for _, s := range probe.Services {
		advertised[s] = true
	}

	unimplemented := map[string]bool{}
	for _, r := range UnimplementedRoutes() {
		unimplemented[r] = true
	}

	for route, service := range serviceForRoute {
		if unimplemented[route] && advertised[service] {
			t.Errorf("meshd advertises service %q but %s returns 501; "+
				"the hosting/inference planners commit to a plan and do not retry elsewhere",
				service, route)
		}
	}

	// ...and the converse: a service must be advertised if its routes are live.
	for route, service := range serviceForRoute {
		if !unimplemented[route] && !advertised[service] {
			t.Errorf("%s is implemented but meshd does not advertise service %q, "+
				"so the scheduler will never send it this work", route, service)
		}
	}
}

// Non-vacuity: the routes we claim are unimplemented really must answer 501, and
// the ones we claim are implemented really must not. A test built on a stale
// route list proves nothing.
func TestUnimplementedRoutesReallyReturn501(t *testing.T) {
	a := &Agent{}
	h := a.routes()
	for _, route := range UnimplementedRoutes() {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, route, strings.NewReader("{}")))
		if rec.Code != http.StatusNotImplemented {
			t.Errorf("%s returned %d, want 501 -- it is listed as unimplemented",
				route, rec.Code)
		}
	}
	// /health is implemented, so it must not be a 501.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code == http.StatusNotImplemented {
		t.Fatalf("/health returned 501; the route table is not wired at all, so the "+
			"assertions above would pass vacuously (code=%d)", rec.Code)
	}
}
