// Package agent wires the meshd pieces together: device identity, control-plane
// enrollment, the heartbeat loop, and the HTTP surface the coordinator calls back on.
//
// This is deliberately a *subset* of the Python daemon (meshpool/agent/daemon.py):
// identity + registration + heartbeat + the pooled blob store. Endpoints that are not
// implemented yet are not faked -- they return 501 with an explicit message, so a
// coordinator that routes work here fails loudly instead of silently getting a wrong
// answer. `sim/scenarios/verify_go_agent_parity.py` pins exactly the subset below.
package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"meshd/internal/blob"
	"meshd/internal/config"
	"meshd/internal/identity"
	"meshd/internal/probe"
	"meshd/internal/tunnel"
	"meshd/internal/wasmfn"
)

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

// wasmPrefix marks a user-supplied function; everything after it is the module's
// sha256. Must stay in sync with meshpool/batch/functions.py WASM_PREFIX.
const wasmPrefix = "wasm:"

// Agent owns every long-lived resource meshd needs.
type Agent struct {
	cfg   config.Config
	id    identity.Identity
	blobs *blob.Store
	cap   probe.Capability
	link  probe.Link
	http  *http.Client
	fns   *wasmfn.Runner

	mu                sync.RWMutex
	nodeID            string
	nodeToken         string
	heartbeatInterval time.Duration
	startedUnix       int64
	reclaimed         bool
}

// New probes the machine and prepares state; it performs no network I/O.
func New(cfg config.Config) (*Agent, error) {
	id, err := identity.LoadOrCreate(cfg.DataDir)
	if err != nil {
		return nil, fmt.Errorf("identity: %w", err)
	}
	store, err := blob.New(probe.BlobRoot(cfg.DataDir))
	if err != nil {
		return nil, fmt.Errorf("blob store: %w", err)
	}
	capability, link := probe.Probe(cfg)
	runner, err := wasmfn.NewRunner(context.Background(), wasmfn.DefaultLimits())
	if err != nil {
		return nil, fmt.Errorf("wasm runtime: %w", err)
	}
	return &Agent{
		cfg:               cfg,
		id:                id,
		blobs:             store,
		cap:               capability,
		link:              link,
		http:              &http.Client{Timeout: 30 * time.Second},
		fns:               runner,
		heartbeatInterval: 3 * time.Second,
		startedUnix:       time.Now().Unix(),
	}, nil
}

// Run serves the agent HTTP API and keeps the coordinator registration alive until
// ctx is cancelled.
func (a *Agent) Run(ctx context.Context) error {
	srv := &http.Server{
		Addr:              fmt.Sprintf("%s:%d", a.cfg.AgentBind, a.cfg.AgentPort),
		Handler:           a.routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ln, err := net.Listen("tcp", srv.Addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", srv.Addr, err)
	}

	errCh := make(chan error, 1)
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()
	log.Printf("meshd online addr=%s advertise=%s device=%s cp=%s",
		srv.Addr, a.cfg.AgentURL(), a.id.DeviceID, a.cfg.ControlPlaneURL)

	go a.registerLoop(ctx)
	go a.heartbeatLoop(ctx)
	// The reverse tunnel is what lets a machine behind CGNAT participate at
	// all. It is started unconditionally rather than only when inbound looks
	// unreachable, because a node cannot reliably detect its own reachability
	// from the inside and a wrong guess strands it silently.
	go (&tunnel.Client{
		ControlPlaneURL: a.cfg.ControlPlaneURL,
		LocalURL:        a.cfg.LocalURL(),
		Session:         a,
	}).Run(ctx)

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	}
}

// --- control-plane enrollment --------------------------------------------- //

type registerRequest struct {
	Name       string           `json:"name"`
	URL        string           `json:"url"`
	Owner      string           `json:"owner"`
	Capability probe.Capability `json:"capability"`
	Link       probe.Link       `json:"link"`
	Policy     policy           `json:"policy"`
	EnrollTok  string           `json:"enroll_token,omitempty"`
	DeviceID   string           `json:"device_id"`
	PublicKey  string           `json:"public_key"`
	Signature  string           `json:"signature"`
	TsMS       int64            `json:"ts_ms"`
}

type policy struct {
	CPUPct      int    `json:"cpu_pct"`
	RAMMB       int    `json:"ram_mb"`
	GPUAllowed  bool   `json:"gpu_allowed"`
	Schedule    string `json:"schedule"`
	IdleMinutes int    `json:"idle_minutes"`
}

type nodeLoad struct {
	CPULoadPct   float64 `json:"cpu_load_pct"`
	RAMUsedMB    int     `json:"ram_used_mb"`
	RunningUnits int     `json:"running_units"`
	IdleSeconds  float64 `json:"idle_seconds"`
}

type heartbeat struct {
	NodeID string   `json:"node_id"`
	Load   nodeLoad `json:"load"`
	Policy policy   `json:"policy"`
	Status string   `json:"status"`
}

func (a *Agent) policy() policy {
	return policy{
		CPUPct:      a.cfg.DonateCPUPct,
		RAMMB:       a.cfg.DonateRAMMB,
		GPUAllowed:  false,
		Schedule:    a.cfg.Schedule,
		IdleMinutes: 10,
	}
}

// registerOnce performs a single signed enrollment attempt. Returns true when the
// coordinator accepted it and the node id is now known.
func (a *Agent) registerOnce(ctx context.Context) bool {
	ts := time.Now().UnixMilli()
	sig, err := a.id.SignEnrollment(ts)
	if err != nil {
		log.Printf("register: sign failed: %v", err)
		return false
	}
	body, err := json.Marshal(registerRequest{
		Name:       a.cfg.Name,
		URL:        a.cfg.AgentURL(),
		Owner:      a.cfg.Owner,
		Capability: a.cap,
		Link:       a.link,
		Policy:     a.policy(),
		EnrollTok:  a.cfg.EnrollToken,
		DeviceID:   a.id.DeviceID,
		PublicKey:  a.id.PublicKey,
		Signature:  sig,
		TsMS:       ts,
	})
	if err != nil {
		log.Printf("register: marshal failed: %v", err)
		return false
	}
	url := strings.TrimRight(a.cfg.ControlPlaneURL, "/") + "/api/v1/nodes/register"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.http.Do(req)
	if err != nil {
		log.Printf("register failed, retrying: %v", err)
		return false
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		log.Printf("register rejected: %s %s", resp.Status, truncate(string(raw), 160))
		return false
	}
	var out struct {
		NodeID            string  `json:"node_id"`
		NodeToken         string  `json:"node_token"`
		HeartbeatInterval float64 `json:"heartbeat_interval_s"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || out.NodeID == "" {
		log.Printf("register: bad response: %s", truncate(string(raw), 160))
		return false
	}
	a.mu.Lock()
	a.nodeID = out.NodeID
	a.nodeToken = out.NodeToken
	if out.HeartbeatInterval > 0 {
		a.heartbeatInterval = time.Duration(out.HeartbeatInterval * float64(time.Second))
	}
	a.mu.Unlock()
	log.Printf("registered with control plane node=%s device=%s", out.NodeID, a.id.DeviceID)
	return true
}

func (a *Agent) registerLoop(ctx context.Context) {
	for {
		if a.NodeID() != "" {
			return
		}
		if a.registerOnce(ctx) {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
}

func (a *Agent) heartbeatLoop(ctx context.Context) {
	for {
		a.mu.RLock()
		nodeID, interval := a.nodeID, a.heartbeatInterval
		a.mu.RUnlock()
		if nodeID != "" {
			a.sendHeartbeat(ctx, nodeID)
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(interval):
		}
	}
}

func (a *Agent) sendHeartbeat(ctx context.Context, nodeID string) {
	body, err := json.Marshal(heartbeat{
		NodeID: nodeID,
		Load:   nodeLoad{IdleSeconds: float64(time.Now().Unix() - a.startedUnix)},
		Policy: a.policy(),
		Status: "online",
	})
	if err != nil {
		return
	}
	url := strings.TrimRight(a.cfg.ControlPlaneURL, "/") + "/api/v1/nodes/heartbeat"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.http.Do(req)
	if err != nil {
		return
	}
	// A coordinator restart invalidates our node id; re-enroll rather than
	// heartbeating into the void forever.
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusUnauthorized {
		a.mu.Lock()
		a.nodeID = ""
		a.mu.Unlock()
		go a.registerLoop(ctx)
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16)) //nolint:errcheck
	resp.Body.Close()
}

// NodeID returns the coordinator-assigned id, or "" before enrollment completes.
func (a *Agent) NodeID() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.nodeID
}

// NodeToken returns the current session token. It changes on re-registration,
// so the tunnel reads it fresh on every reconnect rather than caching it.
func (a *Agent) NodeToken() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.nodeToken
}

// Reregister mints a fresh session after the coordinator has forgotten this
// node. Clearing nodeID first is what makes registerOnce actually re-enroll
// instead of short-circuiting on the stale id.
func (a *Agent) Reregister(ctx context.Context) bool {
	a.mu.Lock()
	a.nodeID = ""
	a.nodeToken = ""
	a.mu.Unlock()
	return a.registerOnce(ctx)
}

// --- HTTP surface ---------------------------------------------------------- //

// unimplementedRoutes are the endpoints the Python daemon serves that meshd does
// not implement yet. Exported (via UnimplementedRoutes) so a test can assert that
// nothing in here is advertised in probe.Services -- otherwise the advertisement
// and the reality drift the moment someone implements one of them.
var unimplementedRoutes = []string{
	"/policy",
	"/start_replica", "/stop_replica", "/replica_logs",
	"/shard/load", "/shard/forward", "/shard/drop", "/shard/unload",
	"/ollama/generate",
}

// UnimplementedRoutes returns a copy of the routes that answer 501.
func UnimplementedRoutes() []string {
	out := make([]string, len(unimplementedRoutes))
	copy(out, unimplementedRoutes)
	return out
}

func (a *Agent) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", a.handleHealth)
	mux.HandleFunc("/blob_put", a.handleBlobPut)
	mux.HandleFunc("/blob_get", a.handleBlobGet)
	mux.HandleFunc("/blob_has", a.handleBlobHas)
	mux.HandleFunc("/blob_delete", a.handleBlobDelete)
	mux.HandleFunc("/blob_pull", a.handleBlobPull)
	mux.HandleFunc("/run_unit", a.handleRunUnit)
	mux.HandleFunc("/reclaim", a.handleReclaim)
	mux.HandleFunc("/resume", a.handleResume)
	// Everything the Python daemon serves but meshd does not implement yet.
	for _, path := range unimplementedRoutes {
		p := path
		mux.HandleFunc(p, func(w http.ResponseWriter, r *http.Request) {
			writeErr(w, http.StatusNotImplemented,
				"meshd does not implement "+p+" yet; use the Python agent for this workload")
		})
	}
	return mux
}

// --- running user functions ------------------------------------------------ //

type runUnitRequest struct {
	JobID        string            `json:"job_id"`
	UnitID       string            `json:"unit_id"`
	Func         string            `json:"func"`
	Payload      map[string]any    `json:"payload"`
	CPUPct       int               `json:"cpu_pct"`
	RAMMb        int               `json:"ram_mb"`
	SecretLeases map[string]string `json:"secret_leases"`
	WasmSources  []string          `json:"wasm_sources"`
}

// handleRunUnit executes one work unit.
//
// meshd runs *user* functions -- wasm:<sha256> modules -- and only those. The curated
// built-ins (square, is_prime, embed, ...) live in the Python registry and are not
// reimplemented here on purpose: two independent implementations of the same named
// function is precisely how a pool starts returning different answers for the same
// input, which would make the coordinator's spot-check oracle slash honest nodes.
// A built-in therefore still gets an explicit 501 rather than a guess.
func (a *Agent) handleRunUnit(w http.ResponseWriter, r *http.Request) {
	var req runUnitRequest
	if !decode(w, r, &req) {
		return
	}
	a.mu.RLock()
	draining := a.reclaimed
	a.mu.RUnlock()
	if draining {
		writeErr(w, http.StatusServiceUnavailable, "node draining (reclaimed by owner)")
		return
	}
	if len(req.SecretLeases) > 0 {
		// Redeeming a lease requires the node-scoped token exchange the Python daemon
		// implements. Running the unit *without* the secrets would hand the user a
		// silently wrong answer, so refuse instead.
		writeErr(w, http.StatusNotImplemented,
			"meshd cannot redeem secret leases yet; schedule secret-bearing units "+
				"onto a Python agent")
		return
	}
	if !strings.HasPrefix(req.Func, wasmPrefix) {
		writeErr(w, http.StatusNotImplemented,
			"meshd runs wasm:<sha256> user functions only; the built-in function "+
				truncate(req.Func, 40)+" is served by the Python agent")
		return
	}
	moduleID := strings.ToLower(strings.TrimSpace(req.Func[len(wasmPrefix):]))
	if !hex64.MatchString(moduleID) {
		writeErr(w, http.StatusBadRequest,
			"malformed wasm function name: expected 'wasm:<64-hex-char sha256>'")
		return
	}

	ctx := r.Context()
	module, ok, err := a.blobs.Get(moduleID)
	if err != nil {
		writeErr(w, http.StatusInternalServerError,
			"could not read the local module cache: "+truncate(err.Error(), 150))
		return
	}
	if !ok {
		if len(req.WasmSources) == 0 {
			writeErr(w, http.StatusNotFound, fmt.Sprintf(
				"wasm module %s is not on this node and no sources were supplied",
				moduleID[:12]))
			return
		}
		pulled, err := a.pullVerified(ctx, moduleID, req.WasmSources)
		if err != nil {
			writeErr(w, http.StatusBadGateway, fmt.Sprintf(
				"could not fetch wasm module %s: %s", moduleID[:12], truncate(err.Error(), 150)))
			return
		}
		module = pulled
		// Best-effort cache for the next unit; a full disk must not fail the unit.
		if _, err := a.blobs.Put(moduleID, module); err != nil {
			log.Printf("meshd: could not cache module %s: %v", moduleID[:12], err)
		}
	}

	// Content addressing is the integrity check. pullVerified already enforces it, but
	// a blob read back from disk is checked again here: bit-rot or a tampered cache
	// file must never become executed code.
	sum := sha256.Sum256(module)
	if hex.EncodeToString(sum[:]) != moduleID {
		_ = a.blobs.Delete(moduleID)
		writeErr(w, http.StatusInternalServerError, fmt.Sprintf(
			"local copy of wasm module %s failed its content-hash check; evicted",
			moduleID[:12]))
		return
	}

	// The guest never sees MeshPool's internal plumbing.
	guest := map[string]any{}
	for k, v := range req.Payload {
		if k == "_wasm_module" {
			continue
		}
		guest[k] = v
	}

	started := time.Now()
	res, err := a.fns.Run(ctx, module, guest)
	elapsed := float64(time.Since(started).Nanoseconds()) / 1e6
	if err != nil {
		var fnErr *wasmfn.Error
		status := http.StatusInternalServerError
		detail := truncate(err.Error(), 200)
		if errors.As(err, &fnErr) {
			// A policy stop is worth retrying on a faster/emptier node; a user bug is
			// not, and 400 tells the coordinator to stop burning the retry budget.
			switch fnErr.Kind {
			case "timeout", "memory", "output":
				status = http.StatusRequestTimeout
			case "invalid_module", "bad_output", "exit", "bad_input":
				status = http.StatusBadRequest
			}
			detail = truncate(fnErr.Kind+": "+fnErr.Message, 200)
		}
		writeErr(w, status, detail)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"unit_id":    req.UnitID,
		"result":     res.Value,
		"compute_ms": elapsed,
	})
}

func (a *Agent) handleReclaim(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	a.mu.Lock()
	a.reclaimed = true
	a.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "draining", "message": "all donated resources reclaimed"})
}

func (a *Agent) handleResume(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	a.mu.Lock()
	a.reclaimed = false
	a.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"status": "online"})
}

func (a *Agent) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		writeErr(w, http.StatusMethodNotAllowed, "GET only")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"node_id":    a.NodeID(),
		"name":       a.cfg.Name,
		"status":     "online",
		"agent":      "meshd-go",
		"capability": a.cap,
		"policy":     a.policy(),
		"load":       nodeLoad{},
		"cpu_slots":  a.cap.CPUCores,
	})
}

type blobRequest struct {
	BlobID string `json:"blob_id"`
	Data   string `json:"data"`
}

func (a *Agent) handleBlobPut(w http.ResponseWriter, r *http.Request) {
	var req blobRequest
	if !decode(w, r, &req) {
		return
	}
	if !hex64.MatchString(req.BlobID) {
		writeErr(w, http.StatusBadRequest, "blob_id must be a sha256 hex digest")
		return
	}
	raw, err := base64.StdEncoding.DecodeString(req.Data)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "data must be valid base64")
		return
	}
	n, err := a.blobs.Put(req.BlobID, raw)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"blob_id": req.BlobID, "stored": true, "size": n})
}

func (a *Agent) handleBlobGet(w http.ResponseWriter, r *http.Request) {
	var req blobRequest
	if !decode(w, r, &req) {
		return
	}
	if !hex64.MatchString(req.BlobID) {
		writeErr(w, http.StatusBadRequest, "blob_id must be a sha256 hex digest")
		return
	}
	data, found, err := a.blobs.Get(req.BlobID)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if !found {
		writeJSON(w, http.StatusOK, map[string]any{
			"blob_id": req.BlobID, "found": false, "data": ""})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"blob_id": req.BlobID, "found": true,
		"data": base64.StdEncoding.EncodeToString(data)})
}

func (a *Agent) handleBlobHas(w http.ResponseWriter, r *http.Request) {
	var req blobRequest
	if !decode(w, r, &req) {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"blob_id": req.BlobID, "has": a.blobs.Has(req.BlobID)})
}

func (a *Agent) handleBlobDelete(w http.ResponseWriter, r *http.Request) {
	var req blobRequest
	if !decode(w, r, &req) {
		return
	}
	if err := a.blobs.Delete(req.BlobID); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"blob_id": req.BlobID, "deleted": true})
}

type pullChunk struct {
	ID      string   `json:"id"`
	Sources []string `json:"sources"`
}

type pullRequest struct {
	BlobID  string      `json:"blob_id"`
	Sources []string    `json:"sources"`
	Chunks  []pullChunk `json:"chunks"`
}

// handleBlobPull fetches a blob straight from a peer that already holds it, so the
// coordinator moves ids and peer URLs rather than bytes. Every piece is verified
// against its SHA-256 id, and a chunked object is re-verified after reassembly.
func (a *Agent) handleBlobPull(w http.ResponseWriter, r *http.Request) {
	var req pullRequest
	if !decode(w, r, &req) {
		return
	}
	if !hex64.MatchString(req.BlobID) {
		writeErr(w, http.StatusBadRequest, "blob_id must be a sha256 hex digest")
		return
	}
	if a.blobs.Has(req.BlobID) {
		writeJSON(w, http.StatusOK, map[string]any{
			"blob_id": req.BlobID, "stored": true, "bytes": 0, "cached": true})
		return
	}
	ctx := r.Context()
	var whole []byte
	if len(req.Chunks) > 0 {
		for _, ch := range req.Chunks {
			if !hex64.MatchString(ch.ID) {
				writeErr(w, http.StatusBadRequest, "chunk id must be a sha256 hex digest")
				return
			}
			part, err := a.pullVerified(ctx, ch.ID, ch.Sources)
			if err != nil {
				writeErr(w, http.StatusBadGateway, err.Error())
				return
			}
			whole = append(whole, part...)
		}
		sum := sha256.Sum256(whole)
		if hex.EncodeToString(sum[:]) != req.BlobID {
			writeErr(w, http.StatusBadGateway, "reassembled object failed integrity check")
			return
		}
	} else {
		part, err := a.pullVerified(ctx, req.BlobID, req.Sources)
		if err != nil {
			writeErr(w, http.StatusBadGateway, err.Error())
			return
		}
		whole = part
	}
	if _, err := a.blobs.Put(req.BlobID, whole); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"blob_id": req.BlobID, "stored": true, "bytes": len(whole), "cached": false})
}

// pullVerified tries each source in turn and returns the first payload whose SHA-256
// matches the requested id. A peer that serves corrupt bytes is skipped, not trusted.
func (a *Agent) pullVerified(ctx context.Context, blobID string, sources []string) ([]byte, error) {
	if len(sources) == 0 {
		return nil, fmt.Errorf("no sources for %s", blobID[:12])
	}
	var lastErr error
	for _, src := range sources {
		url := strings.TrimRight(src, "/") + "/blob_get"
		body, _ := json.Marshal(blobRequest{BlobID: blobID})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			lastErr = err
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := a.http.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		raw, err := io.ReadAll(io.LimitReader(resp.Body, 512<<20))
		resp.Body.Close()
		if err != nil || resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("peer %s: status %d", src, resp.StatusCode)
			continue
		}
		var out struct {
			Found bool   `json:"found"`
			Data  string `json:"data"`
		}
		if err := json.Unmarshal(raw, &out); err != nil || !out.Found {
			lastErr = fmt.Errorf("peer %s does not have %s", src, blobID[:12])
			continue
		}
		data, err := base64.StdEncoding.DecodeString(out.Data)
		if err != nil {
			lastErr = fmt.Errorf("peer %s sent invalid base64", src)
			continue
		}
		sum := sha256.Sum256(data)
		if hex.EncodeToString(sum[:]) != blobID {
			lastErr = fmt.Errorf("peer %s failed the content-hash check", src)
			continue
		}
		return data, nil
	}
	return nil, fmt.Errorf("all sources failed for %s: %v", blobID[:12], lastErr)
}

// --- helpers --------------------------------------------------------------- //

func decode(w http.ResponseWriter, r *http.Request, dst any) bool {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only")
		return false
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, 512<<20))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "unreadable body")
		return false
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		writeErr(w, http.StatusUnprocessableEntity, "invalid json body")
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

// writeErr mirrors FastAPI's {"detail": ...} error envelope so the existing Python
// control plane parses meshd errors exactly as it parses the Python agent's.
func writeErr(w http.ResponseWriter, status int, detail string) {
	writeJSON(w, status, map[string]any{"detail": detail})
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
