package tunnel

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeSession struct {
	mu         sync.Mutex
	id, token  string
	reregister int
}

func (f *fakeSession) NodeID() string    { f.mu.Lock(); defer f.mu.Unlock(); return f.id }
func (f *fakeSession) NodeToken() string { f.mu.Lock(); defer f.mu.Unlock(); return f.token }
func (f *fakeSession) Reregister(context.Context) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.reregister++
	f.token = "fresh-token"
	return true
}

// newClient wires a Client against a fake local agent, exactly as Run would.
func newClient(localURL string) (*Client, *fakeSession) {
	s := &fakeSession{id: "node-1", token: "tok-1"}
	c := &Client{LocalURL: localURL, Session: s}
	c.init()
	return c, s
}

func TestCallFrameHitsLocalAgentAndReportsBody(t *testing.T) {
	var gotPath, gotMethod, gotBody string
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotPath, gotMethod, gotBody = r.URL.Path, r.Method, string(body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ok_field":42}`)
	}))
	defer local.Close()

	c, _ := newClient(local.URL)
	reply := c.handleCall(context.Background(), map[string]any{
		"id": "f1", "op": "call", "method": "POST", "path": "/run_unit",
		"payload": map[string]any{"unit": "u-1"},
	})

	if gotPath != "/run_unit" || gotMethod != "POST" {
		t.Fatalf("local agent saw %s %s", gotMethod, gotPath)
	}
	if !strings.Contains(gotBody, `"unit":"u-1"`) {
		t.Fatalf("payload not forwarded: %s", gotBody)
	}
	if reply["ok"] != true || reply["id"] != "f1" {
		t.Fatalf("reply = %v", reply)
	}
	body, ok := reply["body"].(map[string]any)
	if !ok || body["ok_field"] != float64(42) {
		t.Fatalf("body not round-tripped: %v", reply["body"])
	}
}

func TestCallFrameReportsLocalErrorStatus(t *testing.T) {
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not implemented here", http.StatusNotImplemented)
	}))
	defer local.Close()

	c, _ := newClient(local.URL)
	reply := c.handleCall(context.Background(), map[string]any{
		"id": "f2", "op": "call", "path": "/shard/load",
	})
	if reply["ok"] != false {
		t.Fatalf("a 501 from the local agent must not be reported as success: %v", reply)
	}
	if reply["status"] != http.StatusNotImplemented {
		t.Fatalf("status not propagated: %v", reply["status"])
	}
	if msg, _ := reply["error"].(string); !strings.Contains(msg, "501") {
		t.Fatalf("error text lost the status: %q", msg)
	}
}

func TestCallFrameFailsClosedWhenLocalAgentIsDown(t *testing.T) {
	// Port 1 on loopback is not listening; the frame must fail, not hang or panic.
	c, _ := newClient("http://127.0.0.1:1")
	reply := c.handleCall(context.Background(), map[string]any{"id": "f3", "path": "/health"})
	if reply["ok"] != false {
		t.Fatalf("unreachable local agent reported ok: %v", reply)
	}
}

func TestProxyFrameForwardsMethodQueryHeadersAndBody(t *testing.T) {
	var gotQuery, gotHeader, gotHost, gotBody, gotMethod string
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotQuery, gotHeader, gotHost = r.URL.RawQuery, r.Header.Get("X-Trace"), r.Host
		gotBody, gotMethod = string(body), r.Method
		w.Header().Set("X-Reply", "yes")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprint(w, "hello-body")
	}))
	defer local.Close()

	c, _ := newClient(local.URL)
	reply := c.handleProxy(context.Background(), map[string]any{
		"id": "p1", "op": "proxy", "method": "PUT", "url": local.URL + "/app",
		"params":      []any{[]any{"q", "1"}, []any{"r", "two"}},
		"headers":     map[string]any{"X-Trace": "abc", "Host": "app.example.com"},
		"content_b64": base64.StdEncoding.EncodeToString([]byte("payload-bytes")),
	})

	if gotMethod != "PUT" {
		t.Fatalf("method not forwarded: %s", gotMethod)
	}
	if !strings.Contains(gotQuery, "q=1") || !strings.Contains(gotQuery, "r=two") {
		t.Fatalf("query params lost: %q", gotQuery)
	}
	if gotHeader != "abc" {
		t.Fatalf("header lost: %q", gotHeader)
	}
	if gotHost != "app.example.com" {
		t.Fatalf("Host must be applied to the request, got %q", gotHost)
	}
	if gotBody != "payload-bytes" {
		t.Fatalf("body lost: %q", gotBody)
	}

	body, ok := reply["body"].(map[string]any)
	if !ok {
		t.Fatalf("proxy reply shape: %v", reply)
	}
	if body["status"] != http.StatusCreated {
		t.Fatalf("status lost: %v", body["status"])
	}
	raw, err := base64.StdEncoding.DecodeString(body["content_b64"].(string))
	if err != nil || string(raw) != "hello-body" {
		t.Fatalf("content not round-tripped: %v %q", err, raw)
	}
	hdrs, _ := body["headers"].(map[string]any)
	if hdrs["X-Reply"] != "yes" {
		t.Fatalf("response headers lost: %v", hdrs)
	}
}

func TestPingFramesAreNeverAnswered(t *testing.T) {
	// A reply to a keepalive would be an unsolicited POST the coordinator has
	// no pending request for.
	var replies int32
	var mu sync.Mutex
	cp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/tunnel/stream"):
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			for i := 0; i < 3; i++ {
				fmt.Fprint(w, "{\"op\":\"ping\"}\n")
				flusher.Flush()
			}
			time.Sleep(150 * time.Millisecond)
		case strings.HasSuffix(r.URL.Path, "/tunnel/reply"):
			mu.Lock()
			replies++
			mu.Unlock()
			fmt.Fprint(w, "{}")
		}
	}))
	defer cp.Close()

	c := &Client{ControlPlaneURL: cp.URL, LocalURL: cp.URL, Session: &fakeSession{id: "n", token: "t"}}
	c.init()
	ctx, cancel := context.WithTimeout(context.Background(), 900*time.Millisecond)
	defer cancel()
	c.serve(ctx, cp.URL+"/tunnel/stream", cp.URL+"/tunnel/reply")

	mu.Lock()
	defer mu.Unlock()
	if replies != 0 {
		t.Fatalf("keepalive pings produced %d replies, want 0", replies)
	}
}

func TestStaleTokenTriggersReregistration(t *testing.T) {
	cp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unknown node", http.StatusUnauthorized)
	}))
	defer cp.Close()

	s := &fakeSession{id: "n", token: "stale"}
	c := &Client{ControlPlaneURL: cp.URL, LocalURL: cp.URL, Session: s}
	c.init()
	if healthy := c.serve(context.Background(), cp.URL+"/tunnel/stream", cp.URL+"/tunnel/reply"); healthy {
		t.Fatal("a 401 stream must not count as a healthy connection")
	}
	if s.reregister != 1 {
		t.Fatalf("re-registration attempts = %d, want 1", s.reregister)
	}
}

func TestLargeFrameExceedsDefaultScannerLimit(t *testing.T) {
	// Regression guard: bufio.Scanner defaults to 64 KiB per line, and a single
	// blob-upload frame is far larger. Without the raised buffer the tunnel
	// would drop the connection instead of serving the request.
	big := base64.StdEncoding.EncodeToString(make([]byte, 512*1024))
	frame, _ := json.Marshal(map[string]any{"op": "call", "id": "big", "path": "/blob", "payload": map[string]any{"b": big}})
	if len(frame) <= 64*1024 {
		t.Fatalf("fixture is only %d bytes; it must exceed the 64 KiB default", len(frame))
	}

	var served int32
	var mu sync.Mutex
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		mu.Lock()
		served++
		mu.Unlock()
		fmt.Fprint(w, "{}")
	}))
	defer local.Close()

	cp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/tunnel/stream") {
			w.WriteHeader(http.StatusOK)
			w.Write(append(frame, '\n'))
			w.(http.Flusher).Flush()
			time.Sleep(400 * time.Millisecond)
			return
		}
		fmt.Fprint(w, "{}")
	}))
	defer cp.Close()

	c := &Client{ControlPlaneURL: cp.URL, LocalURL: local.URL, Session: &fakeSession{id: "n", token: "t"}}
	c.init()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	c.serve(ctx, cp.URL+"/tunnel/stream", cp.URL+"/tunnel/reply")

	mu.Lock()
	defer mu.Unlock()
	if served != 1 {
		t.Fatalf("large frame served %d times, want 1", served)
	}
}
