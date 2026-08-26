package quictransport

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestKeepAliveMatchesResidentialNATBudget(t *testing.T) {
	if keepAlive < 20_000_000_000 || keepAlive > 25_000_000_000 {
		t.Fatalf("keepalive=%s, want 20-25s for residential UDP NAT bindings", keepAlive)
	}
}

func TestHandleCallUsesLoopbackAgentAPI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/echo" {
			t.Fatalf("path=%s, want /echo", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		json.NewEncoder(w).Encode(map[string]any{"seen": body["x"]})
	}))
	defer srv.Close()

	c := &Client{LocalURL: srv.URL}
	c.init()
	reply := c.handleCall(context.Background(), map[string]any{
		"op": "call", "path": "echo", "method": "POST", "payload": map[string]any{"x": "ok"},
	})
	if !reply["ok"].(bool) {
		t.Fatalf("reply not ok: %#v", reply)
	}
	body := reply["body"].(map[string]any)
	if body["seen"] != "ok" {
		t.Fatalf("body=%#v, want seen=ok", body)
	}
}

func TestHandleProxyPreservesResponseBytes(t *testing.T) {
	want := []byte("hello over quic")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Test", "yes")
		w.WriteHeader(http.StatusAccepted)
		w.Write(want)
	}))
	defer srv.Close()

	c := &Client{}
	c.init()
	reply := c.handleProxy(context.Background(), map[string]any{
		"op": "proxy", "method": "GET", "url": srv.URL,
	})
	if !reply["ok"].(bool) {
		t.Fatalf("reply not ok: %#v", reply)
	}
	body := reply["body"].(map[string]any)
	if int(body["status"].(int)) != http.StatusAccepted {
		t.Fatalf("status=%v", body["status"])
	}
	got, err := base64.StdEncoding.DecodeString(body["content_b64"].(string))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}
