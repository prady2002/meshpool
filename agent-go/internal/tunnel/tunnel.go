// Package tunnel implements the agent-initiated reverse tunnel.
//
// A consumer PC usually has no reachable inbound address: roughly a third of
// residential broadband is behind CGNAT, where inbound connections are simply
// impossible and no amount of port forwarding helps because the user does not
// control the carrier's NAT. So instead of waiting to be called, the agent
// dials *out* to the coordinator and holds open a streaming GET. The
// coordinator writes framed requests onto that stream; the agent executes each
// one against its own loopback server and POSTs the reply back.
//
// Frames are executed as ordinary loopback HTTP against the same routes the
// direct path uses, so the tunnel is a thin forwarder rather than a second
// implementation of every handler — one code path, one test surface.
//
// This mirrors meshpool/agent/tunnel_client.py frame for frame. The two must
// stay wire-compatible: the same coordinator serves both.
package tunnel

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	// HeaderNodeID and HeaderNodeToken must match meshpool/cloud/tunnel.py.
	HeaderNodeID    = "X-MeshPool-Node-Id"
	HeaderNodeToken = "X-MeshPool-Node-Token"

	maxBackoff  = 10 * time.Second
	baseBackoff = 500 * time.Millisecond

	// Bound on frames executed at once. Python leaves this unbounded, but an
	// unbounded goroutine fan-out on a donated home machine is exactly the
	// behaviour a contributor would notice and uninstall over.
	maxInflight = 32
)

// Session is the slice of agent state the tunnel needs. It is an interface so
// this package does not import the agent package (which imports this one).
type Session interface {
	// NodeID and NodeToken are re-read on every reconnect because
	// re-registration replaces both.
	NodeID() string
	NodeToken() string
	// Reregister mints a fresh session token after the coordinator has
	// forgotten us (it restarted, or our token expired).
	Reregister(ctx context.Context) bool
}

// Client runs the reverse tunnel until its context is cancelled.
type Client struct {
	ControlPlaneURL string
	LocalURL        string
	Session         Session

	// stream must not carry an overall timeout: the whole point is a
	// long-lived connection. Liveness comes from the coordinator's keepalive
	// ping plus the response-header timeout on dial.
	stream *http.Client
	// work carries the short request/reply calls and does have a timeout.
	work *http.Client
	once sync.Once
}

func (c *Client) init() {
	c.once.Do(func() {
		c.stream = &http.Client{
			Timeout: 0,
			Transport: &http.Transport{
				ResponseHeaderTimeout: 15 * time.Second,
				// A donated machine may sleep or change networks; do not let a
				// dead socket linger and look like a healthy idle tunnel.
				IdleConnTimeout: 90 * time.Second,
			},
		}
		c.work = &http.Client{Timeout: 120 * time.Second}
	})
}

// Run dials the coordinator and serves frames, reconnecting with exponential
// backoff. It returns only when ctx is cancelled.
func (c *Client) Run(ctx context.Context) {
	c.init()
	streamURL := strings.TrimRight(c.ControlPlaneURL, "/") + "/tunnel/stream"
	replyURL := strings.TrimRight(c.ControlPlaneURL, "/") + "/tunnel/reply"

	// Wait for the first registration to mint a token.
	for c.Session.NodeID() == "" || c.Session.NodeToken() == "" {
		select {
		case <-ctx.Done():
			return
		case <-time.After(200 * time.Millisecond):
		}
	}

	backoff := baseBackoff
	for ctx.Err() == nil {
		if healthy := c.serve(ctx, streamURL, replyURL); healthy {
			backoff = baseBackoff
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		if backoff < maxBackoff {
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}
}

// serve holds one stream open. It reports whether the connection was healthy
// enough to reset the backoff.
func (c *Client) serve(ctx context.Context, streamURL, replyURL string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, streamURL, nil)
	if err != nil {
		return false
	}
	req.Header.Set(HeaderNodeToken, c.Session.NodeToken())
	req.Header.Set(HeaderNodeID, c.Session.NodeID())

	resp, err := c.stream.Do(req)
	if err != nil {
		if ctx.Err() == nil {
			log.Printf("tunnel disconnected, retrying err=%s", truncate(err.Error(), 140))
		}
		return false
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// healthy
	case http.StatusUnauthorized, http.StatusNotFound:
		io.Copy(io.Discard, resp.Body)
		log.Printf("tunnel token stale; re-registering")
		c.Session.Reregister(ctx)
		return false
	default:
		io.Copy(io.Discard, resp.Body)
		log.Printf("tunnel stream rejected status=%d", resp.StatusCode)
		return false
	}

	log.Printf("tunnel connected cp=%s node=%s", c.ControlPlaneURL, c.Session.NodeID())

	var wg sync.WaitGroup
	sem := make(chan struct{}, maxInflight)
	scanner := bufio.NewScanner(resp.Body)
	// Frames carry base64 request bodies, so the default 64 KiB line cap is far
	// too small; one upload frame would otherwise silently kill the tunnel.
	scanner.Buffer(make([]byte, 0, 64*1024), 32*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal([]byte(line), &frame); err != nil {
			continue
		}
		if op, _ := frame["op"].(string); op == "ping" {
			continue // keepalive; nothing to reply
		}
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			wg.Wait()
			return true
		}
		wg.Add(1)
		go func(f map[string]any) {
			defer wg.Done()
			defer func() { <-sem }()
			c.dispatch(ctx, f, replyURL)
		}(frame)
	}
	wg.Wait()
	if err := scanner.Err(); err != nil && ctx.Err() == nil {
		log.Printf("tunnel disconnected, retrying err=%s", truncate(err.Error(), 140))
	}
	return true
}

func (c *Client) dispatch(ctx context.Context, frame map[string]any, replyURL string) {
	var reply map[string]any
	switch op, _ := frame["op"].(string); op {
	case "call":
		reply = c.handleCall(ctx, frame)
	case "proxy":
		reply = c.handleProxy(ctx, frame)
	default:
		reply = map[string]any{
			"id": frame["id"], "ok": false,
			"error": fmt.Sprintf("unknown op %v", frame["op"]),
		}
	}
	body, err := json.Marshal(reply)
	if err != nil {
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, replyURL, bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderNodeToken, c.Session.NodeToken())
	req.Header.Set(HeaderNodeID, c.Session.NodeID())
	resp, err := c.work.Do(req)
	if err != nil {
		return
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
}

// HandleCall forwards a control-plane API call to this agent's own loopback
// server, so tunnelled and direct requests hit identical handlers.
func (c *Client) handleCall(ctx context.Context, frame map[string]any) map[string]any {
	c.init()
	path := strings.TrimLeft(asString(frame["path"]), "/")
	method := asStringOr(frame["method"], http.MethodPost)
	payload := frame["payload"]
	if payload == nil {
		payload = map[string]any{}
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		raw = []byte("{}")
	}
	target := strings.TrimRight(c.LocalURL, "/") + "/" + path

	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(raw))
	if err != nil {
		return failure(frame["id"], "local call failed: "+truncate(err.Error(), 160))
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.work.Do(req)
	if err != nil {
		return failure(frame["id"], "local call failed: "+truncate(err.Error(), 160))
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)

	if resp.StatusCode >= 400 {
		return map[string]any{
			"id": frame["id"], "ok": false, "status": resp.StatusCode,
			"body":  map[string]any{},
			"error": fmt.Sprintf("HTTP %d: %s", resp.StatusCode, truncate(string(data), 160)),
		}
	}
	// A non-JSON 2xx body is reported as {} rather than an error, matching the
	// Python client: the coordinator only ever reads structured replies.
	var body any = map[string]any{}
	if len(data) > 0 {
		var parsed any
		if json.Unmarshal(data, &parsed) == nil {
			body = parsed
		}
	}
	return map[string]any{
		"id": frame["id"], "ok": true, "status": resp.StatusCode,
		"body": body, "error": nil,
	}
}

// handleProxy forwards hosted-app traffic from the agent's vantage point to a
// loopback replica and returns the raw response verbatim.
func (c *Client) handleProxy(ctx context.Context, frame map[string]any) map[string]any {
	c.init()
	method := asStringOr(frame["method"], http.MethodGet)
	target := asString(frame["url"])

	content, err := base64.StdEncoding.DecodeString(asString(frame["content_b64"]))
	if err != nil {
		content = nil
	}
	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(content))
	if err != nil {
		return failure(frame["id"], "local proxy failed: "+truncate(err.Error(), 160))
	}
	if raw, ok := frame["params"].([]any); ok && len(raw) > 0 {
		q := url.Values{}
		for _, pair := range raw {
			kv, ok := pair.([]any)
			if !ok || len(kv) != 2 {
				continue
			}
			q.Add(asString(kv[0]), asString(kv[1]))
		}
		if encoded := q.Encode(); encoded != "" {
			if req.URL.RawQuery == "" {
				req.URL.RawQuery = encoded
			} else {
				req.URL.RawQuery += "&" + encoded
			}
		}
	}
	if hdrs, ok := frame["headers"].(map[string]any); ok {
		for k, v := range hdrs {
			// Host must be set on the request, not the header map, or Go
			// silently ignores it and virtual-hosted apps route wrong.
			if strings.EqualFold(k, "host") {
				req.Host = asString(v)
				continue
			}
			req.Header.Set(k, asString(v))
		}
	}

	resp, err := c.work.Do(req)
	if err != nil {
		return failure(frame["id"], "local proxy failed: "+truncate(err.Error(), 160))
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)

	headers := map[string]any{}
	for k := range resp.Header {
		headers[k] = resp.Header.Get(k)
	}
	return map[string]any{
		"id": frame["id"], "ok": true,
		"body": map[string]any{
			"status":      resp.StatusCode,
			"headers":     headers,
			"content_b64": base64.StdEncoding.EncodeToString(data),
		},
	}
}

func failure(id any, msg string) map[string]any {
	return map[string]any{"id": id, "ok": false, "error": msg}
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asStringOr(v any, fallback string) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return fallback
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
