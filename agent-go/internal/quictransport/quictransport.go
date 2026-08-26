// Package quictransport implements the agent side of MeshPool's QUIC fabric.
//
// The agent still dials out to the coordinator, preserving the no-port-forwarding
// property of the HTTPS tunnel.  Once connected, coordinator requests arrive on
// independent QUIC streams, avoiding TCP head-of-line blocking and allowing QUIC's
// connection migration to survive NAT rebinding / Wi-Fi changes.
package quictransport

import (
	"bytes"
	"context"
	"crypto/tls"
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

	"meshd/internal/identity"
	"meshd/internal/natprobe"

	quic "github.com/quic-go/quic-go"
)

const (
	alpn = "meshpool-quic-v1"

	// Residential UDP NAT bindings can expire in ~15-30s.  Keepalives at 22s
	// match Tailscale's 20-26s operating range and keep cheap home routers warm.
	keepAlive = natprobe.KeepAliveSeconds * time.Second
	maxIdle   = 90 * time.Second
)

type Session interface {
	NodeID() string
	NodeToken() string
	Reregister(ctx context.Context) bool
}

type Client struct {
	Addr     func() string
	LocalURL string
	Session  Session
	Identity identity.Identity

	work *http.Client
	once sync.Once
}

func (c *Client) init() {
	c.once.Do(func() {
		c.work = &http.Client{Timeout: 120 * time.Second}
	})
}

func (c *Client) Run(ctx context.Context) {
	c.init()
	backoff := 500 * time.Millisecond
	for ctx.Err() == nil {
		for (c.Session.NodeID() == "" || c.Session.NodeToken() == "" || c.addr() == "") && ctx.Err() == nil {
			time.Sleep(200 * time.Millisecond)
		}
		if ctx.Err() != nil {
			return
		}
		if c.serve(ctx) {
			backoff = 500 * time.Millisecond
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		if backoff < 10*time.Second {
			backoff *= 2
			if backoff > 10*time.Second {
				backoff = 10 * time.Second
			}
		}
	}
}

func (c *Client) addr() string {
	if c.Addr == nil {
		return ""
	}
	return c.Addr()
}

func (c *Client) serve(ctx context.Context) bool {
	addr := c.addr()
	tlsConf := &tls.Config{ //nolint:gosec
		NextProtos:         []string{alpn},
		InsecureSkipVerify: true,
	}
	conn, err := quic.DialAddr(ctx, addr, tlsConf, &quic.Config{
		KeepAlivePeriod: keepAlive,
		MaxIdleTimeout:  maxIdle,
	})
	if err != nil {
		log.Printf("quic unavailable, http tunnel remains active addr=%s err=%s", addr, truncate(err.Error(), 140))
		return false
	}
	defer conn.CloseWithError(0, "agent shutdown") //nolint:errcheck

	if !c.authenticate(ctx, conn) {
		return false
	}
	log.Printf("quic connected addr=%s node=%s", addr, c.Session.NodeID())
	for ctx.Err() == nil {
		stream, err := conn.AcceptStream(ctx)
		if err != nil {
			if ctx.Err() == nil {
				log.Printf("quic disconnected, retrying err=%s", truncate(err.Error(), 140))
			}
			return true
		}
		go c.dispatch(ctx, stream)
	}
	return true
}

func (c *Client) authenticate(ctx context.Context, conn *quic.Conn) bool {
	ts := time.Now().UnixMilli()
	challenge := []byte(fmt.Sprintf("meshpool-quic-auth:%s:%s:%d:%s",
		c.Session.NodeID(), c.Identity.DeviceID, ts, c.Session.NodeToken()))
	sig, err := c.Identity.Sign(challenge)
	if err != nil {
		log.Printf("quic auth sign failed: %v", err)
		return false
	}
	body, _ := json.Marshal(map[string]any{
		"op": "auth", "node_id": c.Session.NodeID(), "node_token": c.Session.NodeToken(),
		"device_id": c.Identity.DeviceID, "public_key": c.Identity.PublicKey,
		"ts_ms": ts, "signature": sig,
	})
	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return false
	}
	if _, err := stream.Write(append(body, '\n')); err != nil {
		return false
	}
	stream.Close() //nolint:errcheck
	raw, err := io.ReadAll(io.LimitReader(stream, 1<<20))
	if err != nil {
		return false
	}
	var out struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || !out.OK {
		log.Printf("quic auth rejected: %s", truncate(out.Error, 160))
		return false
	}
	return true
}

func (c *Client) dispatch(ctx context.Context, stream *quic.Stream) {
	defer stream.Close() //nolint:errcheck
	raw, err := io.ReadAll(io.LimitReader(stream, 32*1024*1024))
	if err != nil {
		return
	}
	var frame map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(raw), &frame); err != nil {
		return
	}
	var reply map[string]any
	switch op, _ := frame["op"].(string); op {
	case "call":
		reply = c.handleCall(ctx, frame)
	case "proxy":
		reply = c.handleProxy(ctx, frame)
	default:
		reply = map[string]any{"ok": false, "error": fmt.Sprintf("unknown op %v", frame["op"])}
	}
	body, err := json.Marshal(reply)
	if err != nil {
		return
	}
	stream.Write(append(body, '\n')) //nolint:errcheck
}

func (c *Client) handleCall(ctx context.Context, frame map[string]any) map[string]any {
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
		return failure("local call failed: " + truncate(err.Error(), 160))
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.work.Do(req)
	if err != nil {
		return failure("local call failed: " + truncate(err.Error(), 160))
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return map[string]any{"ok": false, "status": resp.StatusCode, "body": map[string]any{},
			"error": fmt.Sprintf("HTTP %d: %s", resp.StatusCode, truncate(string(data), 160))}
	}
	var body any = map[string]any{}
	if len(data) > 0 {
		var parsed any
		if json.Unmarshal(data, &parsed) == nil {
			body = parsed
		}
	}
	return map[string]any{"ok": true, "status": resp.StatusCode, "body": body, "error": nil}
}

func (c *Client) handleProxy(ctx context.Context, frame map[string]any) map[string]any {
	method := asStringOr(frame["method"], http.MethodGet)
	target := asString(frame["url"])
	content, err := base64.StdEncoding.DecodeString(asString(frame["content_b64"]))
	if err != nil {
		content = nil
	}
	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(content))
	if err != nil {
		return failure("local proxy failed: " + truncate(err.Error(), 160))
	}
	if raw, ok := frame["params"].([]any); ok && len(raw) > 0 {
		q := url.Values{}
		for _, pair := range raw {
			kv, ok := pair.([]any)
			if ok && len(kv) == 2 {
				q.Add(asString(kv[0]), asString(kv[1]))
			}
		}
		req.URL.RawQuery = q.Encode()
	}
	if hdrs, ok := frame["headers"].(map[string]any); ok {
		for k, v := range hdrs {
			if strings.EqualFold(k, "host") {
				req.Host = asString(v)
			} else {
				req.Header.Set(k, asString(v))
			}
		}
	}
	resp, err := c.work.Do(req)
	if err != nil {
		return failure("local proxy failed: " + truncate(err.Error(), 160))
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	headers := map[string]any{}
	for k := range resp.Header {
		headers[k] = resp.Header.Get(k)
	}
	return map[string]any{"ok": true, "body": map[string]any{
		"status": resp.StatusCode, "headers": headers,
		"content_b64": base64.StdEncoding.EncodeToString(data),
	}}
}

func failure(msg string) map[string]any {
	return map[string]any{"ok": false, "error": msg}
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
