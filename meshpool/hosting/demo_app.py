"""A tiny self-contained web app used as the *hosted workload* for the hosting
engine (M4). In production this is any container/Dockerfile the user points at;
here it's a stdlib HTTP server so the placement, load-balancing, health-checking
and failover machinery is fully exercisable with zero dependencies.

Launched by the node agent as a child process (see ``runtime.start_app_process``).
Reads its config from the environment:
  MESHPOOL_APP_PORT   port to bind
  MESHPOOL_APP        app name
  MESHPOOL_REPLICA_ID replica id (shown so you can see the LB spread traffic)
  MESHPOOL_NODE_NAME  which machine it's running on
"""
from __future__ import annotations

import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from meshpool.hosting import site as sitelib

APP = os.environ.get("MESHPOOL_APP", "hello")
REPLICA = os.environ.get("MESHPOOL_REPLICA_ID", "r?")
NODE = os.environ.get("MESHPOOL_NODE_NAME", socket.gethostname())
PORT = int(os.environ.get("MESHPOOL_APP_PORT", "8500"))
SITE_DIR = os.environ.get("MESHPOOL_SITE_DIR", "")

_PAGE = """<!DOCTYPE html><html><head><meta charset=utf-8>
<title>{app} · MeshPool</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0a0e14;color:#e6edf3;
display:grid;place-items:center;height:100vh;margin:0}}
.c{{text-align:center;padding:40px 56px;border:1px solid #2a3346;border-radius:16px;background:#111722;
box-shadow:0 8px 40px rgba(0,0,0,.4)}}
h1{{margin:0 0 6px;font-size:28px;background:linear-gradient(135deg,#4f8cff,#7c5cff);
-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}}
.n{{font-size:14px;color:#8b97a8;margin-top:14px}} .b{{font-weight:700;color:#3fb950}}
code{{background:#1a2130;padding:2px 8px;border-radius:6px;color:#7c5cff}}</style></head>
<body><div class=c><h1>🚀 {app}</h1>
<div class=n>Served by replica <code>{replica}</code></div>
<div class=n>running on machine <span class=b>{node}</span></div>
<div class=n style=margin-top:20px>This app is <b>hosted across your MeshPool</b>.<br>
Kill this machine and the load balancer reroutes you — no dropped requests.</div>
<div class=n style=margin-top:16px;font-size:12px>hits served here: <b id=h>{hits}</b></div>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    hits = 0

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-MeshPool-Node", NODE)
        self.send_header("X-MeshPool-Replica", REPLICA)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/healthz", "/health"):
            self._send(200, b'{"status":"ok"}', "application/json")
            return
        if SITE_DIR:
            hit = sitelib.resolve_request(SITE_DIR, self.path)
            if hit is not None:
                Handler.hits += 1
                self._send(200, hit[0], hit[1])
            else:
                self._send(404, b"404 Not Found", "text/plain; charset=utf-8")
            return
        Handler.hits += 1
        html = _PAGE.format(app=APP, replica=REPLICA, node=NODE, hits=Handler.hits)
        self._send(200, html.encode(), "text/html; charset=utf-8")

    def log_message(self, *args):  # silence
        return


def main() -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
