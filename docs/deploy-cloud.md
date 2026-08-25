# Deploy a public MeshPool coordinator

This runbook makes the roadmap item "Hosted MeshPool Cloud" actionable for an
operator running a single public coordinator (`scale=public`). Agents connect
outbound with `contribute.sh --cloud` / `contribute.ps1 -Cloud`; they do not need
inbound ports or router port-forwarding.

## Prerequisites

- A Linux host or VM with Python 3.13+ for systemd, or Docker Compose for the
  container path.
- A DNS name for the coordinator, for example `meshpool.example.com`.
- A TLS reverse proxy or load balancer that can terminate HTTPS/WSS on port 443
  and proxy to the coordinator's HTTP port (`8080` by default).
- Persistent storage for the coordinator data directory.

Point the DNS record at the host or load balancer before requesting TLS
certificates.

## Configuration

The coordinator is configured by environment variables:

| Variable | Public coordinator value |
|---|---|
| `MESHPOOL_SCALE` | `public` |
| `MESHPOOL_PORT` | Internal HTTP port, usually `8080` |
| `MESHPOOL_HOST` | Bind address, usually `0.0.0.0` behind a proxy |
| `MESHPOOL_CONTROL_PLANE_URL` | Public HTTPS URL agents and browsers use |
| `MESHPOOL_DATA_DIR` | Persistent directory or mounted volume |
| `MESHPOOL_OWNER` | Optional owner label; defaults to `default` |

`MESHPOOL_DATA_DIR` holds the SQLite ledger (`meshpool.db`) and the
coordinator's `identity.json`. Treat it as durable state.

## Option A: Docker Compose

Edit `deploy/docker-compose.yml` and set:

```yaml
MESHPOOL_CONTROL_PLANE_URL: https://meshpool.example.com
```

Then start the coordinator:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

The compose file mounts a named volume at `/data`. For production, put Caddy,
nginx, Traefik, or a cloud load balancer in front of the service and publish
that proxy on `443`. The example compose file includes a commented Caddy
service showing automatic HTTPS.

## Option B: systemd

Install the repo and dependencies on the host:

```bash
sudo useradd --system --home /var/lib/meshpool --shell /usr/sbin/nologin meshpool
sudo mkdir -p /opt/meshpool /var/lib/meshpool
sudo chown -R meshpool:meshpool /opt/meshpool /var/lib/meshpool
cd /opt/meshpool
python3.13 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
```

Copy the repo contents into `/opt/meshpool`, edit
`deploy/meshpool-coordinator.service` so `MESHPOOL_CONTROL_PLANE_URL` is your
public HTTPS URL, then install it:

```bash
sudo cp deploy/meshpool-coordinator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now meshpool-coordinator
```

Place your TLS reverse proxy in front of `127.0.0.1:8080` or the host's private
address on port `8080`.

## TLS reverse proxy

TLS is mandatory for a public coordinator. Contributors dial the coordinator
over outbound HTTPS/WSS reverse tunnels, so the public URL in
`MESHPOOL_CONTROL_PLANE_URL` must be an `https://` origin with a valid
certificate. Terminate TLS at Caddy, nginx, Traefik, or your cloud load
balancer, then proxy to the coordinator's internal HTTP port.

## Verify

Check the health endpoint through the public TLS origin:

```bash
curl https://meshpool.example.com/api/v1/health
```

Then connect a test contributor from another machine:

```bash
./contribute.sh --cloud https://meshpool.example.com
```

On Windows:

```powershell
.\contribute.ps1 -Cloud https://meshpool.example.com
```

The agent uses an outbound connection; do not open inbound ports on the
contributor.

## Persistence and backup

Back up the Docker named volume or systemd data directory regularly. The data
directory contains:

- `meshpool.db` — the SQLite ledger and coordinator state.
- `identity.json` — the coordinator's device identity.

Losing this directory loses balances and changes the coordinator identity.
Never bake `identity.json` or `*.db` files into an image or zip artifact.

## Scaling and limitations

The public posture (`scale=public`) enables Tier-2 behavior: device identities,
the earn-to-spend credit ledger, rate limits, sandboxing, and known-answer
verification. The coordinator is still single-node today; HA, multi-region
control planes, and relay operations are future roadmap work.

## Security operations

- Keep HTTPS/WSS enabled and certificates current.
- Keep the host, Docker engine, Python runtime, and reverse proxy patched.
- Restrict direct access to the coordinator's internal HTTP port.
- Back up `/data` or `/var/lib/meshpool` securely.
- Monitor logs and resource usage; public contributors are untrusted even though
  Tier-2 sandboxing, credits, rate limits, and verification are active.
