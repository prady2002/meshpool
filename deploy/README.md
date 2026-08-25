# Deployment & install

Onboard real machines and run MeshPool across your LAN. The MVP goal (spec §4.2, §9):
**3 machines onboarded in under 5 minutes with zero networking config.**

> **Testing on two machines?** For a guided, copy-paste walkthrough that pools two computers
> and serves a real LLM (deploy → chat → benchmark → failover), see the
> **[two-machine test runbook](../docs/two-machine-test.md)**.

## Fastest path — one command per machine (foreground)

Copy this folder to every machine, then:

**On ONE machine — start the control plane:**

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File deploy\start-controlplane.ps1 -OpenFirewall
```
```bash
# macOS / Linux
bash deploy/start-controlplane.sh          # flags: --port 8080 --scale home|office|enterprise
```

It installs deps (first run), auto-detects this machine's LAN IP, and prints the dashboard URL
plus the **exact join command** for the other machines.

**On EACH other machine — run that printed join command:**

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File deploy\join.ps1 -Server http://<CP-IP>:8080 -Token <token> -Cpu 50 -OpenFirewall
```
```bash
# macOS / Linux
bash deploy/join.sh --server http://<CP-IP>:8080 --token <token> --cpu 50
```

`join` installs deps, **preflights reachability** to the control plane (fails fast with a clear
message if it can't connect), auto-detects this machine's LAN IP, and self-registers. Both
launchers run in the foreground so you can watch logs and stop with `Ctrl-C`.

| Launcher | Key flags |
|---|---|
| `start-controlplane.ps1` | `-Port 8080`, `-Scale home\|office\|enterprise`, `-OpenFirewall` |
| `start-controlplane.sh` | `--port 8080`, `--scale home\|office\|enterprise` |
| `join.ps1` | `-Server <url>` (req), `-Token <tok>`, `-Name <host>`, `-Cpu 50`, `-Port 9001`, `-OpenFirewall` |
| `join.sh` | `--server <url>` (req), `--token <tok>`, `--name <host>`, `--cpu 50`, `--port 9001` |

## Firewall

The control-plane port (8080) and each agent port (9001) must be reachable across the LAN.

- **Windows** blocks inbound to `python.exe` by default. Pass `-OpenFirewall` (adds the inbound
  rule via `netsh`; needs one elevated prompt), **or** choose **Allow** when Windows prompts —
  on **Private** networks, not Public. To do it by hand (Administrator):
  ```powershell
  netsh advfirewall firewall add rule name="MeshPool" dir=in action=allow protocol=TCP localport=8080
  ```
- **macOS / Linux** home firewalls usually allow LAN inbound already. If you run `ufw`/`firewalld`,
  allow the port: `sudo ufw allow 8080/tcp` (control plane) and `sudo ufw allow 9001/tcp` (agent).

## Auto-scaled donation (works out of the box)

Each agent probes its real hardware and, unless you set `MESHPOOL_DONATE_RAM_MB`, donates **half
of detected RAM** (2 GB floor) and 50% CPU (`--cpu` overrides). The RAM cap is a ceiling, not a
reservation. This is why the flagship "shard a model too big for one machine" works on first run —
a fixed tiny cap would be smaller than a single model layer.

## Run as a background service (always-on)

For a machine that should contribute on every boot, register a service instead of the foreground
launcher:

**Linux/macOS**

```bash
# control plane (first machine)
sudo ./install.sh --role controlplane --port 8080

# each additional machine
sudo ./install.sh --role agent --join http://<control-plane-host>:8080 --cpu 50
```

**Windows (PowerShell as admin)**

```powershell
# control plane
./install.ps1 -Role controlplane -Port 8080

# each additional machine
./install.ps1 -Role agent -Join http://<control-plane-host>:8080 -Cpu 50
```

## Onboarding model (zero-config)

1. The **first machine** becomes the control plane (or point everyone at a dedicated always-on box).
2. It prints a **pairing token / join command** (`meshpool onboarding`, the control-plane startup
   banner, or the dashboard's "Add a machine" card).
3. Every other machine runs `join` with that one token — **no manual IP entry, no port-forwarding.**
   On a flat LAN this build uses rendezvous registration; the production build adds mDNS/zeroconf
   so peers discover the control plane automatically.

## Entry points (what the launchers/services run)

| Component | Command | Key env |
|---|---|---|
| Control plane | `python -m meshpool.controlplane.app` | `MESHPOOL_HOST` (bind, default `0.0.0.0`), `MESHPOOL_ADVERTISE_HOST` (LAN IP, auto), `MESHPOOL_PORT`, `MESHPOOL_DATA_DIR`, `MESHPOOL_SCALE` |
| Agent | `python -m meshpool.agent.daemon` | `MESHPOOL_CONTROL_PLANE_URL`, `MESHPOOL_AGENT_HOST` (advertised LAN IP, auto), `MESHPOOL_AGENT_BIND` (default `0.0.0.0`), `MESHPOOL_AGENT_PORT`, `MESHPOOL_NAME`, `MESHPOOL_DONATE_CPU_PCT`, `MESHPOOL_DONATE_RAM_MB` |

> **Bind vs advertise:** services **bind** `0.0.0.0` (all interfaces, LAN-reachable) but **advertise**
> a routable LAN IP that the control plane and peers dial to place work. Both are auto-detected
> (`meshpool/common/netutil.py`); override the advertised address with `MESHPOOL_ADVERTISE_HOST` /
> `MESHPOOL_AGENT_HOST`. They fall back to `127.0.0.1` only for a single-box run (the sim harness
> pins loopback so tests stay deterministic).
>
> On Windows the service installer registers an auto-start **Scheduled Task** (SYSTEM, at startup)
> running an env-baked wrapper — `sc.exe create "python -m ..."` is not a valid Windows service.
> Manage it with `schtasks /End|/Delete /TN MeshPoolAgent`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `join` says "Cannot reach control plane" | Wrong IP/port, control plane not started, or its firewall is blocking. Re-run `start-controlplane` with `-OpenFirewall`; confirm `http://<CP-IP>:8080/api/v1/health` opens in a browser from the joining machine. |
| Node registers but work never lands on it | The agent advertised an unreachable address. Ensure the agent machine's firewall allows its port (`-OpenFirewall`), and that `MESHPOOL_AGENT_HOST` (if set) is a routable LAN IP, not `127.0.0.1`. |
| Control-plane banner shows a `127.x` dashboard URL | LAN IP couldn't be detected (offline / no default route). Set `MESHPOOL_ADVERTISE_HOST` to this machine's LAN IP explicitly and restart. |
| Model deploy: "insufficient pooled memory" | Add more machines, or raise the donation cap (dashboard slider, or `MESHPOOL_DONATE_RAM_MB`). Each node hosts `ram_cap × 0.8 / per-layer-MB` layers. |
| Windows firewall keeps prompting | Run the launcher once elevated with `-OpenFirewall`, or add the rule by hand (see **Firewall** above). |

## Prove the real multi-machine path

```bash
python -m sim.scenarios.verify_lan
```

Runs the control plane and agents on **distinct addresses** with **real hardware probes** and
checks that batch dispatch, hosted-app proxying, and inference activation hops all reach each
node at the address it advertised. Point it at a real LAN with `E2E_CP_HOST` / `E2E_AGENT_HOSTS`.

## Substrate note

`docker-compose.yml` and `Dockerfile` describe the **production** container substrate from the
spec. This environment has no Docker, so the runnable local equivalent is the multi-process
`sim/harness.py` (`python -m sim.harness --agents 3`), which stands up the same topology as
separate OS processes. Everything else — service wrappers, env contract, onboarding — is identical.
