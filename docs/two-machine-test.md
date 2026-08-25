# Two-machine test — run a real LLM across the pool

A copy-paste runbook for pooling **two real machines** on the same network and serving a
real Ollama model (e.g. `qwen2.5-coder:7b`) across both. No Docker, no accounts, no cloud.

This is the guide to hand to the second person along with the zip.

## What this proves

- Two machines join into **one endpoint** (dashboard + OpenAI-compatible API).
- Requests **load-balance across both** machines → higher aggregate throughput than one box.
- Kill one machine mid-use → the other **keeps serving** (failover).

**Honest expectation:** pooling scales *concurrent throughput* and adds *resilience*, and it
lets you run a model too big for one box. It does **not** make a single reply faster than one
machine — that's network physics. The benchmark in Step 5 shows the throughput win under load.

---

## 0. Prerequisites (on **both** machines)

- **Windows 10/11** (macOS/Linux work too — use the `.sh` scripts instead of `.ps1`).
- **Python 3.11+** — install from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"**.
  Verify: `python --version`.
- **Ollama** — install from <https://ollama.com/download>, then pull the **same** model on **both** machines:
  ```powershell
  ollama pull qwen2.5-coder:7b
  ollama list          # confirm it's there
  ```
  Keep Ollama running (it listens on `127.0.0.1:11434`). MeshPool auto-detects it.
- Both machines on the **same Wi-Fi / LAN**.
- The unzipped **meshpool** folder on both. Run every command **from that folder**.

> Machine A = the coordinator (hosts the dashboard). Machine B = the machine that joins.
> Both machines still contribute their Ollama — A serves requests too.

---

## 1. Start the coordinator (Machine A)

From the meshpool folder:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start-controlplane.ps1 -OpenFirewall
```

- If Windows asks about network access, choose **Allow** on **Private networks**.
  (`-OpenFirewall` also adds the rule; run PowerShell **as Administrator** if it can't.)
- You'll see the Ollama preflight line (`==> Ollama detected here: qwen2.5-coder:7b`) and then a banner:

  ```
  ================================================================
    MeshPool control plane is LIVE
    Dashboard   : http://192.168.1.20:8080
    Add a machine (run on each laptop, in the meshpool folder):
      Windows : powershell -ExecutionPolicy Bypass -File deploy\join.ps1 -Server http://192.168.1.20:8080 -Token abc123
      macOS/Linux : bash deploy/join.sh --server http://192.168.1.20:8080 --token abc123
  ================================================================
  ```

- **Copy the Windows join command** (it contains A's real LAN IP + token).
- Open the **Dashboard** URL in a browser on Machine A. **Leave this window running.**

> If the `Dashboard` line shows `127.0.0.1` instead of a `192.168.x.x`/`10.x.x.x` address,
> A couldn't detect its LAN IP. Set it explicitly and restart:
> `$env:MESHPOOL_ADVERTISE_HOST = "192.168.1.20"` (use A's real IP from `ipconfig`).

---

## 2. Join the second machine (Machine B)

From the meshpool folder on **Machine B**, paste the command A printed and **add `-OpenFirewall`**:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\join.ps1 -Server http://192.168.1.20:8080 -Token abc123 -OpenFirewall
```

- Choose **Allow** on the firewall prompt.
- The script does a preflight (`==> Control plane reachable`), prints its own Ollama detection,
  then registers. **Leave it running.**
- On **A's dashboard**, both machines now appear online (with CPU/RAM/health). You now have a 2-node pool.

---

## 3. Deploy the model and chat

### Option A — Dashboard (Machine A)

1. Dashboard → **Use** tab → **"Run a model"** card.
2. The **Model** dropdown lists `qwen2.5-coder:7b — Ollama · 2 nodes`. Select it → **Deploy across the pool**.
3. Type in the chat box and send. Each reply shows **which machine served it** and its **tok/s**.
   Send a few — they alternate between A and B.

### Option B — CLI (Machine A, from the meshpool folder)

```powershell
python -m meshpool.cli.main models available          # both nodes should list the model
python -m meshpool.cli.main models deploy qwen2.5-coder:7b
python -m meshpool.cli.main chat "Write a haiku about distributed systems."
```

Any OpenAI client works too:

```powershell
curl http://192.168.1.20:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen2.5-coder:7b\",\"messages\":[{\"role\":\"user\",\"content\":\"MeshPool is\"}]}"
```

---

## 4. Measure the throughput win (Machine A)

```powershell
python -m meshpool.cli.main bench --model qwen2.5-coder:7b --concurrency 8 --deploy
```

This runs the **same load** twice and prints a table:

- **Baseline** — 8 concurrent requests to **one box's** Ollama (A's `127.0.0.1:11434`).
- **Pooled** — 8 concurrent requests to the **coordinator**, which round-robins across **both** machines.

With both machines serving, **pooled aggregate tok/s** should beat one box under concurrency.
At `--concurrency 1` they're roughly equal — that's expected and honest.

Standalone equivalent (same thing, no CLI wrapper):

```powershell
python -m meshpool.inference.bench --mode both --model qwen2.5-coder:7b -c 8 --deploy
```

---

## 5. Prove failover (kill a machine)

1. With a model deployed and a chat working, go to **Machine B's** join window and press **Ctrl+C**
   (or unplug B's network / turn off Wi-Fi).
2. Back on A, **send another chat message** (wait a couple of seconds first). It's still answered —
   now by **A alone**. The reply's "served by" shows the survivor.
3. Bring B back: **re-run the join command** on B. It rejoins and starts sharing load again.

You can also do this from the dashboard's node list (**reclaim** a node) instead of Ctrl+C.

---

## 6. Teardown

- Press **Ctrl+C** in each window (A and B).
- Nothing is installed system-wide. Local state lives under `~/.meshpool` and can be deleted.
- To remove the firewall rules: `netsh advfirewall firewall delete rule name="MeshPool ControlPlane"`
  (on A) and `... name="MeshPool Agent"` (on B).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Model **not in the dropdown** / not under `models available` | Run `ollama list` on **each** machine and `ollama pull qwen2.5-coder:7b`. The agent detects Ollama **at start** — after pulling, **restart** that machine's `join.ps1`. |
| Only **nano-gpt** shows | No joined machine advertises an Ollama model — Ollama isn't running, or the model isn't pulled on any node. The built-in demo model still works. |
| **B can't reach A** (`Cannot reach control plane`) | Same network? Use A's **LAN IP**, not `127.0.0.1`. Both need the firewall open (`-OpenFirewall` or click **Allow**). Test from B: `curl http://192.168.1.20:8080/api/v1/health`. |
| **Firewall rule failed** | Run PowerShell **as Administrator**, or click **Allow** on the Windows prompt when the app starts. |
| Dashboard banner shows **`127.0.0.1`** | A couldn't auto-detect its LAN IP. Set `$env:MESHPOOL_ADVERTISE_HOST = "<A's IP>"` (from `ipconfig`) and restart the control plane. |
| **Pooled not faster** than one box | Raise `--concurrency`. Confirm **both** machines hold the model (`models available` shows **2 nodes**). One shared box can't beat itself. |
| **Slow** tok/s | CPU-only inference is slow by nature. `qwen2.5-coder:7b` is a good size; `:14b` is ~2× slower. That's the model, not MeshPool. |
| `python` **not recognized** | Reinstall Python with **"Add to PATH"**, or use `py` instead of `python`. Open a fresh terminal after installing. |

---

## What "pooled" means here (the honest architecture)

Each machine runs the **whole** model in its own Ollama. MeshPool gives you a **single endpoint**
that **load-balances** requests across every node that holds the model and **fails over** when a
node drops. That scales **concurrent throughput** and adds **resilience** — the real wins for a
model that already fits on one machine.

For a model **too big for any single machine** (true layer-by-layer sharding across nodes), see the
built-in `nano-gpt` demo in the [quickstart](quickstart.md) and the sharding roadmap in
[inference.md](inference.md).

Next: [quickstart](quickstart.md) · [architecture](architecture.md) · [inference deep-dive](inference.md).
