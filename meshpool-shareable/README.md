# MeshPool - Instant Setup & User Guide

MeshPool turns the machines you already own into one pooled computer. Host apps, run batch jobs, store encrypted files, and run large AI models across your machines.

---

## 1. Instant Run (Single PC / Solo Mode)

### Windows
- **Double-click `start.cmd`** OR run in PowerShell:
  ```powershell
  powershell -ExecutionPolicy Bypass -File start.ps1
  ```

### macOS / Linux
```bash
chmod +x start.sh
./start.sh
```

> **What happens automatically:**
> 1. Installs the bundled pre-built MeshPool artifact (`meshpool-*.whl`).
> 2. Starts the **Control Plane** + a local compute **Agent** on loopback (`http://127.0.0.1:8080`).
> 3. Opens the interactive **Web Dashboard** in your browser.

---

## 2. Pooling Multiple PCs into a Cluster

### Step A - On the Main / Coordinator PC
Start the control plane bound to your network:
```powershell
meshpool-controlplane
```
To view your cluster pairing code and command:
```bash
meshpool onboarding
```

### Step B - On Other PCs (Worker / Donor Nodes)
Copy this folder to any other machine on your local network and run:
```powershell
# Windows (Example: Coordinator is at 192.168.1.20)
powershell -ExecutionPolicy Bypass -File join.ps1 -Server http://192.168.1.20:8080 -Token <PAIRING_TOKEN> -Cpu 50

# Linux / macOS
bash join.sh --server http://192.168.1.20:8080 --token <PAIRING_TOKEN> --cpu 50
```

---

## 3. Everything You Can Do with MeshPool

### Web & Dynamic App Hosting
```bash
# Deploy a static website (HTML/JS/React dist folder) with 2 replicas:
meshpool deploy ./my-site --replicas 2

# Deploy a dynamic Python/Node API app:
meshpool deploy ./my-api --cmd "python server.py" --replicas 2

# Manage deployments & logs:
meshpool deployments
meshpool logs <deployment_id>
meshpool undeploy <deployment_id>
```

### Distributed AI & Large Model Inference (OpenAI-Compatible)
```bash
# List available models
meshpool models available

# Deploy and query a model across the cluster
meshpool models deploy nano-gpt
meshpool chat "Explain quantum computing in simple terms"
meshpool run nano-gpt "Once upon a time" --tokens 64

# Start local OpenAI-compatible proxy (for Open-WebUI, Cursor, LangChain, etc.)
meshpool serve nano-gpt --port 11435

# Benchmark single-box vs multi-node pooled inference speed
meshpool bench --model qwen2.5-coder:7b --concurrency 8
```

### Distributed Batch Compute & DAG Pipelines
```bash
# Run parallel compute units across nodes:
meshpool jobs submit sha_pow --count 50 --cpu 20
meshpool jobs list

# Run multi-stage DAG pipelines:
meshpool dag submit ./pipeline.json --wait
meshpool dag list
```

### WebAssembly (WASM) Serverless Functions
```bash
meshpool fn deploy ./handler.wasm --name myfunc
meshpool fn run wasm:<id> --payload '{"input": 42}' --count 10
meshpool fn builtin
```

### Pooled Encrypted Storage & S3-Compatible Store
```bash
# Store an encrypted file replicated across 2 nodes:
meshpool storage put my-database.db --replication 2
meshpool storage ls
meshpool storage get <blob_id> ./restored.db

# S3 compatibility:
meshpool store cp ./dataset.csv s3://mybucket/dataset.csv
meshpool store cp s3://mybucket/dataset.csv ./downloaded.csv
```

### Security, Resource Control & Ledger
```bash
# View all nodes & their live status:
meshpool nodes
meshpool mesh --watch

# View compute donated vs consumed:
meshpool ledger

# Emergency stop donating compute on a machine (kill-switch):
meshpool reclaim <node_id>
meshpool resume <node_id>

# Mint scoped capability tokens:
meshpool tokens mint "ci-worker" -s jobs:read -s jobs:write
meshpool tokens list
```
