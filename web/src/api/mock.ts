import type {
  AuthResponse,
  BatchJob,
  BlobMeta,
  ClusterResponse,
  Deployment,
  EventMessage,
  LedgerResponse,
  ModelDeployment,
  Node,
  OnboardingResponse,
  WalletResponse,
} from './types'

const now = Date.now()

export const mockNodes: Node[] = [
  { id: 'node-abc123', name: 'Alices-MacBook-Pro', url: 'https://node-a.mesh', owner: 'default', status: 'online', capability: { cpu_cores: 8, cpu_arch: 'arm64', ram_mb: 16384, gpu: false, vram_mb: 0, disk_mb: 102400, flops_estimate: 42.5, ollama_models: ['qwen2.5:7b'] }, link: { latency_ms: 12, bandwidth_mbps: 450 }, policy: { cpu_pct: 50, ram_mb: 4096, gpu_allowed: false, schedule: 'always', idle_minutes: 10 }, load: { cpu_load_pct: 23, ram_used_mb: 2048, running_units: 2, idle_seconds: 0 }, trust_score: 0.97, last_heartbeat_ms: now, registered_ms: now - 86400000 },
  { id: 'node-def456', name: 'Bobs-Gaming-PC', url: 'https://node-b.mesh', owner: 'default', status: 'online', capability: { cpu_cores: 16, cpu_arch: 'x86_64', ram_mb: 32768, gpu: true, vram_mb: 8192, disk_mb: 512000, flops_estimate: 128, ollama_models: [] }, link: { latency_ms: 8, bandwidth_mbps: 900 }, policy: { cpu_pct: 60, ram_mb: 8192, gpu_allowed: true, schedule: 'idle_only', idle_minutes: 5 }, load: { cpu_load_pct: 45, ram_used_mb: 6144, running_units: 5, idle_seconds: 0 }, trust_score: 0.89, last_heartbeat_ms: now - 5000, registered_ms: now - 172800000 },
  { id: 'node-ghi789', name: 'Carols-Home-Server', url: 'https://node-c.mesh', owner: 'default', status: 'online', capability: { cpu_cores: 4, cpu_arch: 'x86_64', ram_mb: 8192, gpu: false, vram_mb: 0, disk_mb: 204800, flops_estimate: 18.2, ollama_models: [] }, link: { latency_ms: 45, bandwidth_mbps: 100 }, policy: { cpu_pct: 80, ram_mb: 6144, gpu_allowed: false, schedule: 'always', idle_minutes: 10 }, load: { cpu_load_pct: 12, ram_used_mb: 1024, running_units: 1, idle_seconds: 300 }, trust_score: 1, last_heartbeat_ms: now - 2000, registered_ms: now - 604800000 },
  { id: 'node-jkl012', name: 'Davids-Laptop', url: 'https://node-d.mesh', owner: 'default', status: 'draining', capability: { cpu_cores: 4, cpu_arch: 'x86_64', ram_mb: 8192, gpu: false, vram_mb: 0, disk_mb: 51200, flops_estimate: 15, ollama_models: [] }, link: { latency_ms: 120, bandwidth_mbps: 50 }, policy: { cpu_pct: 30, ram_mb: 2048, gpu_allowed: false, schedule: 'idle_only', idle_minutes: 15 }, load: { cpu_load_pct: 5, ram_used_mb: 512, running_units: 0, idle_seconds: 1200 }, trust_score: 0.75, last_heartbeat_ms: now - 8000, registered_ms: now - 259200000 },
  { id: 'node-mno345', name: 'Eves-Raspberry-Pi', url: 'https://node-e.mesh', owner: 'alice@example.com', status: 'offline', capability: { cpu_cores: 4, cpu_arch: 'arm64', ram_mb: 4096, gpu: false, vram_mb: 0, disk_mb: 32768, flops_estimate: 5.1, ollama_models: [] }, link: { latency_ms: 0, bandwidth_mbps: 0 }, policy: { cpu_pct: 90, ram_mb: 3072, gpu_allowed: false, schedule: 'always', idle_minutes: 10 }, load: { cpu_load_pct: 0, ram_used_mb: 0, running_units: 0, idle_seconds: 0 }, trust_score: 0.62, last_heartbeat_ms: now - 3600000, registered_ms: now - 432000000 },
]

export const mockModels: ModelDeployment[] = [
  { id: 'mdep-aaa111', model: 'nano-gpt', backend: 'meshshard', n_layers: 12, partition: [
    { node_id: 'node-abc123', url: 'http://node-a', start_layer: 0, end_layer: 4, is_first: true, is_last: false },
    { node_id: 'node-def456', url: 'http://node-b', start_layer: 4, end_layer: 8, is_first: false, is_last: false },
    { node_id: 'node-ghi789', url: 'http://node-c', start_layer: 8, end_layer: 12, is_first: false, is_last: true },
  ], owner: 'default', status: 'running', tokens_per_sec: 47.3, created_ms: now - 3600000 },
  { id: 'mdep-bbb222', model: 'qwen2.5:7b', backend: 'ollama', n_layers: 32, partition: [
    { node_id: 'node-abc123', url: 'http://node-a', start_layer: 0, end_layer: 32, is_first: true, is_last: true },
  ], owner: 'default', status: 'warming', tokens_per_sec: 18.1, created_ms: now - 900000 },
]

export const mockCluster: ClusterResponse = {
  nodes: mockNodes,
  summary: {
    total_nodes: mockNodes.length,
    online: mockNodes.filter((n) => n.status === 'online').length,
    total_cores: mockNodes.reduce((sum, n) => sum + n.capability.cpu_cores, 0),
    total_ram_mb: mockNodes.reduce((sum, n) => sum + n.capability.ram_mb, 0),
  },
}

export const mockJobs: BatchJob[] = [  { id: 'job-bbb222', name: 'embed-docs', func: 'embed', status: 'running', progress: { pending: 20, assigned: 5, running: 8, done: 67, failed: 0, total: 100 }, owner: 'default', created_ms: now - 600000, units: Array.from({ length: 8 }, (_, index) => ({ id: `unit-run-${index}`, status: 'running' as const, assigned_node: mockNodes[index % 3].id, attempts: 1 })) },
  { id: 'job-ccc333', name: 'batch-transform', func: 'square', status: 'completed_with_failures', progress: { pending: 0, assigned: 0, running: 0, done: 50, failed: 2, total: 52 }, owner: 'default', created_ms: now - 7200000, units: Array.from({ length: 10 }, (_, index) => ({ id: `unit-done-${index}`, status: (index < 8 ? 'done' : 'failed') as 'done' | 'failed', assigned_node: mockNodes[index % 4].id, attempts: index >= 8 ? 5 : 1, result: index < 8 ? { squared: index * index } : null, error: index >= 8 ? 'OOM retry suggested' : null })) },
  { id: 'job-ddd444', name: 'pending-analysis', func: 'wasm:9f2c1a4b6d8e0f13579bdf2468ace013579bdf2468ace013579bdf2468ace013', status: 'pending', progress: { pending: 200, assigned: 0, running: 0, done: 0, failed: 0, total: 200 }, owner: 'default', created_ms: now - 60000, units: [] },
]

export const mockBlobs: BlobMeta[] = [
  { id: 'sha256:aabbcc', owner: 'default', size: 1048576, name: 'dataset.csv', replicas: ['node-abc123', 'node-ghi789'], replication_desired: 2, status: 'stored', chunks: [], created_ms: now - 86400000 },
  { id: 'sha256:ddeeff', owner: 'default', size: 52428800, name: 'model-weights.npz', replicas: ['node-abc123'], replication_desired: 2, status: 'degraded', chunks: [], created_ms: now - 172800000 },
  ...Array.from({ length: 8 }, (_, index) => ({ id: `sha256:fixture-${index}`, owner: 'default', size: (index + 1) * 1250000, name: `artifact-${index}.bin`, replicas: [mockNodes[index % 4].id], replication_desired: 2, status: index % 3 === 0 ? 'replicating' : 'stored', chunks: [], created_ms: now - index * 3600000 })),
]

export const mockWallet: WalletResponse = {
  owner: 'default', balance: 142.5, donated: 287.3, consumed: 144.8, granted: 0, reputation: 0.94, gating: false, nodes: ['node-abc123', 'node-ghi789'],
}

export const mockLedger: LedgerResponse = {
  owners: [{ owner: 'default', balance: 142.5 }, { owner: 'alice@example.com', balance: 17.2 }],
  summary: { total_earned: 287.3, total_spent: 144.8, pending_payout: 18.4 },
  by_node: mockNodes.map((node, index) => ({ node_id: node.id, earned: 22 + index * 11.2, uptime_hours: 15 + index * 6 })),
  credits_by_owner: [{ owner: 'default', credits: 842 }, { owner: 'alice@example.com', credits: 110 }],
  fair_share: [{ owner: 'default', share: 0.78 }, { owner: 'alice@example.com', share: 0.22 }],
  recent: [
    { id: 'txn-1', ts: now - 3600000, type: 'earning', amount: 4.2, node_id: 'node-def456' },
    { id: 'txn-2', ts: now - 5400000, type: 'consume', amount: -1.8, node_id: 'node-abc123' },
    { id: 'txn-3', ts: now - 7200000, type: 'payout', amount: 12.6 },
  ],
}

export const mockDeployments: Deployment[] = [
  { id: 'dep-web-1', name: 'vision-worker', owner: 'default', status: 'running', replicas: 3, created_ms: now - 5400000 },
  { id: 'dep-web-2', name: 'ocr-sidecar', owner: 'default', status: 'updating', replicas: 1, created_ms: now - 1800000 },
]

export const mockOnboarding: OnboardingResponse = {
  control_plane_url: 'https://meshpool.local',
  invite_token: 'INVITE-MESH-2026',
  join_windows: 'meshpool-node.exe --join INVITE-MESH-2026',
  join_unix: './meshpool-node --join INVITE-MESH-2026',
  join_command: 'meshpool-node --join INVITE-MESH-2026 --owner default',
}

export const mockSecrets = [
  { name: 'OPENAI_API_KEY', owner: 'default', created_ms: now - 100000 },
  { name: 'S3_UPLOAD_TOKEN', owner: 'default', created_ms: now - 200000 },
]

export const mockEvents: EventMessage[] = [
  { type: 'node.heartbeat', ts: now, payload: { node_id: 'node-abc123', cpu: 23 } },
  { type: 'inference.assigned', ts: now, payload: { model_id: 'mdep-aaa111', nodes: ['node-abc123', 'node-def456', 'node-ghi789'] } },
]

const fixtureChat = 'MeshPool routed this prompt across three community nodes, balancing latency and available memory before streaming the final answer back to you.'

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })
}

function streamFromText(text: string) {
  const encoder = new TextEncoder()
  const tokens = text.split(/(\s+)/).filter(Boolean)
  return new ReadableStream<Uint8Array>({
    start(controller) {
      let index = 0
      const pump = () => {
        if (index >= tokens.length) {
          controller.close()
          return
        }
        controller.enqueue(encoder.encode(tokens[index]))
        index += 1
        setTimeout(pump, 45)
      }
      pump()
    },
  })
}

export function createMockResponse(path: string, init?: RequestInit): Response {
  if (path.endsWith('/api/v1/health')) return json({ ok: true, ts: Date.now(), scale: mockNodes.length })
  if (path.endsWith('/api/v1/auth/local') || path.endsWith('/api/v1/auth/device')) return json({ owner: 'default', session_token: 'mock-session-token', device_id: 'dev-1', balance: 142.5, gating: false, tier: 'pro' } as AuthResponse)
  if (path.endsWith('/api/v1/cluster')) return json({ nodes: mockNodes, summary: { total_nodes: mockNodes.length, online: mockNodes.filter((node) => node.status === 'online').length, total_cores: mockNodes.reduce((sum, node) => sum + node.capability.cpu_cores, 0), total_ram_mb: mockNodes.reduce((sum, node) => sum + node.capability.ram_mb, 0) } } as ClusterResponse)
  if (path.endsWith('/api/v1/nodes')) return json({ nodes: mockNodes })
  if (path.endsWith('/api/v1/onboarding')) return json(mockOnboarding)
  if (path.endsWith('/api/v1/wallet')) return json(mockWallet)
  if (path.endsWith('/api/v1/ledger')) return json(mockLedger)
  if (path.endsWith('/api/v1/jobs') && init?.method === 'POST') return json({ ...mockJobs[0], id: `job-${Date.now()}`, name: 'new-job', status: 'pending' })
  if (path.endsWith('/api/v1/jobs')) return json({ jobs: mockJobs })
  if (path.includes('/api/v1/jobs/')) return json(mockJobs.find((job) => path.endsWith(job.id)) ?? mockJobs[0])
  if (path.endsWith('/api/v1/models/available')) return json(['nano-gpt', 'qwen2.5:7b', 'llama3.1:8b'])
  if (path.endsWith('/api/v1/models') && init?.method === 'POST') return json({ ...mockModels[0], id: `mdep-${Date.now()}`, model: 'llama3.1:8b', status: 'deploying' })
  if (path.endsWith('/api/v1/models')) return json({ models: mockModels })
  if (path.includes('/api/v1/models/') && init?.method === 'DELETE') return json({ status: 'deleted' })
  if (path.includes('/api/v1/models/')) return json(mockModels[0])
  if (path.endsWith('/api/v1/deployments')) return json({ deployments: mockDeployments })
  if (path.includes('/api/v1/deployments/') && init?.method === 'DELETE') return json({ status: 'deleted' })
  if (path.includes('/api/v1/deployments/')) return json(mockDeployments[0])
  if (path.endsWith('/api/v1/blobs') && init?.method === 'POST') return json({ ...mockBlobs[0], id: `sha256:uploaded-${Date.now()}`, name: 'upload.bin' })
  if (path.endsWith('/api/v1/blobs')) return json({ blobs: mockBlobs })
  if (path.includes('/api/v1/blobs/') && init?.method === 'DELETE') return json({ status: 'deleted' })
  if (path.includes('/api/v1/blobs/')) return json(mockBlobs[0])
  if (path.includes('/reclaim')) return json({ status: 'draining' })
  if (path.includes('/resume')) return json({ status: 'online' })
  if (path.includes('/policy')) return json(mockNodes[0])
  if (path.endsWith('/api/v1/secrets') && init?.method === 'POST') return json({ name: 'NEW_SECRET', owner: 'default', created_ms: Date.now() })
  if (path.endsWith('/api/v1/secrets')) return json({ secrets: mockSecrets })
  if (path.includes('/api/v1/secrets/')) return json({ status: 'deleted' })
  if (path.endsWith('/api/v1/inference/chat')) return new Response(streamFromText(fixtureChat), { headers: { 'Content-Type': 'text/event-stream' } })
  if (path.endsWith('/v1/chat/completions')) return json({ id: 'chatcmpl-mock', choices: [{ message: { role: 'assistant', content: fixtureChat } }] })
  return json({ message: 'not found in mock' }, 404)
}

export function setupMock() {
  const originalFetch = globalThis.fetch.bind(globalThis)
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    if (url.includes('/api/') || url.includes('/v1/')) {
      await new Promise((resolve) => setTimeout(resolve, 120))
      return createMockResponse(url, init)
    }
    return originalFetch(input, init)
  }

  class MockEventSource implements EventSource {
    static readonly CONNECTING = 0
    static readonly OPEN = 1
    static readonly CLOSED = 2
    readonly CONNECTING = 0
    readonly OPEN = 1
    readonly CLOSED = 2
    onerror: ((this: EventSource, ev: Event) => unknown) | null = null
    onmessage: ((this: EventSource, ev: MessageEvent<string>) => unknown) | null = null
    onopen: ((this: EventSource, ev: Event) => unknown) | null = null
    readyState = 1
    url: string
    withCredentials = false
    private timer?: number

    constructor(url: string) {
      this.url = url
      setTimeout(() => this.onopen?.call(this, new Event('open')), 20)
      this.timer = window.setInterval(() => {
        const event = mockEvents[Math.floor(Math.random() * mockEvents.length)]
        this.onmessage?.call(this, new MessageEvent('message', { data: JSON.stringify({ ...event, ts: Date.now() }) }))
      }, 1800)
    }

    addEventListener() {}
    removeEventListener() {}
    dispatchEvent() { return true }
    close() { if (this.timer) window.clearInterval(this.timer); this.readyState = 2 }
  }

  Object.defineProperty(globalThis, 'EventSource', {
    configurable: true,
    value: MockEventSource,
  })
}
