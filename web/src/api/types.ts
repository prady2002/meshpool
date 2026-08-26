export type NodeStatus = 'online' | 'draining' | 'offline'

export interface Node {
  id: string
  name: string
  url: string
  owner: string
  status: NodeStatus
  capability: {
    cpu_cores: number
    cpu_arch: string
    ram_mb: number
    gpu: boolean
    vram_mb: number
    disk_mb: number
    flops_estimate: number
    ollama_models: string[]
  }
  link: { latency_ms: number; bandwidth_mbps: number }
  policy: { cpu_pct: number; ram_mb: number; gpu_allowed: boolean; schedule: string; idle_minutes: number }
  load: { cpu_load_pct: number; ram_used_mb: number; running_units: number; idle_seconds: number }
  trust_score: number
  last_heartbeat_ms: number
  registered_ms: number
}

export interface ClusterSummary {
  total_nodes: number
  online: number
  total_cores: number
  total_ram_mb: number
}

export interface ClusterResponse {
  nodes: Node[]
  summary: ClusterSummary
}

export interface WalletResponse {
  owner: string
  balance: number
  donated: number
  consumed: number
  granted: number
  reputation: number
  gating: boolean
  nodes: string[]
}

export interface LedgerSummary {
  total_earned: number
  total_spent: number
  pending_payout: number
}

export interface LedgerResponse {
  owners: Array<{ owner: string; balance: number }>
  summary: LedgerSummary
  by_node: Array<{ node_id: string; earned: number; uptime_hours: number }>
  credits_by_owner: Array<{ owner: string; credits: number }>
  fair_share: Array<{ owner: string; share: number }>
  recent: Array<{ id: string; ts: number; type: string; amount: number; node_id?: string }>
}

// Mirrors meshpool.common.models.WorkUnit. Do not add fields the API does not
// actually return — a field that is always undefined silently renders as blank.
export interface WorkUnit {
  id: string
  status: 'pending' | 'assigned' | 'running' | 'done' | 'failed'
  payload?: Record<string, unknown>
  assigned_node?: string | null
  attempts?: number
  result?: unknown
  error?: string | null
}

export interface BatchJob {
  id: string
  name: string
  func: string
  units: WorkUnit[]
  owner: string
  status: string
  created_ms: number
  progress: { pending: number; assigned: number; running: number; done: number; failed: number; total: number }
}

export interface JobsResponse {
  jobs: BatchJob[]
}

export interface TimelineEvent {
  type: 'span' | 'event'
  trace_id: string
  span_id: string | null
  parent_span_id: string | null
  job_id: string
  unit_id: string | null
  node_id: string | null
  attempt: number | null
  name: string
  start_ms: number
  end_ms: number | null
  duration_ms: number | null
  outcome: string | null
  error: string | null
  attributes: Record<string, unknown>
}

export interface TimelineLane {
  node_id: string
  node_name: string
  spans: TimelineEvent[]
}

export interface JobTimeline {
  job_id: string
  trace_id: string
  status: string
  created_ms: number
  start_ms: number
  end_ms: number | null
  duration_ms: number
  units: Array<{ unit_id: string; status: string; node_id: string | null; attempts: number; outcome: string; error: string | null }>
  events: TimelineEvent[]
  lanes: TimelineLane[]
}

export interface JobSubmitRequest {
  name: string
  func: string
  payloads: unknown[]
  cpu_pct: number
  ram_mb: number
  owner: string
  secret_names: string[]
}

export interface PartitionSlice {
  node_id: string
  url: string
  start_layer: number
  end_layer: number
  is_first: boolean
  is_last: boolean
}

export interface ModelDeployment {
  id: string
  model: string
  backend: string
  n_layers: number
  partition: PartitionSlice[]
  owner: string
  status: string
  tokens_per_sec: number | null
  created_ms: number
}

export interface ModelsResponse {
  models: ModelDeployment[]
}

/** Shape of GET /api/v1/models/available -- an object of named lists, not an array. */
export interface AvailableModelsResponse {
  builtin?: string[]
  ollama?: string[]
}

export interface DeployModelRequest {
  model: string
  owner: string
}

export interface Deployment {
  id: string
  name: string
  owner: string
  status: string
  replicas: number
  created_ms: number
}

export interface DeploymentsResponse {
  deployments: Deployment[]
}

export interface BlobMeta {
  id: string
  owner: string
  size: number
  name: string
  replicas: string[]
  replication_desired: number
  status: string
  chunks: string[]
  created_ms: number
}

export interface BlobsResponse {
  blobs: BlobMeta[]
}

/** Response of `POST /api/v1/functions` — a user-uploaded WebAssembly module. */
export interface FunctionUpload {
  function: string
  sha256: string
  size_bytes: number
  abi: string
  imports: string[]
  exports: string[]
  name: string | null
  owner: string
}

export interface AuthResponse {
  owner: string
  session_token: string
  device_id?: string
  balance: number
  gating: boolean
  tier?: string
}

export interface HealthResponse {
  ok: boolean
  ts: number
  scale: number
}

export interface OnboardingResponse {
  control_plane_url: string
  invite_token: string
  join_windows: string
  join_unix: string
  join_command: string
}

export interface DonationPolicy {
  cpu_pct: number
  ram_mb: number
  gpu_allowed: boolean
  schedule: string
  idle_minutes: number
}

export interface SecretRecord {
  name: string
  owner: string
  created_ms: number
}

/**
 * GET /api/v1/secrets returns `{names: string[], active_leases: number}`.
 *
 * It deliberately returns only NAMES -- the control plane never serves secret
 * values back over the API, and it does not expose owner/created metadata here.
 * The console previously declared this as `{secrets: SecretRecord[]}`, so
 * `.secrets` was `undefined` and the Account page crashed on `undefined.map`.
 */
export interface SecretsResponse {
  names?: string[]
  active_leases?: number
}

/**
 * GET /api/v1/account/spend-cap. `cap`/`remaining` are null when the owner is
 * uncapped; `enforced` is the honest signal the UI must render instead of
 * inventing a limit.
 */
export interface SpendCapStatus {
  owner: string
  cap: number | null
  spent: number
  remaining: number | null
  enforced: boolean
  exceeded: boolean
}

export interface EventMessage {
  type: string
  ts: number
  payload: Record<string, unknown>
}

export interface ChatRequest {
  prompt: string
  max_new_tokens: number
  model_id: string
  temperature: number
}

export interface ChatCompletionRequest {
  model: string
  messages: Array<{ role: string; content: string }>
  max_tokens: number
  temperature: number
}

export interface MeshAggregate {
  mode: 'individual' | 'dense' | 'clustered'
  items: Array<{
    id: string
    label: string
    count: number
    status: NodeStatus
    totalCompute: number
    nodes: Node[]
  }>
}

/**
 * Node role class inferred client-side from observable telemetry.
 * Not part of the wire API — this is a calibrated projection.
 *
 *   anchor       — long-running, high-trust, low-latency dedicated node
 *   relay        — high-bandwidth node currently idle (routing specialist)
 *   constellation — general pool member (the majority class)
 *   drifter      — new, high-latency, or low-trust transient contributor
 */
export type NodeClass = 'anchor' | 'relay' | 'constellation' | 'drifter'

/**
 * A latency neighbourhood: nodes whose control-plane RTT falls within a 25-ms
 * bucket. Used as a geographic-proximity proxy — MeshPool has no geoIP.
 * Any capacity figure derived from this grouping is a 'calibrated projection'.
 * `source: 'approximated'` — the RTT is node→control-plane only; pairwise
 * inter-node latency is not measured. The scheduler uses Vivaldi coordinates
 * for confirmed constellations; until that data reaches the client, groups from
 * `groupConstellations()` are an approximation and must not be labelled as
 * measured pairwise neighbourhoods.
 */
export interface ConstellationGroup {
  id: string
  latencyBucketMs: number
  nodeIds: string[]
  totalCompute: number
  onlineCount: number
  label: string
  source: 'measured' | 'approximated'
}
