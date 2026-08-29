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

/**
 * What `GET /api/v1/jobs` returns for each job.
 *
 * This is deliberately NOT the same shape as `GET /api/v1/jobs/{id}`. The list
 * endpoint omits `units` because it is unbounded; it projects `active_nodes`
 * and `units_total` instead. Keeping the two shapes as one interface is what
 * made `job.units` undefined in the list views, and the `?? []` fallbacks hid
 * it -- the mesh map's live-work highlight and the jobs task grid rendered
 * empty forever while every test passed.
 */
export interface BatchJobSummary {
  id: string
  name: string
  func: string
  owner: string
  status: string
  created_ms: number
  active_nodes: string[]
  units_total: number
  progress: { pending: number; assigned: number; running: number; done: number; failed: number; total: number }
}

/** What `GET /api/v1/jobs/{id}` returns: the summary plus the full unit list. */
export interface BatchJob extends BatchJobSummary {
  units: WorkUnit[]
}

export interface JobsResponse {
  jobs: BatchJobSummary[]
}

/**
 * What `POST /api/v1/jobs` returns. Note `units` is a *count* here, not a list:
 * this is a third distinct job shape, and pretending it was a `BatchJob` is why
 * the caller had to cast through `unknown` to reach `job_id`.
 */
export interface JobSubmitResponse {
  job_id: string
  units: number
  owner: string
  trace_id: string | null
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

/**
 * One pipeline stage: the layer range a single machine holds.
 *
 * Transcribed field-for-field from InferenceEngine.view() in
 * meshpool/inference/engine.py. It was previously declared as `PartitionSlice`
 * with `start_layer`/`end_layer`/`url` under a `partition` key -- a shape the
 * server has never sent. Because `request<T>()` is a type assertion over
 * network data rather than a validation, tsc, eslint and 119 vitest tests were
 * all green while every screen that touched it crashed the moment a model was
 * actually deployed.
 */
export interface ModelStage {
  node_id: string
  node_name: string
  first: number
  last: number
  is_first: boolean
  is_last: boolean
}

export interface ModelDeployment {
  id: string
  model: string
  backend: string
  deployed: boolean
  pooled: boolean
  weight_groups: string[] | null
  n_layers: number
  replicas: number
  /** 'ready' | 'loading' | 'degraded'. Never 'running' -- that is a hosting status. */
  status: string
  tokens_per_sec: number | null
  protected_stages: number
  /** stage node name -> the warm standbys holding the same layers. */
  spares: Record<string, string[]>
  stages: ModelStage[]
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
  /** Optional. The control plane resolves the real owner from the session; a
   *  client-supplied owner is a request, not an identity. Redeploy omits it. */
  owner?: string
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

export interface CapabilityToken {
  id: string
  label: string
  scopes: string[]
  expires_at_ms: number
  created_ms: number
  last_used_ms: number | null
  spend_cap: number | null
  spent: number
}

export interface CapabilityTokenCreate {
  label: string
  scopes: string[]
  expires_in_seconds: number
  spend_cap?: number
}

export interface CapabilityTokenMinted extends CapabilityToken {
  owner: string
  token: string
}

export interface CapabilityTokensResponse {
  tokens: CapabilityToken[]
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
  stream?: boolean
}

/** What the engine reports about a finished run. Mirrors InferenceEngine.chat(). */
export interface ChatRunResult {
  model: string
  deployment_id: string
  prompt: string
  text: string
  tokens: number
  finish_reason: string
  tokens_per_sec: number
  elapsed_s: number
  batch?: { mean_batch: number; queued_s: number; ttft_s: number; max_batch: number }
  stages?: Array<{ node_name: string; first: number; last: number }>
}

/** One NDJSON frame from POST /api/v1/inference/chat with `stream: true`. */
export interface ChatStreamEvent {
  delta?: string
  done?: boolean
  result?: ChatRunResult
  error?: string
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
