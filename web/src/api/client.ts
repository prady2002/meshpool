import type {
  AuthResponse,
  AvailableModelsResponse,
  BatchJob,
  BlobMeta,
  BlobsResponse,
  ChatRequest,
  ClusterResponse,
  DeployModelRequest,
  Deployment,
  DeploymentsResponse,
  DonationPolicy,
  EventMessage,
  FunctionUpload,
  HealthResponse,
  JobSubmitRequest,
  JobTimeline,
  JobsResponse,
  LedgerResponse,
  ModelDeployment,
  ModelsResponse,
  Node,
  OnboardingResponse,
  SecretRecord,
  SecretsResponse,
  SpendCapStatus,
  WalletResponse,
} from './types'

const SESSION_KEY = 'meshpool.session'
const OWNER_KEY = 'meshpool.owner'

const jsonHeaders = {
  'Content-Type': 'application/json',
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export class ApiClient {
  baseUrl = import.meta.env.VITE_API_BASE ?? ''
  mockEnabled = import.meta.env.VITE_MOCK === 'true'

  getSessionToken(): string | null {
    return localStorage.getItem(SESSION_KEY)
  }

  setSessionToken(token: string, owner?: string) {
    localStorage.setItem(SESSION_KEY, token)
    if (owner) {
      localStorage.setItem(OWNER_KEY, owner)
    }
  }

  getOwner(): string {
    return localStorage.getItem(OWNER_KEY) ?? 'default'
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers)
    const token = this.getSessionToken()

    if (token) {
      headers.set('X-MeshPool-Session', token)
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers,
    })

    if (!response.ok) {
      const text = await response.text()
      throw new ApiError(text || response.statusText, response.status)
    }

    return response.json() as Promise<T>
  }

  health() {
    return this.request<HealthResponse>('/api/v1/health')
  }

  async authLocal(owner: string, password: string) {
    const response = await this.request<AuthResponse>('/api/v1/auth/local', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ owner, password }),
    })
    this.setSessionToken(response.session_token, response.owner)
    return response
  }

  async authDevice(device_code: string) {
    const response = await this.request<AuthResponse>('/api/v1/auth/device', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ device_code }),
    })
    this.setSessionToken(response.session_token, response.owner)
    return response
  }

  getCluster() { return this.request<ClusterResponse>('/api/v1/cluster') }
  async getNodes() { const data = await this.request<{ nodes: Node[] }>('/api/v1/nodes'); return data.nodes }
  getWallet() { return this.request<WalletResponse>('/api/v1/wallet') }
  getLedger() { return this.request<LedgerResponse>('/api/v1/ledger') }
  async getJobs() { return (await this.request<JobsResponse>('/api/v1/jobs')).jobs }
  getJob(id: string) { return this.request<BatchJob>(`/api/v1/jobs/${id}`) }
  getJobTimeline(id: string) { return this.request<JobTimeline>(`/api/v1/jobs/${id}/timeline`) }
  submitJob(body: JobSubmitRequest) { return this.request<BatchJob>('/api/v1/jobs', { method: 'POST', headers: jsonHeaders, body: JSON.stringify(body) }) }
  async getModels() { return (await this.request<ModelsResponse>('/api/v1/models')).models }
  /**
   * The endpoint returns `{builtin: string[], ollama: string[]}`, NOT a bare array.
   *
   * This was previously declared as `request<string[]>`, which is a *type
   * assertion* over untrusted network data, not a validation -- so `tsc` was
   * perfectly happy while the Models page called `.map()` on an object at
   * runtime and rendered a blank screen. Unwrap the real shape here, and stay
   * tolerant of a server that returns a plain array or omits a key.
   */
  async getAvailableModels(): Promise<string[]> {
    const raw = await this.request<AvailableModelsResponse | string[]>(
      '/api/v1/models/available',
    )
    if (Array.isArray(raw)) return raw.filter((m): m is string => typeof m === 'string')
    if (!raw || typeof raw !== 'object') return []
    const names = [
      ...(Array.isArray(raw.builtin) ? raw.builtin : []),
      ...(Array.isArray(raw.ollama) ? raw.ollama : []),
    ].filter((m): m is string => typeof m === 'string')
    return [...new Set(names)]
  }
  deployModel(body: DeployModelRequest) { return this.request<ModelDeployment>('/api/v1/models', { method: 'POST', headers: jsonHeaders, body: JSON.stringify(body) }) }
  deleteModel(id: string) { return this.request<{ status: string }>(`/api/v1/models/${id}`, { method: 'DELETE' }) }
  async getDeployments() { return (await this.request<DeploymentsResponse>('/api/v1/deployments')).deployments }
  getDeployment(id: string) { return this.request<Deployment>(`/api/v1/deployments/${id}`) }
  deleteDeployment(id: string) { return this.request<{ status: string }>(`/api/v1/deployments/${id}`, { method: 'DELETE' }) }
  async getBlobs() { return (await this.request<BlobsResponse>('/api/v1/blobs')).blobs }
  /** Upload a wasm32-wasip1 command module; the pool returns `wasm:<sha256>`. */
  uploadFunction(bytes: ArrayBuffer, name?: string) {
    const q = name ? `?name=${encodeURIComponent(name)}` : ''
    return this.request<FunctionUpload>(`/api/v1/functions${q}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/wasm' },
      body: bytes,
    })
  }  uploadBlob(file: File) {
    const form = new FormData()
    form.append('file', file)
    return this.request<BlobMeta>('/api/v1/blobs', { method: 'POST', body: form })
  }
  deleteBlob(id: string) { return this.request<{ status: string }>(`/api/v1/blobs/${id}`, { method: 'DELETE' }) }
  reclaimNode(id: string) { return this.request<{ status: 'draining' }>(`/api/v1/nodes/${id}/reclaim`, { method: 'POST' }) }
  resumeNode(id: string) { return this.request<{ status: 'online' }>(`/api/v1/nodes/${id}/resume`, { method: 'POST' }) }
  setNodePolicy(id: string, body: DonationPolicy) { return this.request<Node>(`/api/v1/nodes/${id}/policy`, { method: 'POST', headers: jsonHeaders, body: JSON.stringify(body) }) }
  getOnboarding() { return this.request<OnboardingResponse>('/api/v1/onboarding') }
  /**
   * Returns the secret NAMES held for this owner.
   *
   * The endpoint answers `{names, active_leases}` -- values are never served
   * back. This was declared as `{secrets: SecretRecord[]}`, so `.secrets` was
   * `undefined` and the Account page died on `undefined.map(...)`. Guard the
   * shape here so a server that omits the key cannot blank the page.
   */
  async getSecrets(): Promise<string[]> {
    const raw = await this.request<SecretsResponse>('/api/v1/secrets')
    const names = raw && Array.isArray(raw.names) ? raw.names : []
    return names.filter((n): n is string => typeof n === 'string')
  }
  putSecret(name: string, value: string, owner = this.getOwner()) { return this.request<SecretRecord>('/api/v1/secrets', { method: 'POST', headers: jsonHeaders, body: JSON.stringify({ name, value, owner }) }) }
  deleteSecret(name: string) { return this.request<{ status: string }>(`/api/v1/secrets/${name}`, { method: 'DELETE' }) }

  /**
   * Hard spend cap. `enforced: false` means no cap exists -- the UI must say so
   * rather than render a number nothing enforces (this page previously showed a
   * hardcoded "200.0" cap that was pure fiction).
   */
  async getSpendCap(): Promise<SpendCapStatus> {
    const raw = await this.request<SpendCapStatus>('/api/v1/account/spend-cap')
    return {
      owner: raw?.owner ?? '',
      cap: typeof raw?.cap === 'number' ? raw.cap : null,
      spent: typeof raw?.spent === 'number' ? raw.spent : 0,
      remaining: typeof raw?.remaining === 'number' ? raw.remaining : null,
      enforced: raw?.enforced === true,
      exceeded: raw?.exceeded === true,
    }
  }
  setSpendCap(cap: number | null) { return this.request<SpendCapStatus>('/api/v1/account/spend-cap', { method: 'PUT', headers: jsonHeaders, body: JSON.stringify({ cap }) }) }
  resetSpendCap() { return this.request<SpendCapStatus>('/api/v1/account/spend-cap/reset', { method: 'POST' }) }

  async chat(prompt: string, options: Omit<ChatRequest, 'prompt'>): Promise<ReadableStreamDefaultReader<string>> {
    const token = this.getSessionToken()
    const response = await fetch(`${this.baseUrl}/api/v1/inference/chat`, {
      method: 'POST',
      headers: {
        ...jsonHeaders,
        ...(token ? { 'X-MeshPool-Session': token } : {}),
      },
      body: JSON.stringify({ prompt, ...options }),
    })

    if (!response.ok || !response.body) {
      throw new ApiError('Unable to start chat stream', response.status)
    }

    const decoder = new TextDecoder()
    const reader = response.body.getReader()

    return {
      async read(): Promise<ReadableStreamReadResult<string>> {
        const { done, value } = await reader.read()
        // The discriminated union requires done:true to carry value:undefined and
        // done:false to carry a value, so the two cases must be built separately.
        if (done) return { done: true, value: undefined }
        return { done: false, value: decoder.decode(value) }
      },
      releaseLock() { reader.releaseLock() },
      cancel(reason?: unknown) { return reader.cancel(reason) },
      closed: Promise.resolve(undefined),
    }
  }

  subscribeEvents(onMessage: (event: EventMessage) => void) {
    const source = new EventSource(`${this.baseUrl}/api/v1/events`)
    source.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data) as EventMessage)
      } catch {
        onMessage({ type: 'message', ts: Date.now(), payload: { raw: event.data } })
      }
    }
    return source
  }
}

export const apiClient = new ApiClient()
