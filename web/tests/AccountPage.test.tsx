/**
 * The Account page rendered a blank white screen in production.
 *
 * Cause: `GET /api/v1/secrets` answers `{names, active_leases}` -- it returns
 * only NAMES, because the control plane never serves secret values back. The
 * console declared the response as `{secrets: SecretRecord[]}`, so `.secrets`
 * was `undefined`, `setSecrets(undefined)` ran, and `secrets.map(...)` threw
 * "Cannot read properties of undefined (reading 'map')".
 *
 * Same root cause as the Models page: `request<T>()` is a type *assertion* over
 * untrusted network data, so `tsc` cannot catch a server/client shape mismatch.
 * See `sim/scenarios/verify_console_api_contract.py`, which checks these shapes
 * against a REAL running control plane rather than against a mock.
 */
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { apiClient } from '../src/api/client'
import { mockCluster } from '../src/api/mock'
import type { SpendCapStatus, WalletResponse } from '../src/api/types'

const uncapped: SpendCapStatus = {
  owner: 'default',
  cap: null,
  spent: 105.03,
  remaining: null,
  enforced: false,
  exceeded: false,
}

const wallet: WalletResponse = {
  owner: 'default',
  balance: -105.03,
  donated: 0,
  consumed: 105.03,
  granted: 0,
  reputation: 1,
  gating: false,
  nodes: [],
} as WalletResponse

beforeEach(() => {
  vi.restoreAllMocks()
  vi.spyOn(apiClient, 'getCluster').mockResolvedValue(mockCluster)
  vi.spyOn(apiClient, 'getJobs').mockResolvedValue([])
  vi.spyOn(apiClient, 'getBlobs').mockResolvedValue([])
  vi.spyOn(apiClient, 'getModels').mockResolvedValue([])
  vi.spyOn(apiClient, 'getSpendCap').mockResolvedValue(uncapped)
})

function stubFetch(body: unknown) {
  return vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }),
  )
}

describe('getSecrets reads the real response shape', () => {
  it('returns the names the server actually sends', async () => {
    stubFetch({ names: ['openai_key', 'db_url'], active_leases: 0 })
    await expect(apiClient.getSecrets()).resolves.toEqual(['openai_key', 'db_url'])
  })

  it('never returns something .map() would throw on', async () => {
    // The pre-fix code produced `undefined` for every one of these.
    for (const body of [{}, { names: null }, { secrets: [] }, null]) {
      stubFetch(body)
      const got = await apiClient.getSecrets()
      expect(Array.isArray(got)).toBe(true)
    }
  })
})

describe('the Account page renders against the real payload', () => {
  it('lists secret names and stays up when there are none', async () => {
    vi.spyOn(apiClient, 'getWallet').mockResolvedValue(wallet)
    vi.spyOn(apiClient, 'getSecrets').mockResolvedValue([])
    render(
      <MemoryRouter initialEntries={['/account']}>
        <App />
      </MemoryRouter>,
    )
    expect(
      await screen.findByRole('heading', { name: /usage, spend caps, and api keys/i }),
    ).toBeInTheDocument()
    // An empty list must be an explained empty state, not a crash and not a void.
    expect(await screen.findByText(/no secrets yet/i)).toBeInTheDocument()
    // No error boundary should have been triggered.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('renders each secret name with a delete control', async () => {
    vi.spyOn(apiClient, 'getWallet').mockResolvedValue(wallet)
    vi.spyOn(apiClient, 'getSecrets').mockResolvedValue(['openai_key'])
    render(
      <MemoryRouter initialEntries={['/account']}>
        <App />
      </MemoryRouter>,
    )
    expect(await screen.findByText('openai_key')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument()
  })

  it('does not fabricate a spend cap the backend does not enforce', async () => {
    vi.spyOn(apiClient, 'getWallet').mockResolvedValue(wallet)
    vi.spyOn(apiClient, 'getSecrets').mockResolvedValue([])
    render(
      <MemoryRouter initialEntries={['/account']}>
        <App />
      </MemoryRouter>,
    )
    expect(await screen.findByText(/hard spend cap/i)).toBeInTheDocument()
    // The page used to print a hardcoded "200.0", which a user would reasonably
    // read as a real, enforced limit on their spend. When no cap is set the page
    // must say so rather than invent a number.
    expect(await screen.findByText(/no cap set/i)).toBeInTheDocument()
    expect(screen.queryByText('200.0')).not.toBeInTheDocument()
  })

  it('shows a real enforced cap with remaining budget', async () => {
    vi.spyOn(apiClient, 'getWallet').mockResolvedValue(wallet)
    vi.spyOn(apiClient, 'getSecrets').mockResolvedValue([])
    vi.spyOn(apiClient, 'getSpendCap').mockResolvedValue({
      owner: 'default', cap: 250, spent: 100, remaining: 150,
      enforced: true, exceeded: false,
    })
    render(
      <MemoryRouter initialEntries={['/account']}>
        <App />
      </MemoryRouter>,
    )
    expect(await screen.findByText('250.0')).toBeInTheDocument()
    expect(await screen.findByText(/150\.0 remaining/i)).toBeInTheDocument()
    expect(screen.queryByText(/no cap set/i)).not.toBeInTheDocument()
  })

  it('says work is stopped -- not merely flagged -- when the cap is reached', async () => {
    vi.spyOn(apiClient, 'getWallet').mockResolvedValue(wallet)
    vi.spyOn(apiClient, 'getSecrets').mockResolvedValue([])
    vi.spyOn(apiClient, 'getSpendCap').mockResolvedValue({
      owner: 'default', cap: 100, spent: 100, remaining: 0,
      enforced: true, exceeded: true,
    })
    render(
      <MemoryRouter initialEntries={['/account']}>
        <App />
      </MemoryRouter>,
    )
    // The whole product claim is that a cap stops work; the copy must not
    // soften that into an alert.
    expect(await screen.findByText(/work is stopped, not just flagged/i)).toBeInTheDocument()
  })

  it('survives a spend-cap payload with missing fields', async () => {
    vi.spyOn(apiClient, 'getWallet').mockResolvedValue(wallet)
    vi.spyOn(apiClient, 'getSecrets').mockResolvedValue([])
    vi.spyOn(apiClient, 'getSpendCap').mockResolvedValue({} as SpendCapStatus)
    render(
      <MemoryRouter initialEntries={['/account']}>
        <App />
      </MemoryRouter>,
    )
    expect(
      await screen.findByRole('heading', { name: /usage, spend caps, and api keys/i }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('survives a wallet payload missing numeric fields', async () => {
    // `wallet?.balance.toFixed(1)` only guarded `wallet`, not `balance`.
    vi.spyOn(apiClient, 'getWallet').mockResolvedValue({} as WalletResponse)
    vi.spyOn(apiClient, 'getSecrets').mockResolvedValue([])
    render(
      <MemoryRouter initialEntries={['/account']}>
        <App />
      </MemoryRouter>,
    )
    expect(
      await screen.findByRole('heading', { name: /usage, spend caps, and api keys/i }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
