/**
 * The Models page rendered a blank white screen in production.
 *
 * Cause: `GET /api/v1/models/available` returns `{builtin: [...], ollama: [...]}`
 * -- an object -- but `client.ts` declared it as `request<string[]>`. That is a
 * TypeScript *assertion* over untrusted network data, not a validation, so
 * `tsc -b` and `vite build` both passed while `available.map(...)` threw
 * "n.map is not a function" at runtime.
 *
 * It was worse than one broken page: there was no error boundary anywhere in
 * the app, so React unmounted the entire tree. Models, Jobs, Storage, the
 * navigation and the header all went blank, leaving no way to click out.
 *
 * These tests pin down both halves: the client must unwrap the real shape, and
 * a crashing page must not be able to take the shell with it.
 */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { Layout } from '../src/components/Layout'
import { apiClient } from '../src/api/client'
import { mockCluster } from '../src/api/mock'

beforeEach(() => {
  vi.restoreAllMocks()
  vi.spyOn(apiClient, 'getCluster').mockResolvedValue(mockCluster)
  vi.spyOn(apiClient, 'getJobs').mockResolvedValue([])
  vi.spyOn(apiClient, 'getBlobs').mockResolvedValue([])
})

describe('getAvailableModels unwraps the real response shape', () => {
  function stubFetch(body: unknown) {
    return vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
  }

  it('flattens {builtin, ollama} into a single array of names', async () => {
    stubFetch({ builtin: ['nano-gpt', 'qwen2-tiny-random'], ollama: ['llama3'] })
    await expect(apiClient.getAvailableModels()).resolves.toEqual([
      'nano-gpt',
      'qwen2-tiny-random',
      'llama3',
    ])
  })

  it('always returns something .map() can be called on', async () => {
    // Every one of these would have crashed the page before the fix.
    for (const body of [{}, { builtin: null }, { ollama: 'nope' }, null]) {
      stubFetch(body)
      const got = await apiClient.getAvailableModels()
      expect(Array.isArray(got)).toBe(true)
      expect(() => got.map((m) => m)).not.toThrow()
    }
  })

  it('still accepts a plain array, in case the server shape changes back', async () => {
    stubFetch(['nano-gpt'])
    await expect(apiClient.getAvailableModels()).resolves.toEqual(['nano-gpt'])
  })

  it('de-duplicates a name offered by both sources', async () => {
    stubFetch({ builtin: ['nano-gpt'], ollama: ['nano-gpt'] })
    await expect(apiClient.getAvailableModels()).resolves.toEqual(['nano-gpt'])
  })
})

describe('the Models page renders against the real payload', () => {
  it('shows every available model as a deploy option', async () => {
    vi.spyOn(apiClient, 'getModels').mockResolvedValue([])
    vi.spyOn(apiClient, 'getAvailableModels').mockResolvedValue([
      'nano-gpt',
      'qwen2-tiny-random',
    ])
    render(
      <MemoryRouter initialEntries={['/models']}>
        <App />
      </MemoryRouter>,
    )
    expect(
      await screen.findByRole('option', { name: 'qwen2-tiny-random' }),
    ).toBeInTheDocument()
    expect(await screen.findByRole('option', { name: 'nano-gpt' })).toBeInTheDocument()
  })
})

describe('a crashing page cannot destroy the console', () => {
  function Boom(): never {
    throw new Error('simulated render crash')
  }

  it('keeps the navigation usable and explains what happened', async () => {
    // React logs the caught error; silence it so the test output stays readable.
    vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <MemoryRouter initialEntries={['/boom']}>
        <Routes>
          <Route element={<Layout nodes={[]} />}>
            <Route path="/boom" element={<Boom />} />
          </Route>
        </Routes>
      </MemoryRouter>,
    )

    // The failure is reported...
    expect(await screen.findByRole('alert')).toBeInTheDocument()
    expect(screen.getByText(/simulated render crash/i)).toBeInTheDocument()
    // ...and the shell survives, so the user can navigate away.
    await waitFor(() => {
      expect(screen.getByRole('navigation', { name: /primary/i })).toBeInTheDocument()
    })
    expect(screen.getByRole('link', { name: /jobs/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry this page/i })).toBeInTheDocument()
  })

  it('NON-VACUITY: the same crash with no boundary really does blank everything', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    // Rendering Boom outside the Layout (hence outside the boundary) must throw
    // out of render entirely. If this ever stops throwing, the test above is
    // no longer evidence that the boundary is doing anything.
    expect(() =>
      render(
        <MemoryRouter initialEntries={['/boom']}>
          <Boom />
        </MemoryRouter>,
      ),
    ).toThrow(/simulated render crash/)
  })
})
