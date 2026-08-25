/**
 * The console must actually BE the console.
 *
 * This file exists because `web/` type-checked and built cleanly for its entire
 * life while `src/App.tsx` was still the untouched Vite starter template: seven
 * finished pages sat in `src/pages/` and none of them were ever mounted. A build
 * that succeeds proves nothing about what the build contains.
 *
 * Every test below is written so that it FAILS against that starter template.
 */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { apiClient } from '../src/api/client'
import { mockCluster } from '../src/api/mock'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.spyOn(apiClient, 'getCluster').mockResolvedValue(mockCluster)
  vi.spyOn(apiClient, 'getModels').mockResolvedValue([])
  vi.spyOn(apiClient, 'getJobs').mockResolvedValue([])
  vi.spyOn(apiClient, 'getBlobs').mockResolvedValue([])
})

describe('console shell', () => {
  it('mounts the MeshPool console, not the Vite starter', async () => {
    renderAt('/')
    expect(
      await screen.findByRole('heading', { name: /community compute console/i }),
    ).toBeInTheDocument()
    // The starter template's signature elements must be gone.
    expect(screen.queryByText(/Edit .*App\.tsx.* and save to test/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /count is/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /explore vite/i })).not.toBeInTheDocument()
  })

  it('renders every primary navigation destination', async () => {
    renderAt('/')
    const nav = await screen.findByRole('navigation', { name: /primary/i })
    for (const label of ['Playground', 'Mesh Map', 'Models', 'Functions', 'Jobs', 'Storage', 'Contribute', 'Account']) {
      expect(nav).toHaveTextContent(label)
    }
  })

  it('lands a new visitor on the playground rather than a dashboard', async () => {
    renderAt('/')
    expect(await screen.findByRole('heading', { name: /distributed llm chat/i })).toBeInTheDocument()
  })

  it('redirects unknown paths instead of rendering nothing', async () => {
    renderAt('/this-route-does-not-exist')
    expect(await screen.findByRole('heading', { name: /distributed llm chat/i })).toBeInTheDocument()
  })
})

describe('routes resolve to their real pages', () => {
  const cases: Array<[string, RegExp]> = [
    ['/jobs', /submit work and watch progress/i],
    ['/storage', /objects, replicas, shard locations/i],
    ['/functions', /run your own code on the pool/i],
  ]

  for (const [path, heading] of cases) {
    it(`${path} renders its page`, async () => {
      renderAt(path)
      expect(await screen.findByRole('heading', { name: heading })).toBeInTheDocument()
    })
  }
})

describe('live cluster data reaches the shell', () => {
  it('requests the cluster once mounted', async () => {
    renderAt('/')
    await waitFor(() => expect(apiClient.getCluster).toHaveBeenCalled())
  })
})
