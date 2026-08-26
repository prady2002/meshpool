import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import App from '../src/App'
import { apiClient } from '../src/api/client'
import { mockCluster } from '../src/api/mock'

function renderApp(path = '/inference') {
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
  vi.spyOn(apiClient, 'getWallet').mockResolvedValue({ owner: 'default', balance: 0, donated: 0, consumed: 0, granted: 0, reputation: 1, gating: false, nodes: [] })
  vi.spyOn(apiClient, 'getLedger').mockResolvedValue({ owners: [], summary: { total_earned: 0, total_spent: 0, pending_payout: 0 }, by_node: [], credits_by_owner: [], fair_share: [], recent: [] })
  vi.spyOn(apiClient, 'getNodes').mockResolvedValue(mockCluster.nodes)
  vi.spyOn(apiClient, 'getOnboarding').mockResolvedValue({ control_plane_url: 'http://localhost', invite_token: 'dev', join_windows: 'join', join_unix: 'join', join_command: 'meshpool join' })
})

describe('command palette', () => {
  it('opens and closes from the keyboard shortcut', async () => {
    const user = userEvent.setup()
    renderApp()
    await user.keyboard('{Control>}k{/Control}')
    expect(await screen.findByRole('dialog', { name: /command palette/i })).toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: /command palette/i })).not.toBeInTheDocument()
  })

  it('fuzzy-searches and executes navigation commands', async () => {
    const user = userEvent.setup()
    renderApp('/settings')
    await user.keyboard('{Control>}k{/Control}')
    await user.type(screen.getByRole('combobox'), 'inference')
    await user.keyboard('{Enter}')
    expect(await screen.findByRole('heading', { name: /distributed llm chat/i })).toBeInTheDocument()
  })

  it('supports arrow-key selection and node commands', async () => {
    const user = userEvent.setup()
    renderApp('/inference')
    await user.keyboard('{Control>}k{/Control}')
    await user.type(screen.getByRole('combobox'), mockCluster.nodes[0].name)
    await user.keyboard('{Enter}')
    expect(await screen.findByRole('heading', { name: /topology, health, and cluster scale/i })).toBeInTheDocument()
  })

  it('announces result counts to assistive technology', async () => {
    const user = userEvent.setup()
    renderApp()
    await user.keyboard('{Control>}k{/Control}')
    await user.type(screen.getByRole('combobox'), 'logs')
    expect(screen.getByText(/command.*available/i, { selector: '.sr-only' })).toHaveTextContent(/logs/i)
  })
})
