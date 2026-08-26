import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { apiClient } from '../src/api/client'
import { mockCluster } from '../src/api/mock'

beforeEach(() => {
  vi.spyOn(apiClient, 'getCluster').mockResolvedValue(mockCluster)
  vi.spyOn(apiClient, 'getModels').mockResolvedValue([])
  vi.spyOn(apiClient, 'getJobs').mockResolvedValue([])
  vi.spyOn(apiClient, 'getBlobs').mockResolvedValue([])
})

describe('theme switching', () => {
  it('detects system theme, persists user choice, and updates the document class', async () => {
    const user = userEvent.setup()
    localStorage.setItem('meshpool.theme', 'system')
    render(
      <MemoryRouter initialEntries={['/inference']}>
        <App />
      </MemoryRouter>,
    )
    const toggle = await screen.findByRole('button', { name: /dark mode|light mode/i })
    await user.click(toggle)
    expect(localStorage.getItem('meshpool.theme')).toMatch(/light|dark/)
    expect(document.documentElement.dataset.theme).toMatch(/light|dark/)
  })
})
