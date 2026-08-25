/**
 * MeshMap at real mesh scale.
 *
 * The Mesh Map is MeshPool's signature screen, and MeshPool's premise is
 * "1000 strangers' PCs". That scale was never actually rendered in a test --
 * and when it finally was, the component turned out to simulate one marker per
 * node (aggregation buckets on `owner`, and 1000 strangers means 1000 owners)
 * while its accessible table resolved each row's marker with a nested scan.
 *
 * These tests render the component at that scale and put a wall-clock bound on
 * it, so the regression cannot come back quietly.
 */
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MeshMap } from '../src/components/MeshMap'
import { mockCluster } from '../src/api/mock'
import type { Node } from '../src/api/types'

afterEach(cleanup)

function bigMesh(n: number): Node[] {
  const template = mockCluster.nodes[0]
  return Array.from({ length: n }, (_, i) => ({
    ...template,
    id: `node-${i}`,
    name: `node-${i}`,
    owner: `acct:local:owner-${i}`,
    status: i % 11 === 0 ? 'offline' : 'online',
  })) as Node[]
}

describe('MeshMap at 1000+ nodes', () => {
  it('renders 1000 nodes without hanging', () => {
    const t0 = performance.now()
    render(<MeshMap nodes={bigMesh(1000)} />)
    const ms = performance.now() - t0
    expect(screen.getByRole('img')).toBeInTheDocument()
    expect(ms, `initial render took ${ms.toFixed(0)}ms`).toBeLessThan(4000)
  })

  it('renders 5000 nodes without hanging', () => {
    const t0 = performance.now()
    render(<MeshMap nodes={bigMesh(5000)} />)
    const ms = performance.now() - t0
    expect(screen.getByRole('img')).toBeInTheDocument()
    expect(ms, `initial render took ${ms.toFixed(0)}ms`).toBeLessThan(6000)
  })

  it('reports the true node count even though rows are capped', () => {
    render(<MeshMap nodes={bigMesh(1000)} />)
    const canvas = screen.getByRole('img')
    expect(canvas).toHaveAttribute('aria-label', expect.stringMatching(/1000 nodes/i))
  })

  it('does not emit a DOM row for every node at scale', () => {
    const { container } = render(<MeshMap nodes={bigMesh(5000)} />)
    const rows = container.querySelectorAll('tbody tr')
    expect(rows.length).toBeLessThanOrEqual(250)
    expect(rows.length).toBeGreaterThan(0)
  })

  it('is honest that the table is truncated', () => {
    render(<MeshMap nodes={bigMesh(1000)} />)
    expect(screen.getByText(/more are not listed/i)).toBeInTheDocument()
  })

  it('does not truncate or warn on a small mesh', () => {
    const { container } = render(<MeshMap nodes={bigMesh(12)} />)
    expect(container.querySelectorAll('tbody tr')).toHaveLength(12)
    expect(screen.queryByText(/more are not listed/i)).not.toBeInTheDocument()
  })
})
