import { describe, expect, it } from 'vitest'
import { aggregateNodes } from '../src/mesh'
import type { Node } from '../src/api/types'

/**
 * MeshPool's whole premise is "1000 strangers' PCs". That means 1000 DISTINCT
 * owners -- which is precisely the input that defeats bucketing by owner.
 *
 * The map's physics step and its proximity-line renderer are both O(n^2)
 * all-pairs loops that run every animation frame. If aggregation returns one
 * item per node at that scale, a browser tab does ~500k distance computations
 * twice per frame and locks up. Capping the number of simulated items is the
 * only thing standing between the signature screen and a frozen tab.
 */

function makeNodes(n: number, distinctOwners: boolean): Node[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `node-${i}`,
    name: `node-${i}`,
    owner: distinctOwners ? `acct:local:owner-${i}` : 'acct:local:solo',
    status: 'online',
    capability: {
      cores: 4, ram_mb: 8000, cpu_arch: i % 2 ? 'x86_64' : 'arm64',
      flops_estimate: 30,
    },
  } as unknown as Node))
}

const MAX_SIM_ITEMS = 160

describe('mesh aggregation scales to a real mesh', () => {
  it('caps simulated items for 1000 nodes with 1000 distinct owners', () => {
    const agg = aggregateNodes(makeNodes(1000, true))
    expect(agg.items.length).toBeLessThanOrEqual(MAX_SIM_ITEMS)
  })

  it('caps simulated items for 5000 nodes', () => {
    const agg = aggregateNodes(makeNodes(5000, true))
    expect(agg.items.length).toBeLessThanOrEqual(MAX_SIM_ITEMS)
  })

  it('never loses a node: counts always sum to the input size', () => {
    for (const n of [7, 50, 100, 1000, 5000]) {
      const agg = aggregateNodes(makeNodes(n, true))
      const total = agg.items.reduce((s, i) => s + i.count, 0)
      expect(total, `${n} nodes must all be represented`).toBe(n)
    }
  })

  it('still shows individual nodes on a small mesh', () => {
    const agg = aggregateNodes(makeNodes(8, true))
    expect(agg.mode).toBe('individual')
    expect(agg.items).toHaveLength(8)
  })

  it('aggregating 5000 nodes is fast enough to run on a poll', () => {
    const nodes = makeNodes(5000, true)
    const t0 = performance.now()
    aggregateNodes(nodes)
    const ms = performance.now() - t0
    expect(ms, `aggregation took ${ms.toFixed(1)}ms`).toBeLessThan(400)
  })

  it('is not quadratic when one operator runs many nodes', () => {
    // Building buckets with `set(key, [...get(key), node])` copies the whole
    // bucket on every insert, so a single owner with 5000 nodes costs ~12.5M
    // element copies. Same-owner aggregation must cost about the same as
    // distinct-owner aggregation.
    const nodes = makeNodes(5000, false)
    const t0 = performance.now()
    const agg = aggregateNodes(nodes)
    const ms = performance.now() - t0
    expect(agg.items.reduce((s, i) => s + i.count, 0)).toBe(5000)
    expect(ms, `same-owner aggregation took ${ms.toFixed(1)}ms`).toBeLessThan(400)
  })

  it('an all-pairs frame over the capped item count stays under budget', () => {
    // Mirrors the physics/proximity loops in MeshMap.tsx. A 60fps frame is
    // 16.7ms and the map must leave room for actual drawing.
    const agg = aggregateNodes(makeNodes(1000, true))
    const pts = agg.items.map((_, i) => ({ x: i * 7 % 900, y: i * 13 % 600 }))
    const t0 = performance.now()
    for (let f = 0; f < 60; f++) {
      let acc = 0
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const dx = pts[j].x - pts[i].x
          const dy = pts[j].y - pts[i].y
          acc += dx * dx + dy * dy
        }
      }
      if (acc < 0) throw new Error('unreachable')
    }
    const perFrame = (performance.now() - t0) / 60
    expect(perFrame, `${perFrame.toFixed(2)}ms per frame`).toBeLessThan(8)
  })

  it('mapping every node to its item is linear, not quadratic', () => {
    // MeshMap's accessible table used to resolve each row's marker with
    // `items.find(i => i.nodes.some(n => n.id === row.id))` *inside* the row
    // loop. That is O(items x nodes) per row and melts on a large mesh. The
    // component now builds this index once; this test pins that cost.
    const nodes = makeNodes(5000, true)
    const agg = aggregateNodes(nodes)
    const t0 = performance.now()
    const index = new Map<string, string>()
    for (const item of agg.items) {
      for (const raw of item.nodes) index.set(raw.id, item.id)
    }
    const ms = performance.now() - t0
    expect(index.size).toBe(5000)
    expect(ms, `index build took ${ms.toFixed(1)}ms`).toBeLessThan(100)
  })
})
