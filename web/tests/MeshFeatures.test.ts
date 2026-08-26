/**
 * MeshFeatures — unit tests for inferNodeClass() and groupConstellations().
 *
 * Both functions are pure; tests are fast and don't need a browser.
 *
 * MUTATION CONTRACT (two properties verified with real mutation runs):
 *
 *   Property 1 – constellation bucketing uses 25-ms windows:
 *     mutate: `Math.floor(node.link.latency_ms / 25) * 25`
 *          → `node.link.latency_ms`  (each node its own bucket)
 *     expected: 'puts nodes at 10 ms and 15 ms in the same bucket' FAILS
 *
 *   Property 2 – anchor classification uses trust threshold 0.93:
 *     mutate: `node.trust_score >= 0.93` → `node.trust_score >= 1.01`
 *     expected: 'classifies anchor node correctly' FAILS
 */
import { describe, expect, it } from 'vitest'
import { groupConstellations, inferNodeClass } from '../src/mesh'
import type { Node } from '../src/api/types'

// ── helpers ────────────────────────────────────────────────────────────────────

const now = Date.now()

function makeNode(overrides: Partial<Node> = {}): Node {
  return {
    id: 'node-test',
    name: 'Test Node',
    url: 'https://test.mesh',
    owner: 'test-owner',
    status: 'online',
    capability: {
      cpu_cores: 8,
      cpu_arch: 'x86_64',
      ram_mb: 16384,
      gpu: false,
      vram_mb: 0,
      disk_mb: 102400,
      flops_estimate: 42.5,
      ollama_models: [],
    },
    link: { latency_ms: 15, bandwidth_mbps: 300 },
    policy: { cpu_pct: 50, ram_mb: 4096, gpu_allowed: false, schedule: 'always', idle_minutes: 10 },
    load: { cpu_load_pct: 20, ram_used_mb: 2048, running_units: 1, idle_seconds: 0 },
    trust_score: 0.85,
    last_heartbeat_ms: now,
    registered_ms: now - 14 * 86_400_000,  // 14 days old
    ...overrides,
  } as Node
}

// ── inferNodeClass ─────────────────────────────────────────────────────────────

describe('inferNodeClass', () => {
  it('classifies anchor node correctly', () => {
    const node = makeNode({
      trust_score: 0.95,
      registered_ms: now - 14 * 86_400_000,
      link: { latency_ms: 15, bandwidth_mbps: 300 },
    })
    // MUTATION GUARD: change `>= 0.93` to `>= 1.01` and this test fails
    expect(inferNodeClass(node)).toBe('anchor')
  })

  it('anchor requires trust ≥ 0.93 — node at exactly 0.93 is still anchor', () => {
    const node = makeNode({ trust_score: 0.93, link: { latency_ms: 20, bandwidth_mbps: 100 } })
    expect(inferNodeClass(node)).toBe('anchor')
  })

  it('anchor requires trust < 0.93 to be constellation (trust 0.92 → not anchor)', () => {
    const node = makeNode({ trust_score: 0.92, link: { latency_ms: 20, bandwidth_mbps: 100 } })
    expect(inferNodeClass(node)).not.toBe('anchor')
  })

  it('classifies a high-bandwidth idle node as relay', () => {
    const node = makeNode({
      trust_score: 0.85,
      link: { latency_ms: 20, bandwidth_mbps: 600 },
      load: { cpu_load_pct: 5, ram_used_mb: 512, running_units: 0, idle_seconds: 300 },
    })
    expect(inferNodeClass(node)).toBe('relay')
  })

  it('relay requires running_units === 0 — a busy high-bandwidth node is not relay', () => {
    const node = makeNode({
      trust_score: 0.85,
      link: { latency_ms: 20, bandwidth_mbps: 600 },
      load: { cpu_load_pct: 50, ram_used_mb: 4096, running_units: 3, idle_seconds: 0 },
    })
    expect(inferNodeClass(node)).not.toBe('relay')
  })

  it('classifies a fresh (<1 day) node as drifter regardless of trust', () => {
    const node = makeNode({ trust_score: 0.99, registered_ms: now - 3_600_000 })
    expect(inferNodeClass(node)).toBe('drifter')
  })

  it('classifies a low-trust node as drifter', () => {
    const node = makeNode({ trust_score: 0.60 })
    expect(inferNodeClass(node)).toBe('drifter')
  })

  it('classifies a high-latency node as drifter', () => {
    const node = makeNode({ link: { latency_ms: 150, bandwidth_mbps: 100 } })
    expect(inferNodeClass(node)).toBe('drifter')
  })

  it('drifter boundary: trust 0.69 (strictly below 0.70) is drifter', () => {
    const node = makeNode({ trust_score: 0.69, link: { latency_ms: 20, bandwidth_mbps: 50 } })
    expect(inferNodeClass(node)).toBe('drifter')
  })

  it('trust at exactly 0.70 is not a drifter (boundary is exclusive)', () => {
    const node = makeNode({ trust_score: 0.70, link: { latency_ms: 20, bandwidth_mbps: 50 } })
    expect(inferNodeClass(node)).not.toBe('drifter')
  })

  it('trust just above 0.70 (0.71) with normal latency is not a drifter', () => {
    const node = makeNode({ trust_score: 0.71, link: { latency_ms: 40, bandwidth_mbps: 100 } })
    expect(inferNodeClass(node)).not.toBe('drifter')
  })

  it('classifies a typical pool node as constellation', () => {
    const node = makeNode({
      trust_score: 0.82,
      link: { latency_ms: 40, bandwidth_mbps: 200 },
      load: { cpu_load_pct: 30, ram_used_mb: 4000, running_units: 2, idle_seconds: 0 },
    })
    expect(inferNodeClass(node)).toBe('constellation')
  })

  it('drifter beats anchor when latency is high (order of checks matters)', () => {
    // High trust + old node, but very high latency → drifter wins
    const node = makeNode({
      trust_score: 0.99,
      registered_ms: now - 30 * 86_400_000,
      link: { latency_ms: 200, bandwidth_mbps: 100 },
    })
    expect(inferNodeClass(node)).toBe('drifter')
  })
})

// ── groupConstellations ────────────────────────────────────────────────────────

describe('groupConstellations', () => {
  it('returns empty array for empty input', () => {
    expect(groupConstellations([])).toEqual([])
  })

  it('puts nodes at 10 ms and 15 ms in the same 25-ms bucket', () => {
    // MUTATION GUARD: replace `/ 25) * 25` with just `latency_ms` and this fails
    const nodes = [
      makeNode({ id: 'a', link: { latency_ms: 10, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', link: { latency_ms: 15, bandwidth_mbps: 100 } }),
    ]
    const groups = groupConstellations(nodes)
    expect(groups).toHaveLength(1)
    expect(groups[0].nodeIds).toContain('a')
    expect(groups[0].nodeIds).toContain('b')
  })

  it('puts nodes at 10 ms and 30 ms in different buckets', () => {
    const nodes = [
      makeNode({ id: 'a', link: { latency_ms: 10, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', link: { latency_ms: 30, bandwidth_mbps: 100 } }),
    ]
    const groups = groupConstellations(nodes)
    expect(groups).toHaveLength(2)
    expect(groups.find((g) => g.nodeIds.includes('a'))).toBeTruthy()
    expect(groups.find((g) => g.nodeIds.includes('b'))).toBeTruthy()
  })

  it('bucket boundary: 24 ms and 25 ms land in different buckets', () => {
    const nodes = [
      makeNode({ id: 'a', link: { latency_ms: 24, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', link: { latency_ms: 25, bandwidth_mbps: 100 } }),
    ]
    const groups = groupConstellations(nodes)
    expect(groups).toHaveLength(2)
  })

  it('includes every node in exactly one group', () => {
    const nodes = Array.from({ length: 20 }, (_, i) =>
      makeNode({ id: `n${i}`, link: { latency_ms: i * 10, bandwidth_mbps: 100 } }),
    )
    const groups = groupConstellations(nodes)
    const allIds = groups.flatMap((g) => g.nodeIds)
    expect(allIds).toHaveLength(20)
    expect(new Set(allIds).size).toBe(20)  // no duplicates
  })

  it('latencyBucketMs is always a multiple of 25', () => {
    const nodes = [
      makeNode({ id: 'a', link: { latency_ms: 37, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', link: { latency_ms: 99, bandwidth_mbps: 100 } }),
    ]
    for (const g of groupConstellations(nodes)) {
      expect(g.latencyBucketMs % 25).toBe(0)
    }
    expect(groupConstellations([makeNode({ link: { latency_ms: 37, bandwidth_mbps: 100 } })])[0].latencyBucketMs).toBe(25)
    expect(groupConstellations([makeNode({ link: { latency_ms: 99, bandwidth_mbps: 100 } })])[0].latencyBucketMs).toBe(75)
  })

  it('computes onlineCount correctly', () => {
    const nodes = [
      makeNode({ id: 'a', status: 'online',  link: { latency_ms: 10, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', status: 'offline', link: { latency_ms: 12, bandwidth_mbps: 100 } }),
      makeNode({ id: 'c', status: 'online',  link: { latency_ms: 14, bandwidth_mbps: 100 } }),
    ]
    const groups = groupConstellations(nodes)
    expect(groups).toHaveLength(1)
    expect(groups[0].onlineCount).toBe(2)
    expect(groups[0].nodeIds).toHaveLength(3)
  })

  it('sums totalCompute for all members', () => {
    const base = makeNode().capability
    const nodes = [
      makeNode({ id: 'a', capability: { ...base, flops_estimate: 10 }, link: { latency_ms: 5, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', capability: { ...base, flops_estimate: 20 }, link: { latency_ms: 8, bandwidth_mbps: 100 } }),
    ]
    const groups = groupConstellations(nodes)
    expect(groups[0].totalCompute).toBeCloseTo(30)
  })

  it('single-node group has the node name as label', () => {
    const node = makeNode({ id: 'solo', name: 'Solo Node' })
    const groups = groupConstellations([node])
    expect(groups[0].label).toBe('Solo Node')
  })

  it('multi-node group label contains neighbourhood and count', () => {
    const nodes = [
      makeNode({ id: 'a', link: { latency_ms: 5, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', link: { latency_ms: 5, bandwidth_mbps: 100 } }),
    ]
    const groups = groupConstellations(nodes)
    expect(groups[0].label).toMatch(/neighbourhood/i)
    expect(groups[0].label).toMatch(/2 nodes/)
  })

  it('group id is deterministic based on latency bucket', () => {
    // Two nodes with different latency values but the same bucket (0-24ms)
    const a = groupConstellations([makeNode({ link: { latency_ms: 5, bandwidth_mbps: 100 } })])[0].id
    const b = groupConstellations([makeNode({ link: { latency_ms: 19, bandwidth_mbps: 100 } })])[0].id
    expect(a).toBe(b)
    expect(a).toBe('const-0')
  })

  it('grouping 1000 nodes is fast enough to run on every 10-s poll', () => {
    const nodes = Array.from({ length: 1000 }, (_, i) =>
      makeNode({ id: `n${i}`, link: { latency_ms: (i % 50) * 5, bandwidth_mbps: 100 } }),
    )
    const t0 = performance.now()
    const groups = groupConstellations(nodes)
    const ms = performance.now() - t0
    expect(ms, `groupConstellations(1000 nodes) took ${ms.toFixed(1)}ms`).toBeLessThan(50)
    expect(groups.length).toBeGreaterThan(0)
  })

  it('grouping does not lose any node at scale', () => {
    const n = 5000
    const nodes = Array.from({ length: n }, (_, i) =>
      makeNode({ id: `n${i}`, link: { latency_ms: Math.floor(Math.random() * 200), bandwidth_mbps: 100 } }),
    )
    const groups = groupConstellations(nodes)
    const total = groups.reduce((s, g) => s + g.nodeIds.length, 0)
    expect(total).toBe(n)
  })
})
