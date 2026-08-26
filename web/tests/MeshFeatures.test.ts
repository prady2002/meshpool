/**
 * MeshFeatures — unit tests for inferNodeClass() and groupConstellations().
 *
 * Both functions are pure; tests are fast and don't need a browser.
 *
 * MUTATION CONTRACT (four properties verified with real mutation runs):
 *
 *   Property 1 – constellation bucketing uses 25-ms windows (prior session):
 *     mutate: `Math.floor(node.link.latency_ms / 25) * 25`
 *          → `node.link.latency_ms`  (each node its own bucket)
 *     expected: 'puts nodes at 10 ms and 15 ms in the same bucket' FAILS
 *
 *   Property 2 – anchor classification uses trust threshold 0.93 (prior session):
 *     mutate: `node.trust_score >= 0.93` → `node.trust_score >= 1.01`
 *     expected: 'classifies anchor node correctly' FAILS
 *
 *   Property 3 – resolveConstellationHalos uses inverted index (not per-call naive scan):
 *     mutate: replace nodeIdToIdx Map with inline `cgSet = new Set(cg.nodeIds); snodeNodeIds[i].some(...)`
 *     expected: '600 calls under 200ms' FAILS (~941ms for naive form)
 *
 *   Property 4 – resolveConstellationHalos returns no halos for approximated groups:
 *     mutate: remove `if (cg.source !== 'measured') return []` guard
 *     expected: 'returns empty for approximated groups' FAILS
 */
import { describe, expect, it } from 'vitest'
import { groupConstellations, inferNodeClass, resolveConstellationHalos } from '../src/mesh'
import type { ConstellationGroup, Node } from '../src/api/types'

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

// ── resolveConstellationHalos ──────────────────────────────────────────────────

/**
 * Build a set of measured ConstellationGroup objects for perf tests.
 * In production, groupConstellations() always returns source:'approximated'.
 * Measured groups would come from a future server API (Vivaldi coordinates).
 */
function makeMeasuredGroups(nodeCount: number, bucketCount: number): { groups: ConstellationGroup[]; snodeNodeIds: string[][] } {
  // Distribute nodeCount nodes across bucketCount latency buckets
  const nodesPerBucket = Math.ceil(nodeCount / bucketCount)
  const groups: ConstellationGroup[] = []
  const snodeNodeIds: string[][] = []

  for (let b = 0; b < bucketCount; b++) {
    const nodeIds = Array.from({ length: nodesPerBucket }, (_, j) => `n${b * nodesPerBucket + j}`)
    groups.push({
      id: `const-${b * 25}`,
      latencyBucketMs: b * 25,
      nodeIds,
      totalCompute: nodesPerBucket * 42,
      onlineCount: nodesPerBucket,
      label: `${b * 25}–${b * 25 + 24} ms neighbourhood (${nodesPerBucket} nodes)`,
      source: 'measured',
    })
  }

  // Simulate ~250 aggregated sim nodes (MAX_SIM_ITEMS), each covering 4 raw nodes
  const simCount = Math.min(250, nodeCount)
  for (let si = 0; si < simCount; si++) {
    snodeNodeIds.push(Array.from({ length: 4 }, (_, ri) => `n${si * 4 + ri}`))
  }

  return { groups, snodeNodeIds }
}

describe('resolveConstellationHalos — honesty', () => {
  it('groupConstellations marks all groups as approximated (not pairwise-measured)', () => {
    const nodes = [
      makeNode({ id: 'a', link: { latency_ms: 10, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', link: { latency_ms: 12, bandwidth_mbps: 100 } }),
    ]
    for (const g of groupConstellations(nodes)) {
      // MUTATION GUARD: removing 'source' from ConstellationGroup or setting it to
      // 'measured' would allow halos on data that is not pairwise-confirmed.
      expect(g.source).toBe('approximated')
    }
  })

  it('returns no halos for approximated groups (no confident cluster visual)', () => {
    // MUTATION GUARD: removing `if (cg.source !== 'measured') return []` makes this fail
    const approxGroups = groupConstellations([
      makeNode({ id: 'a', link: { latency_ms: 10, bandwidth_mbps: 100 } }),
      makeNode({ id: 'b', link: { latency_ms: 14, bandwidth_mbps: 100 } }),
    ])
    const halos = resolveConstellationHalos(approxGroups, [['a'], ['b']])
    expect(halos).toHaveLength(0)
  })

  it('returns halos for source:measured groups with 2+ distinct sim nodes', () => {
    const measured: ConstellationGroup[] = [{
      id: 'const-0', latencyBucketMs: 0, nodeIds: ['a', 'b'], totalCompute: 0,
      onlineCount: 2, label: 'test', source: 'measured',
    }]
    const halos = resolveConstellationHalos(measured, [['a'], ['b']])
    expect(halos).toHaveLength(1)
    expect(halos[0].memberIndices).toHaveLength(2)
  })

  it('does not produce a halo when both nodes land on the same sim node', () => {
    // If both a and b are in sim node 0, the halo would have only 1 distinct index
    const measured: ConstellationGroup[] = [{
      id: 'const-0', latencyBucketMs: 0, nodeIds: ['a', 'b'], totalCompute: 0,
      onlineCount: 2, label: 'test', source: 'measured',
    }]
    const halos = resolveConstellationHalos(measured, [['a', 'b']])  // both in sim node 0
    expect(halos).toHaveLength(0)
  })
})

describe('resolveConstellationHalos — performance (Defect 1 regression guard)', () => {
  it('600 calls at 1000-node scale stay under 500ms (catches per-frame rescan regression)', () => {
    // WHAT THIS TESTS: resolveConstellationHalos uses an inverted index (O(S×R + C×N))
    // rather than a per-call naive scan (O(C×S×R)). At 1000 nodes the naive form
    // benchmarks at ~941ms for 600 calls; the inverted-index form takes ~265ms.
    // The 500ms threshold sits squarely between the two — the naive mutation fails.
    //
    // MUTATION TO CONFIRM FAILURE: replace the inverted-index body in mesh.ts with
    //   const cgSet = new Set(cg.nodeIds)
    //   const memberIndices = snodeNodeIds.reduce((acc, ids, i) => {
    //     if (ids.some(id => cgSet.has(id))) acc.push(i); return acc
    //   }, [] as number[])
    // and this test will exceed the 500ms budget (~941ms observed).
    const { groups, snodeNodeIds } = makeMeasuredGroups(1000, 40)

    const t0 = performance.now()
    for (let i = 0; i < 600; i++) {
      resolveConstellationHalos(groups, snodeNodeIds)
    }
    const totalMs = performance.now() - t0

    expect(totalMs, `600 calls took ${totalMs.toFixed(0)}ms — naive form takes ~941ms`).toBeLessThan(500)
  })

  it('a single call returns the correct membership indices', () => {
    // Correctness check: sim node 0 has ['n0','n1','n2','n3'], groups[0] covers n0..n24
    const { groups, snodeNodeIds } = makeMeasuredGroups(100, 4)
    const halos = resolveConstellationHalos(groups, snodeNodeIds)
    // At least one halo should reference sim node 0
    const allIndices = halos.flatMap((h) => h.memberIndices)
    expect(allIndices).toContain(0)
  })
})
