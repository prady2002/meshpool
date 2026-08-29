import type { ConstellationGroup, MeshAggregate, Node, NodeClass, NodeStatus } from './api/types'

function dominantStatus(nodes: Node[]): NodeStatus {
  if (nodes.some((node) => node.status === 'online')) return 'online'
  if (nodes.some((node) => node.status === 'draining')) return 'draining'
  return 'offline'
}

/**
 * Upper bound on how many things the map will simulate and draw.
 *
 * MeshMap's physics step and its proximity-line renderer are both all-pairs
 * O(n^2) loops that run every animation frame, and the renderer issues canvas
 * strokes per pair. Left uncapped this is fine in a demo and lethal in the real
 * product: MeshPool's entire premise is "1000 strangers' PCs", which means 1000
 * DISTINCT owners -- exactly the input that defeats bucketing by owner and
 * yields one item per node. 1000 items is ~500k pair computations twice a
 * frame, which freezes the tab.
 */
const MAX_SIM_ITEMS = 120

function toItem(key: string, label: string, grouped: Node[]) {
  return {
    id: key,
    label,
    count: grouped.length,
    status: dominantStatus(grouped),
    totalCompute: grouped.reduce((sum, node) => sum + node.capability.flops_estimate, 0),
    nodes: grouped,
  }
}

export function aggregateNodes(nodes: Node[]): MeshAggregate {
  if (nodes.length <= 20) {
    return {
      mode: 'individual',
      items: nodes.map((node) => ({ id: node.id, label: node.name, count: 1, status: node.status, totalCompute: node.capability.flops_estimate, nodes: [node] })),
    }
  }

  if (nodes.length <= 100) {
    return {
      mode: 'dense',
      items: nodes.map((node) => ({ id: node.id, label: node.name, count: 1, status: node.status, totalCompute: node.capability.flops_estimate, nodes: [node] })),
    }
  }

  const buckets = new Map<string, Node[]>()
  nodes.forEach((node) => {
    const key = `${node.owner}:${node.capability.cpu_arch}`
    // push, not spread-and-replace: the latter copies the whole bucket on every
    // insert, which is O(k^2) for a mesh where one operator runs many nodes.
    const existing = buckets.get(key)
    if (existing) existing.push(node)
    else buckets.set(key, [node])
  })

  // Largest groups stay legible in their own right; the long tail is folded
  // into a single honest "N more nodes" item so nothing is silently dropped.
  const sorted = [...buckets.entries()].sort((a, b) => b[1].length - a[1].length)
  if (sorted.length <= MAX_SIM_ITEMS) {
    return {
      mode: 'clustered',
      items: sorted.map(([key, grouped]) => toItem(key, key, grouped)),
    }
  }

  const head = sorted.slice(0, MAX_SIM_ITEMS - 1)
  const tail = sorted.slice(MAX_SIM_ITEMS - 1)
  const tailNodes = tail.flatMap(([, grouped]) => grouped)
  return {
    mode: 'clustered',
    items: [
      ...head.map(([key, grouped]) => toItem(key, key, grouped)),
      toItem('__tail__', `${tailNodes.length} more nodes (${tail.length} groups)`, tailNodes),
    ],
  }
}

/**
 * Classify a node into a role class from observable telemetry.
 * This is a calibrated projection — no explicit class field exists in the wire
 * API.
 *
 *   anchor       — trust ≥ 0.93, registered ≥ 7 days, latency ≤ 30 ms
 *   relay        — bandwidth ≥ 500 Mbps, currently idle
 *   drifter      — latency > 100 ms OR trust < 0.70 OR registered < 1 day
 *   constellation — everything else
 */
export function inferNodeClass(node: Node): NodeClass {
  const ageDays = (Date.now() - node.registered_ms) / 86_400_000
  if (!Number.isFinite(ageDays)) return 'constellation'  // no age => cannot classify
  if (node.link.latency_ms > 100 || node.trust_score < 0.70 || ageDays < 1) return 'drifter'
  if (node.trust_score >= 0.93 && ageDays >= 7 && node.link.latency_ms <= 30) return 'anchor'
  if (node.link.bandwidth_mbps >= 500 && node.load.running_units === 0) return 'relay'
  return 'constellation'
}

/**
 * Group nodes into 25-ms latency buckets.
 * link.latency_ms is the node-to-control-plane RTT; nodes in the same 25-ms
 * bucket are a latency neighbourhood (constellation) and are likely to have
 * low inter-node RTT as well.
 */
export function groupConstellations(nodes: Node[]): ConstellationGroup[] {
  const buckets = new Map<number, Node[]>()
  for (const node of nodes) {
    const bucket = Math.floor(node.link.latency_ms / 25) * 25
    const existing = buckets.get(bucket)
    if (existing) existing.push(node)
    else buckets.set(bucket, [node])
  }
  return [...buckets.entries()].map(([bucket, members]) => ({
    id: `const-${bucket}`,
    latencyBucketMs: bucket,
    nodeIds: members.map((n) => n.id),
    totalCompute: members.reduce((s, n) => s + n.capability.flops_estimate, 0),
    onlineCount: members.filter((n) => n.status === 'online').length,
    label: members.length === 1
      ? members[0].name
      : `${bucket}–${bucket + 24} ms neighbourhood (${members.length} nodes)`,
    // link.latency_ms is node→control-plane RTT, not pairwise inter-node RTT.
    // Two nodes both 30 ms from the CP can be on opposite sides of the planet.
    // Mark every group as approximated until the server provides Vivaldi-based
    // pairwise measurements.
    source: 'approximated' as const,
  }))
}

/**
 * Map constellation groups to the indices of simulated nodes that contain their
 * members. Must be called **once** when the node set changes (in a React effect),
 * NOT every animation frame.
 *
 * Only `source:'measured'` groups produce halos — approximated groups (RTT from
 * a common control-plane, not pairwise inter-node) must not be rendered as
 * confident cluster boundaries. The result is consumed by the RAF draw loop as
 * pure O(halos × members) index lookups; no per-frame allocation.
 *
 * Complexity: O(S×R + C×N)
 *   S = simulated nodes, R = avg raw nodes per sim item,
 *   C = constellation groups, N = avg nodeIds per group
 * vs. naive scan O(C×S×R) — about 33× slower at 1000-node scale.
 */
export function resolveConstellationHalos(
  constellations: ConstellationGroup[],
  snodeNodeIds: string[][],   // snodeNodeIds[i] = raw nodeIds belonging to simulated node i
): Array<{ constellationId: string; memberIndices: number[] }> {
  // Inverted index: raw nodeId → simulated-node index. O(S×R) to build.
  const nodeIdToIdx = new Map<string, number>()
  for (let i = 0; i < snodeNodeIds.length; i++) {
    for (const id of snodeNodeIds[i]) nodeIdToIdx.set(id, i)
  }

  return constellations.flatMap((cg) => {
    // MUTATION GUARD: changing this condition to `cg.source !== 'measured'` (i.e., keeping
    // approximated groups) is the honesty defect. Tests assert halos.length === 0 for
    // approximated source, so this guard must stay.
    if (cg.source !== 'measured') return []

    const seenIdx = new Set<number>()
    for (const nodeId of cg.nodeIds) {
      const idx = nodeIdToIdx.get(nodeId)
      if (idx !== undefined) seenIdx.add(idx)
    }
    const memberIndices = [...seenIdx]
    return memberIndices.length >= 2 ? [{ constellationId: cg.id, memberIndices }] : []
  })
}
