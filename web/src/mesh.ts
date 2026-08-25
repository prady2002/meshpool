import type { MeshAggregate, Node, NodeStatus } from './api/types'

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
