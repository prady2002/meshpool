import { cn } from '../utils'
import type { NodeStatus } from '../api/types'

const statusClasses: Record<NodeStatus, string> = {
  online: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30',
  draining: 'bg-amber-500/15 text-amber-300 ring-amber-500/30',
  offline: 'bg-rose-500/15 text-rose-300 ring-rose-500/30',
}

export function StatusBadge({ status }: { status: NodeStatus }) {
  return <span className={cn('inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold capitalize ring-1', statusClasses[status])}>{status}</span>
}
