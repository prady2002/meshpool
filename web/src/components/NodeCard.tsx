import type { Node } from '../api/types'
import { StatusBadge } from './StatusBadge'
import { Metric } from './ui'

export function NodeCard({ node, active }: { node: Node; active?: boolean }) {
  return (
    <article className={`rounded-3xl border border-slate-800 bg-slate-950/70 p-4 shadow-lg shadow-slate-950/40 transition ${active ? 'ring-2 ring-indigo-500/60' : ''}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm text-slate-400">{node.owner}</p>
          <h3 className="text-base font-semibold text-slate-100">{node.name}</h3>
        </div>
        <StatusBadge status={node.status} />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3">
        <Metric label="CPU" value={node.capability.cpu_cores} unit="cores" tier="measured" density="compact" />
        <Metric label="RAM" value={(node.capability.ram_mb / 1024).toFixed(1)} unit="GB" tier="measured" density="compact" />
        <Metric label="Load" value={node.load.cpu_load_pct} unit="%" tier="measured" density="compact" />
        <Metric label="Trust" value={Math.round(node.trust_score * 100)} unit="%" tier="measured" density="compact" />
      </div>
    </article>
  )
}
