import type { Node } from '../api/types'
import { formatNumber } from '../utils'
import { StatusBadge } from './StatusBadge'

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
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm text-slate-300">
        <div><dt className="text-slate-500">CPU</dt><dd>{node.capability.cpu_cores} cores</dd></div>
        <div><dt className="text-slate-500">RAM</dt><dd>{formatNumber(node.capability.ram_mb / 1024, 1)} GB</dd></div>
        <div><dt className="text-slate-500">Load</dt><dd>{node.load.cpu_load_pct}%</dd></div>
        <div><dt className="text-slate-500">Trust</dt><dd>{Math.round(node.trust_score * 100)}%</dd></div>
      </dl>
    </article>
  )
}
