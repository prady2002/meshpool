import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import type { ModelDeployment } from '../api/types'

export function ModelsPage() {
  const [models, setModels] = useState<ModelDeployment[]>([])
  const [available, setAvailable] = useState<string[]>([])
  const [selected, setSelected] = useState('nano-gpt')

  const load = () => {
    apiClient.getModels().then(setModels)
    apiClient.getAvailableModels().then(setAvailable)
  }

  useEffect(load, [])

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Models & Deployments</p>
          <h2 className="mt-2 text-3xl font-semibold">Shard-aware deployments</h2>
        </div>
        <div className="flex gap-3">
          <select value={selected} onChange={(event) => setSelected(event.target.value)} className="rounded-2xl border border-slate-200 bg-transparent px-3 py-3 dark:border-slate-800">{available.map((model) => <option key={model}>{model}</option>)}</select>
          <button type="button" onClick={() => apiClient.deployModel({ model: selected, owner: 'default' }).then(load)} className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white">Deploy model</button>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {models.map((model) => (
          <article key={model.id} className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{model.backend}</p>
                <h3 className="text-xl font-semibold">{model.model}</h3>
              </div>
              <div className="text-right"><p className="text-sm text-slate-400">Tokens/sec</p><p className="text-2xl font-semibold text-emerald-300">{model.tokens_per_sec?.toFixed(1) ?? '—'}</p></div>
            </div>
            <div className="mt-5 flex gap-3 overflow-x-auto pb-2">
              {model.partition.map((slice) => (
                <div key={`${model.id}-${slice.node_id}-${slice.start_layer}`} className="min-w-44 rounded-3xl border border-slate-800 bg-slate-900/70 p-4">
                  <p className="text-sm text-slate-400">{slice.node_id}</p>
                  <p className="mt-2 text-lg font-semibold text-slate-100">Layers {slice.start_layer}–{slice.end_layer}</p>
                  <p className="mt-2 text-xs text-slate-500">{slice.is_first ? 'Ingress' : slice.is_last ? 'Egress' : 'Relay'} stage</p>
                </div>
              ))}
            </div>
            <div className="mt-5 flex gap-3">
              <button type="button" onClick={() => apiClient.deployModel({ model: model.model, owner: model.owner }).then(load)} className="rounded-2xl border border-slate-700 px-4 py-2 text-sm">Redeploy</button>
              <button type="button" onClick={() => apiClient.deleteModel(model.id).then(load)} className="rounded-2xl border border-rose-500/30 px-4 py-2 text-sm text-rose-300">Remove</button>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}
