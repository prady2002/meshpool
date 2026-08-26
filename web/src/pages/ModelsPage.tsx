import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import type { ModelDeployment } from '../api/types'
import { Button, Card, EmptyState, Metric, Select, Skeleton } from '../components/ui'

export function ModelsPage() {
  const [models, setModels] = useState<ModelDeployment[]>([])
  const [available, setAvailable] = useState<string[]>([])
  const [selected, setSelected] = useState('nano-gpt')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = () => {
    setLoading(true)
    setError(null)
    Promise.all([apiClient.getModels(), apiClient.getAvailableModels()])
      .then(([nextModels, nextAvailable]) => {
        setModels(nextModels)
        setAvailable(nextAvailable)
        setSelected((current) => nextAvailable.includes(current) ? current : (nextAvailable[0] ?? 'nano-gpt'))
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Unable to load models'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 rounded-[28px] border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Models & Deployments</p>
          <h2 className="mt-2 text-3xl font-semibold">Shard-aware deployments</h2>
          <p className="mt-2 max-w-2xl text-sm text-[var(--color-muted)]">Only deployments returned by the control plane are shown. Performance numbers are labeled by tier.</p>
        </div>
        <div className="flex gap-3">
          <Select value={selected} onChange={(event) => setSelected(event.target.value)} aria-label="Available model">{available.map((model) => <option key={model}>{model}</option>)}</Select>
          <Button type="button" disabled={!selected} onClick={() => apiClient.deployModel({ model: selected, owner: apiClient.getOwner() }).then(load)}>Deploy model</Button>
        </div>
      </div>
      {error ? <div role="alert" className="rounded-3xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">{error}</div> : null}
      {loading ? <div className="grid gap-4 xl:grid-cols-2"><Skeleton className="h-64" /><Skeleton className="h-64" /></div> : null}
      {!loading && models.length === 0 ? <EmptyState title="No model deployments yet" body="Deploy one of the models the API reports as available. Until then the inference playground shows an explicit unavailable state instead of fake chat capacity." /> : null}
      <div className="grid gap-4 xl:grid-cols-2">
        {models.map((model) => (
          <Card key={model.id}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-[var(--color-muted)]">{model.backend}</p>
                <h3 className="text-xl font-semibold">{model.model}</h3>
              </div>
              <Metric label="Tokens/sec" value={model.tokens_per_sec != null ? model.tokens_per_sec.toFixed(1) : '—'} tier="measured" />
            </div>
            <div className="mt-5 flex gap-3 overflow-x-auto pb-2">
              {model.partition.map((slice) => (
                <div key={`${model.id}-${slice.node_id}-${slice.start_layer}`} className="min-w-44 rounded-3xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                  <p className="text-sm text-[var(--color-muted)]">{slice.node_id}</p>
                  <p className="mt-2 text-lg font-semibold">Layers {slice.start_layer}–{slice.end_layer}</p>
                  <p className="mt-2 text-xs text-[var(--color-muted)]">{slice.is_first ? 'Ingress' : slice.is_last ? 'Egress' : 'Relay'} stage</p>
                </div>
              ))}
            </div>
            <div className="mt-5 flex gap-3">
              <Button type="button" className="bg-transparent text-[var(--color-text)] ring-1 ring-[var(--color-border)] hover:bg-[var(--color-surface)]" onClick={() => apiClient.deployModel({ model: model.model, owner: model.owner }).then(load)}>Redeploy</Button>
              <Button type="button" className="bg-transparent text-rose-300 ring-1 ring-rose-500/30 hover:bg-rose-500/10" onClick={() => apiClient.deleteModel(model.id).then(load)}>Remove</Button>
            </div>
          </Card>
        ))}
      </div>
    </section>
  )
}
