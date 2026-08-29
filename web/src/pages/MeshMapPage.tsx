import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import { MeshMap } from '../components/MeshMap'
import { Skeleton } from '../components/Skeleton'
import { useCluster } from '../hooks/useCluster'
import type { BatchJobSummary, ModelDeployment } from '../api/types'

export function MeshMapPage() {
  const { cluster, loading, error } = useCluster()
  const [jobs, setJobs]     = useState<BatchJobSummary[]>([])
  const [models, setModels] = useState<ModelDeployment[]>([])

  useEffect(() => {
    const load = () => {
      apiClient.getJobs().then(setJobs).catch(() => {})
      apiClient.getModels().then(setModels).catch(() => {})
    }
    load()
    const timer = window.setInterval(load, 10000)
    return () => window.clearInterval(timer)
  }, [])

  return (
    <section className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.28em] text-indigo-600 dark:text-indigo-400">Live Mesh Map</p>
        <h2 className="mt-2 text-3xl font-semibold">Topology, health, and cluster scale</h2>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Topology view — node positions are derived from mesh relationships, not geographic
          coordinates. Aggregates automatically once the mesh grows beyond 100 nodes.
        </p>
      </div>
      {loading ? <Skeleton className="h-[560px] rounded-[28px]" /> : null}
      {error ? (
        <div className="rounded-3xl border border-rose-500/30 bg-rose-500/10 p-4 text-rose-200">
          Control plane unreachable: {error}
        </div>
      ) : null}
      {!loading ? (
        <MeshMap
          nodes={cluster?.nodes ?? []}
          jobs={jobs}
          models={models}
        />
      ) : null}
    </section>
  )
}
