import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiClient } from '../api/client'
import type { BatchJob } from '../api/types'
import { Metric } from '../components/ui'
import { ProgressBar } from '../components/ProgressBar'

export function JobsPage() {
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const load = () => apiClient.getJobs().then(setJobs)
  // useEffect must return a cleanup function or nothing — never a Promise, or React
  // would treat the promise as the cleanup callback.
  useEffect(() => { void load() }, [])

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Jobs & Batch</p>
          <h2 className="mt-2 text-3xl font-semibold">Submit work and watch progress</h2>
        </div>
        <button type="button" onClick={() => apiClient.submitJob({ name: 'demo-job', func: 'embed', payloads: [{ id: 1 }], cpu_pct: 40, ram_mb: 2048, owner: 'default', secret_names: [] }).then(load)} className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white">Submit job</button>
      </div>
      <div className="space-y-4">
        {jobs.map((job) => {
          const percent = job.progress.total ? Math.round(((job.progress.done + job.progress.failed) / job.progress.total) * 100) : 0
          return (
            <article key={job.id} className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">{job.func}</p>
                  <h3 className="text-xl font-semibold">{job.name}</h3>
                  <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{job.status} · {job.progress.total} total tasks</p>
                </div>
                <Metric label="Complete" value={percent} unit="%" tier="measured" density="compact" />
              </div>
              <div className="mt-5"><ProgressBar value={percent} /></div>
              <div className="mt-5 grid gap-3 md:grid-cols-5">
                {Object.entries(job.progress).map(([key, value]) => <div key={key} className="rounded-2xl border border-slate-800 p-3 text-sm"><p className="text-slate-500">{key}</p><p className="mt-1 text-lg font-semibold text-slate-100">{value}</p></div>)}
              </div>
              <div className="mt-5 rounded-3xl border border-slate-800 p-4">
                <h4 className="text-sm font-semibold text-slate-200">Per-task timeline</h4>
                <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                  {(job.units ?? []).slice(0, 9).map((unit) => <div key={unit.id} className="rounded-2xl bg-slate-900/80 p-3 text-sm text-slate-300">{unit.id} · {unit.status}{unit.error ? ` · ${unit.error}` : ''}</div>)}
                  {!(job.units ?? []).length ? <div className="rounded-2xl bg-slate-900/80 p-3 text-sm text-slate-400">Open the trace waterfall for assignment, retry, and node timing detail.</div> : null}
                </div>
                <Link to={`/jobs/${job.id}/timeline`} className="mt-4 inline-flex rounded-2xl border border-indigo-400/40 px-4 py-2 text-sm font-semibold text-indigo-300 hover:bg-indigo-500/10">
                  View trace timeline
                </Link>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
