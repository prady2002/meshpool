import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { apiClient } from '../api/client'
import type { JobTimeline, TimelineEvent } from '../api/types'
import { Metric } from '../components/ui'

function fmtTime(ms: number | null) {
  if (ms == null) return 'running'
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function fmtDuration(ms: number | null) {
  if (ms == null) return 'in flight'
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

function eventTone(event: TimelineEvent) {
  if (event.outcome === 'success' || event.outcome === 'completed') return 'bg-emerald-400'
  if (event.outcome === 'retry' || event.outcome === 'failed' || event.error) return 'bg-rose-400'
  if (event.type === 'event') return 'bg-sky-400'
  return 'bg-indigo-400'
}

export function JobTimelinePage() {
  const { jobId = '' } = useParams()
  const [timeline, setTimeline] = useState<JobTimeline | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    apiClient.getJobTimeline(jobId)
      .then((data) => { if (!cancelled) setTimeline(data) })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)) })
    return () => { cancelled = true }
  }, [jobId])

  const bounds = useMemo(() => {
    const start = timeline?.start_ms ?? Date.now()
    const end = Math.max(timeline?.end_ms ?? Date.now(), start + 1)
    return { start, span: end - start }
  }, [timeline])

  if (error) {
    return <section className="rounded-[28px] border border-rose-400/40 bg-rose-500/10 p-6 text-rose-100">Unable to load timeline: {error}</section>
  }

  if (!timeline) {
    return <section className="rounded-[28px] border border-slate-800 bg-slate-950/75 p-6 text-slate-400">Loading job timeline…</section>
  }

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
        <div>
          <Link to="/jobs" className="text-sm font-medium text-indigo-400">← Back to jobs</Link>
          <p className="mt-3 text-sm uppercase tracking-[0.28em] text-indigo-400">Distributed trace</p>
          <h2 className="mt-2 text-3xl font-semibold">Job timeline</h2>
          <p className="mt-2 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
            {timeline.job_id} · trace {timeline.trace_id.slice(0, 12)}… · {timeline.status} · {fmtDuration(timeline.duration_ms)}
          </p>
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          <Metric label="Duration" value={fmtDuration(timeline.duration_ms)} tier="measured" density="compact" />
          <Metric label="Events" value={timeline.events.length} tier="measured" density="compact" />
          <Metric label="Lanes" value={timeline.lanes.length} tier="measured" density="compact" />
        </div>
      </div>

      <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
        <div className="mb-4 flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-500">
          <span>{fmtTime(timeline.start_ms)}</span>
          <span>{fmtDuration(timeline.duration_ms)}</span>
          <span>{fmtTime(timeline.end_ms)}</span>
        </div>
        <div className="space-y-4">
          {timeline.lanes.map((lane) => (
            <div key={lane.node_id} className="grid gap-3 md:grid-cols-[220px_minmax(0,1fr)]">
              <div className="rounded-2xl border border-slate-200 p-3 dark:border-slate-800">
                <p className="text-sm font-semibold">{lane.node_name}</p>
                <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">{lane.node_id}</p>
              </div>
              <div className="relative min-h-20 rounded-2xl border border-slate-200 bg-slate-100/70 p-3 dark:border-slate-800 dark:bg-slate-900/70">
                <div className="absolute left-3 right-3 top-1/2 h-px bg-slate-300 dark:bg-slate-700" />
                {lane.spans.map((event, index) => {
                  const left = ((event.start_ms - bounds.start) / bounds.span) * 100
                  const width = Math.max(2, ((event.duration_ms ?? 50) / bounds.span) * 100)
                  return (
                    <div
                      key={`${event.name}-${event.unit_id ?? 'job'}-${event.attempt ?? index}-${event.start_ms}`}
                      className={`absolute top-5 h-10 rounded-xl px-3 py-2 text-xs font-semibold text-slate-950 shadow-lg ${eventTone(event)}`}
                      style={{ left: `${Math.min(98, Math.max(0, left))}%`, width: `${Math.min(100 - Math.max(0, left), width)}%` }}
                      title={`${event.name} ${event.unit_id ?? ''} ${event.outcome ?? ''} ${event.error ?? ''}`}
                    >
                      <span className="block truncate">{event.unit_id ? `${event.unit_id.slice(0, 8)} #${event.attempt}` : event.name}</span>
                      <span className="block truncate font-normal">{event.outcome ?? fmtDuration(event.duration_ms)}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
        <article className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-lg font-semibold">Ordered events</h3>
          <div className="mt-4 space-y-3">
            {timeline.events.map((event, index) => (
              <div key={`${event.name}-${index}`} className="rounded-2xl border border-slate-200 p-4 text-sm dark:border-slate-800">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="font-semibold">{event.name}{event.unit_id ? ` · ${event.unit_id}` : ''}</p>
                  <p className="text-slate-500 dark:text-slate-400">{fmtTime(event.start_ms)} · {fmtDuration(event.duration_ms)}</p>
                </div>
                <p className="mt-2 text-slate-500 dark:text-slate-400">
                  node {event.node_id ?? 'control-plane'} · attempt {event.attempt ?? '—'} · outcome {event.outcome ?? 'observed'}
                </p>
                {event.error ? <p className="mt-2 rounded-xl bg-rose-500/10 p-2 text-rose-300">{event.error}</p> : null}
              </div>
            ))}
          </div>
        </article>
        <aside className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-lg font-semibold">Unit outcomes</h3>
          <div className="mt-4 space-y-2">
            {timeline.units.map((unit) => (
              <div key={unit.unit_id} className="rounded-2xl bg-slate-100 p-3 text-sm dark:bg-slate-900">
                <p className="font-semibold">{unit.unit_id}</p>
                <p className="mt-1 text-slate-500 dark:text-slate-400">{unit.outcome} · attempts {unit.attempts} · {unit.node_id ?? 'unassigned'}</p>
                {unit.error ? <p className="mt-1 text-rose-300">{unit.error}</p> : null}
              </div>
            ))}
          </div>
        </aside>
      </div>
    </section>
  )
}
