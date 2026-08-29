import { useCallback, useEffect, useRef, useState } from 'react'
import { apiClient } from '../api/client'
import type { BatchJob, FunctionUpload } from '../api/types'
import { formatBytes } from '../utils'

const STORE_KEY = 'meshpool.functions'

interface StoredFn {
  id: string
  label: string
  size: number
  uploadedAt: number
}

function loadStored(): StoredFn[] {
  try {
    const raw = localStorage.getItem(STORE_KEY)
    return raw ? (JSON.parse(raw) as StoredFn[]) : []
  } catch {
    return []
  }
}

export function FunctionsPage() {
  const [fns, setFns] = useState<StoredFn[]>(loadStored)
  const [selected, setSelected] = useState<string>(() => loadStored()[0]?.id ?? '')
  const [payload, setPayload] = useState('{"x": 7}')
  const [units, setUnits] = useState(6)
  const [job, setJob] = useState<BatchJob | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<FunctionUpload | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const pollRef = useRef<number | undefined>(undefined)

  const remember = useCallback((next: StoredFn[]) => {
    setFns(next)
    localStorage.setItem(STORE_KEY, JSON.stringify(next.slice(0, 25)))
  }, [])

  const onUpload = async (file: File) => {
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const info = await apiClient.uploadFunction(await file.arrayBuffer(), file.name)
      const entry: StoredFn = {
        id: info.function,
        label: info.name || file.name,
        size: info.size_bytes,
        uploadedAt: Date.now(),
      }
      remember([entry, ...fns.filter((f) => f.id !== entry.id)])
      setSelected(entry.id)
      setNotice(info)
    } catch (err) {
      // The pool validates the module at upload time, so the message here is the
      // real reason it was refused — show it verbatim rather than "upload failed".
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  const onRun = async () => {
    setError(null)
    let base: Record<string, unknown>
    try {
      base = JSON.parse(payload) as Record<string, unknown>
    } catch (err) {
      setError(`Payload must be JSON: ${err instanceof Error ? err.message : ''}`)
      return
    }
    setBusy(true)
    try {
      const payloads = Array.from({ length: Math.max(1, units) }, (_, i) => ({ ...base, index: i }))
      const created = await apiClient.submitJob({
        name: selected.slice(0, 40),
        func: selected,
        payloads,
        cpu_pct: 15,
        ram_mb: 256,
        owner: apiClient.getOwner(),
        secret_names: [],
      })
      const id = created.job_id
      setJob(null)
      pollRef.current = window.setInterval(async () => {
        const detail = await apiClient.getJob(id)
        setJob(detail)
        if (String(detail.status).startsWith('completed')) {
          window.clearInterval(pollRef.current)
          setBusy(false)
        }
      }, 900)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submit failed')
      setBusy(false)
    }
  }

  useEffect(() => () => window.clearInterval(pollRef.current), [])

  const nodesUsed = new Set((job?.units ?? []).map((u) => u.assigned_node).filter(Boolean)).size

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-indigo-600 dark:text-indigo-400">Functions</p>
          <h2 className="mt-2 text-3xl font-semibold">Run your own code on the pool</h2>
          <p className="mt-2 max-w-2xl text-sm text-slate-600 dark:text-slate-400">
            Compile anything that targets <code>wasm32-wasip1</code> — Rust, Go, Zig, C.
            Your module reads its JSON payload on stdin and writes JSON to stdout. It runs
            sandboxed on donated machines: no filesystem, no sockets, no host calls, with
            fuel, memory and wall-clock ceilings.
          </p>
        </div>
        <div className="flex gap-3">
          <input
            ref={fileRef}
            type="file"
            accept=".wasm,application/wasm"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void onUpload(f) }}
          />
          <button
            type="button"
            disabled={busy}
            onClick={() => fileRef.current?.click()}
            className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white disabled:opacity-50"
          >
            Upload .wasm
          </button>
        </div>
      </div>

      {error ? (
        <div role="alert" className="rounded-[28px] border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      ) : null}

      {notice ? (
        <div className="rounded-[28px] border border-emerald-500/40 bg-emerald-500/10 p-5 text-sm">
          <p className="font-semibold text-emerald-700 dark:text-emerald-300">Module accepted</p>
          <p className="mt-2 break-all font-mono text-xs text-slate-700 dark:text-slate-300">{notice.function}</p>
          <dl className="mt-3 grid grid-cols-2 gap-3 text-slate-600 dark:text-slate-400 sm:grid-cols-4">
            <div><dt className="text-slate-600 dark:text-slate-400">Size</dt><dd>{formatBytes(notice.size_bytes)}</dd></div>
            <div><dt className="text-slate-600 dark:text-slate-400">ABI</dt><dd>{notice.abi}</dd></div>
            <div><dt className="text-slate-600 dark:text-slate-400">Exports</dt><dd>{notice.exports.join(', ') || '—'}</dd></div>
            <div><dt className="text-slate-600 dark:text-slate-400">Imports</dt><dd>{notice.imports.length}</dd></div>
          </dl>
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,360px)_minmax(0,1fr)]">
        <div className="space-y-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-lg font-semibold">Your functions</h3>
          {fns.length === 0 ? (
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Nothing uploaded yet. A module is identified by the SHA-256 of its bytes, so
              the same file always yields the same name and every node re-verifies it
              before running.
            </p>
          ) : (
            <ul className="space-y-2">
              {fns.map((f) => (
                <li key={f.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(f.id)}
                    className={`w-full rounded-2xl border px-4 py-3 text-left text-sm transition ${selected === f.id ? 'border-indigo-400 bg-indigo-500/10' : 'border-slate-200 dark:border-slate-800'}`}
                  >
                    <span className="block font-medium">{f.label}</span>
                    <span className="block truncate font-mono text-xs text-slate-600 dark:text-slate-400">{f.id}</span>
                    <span className="block text-xs text-slate-600 dark:text-slate-400">{formatBytes(f.size)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="space-y-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-lg font-semibold">Run</h3>
          <label className="block text-sm">
            <span className="text-slate-600 dark:text-slate-400">Payload (JSON)</span>
            <textarea
              value={payload}
              onChange={(e) => setPayload(e.target.value)}
              rows={4}
              spellCheck={false}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-transparent p-3 font-mono text-sm dark:border-slate-800"
            />
          </label>
          <label className="block text-sm">
            <span className="text-slate-600 dark:text-slate-400">Work units (each gets a distinct <code>index</code>)</span>
            <input
              type="number"
              min={1}
              max={256}
              value={units}
              onChange={(e) => setUnits(Number(e.target.value))}
              className="mt-2 w-32 rounded-2xl border border-slate-200 bg-transparent p-3 text-sm dark:border-slate-800"
            />
          </label>
          <button
            type="button"
            disabled={busy || !selected}
            onClick={() => void onRun()}
            className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white disabled:opacity-50"
          >
            {busy ? 'Running on the pool…' : 'Run across the pool'}
          </button>

          {job ? (
            <div className="space-y-3">
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {job.status} · {job.progress?.done ?? 0} done · {job.progress?.failed ?? 0} failed
                {nodesUsed > 0 ? ` · spread over ${nodesUsed} machine${nodesUsed === 1 ? '' : 's'}` : ''}
              </p>
              <div className="max-h-80 overflow-auto rounded-2xl border border-slate-200 dark:border-slate-800">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 bg-slate-100 text-xs uppercase tracking-wide text-slate-600 dark:text-slate-400 dark:bg-slate-900">
                    <tr><th className="px-3 py-2">Unit</th><th className="px-3 py-2">Node</th><th className="px-3 py-2">Result</th></tr>
                  </thead>
                  <tbody>
                    {job.units.map((u) => (
                      <tr key={u.id} className="border-t border-slate-200 dark:border-slate-800">
                        <td className="px-3 py-2 font-mono text-xs">{u.id}</td>
                        <td className="px-3 py-2 font-mono text-xs">{u.assigned_node ?? '—'}</td>
                        <td className="px-3 py-2 font-mono text-xs">
                          {u.result != null ? JSON.stringify(u.result) : <span className="text-rose-600 dark:text-rose-400">{u.error ?? u.status}</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  )
}
