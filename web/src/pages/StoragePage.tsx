import { useEffect, useRef, useState } from 'react'
import { apiClient } from '../api/client'
import type { BlobMeta } from '../api/types'
import { formatBytes } from '../utils'

export function StoragePage() {
  const [blobs, setBlobs] = useState<BlobMeta[]>([])
  const inputRef = useRef<HTMLInputElement | null>(null)
  const load = () => apiClient.getBlobs().then(setBlobs)
  useEffect(() => { void load() }, [])

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Storage</p>
          <h2 className="mt-2 text-3xl font-semibold">Objects, replicas, shard locations</h2>
        </div>
        <div className="flex gap-3">
          <input ref={inputRef} type="file" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) apiClient.uploadBlob(file).then(load) }} />
          <button type="button" onClick={() => inputRef.current?.click()} className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white">Upload blob</button>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {blobs.map((blob) => (
          <article key={blob.id} className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{blob.id}</p>
                <h3 className="text-xl font-semibold">{blob.name}</h3>
              </div>
              <button type="button" onClick={() => apiClient.deleteBlob(blob.id).then(load)} className="rounded-2xl border border-rose-500/30 px-4 py-2 text-sm text-rose-300">Delete</button>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm text-slate-300">
              <div><dt className="text-slate-500">Size</dt><dd>{formatBytes(blob.size)}</dd></div>
              <div><dt className="text-slate-500">State</dt><dd>{blob.status}</dd></div>
              <div><dt className="text-slate-500">Replicas</dt><dd>{blob.replicas.length}/{blob.replication_desired}</dd></div>
              <div><dt className="text-slate-500">Owner</dt><dd>{blob.owner}</dd></div>
            </dl>
            <div className="mt-4 rounded-3xl border border-slate-800 p-4 text-sm text-slate-300">
              Shard locations: {blob.replicas.join(', ')}
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}
