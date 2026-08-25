import { useEffect, useMemo, useState } from 'react'
import * as Slider from '@radix-ui/react-slider'
import { Link } from 'react-router-dom'
import { apiClient } from '../api/client'
import type { ModelDeployment, Node } from '../api/types'
import { ChatMessage } from '../components/ChatMessage'
import { NodeCard } from '../components/NodeCard'
import { Skeleton } from '../components/Skeleton'
import { useChat } from '../hooks/useChat'

export function Playground() {
  const [models, setModels] = useState<ModelDeployment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('Show how MeshPool would route a customer support summarization job across the mesh.')
  const [temperature, setTemperature] = useState([0.7])
  const [maxTokens, setMaxTokens] = useState([256])
  const [selectedModel, setSelectedModel] = useState('')
  const { messages, streaming, error: chatError, sendMessage } = useChat()

  useEffect(() => {
    apiClient.getModels()
      .then((items) => {
        setModels(items)
        setSelectedModel(items[0]?.id ?? '')
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load models'))
      .finally(() => setLoading(false))
  }, [])

  const selectedDeployment = useMemo(() => models.find((model) => model.id === selectedModel) ?? null, [models, selectedModel])
  const assignedNodes = useMemo<Node[]>(() => {
    if (!selectedDeployment) return []
    const ids = new Set(selectedDeployment.partition.map((slice) => slice.node_id))
    return selectedDeployment.partition.map((slice): Node => ({ id: slice.node_id, name: slice.node_id, owner: selectedDeployment.owner, url: slice.url, status: 'online', capability: { cpu_cores: 0, cpu_arch: 'unknown', ram_mb: 0, gpu: false, vram_mb: 0, disk_mb: 0, flops_estimate: 0, ollama_models: [] }, link: { latency_ms: 0, bandwidth_mbps: 0 }, policy: { cpu_pct: 0, ram_mb: 0, gpu_allowed: false, schedule: 'always', idle_minutes: 0 }, load: { cpu_load_pct: 0, ram_used_mb: 0, running_units: 0, idle_seconds: 0 }, trust_score: 1, last_heartbeat_ms: Date.now(), registered_ms: Date.now() }))
      .filter((node) => ids.has(node.id))
  }, [selectedDeployment])

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_360px]">
      <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-xl shadow-slate-200/40 dark:border-slate-800 dark:bg-slate-950/75 dark:shadow-slate-950/40">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Playground</p>
            <h2 className="mt-1 text-3xl font-semibold">Distributed LLM chat</h2>
            <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">Stream tokens live while seeing which nodes serve each stage.</p>
          </div>
        </div>
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <label className="rounded-3xl border border-slate-200 p-4 dark:border-slate-800">
            <span className="text-sm text-slate-500 dark:text-slate-400">Model</span>
            {loading ? <Skeleton className="mt-3 h-11" /> : (
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} className="mt-3 w-full rounded-2xl border border-slate-200 bg-transparent px-3 py-3 dark:border-slate-700">
                {models.map((model) => <option key={model.id} value={model.id}>{model.model} · {model.status}</option>)}
              </select>
            )}
          </label>
          <div className="rounded-3xl border border-slate-200 p-4 dark:border-slate-800">
            <div className="flex justify-between text-sm text-slate-500 dark:text-slate-400"><span>Temperature</span><span>{temperature[0].toFixed(1)}</span></div>
            <Slider.Root value={temperature} min={0} max={1} step={0.1} onValueChange={setTemperature} className="mt-5 flex h-5 items-center">
              <Slider.Track className="relative h-2 grow rounded-full bg-slate-800"><Slider.Range className="absolute h-full rounded-full bg-indigo-500" /></Slider.Track><Slider.Thumb className="block h-5 w-5 rounded-full bg-white shadow" />
            </Slider.Root>
          </div>
          <div className="rounded-3xl border border-slate-200 p-4 dark:border-slate-800">
            <div className="flex justify-between text-sm text-slate-500 dark:text-slate-400"><span>Max tokens</span><span>{maxTokens[0]}</span></div>
            <Slider.Root value={maxTokens} min={64} max={1024} step={32} onValueChange={setMaxTokens} className="mt-5 flex h-5 items-center">
              <Slider.Track className="relative h-2 grow rounded-full bg-slate-800"><Slider.Range className="absolute h-full rounded-full bg-violet-500" /></Slider.Track><Slider.Thumb className="block h-5 w-5 rounded-full bg-white shadow" />
            </Slider.Root>
          </div>
        </div>
        <div className="mt-6 rounded-3xl border border-slate-200 p-4 dark:border-slate-800">
          <label className="block text-sm text-slate-500 dark:text-slate-400">Prompt</label>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} className="mt-3 w-full rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-700" />
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button type="button" disabled={!selectedModel || streaming} onClick={() => sendMessage(prompt, { modelId: selectedModel, temperature: temperature[0], maxNewTokens: maxTokens[0] })} className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60">
              {streaming ? 'Streaming…' : 'Send to mesh'}
            </button>
            <span className="text-sm text-slate-500 dark:text-slate-400">Honest live stream. No synthetic latency masking.</span>
          </div>
          {error ? <p className="mt-3 text-sm text-rose-400">{error}</p> : null}
          {chatError ? <p className="mt-3 text-sm text-rose-400">{chatError}</p> : null}
          {!loading && models.length === 0 ? (
            <div className="mt-6 rounded-3xl border border-dashed border-slate-700 px-6 py-10 text-center text-slate-400">
              <p className="text-4xl">🛰️</p>
              <p className="mt-3 text-lg font-medium text-slate-200">Deploy a model first</p>
              <Link to="/models" className="mt-3 inline-flex text-indigo-300 underline">Open Models page</Link>
            </div>
          ) : null}
        </div>
        <div className="mt-6 space-y-4">
          {messages.map((message, index) => <ChatMessage key={`${message.role}-${index}`} role={message.role} content={message.content} />)}
        </div>
      </section>
      <aside className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-xl shadow-slate-200/40 dark:border-slate-800 dark:bg-slate-950/75 dark:shadow-slate-950/40">
        <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Computing on</p>
        <h3 className="mt-2 text-2xl font-semibold">Pipeline stages</h3>
        <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">Node assignments per layer slice. Estimates are labeled when the backend does not publish exact metrics.</p>
        <div className="mt-5 space-y-4">
          {selectedDeployment?.partition.map((slice, index) => (
            <div key={`${slice.node_id}-${index}`} className="rounded-3xl border border-slate-200 p-4 dark:border-slate-800">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Stage {index + 1}</p>
                  <h4 className="font-semibold text-slate-100">{slice.node_id}</h4>
                </div>
                <span className="rounded-full bg-indigo-500/15 px-2 py-1 text-xs text-indigo-300">Layers {slice.start_layer}–{slice.end_layer}</span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full w-full animate-pulse rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 motion-reduce:animate-none" /></div>
            </div>
          ))}
          {assignedNodes.map((node) => <NodeCard key={node.id} node={node} active={streaming} />)}
        </div>
      </aside>
    </div>
  )
}
