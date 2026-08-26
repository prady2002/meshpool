import * as Dialog from '@radix-ui/react-dialog'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../api/client'
import type { ModelDeployment, Node } from '../api/types'
import { useAppStore } from '../store/appStore'
import { Badge } from './ui'

type CommandKind = 'navigation' | 'action' | 'model' | 'node' | 'logs'

interface CommandItem {
  id: string
  label: string
  detail: string
  keywords: string[]
  scope: string
  kind: CommandKind
  run: () => void
}

const nav = [
  ['Overview', '/overview'],
  ['Inference', '/inference'],
  ['Apps', '/apps'],
  ['Functions', '/functions'],
  ['Store', '/store'],
  ['Mesh', '/mesh'],
  ['Usage', '/usage'],
  ['Settings', '/settings'],
] as const

function fuzzyScore(query: string, item: CommandItem) {
  const haystack = [item.label, item.detail, item.scope, ...item.keywords].join(' ').toLowerCase()
  const q = query.trim().toLowerCase()
  if (!q) return 1
  let cursor = 0
  let score = 0
  for (const char of q) {
    const found = haystack.indexOf(char, cursor)
    if (found < 0) return 0
    score += found === cursor ? 3 : 1
    cursor = found + 1
  }
  if (haystack.includes(q)) score += 10
  return score
}

export function CommandPalette({ nodes }: { nodes: Node[] }) {
  const navigate = useNavigate()
  const paletteOpen = useAppStore((state) => state.paletteOpen)
  const setPaletteOpen = useAppStore((state) => state.setPaletteOpen)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const [models, setModels] = useState<ModelDeployment[]>([])
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setPaletteOpen(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setPaletteOpen])

  useEffect(() => {
    if (!paletteOpen) return
    setQuery('')
    setActive(0)
    void apiClient.getModels().then(setModels).catch(() => setModels([]))
    window.setTimeout(() => inputRef.current?.focus(), 0)
  }, [paletteOpen])

  const commands = useMemo<CommandItem[]>(() => {
    const go = (to: string) => () => { navigate(to); setPaletteOpen(false) }
    return [
      ...nav.map(([label, to]): CommandItem => ({ id: `nav-${label}`, label: `Open ${label}`, detail: to, keywords: [label, to], scope: 'Navigate', kind: 'navigation', run: go(to) })),
      { id: 'deploy-model', label: 'Deploy a model', detail: 'Opens the model deployment surface backed by /api/v1/models', keywords: ['deploy', 'model', 'inference'], scope: 'Actions', kind: 'action', run: go('/models') },
      { id: 'run-inference', label: 'Run inference prompt', detail: 'Focus the real chat playground; requires a deployed model', keywords: ['chat', 'prompt', 'llm'], scope: 'Actions', kind: 'action', run: go('/inference') },
      { id: 'upload-function', label: 'Upload a WASM function', detail: 'Uses POST /api/v1/functions; no fake templates', keywords: ['wasm', 'function', 'deploy'], scope: 'Actions', kind: 'action', run: go('/functions') },
      { id: 'open-logs', label: 'Open job logs and trace timeline', detail: 'Shows submitted jobs from /api/v1/jobs', keywords: ['logs', 'trace', 'timeline', 'jobs'], scope: 'Logs', kind: 'logs', run: go('/apps') },
      ...models.map((model): CommandItem => ({ id: `model-${model.id}`, label: `Switch model: ${model.model}`, detail: `${model.status} · ${model.partition.length} pipeline stage${model.partition.length === 1 ? '' : 's'}`, keywords: [model.id, model.model, model.backend], scope: 'Models', kind: 'model', run: go('/inference') })),
      ...nodes.slice(0, 50).map((node): CommandItem => ({ id: `node-${node.id}`, label: `Jump to node: ${node.name}`, detail: `${node.status} · ${node.owner} · ${node.capability.cpu_cores} cores`, keywords: [node.id, node.name, node.owner, node.status], scope: 'Nodes', kind: 'node', run: go('/mesh') })),
    ]
  }, [models, navigate, nodes, setPaletteOpen])

  const results = useMemo(() => commands
    .map((item) => ({ item, score: fuzzyScore(query, item) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.item.label.localeCompare(b.item.label))
    .slice(0, 12)
    .map(({ item }) => item), [commands, query])

  useEffect(() => { setActive(0) }, [query])

  const execute = (item = results[active]) => item?.run()

  return (
    <Dialog.Root open={paletteOpen} onOpenChange={setPaletteOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-950/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-[14%] z-50 w-[min(760px,92vw)] -translate-x-1/2 rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-panel)] p-4 text-[var(--color-text)] shadow-[var(--shadow-elevated)] outline-none" aria-describedby="command-palette-description">
          <Dialog.Title className="text-lg font-semibold">Command palette</Dialog.Title>
          <Dialog.Description id="command-palette-description" className="mt-1 text-sm text-[var(--color-muted)]">Search navigation, actions, deployed models, nodes, and logs. Use arrow keys and Enter.</Dialog.Description>
          <div className="sr-only" aria-live="polite">{results.length} command{results.length === 1 ? '' : 's'} available. {results[active]?.label ?? 'No command selected'}.</div>
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowDown') { event.preventDefault(); setActive((value) => Math.min(results.length - 1, value + 1)) }
              if (event.key === 'ArrowUp') { event.preventDefault(); setActive((value) => Math.max(0, value - 1)) }
              if (event.key === 'Enter') { event.preventDefault(); execute() }
              if (event.key === 'Escape') setPaletteOpen(false)
            }}
            role="combobox"
            aria-expanded="true"
            aria-controls="command-results"
            aria-activedescendant={results[active]?.id}
            placeholder="Type a command, model, node, or scope like logs…"
            className="mt-4 w-full rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-[var(--color-text)] outline-none ring-0 focus-visible:ring-2 focus-visible:ring-[var(--color-focus)]"
          />
          <ul id="command-results" role="listbox" className="mt-4 max-h-80 space-y-2 overflow-y-auto">
            {results.map((result, index) => (
              <li key={result.id} role="option" aria-selected={active === index} id={result.id}>
                <button type="button" className={`w-full rounded-2xl border px-4 py-3 text-left text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)] motion-reduce:transition-none ${active === index ? 'border-[var(--color-focus)] bg-[var(--color-surface)]' : 'border-transparent hover:border-[var(--color-border)] hover:bg-[var(--color-surface)]'}`} onMouseEnter={() => setActive(index)} onClick={() => execute(result)}>
                  <span className="flex items-center justify-between gap-3"><span className="font-medium">{result.label}</span><Badge>{result.scope}</Badge></span>
                  <span className="mt-1 block text-xs text-[var(--color-muted)]">{result.detail}</span>
                </button>
              </li>
            ))}
            {results.length === 0 ? <li className="rounded-2xl border border-dashed border-[var(--color-border)] px-4 py-6 text-center text-sm text-[var(--color-muted)]">No command found. Unavailable surfaces stay explicit instead of showing fake data.</li> : null}
          </ul>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
