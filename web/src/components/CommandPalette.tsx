import * as Dialog from '@radix-ui/react-dialog'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Node } from '../api/types'
import { useAppStore } from '../store/appStore'

const actions = [
  { id: 'route-playground', label: 'Go to Playground', run: '/playground' },
  { id: 'route-mesh', label: 'Go to Mesh Map', run: '/mesh' },
  { id: 'route-models', label: 'Go to Models', run: '/models' },
  { id: 'route-jobs', label: 'Go to Jobs', run: '/jobs' },
  { id: 'route-storage', label: 'Go to Storage', run: '/storage' },
  { id: 'route-contribute', label: 'Go to Contributor View', run: '/contribute' },
  { id: 'route-account', label: 'Go to Account', run: '/account' },
  { id: 'action-deploy', label: 'Quick action: Deploy model', run: '/models' },
  { id: 'action-job', label: 'Quick action: Submit job', run: '/jobs' },
  { id: 'action-upload', label: 'Quick action: Upload blob', run: '/storage' },
]

export function CommandPalette({ nodes }: { nodes: Node[] }) {
  const navigate = useNavigate()
  const paletteOpen = useAppStore((state) => state.paletteOpen)
  const setPaletteOpen = useAppStore((state) => state.setPaletteOpen)
  const [query, setQuery] = useState('')

  const results = useMemo(() => {
    const lowered = query.trim().toLowerCase()
    const actionMatches = actions.filter((item) => item.label.toLowerCase().includes(lowered))
    const nodeMatches = nodes
      .filter((node) => node.name.toLowerCase().includes(lowered) || node.id.toLowerCase().includes(lowered))
      .slice(0, 5)
      .map((node) => ({ id: node.id, label: `Find node: ${node.name}`, run: '/mesh' }))
    return lowered ? [...actionMatches, ...nodeMatches] : [...actions, ...nodeMatches]
  }, [nodes, query])

  return (
    <Dialog.Root open={paletteOpen} onOpenChange={setPaletteOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-slate-950/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-[18%] w-[min(720px,92vw)] -translate-x-1/2 rounded-3xl border border-slate-800 bg-slate-950 p-4 shadow-2xl outline-none">
          <Dialog.Title className="text-lg font-semibold text-slate-100">Command palette</Dialog.Title>
          <input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Jump to routes, actions, or nodes…" className="mt-4 w-full rounded-2xl border border-slate-700 bg-slate-900 px-4 py-3 text-slate-100 outline-none ring-0" />
          <ul className="mt-4 max-h-80 space-y-2 overflow-y-auto">
            {results.map((result) => (
              <li key={result.id}>
                <button type="button" className="w-full rounded-2xl border border-transparent bg-slate-900 px-4 py-3 text-left text-sm text-slate-200 hover:border-slate-700" onClick={() => { navigate(result.run); setPaletteOpen(false) }}>
                  {result.label}
                </button>
              </li>
            ))}
          </ul>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
