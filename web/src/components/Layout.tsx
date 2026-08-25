import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { CommandPalette } from './CommandPalette'
import { ErrorBoundary } from './ErrorBoundary'
import { useAppStore } from '../store/appStore'
import type { Node } from '../api/types'

const navItems = [
  ['Playground', '/playground', '◈'],
  ['Mesh Map', '/mesh', '◎'],
  ['Models', '/models', '◫'],
  ['Functions', '/functions', '⌁'],
  ['Jobs', '/jobs', '▣'],
  ['Storage', '/storage', '⬢'],
  ['Contribute', '/contribute', '✦'],
  ['Account', '/account', '◌'],
] as const

export function Layout({ nodes }: { nodes: Node[] }) {
  const darkMode = useAppStore((state) => state.darkMode)
  const toggleDarkMode = useAppStore((state) => state.toggleDarkMode)
  const sseConnected = useAppStore((state) => state.sseConnected)
  const mockMode = useAppStore((state) => state.mockMode)
  const setPaletteOpen = useAppStore((state) => state.setPaletteOpen)
  const location = useLocation()

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="grid min-h-screen lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className="border-b border-slate-200/70 bg-white/90 p-5 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90 lg:border-b-0 lg:border-r">
          <div className="rounded-3xl border border-slate-200 bg-gradient-to-br from-indigo-500 to-violet-500 p-5 text-white dark:border-slate-800">
            <p className="text-xs uppercase tracking-[0.24em] text-white/75">MeshPool</p>
            <h1 className="mt-2 text-2xl font-semibold">Community compute console</h1>
            <p className="mt-2 text-sm text-white/80">Realtime orchestration, contributor earnings, and distributed inference.</p>
          </div>
          <nav className="mt-6 space-y-2" aria-label="Primary">
            {navItems.map(([label, to, icon]) => (
              <NavLink key={to} to={to} className={({ isActive }) => `flex items-center gap-3 rounded-2xl px-4 py-3 text-sm transition ${isActive ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950' : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900'}`}>
                <span aria-hidden="true">{icon}</span>
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
          <button type="button" onClick={() => setPaletteOpen(true)} className="mt-6 w-full rounded-2xl border border-slate-200 px-4 py-3 text-left text-sm text-slate-600 dark:border-slate-800 dark:text-slate-300">
            ⌘ / Ctrl + K command palette
          </button>
        </aside>
        <div className="flex min-h-screen flex-col">
          <header className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-slate-200/70 bg-white/85 px-6 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/85">
            <div className="flex items-center gap-3 text-sm">
              <span className={`inline-flex h-3 w-3 rounded-full ${sseConnected ? 'bg-emerald-400' : mockMode ? 'bg-amber-400' : 'bg-rose-400'}`} />
              <span>{sseConnected ? 'Realtime connected' : mockMode ? 'Fixture mode' : 'Offline / reconnecting'}</span>
              {mockMode ? <span className="rounded-full bg-indigo-500/15 px-2 py-1 text-xs font-medium text-indigo-300">Fixture mode</span> : null}
            </div>
            <button type="button" onClick={toggleDarkMode} className="rounded-2xl border border-slate-200 px-3 py-2 text-sm dark:border-slate-800">
              {darkMode ? 'Light mode' : 'Dark mode'}
            </button>
          </header>
          <main className="flex-1 px-4 py-4 sm:px-6 sm:py-6">
            {/* Scoped to the routed view so a crashing page never takes the
                navigation, header, or command palette down with it. Keyed on
                the pathname so moving to another page clears the error. */}
            <ErrorBoundary resetKey={location.pathname}>
              <Outlet />
            </ErrorBoundary>
          </main>
        </div>
      </div>
      <CommandPalette nodes={nodes} />
    </div>
  )
}
