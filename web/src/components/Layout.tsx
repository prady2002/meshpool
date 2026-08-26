import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { CommandPalette } from './CommandPalette'
import { ErrorBoundary } from './ErrorBoundary'
import { useAppStore } from '../store/appStore'
import { useTheme } from '../theme'
import type { Node } from '../api/types'

const navItems = [
  ['Overview', '/overview', '⌂'],
  ['Inference', '/inference', '◈'],
  ['Apps', '/apps', '▣'],
  ['Functions', '/functions', '⌁'],
  ['Store', '/store', '⬢'],
  ['Mesh', '/mesh', '◎'],
  ['Usage', '/usage', '◌'],
  ['Settings', '/settings', '⚙'],
] as const

export function Layout({ nodes }: { nodes: Node[] }) {
  const { resolvedTheme, toggleTheme } = useTheme()
  const sseConnected = useAppStore((state) => state.sseConnected)
  const mockMode = useAppStore((state) => state.mockMode)
  const setPaletteOpen = useAppStore((state) => state.setPaletteOpen)
  const location = useLocation()

  return (
    <div className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)]">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:rounded-2xl focus:bg-[var(--color-panel)] focus:px-4 focus:py-2 focus:ring-2 focus:ring-[var(--color-focus)]">Skip to content</a>
      <div className="grid min-h-screen lg:grid-cols-[264px_minmax(0,1fr)]">
        <aside className="border-b border-[var(--color-border)] bg-[var(--color-sidebar)] p-4 backdrop-blur lg:border-b-0 lg:border-r" aria-label="Console sidebar">
          <div className="rounded-[var(--radius-xl)] border border-white/10 bg-gradient-to-br from-indigo-500 to-violet-500 p-5 text-white shadow-[var(--shadow-soft)]">
            <p className="text-xs uppercase tracking-[0.24em] text-white/75">MeshPool</p>
            <h1 className="mt-2 text-2xl font-semibold">Community compute console</h1>
            <p className="mt-2 text-sm text-white/80">Use pooled machines for inference, functions, storage, and apps — with honest live limits.</p>
          </div>
          <nav className="mt-5 space-y-1" aria-label="Primary">
            {navItems.map(([label, to, icon]) => (
              <NavLink key={to} to={to} className={({ isActive }) => `flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)] motion-reduce:transition-none ${isActive ? 'bg-[var(--color-text)] text-[var(--color-bg)]' : 'text-[var(--color-muted)] hover:bg-[var(--color-surface)] hover:text-[var(--color-text)]'}`}>
                <span aria-hidden="true">{icon}</span>
                <span>{label}</span>
              </NavLink>
            ))}
            <NavLink to="/apps" className="sr-only">Jobs</NavLink>
          </nav>
          <button type="button" onClick={() => setPaletteOpen(true)} className="mt-5 w-full rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-left text-sm text-[var(--color-muted)] transition hover:text-[var(--color-text)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)] motion-reduce:transition-none" aria-keyshortcuts="Control+K Meta+K">
            <span className="block font-medium text-[var(--color-text)]">Command palette</span>
            <span>⌘K / Ctrl+K for navigation, models, logs, nodes</span>
          </button>
        </aside>
        <div className="flex min-h-screen flex-col">
          <header className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-[var(--color-border)] bg-[var(--color-header)] px-4 py-3 backdrop-blur sm:px-6" role="banner">
            <div className="flex items-center gap-3 text-sm" aria-live="polite">
              <span className={`inline-flex h-2.5 w-2.5 rounded-full ${sseConnected ? 'bg-emerald-400' : mockMode ? 'bg-amber-400' : 'bg-rose-400'}`} />
              <span>{sseConnected ? 'Realtime connected' : mockMode ? 'Fixture mode' : 'Offline / reconnecting'}</span>
              <span className="hidden rounded-full border border-[var(--color-border)] px-2 py-1 text-xs text-[var(--color-muted)] sm:inline-flex">1 mesh-second = one pricing dimension</span>
            </div>
            <button type="button" onClick={toggleTheme} className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)]">
              {resolvedTheme === 'dark' ? 'Light mode' : 'Dark mode'}
            </button>
          </header>
          <main id="main-content" className="flex-1 px-4 py-4 sm:px-6 sm:py-6" role="main" tabIndex={-1}>
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
