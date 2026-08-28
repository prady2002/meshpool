import { Component, type ErrorInfo, type ReactNode } from 'react'

/**
 * Keeps one broken page from destroying the whole console.
 *
 * This exists because of a real, shipped failure: `getAvailableModels()` was
 * type-asserted as `string[]` while the API actually returns
 * `{builtin, ollama}`. The Models page called `.map()` on an object, React
 * unmounted the entire tree, and the user got a pure white screen -- on Models
 * AND on every other page, because the crash took the router down with it. The
 * navigation, the header, and the theme toggle all vanished, so there was no
 * way to click out of it.
 *
 * An uncaught render error in one route should cost the user that route, not
 * the application. React only stops tree-wide unmounting if an error boundary
 * is present, and there was none anywhere in the app.
 */
interface Props {
  children: ReactNode
  /** Changing this resets the boundary -- pass the route key so navigation recovers. */
  resetKey?: string
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null })
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep the detail in the console for bug reports; the UI stays calm.
    console.error('MeshPool console: unhandled render error', error, info)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <section
        role="alert"
        className="rounded-[28px] border border-rose-200 bg-rose-50/80 p-6 dark:border-rose-900 dark:bg-rose-950/40"
      >
        <p className="text-sm uppercase tracking-[0.28em] text-rose-500">
          This page hit an error
        </p>
        <h2 className="mt-2 text-2xl font-semibold">
          The rest of the console still works
        </h2>
        <p className="mt-3 max-w-2xl text-sm text-slate-600 dark:text-slate-300">
          Something in this view failed to render. Your mesh, your jobs, and your
          data are unaffected — this is a bug in the console, not in the cluster.
          Use the navigation on the left to continue, or retry this page.
        </p>
        <pre tabIndex={0} className="mt-4 max-w-full overflow-x-auto rounded-2xl bg-slate-900/90 p-4 text-xs text-rose-200">
          {error.message}
        </pre>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="mt-4 rounded-2xl bg-rose-500 px-4 py-2.5 text-sm font-semibold text-white"
        >
          Retry this page
        </button>
      </section>
    )
  }
}
