/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type ThemeMode = 'light' | 'dark' | 'system'
const THEME_KEY = 'meshpool.theme'

interface ThemeContextValue {
  theme: ThemeMode
  resolvedTheme: 'light' | 'dark'
  setTheme: (theme: ThemeMode) => void
  toggleTheme: () => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

function systemTheme(): 'light' | 'dark' {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function storedTheme(): ThemeMode {
  const raw = localStorage.getItem(THEME_KEY)
  return raw === 'light' || raw === 'dark' || raw === 'system' ? raw : 'system'
}

function applyTheme(theme: ThemeMode) {
  const resolved = theme === 'system' ? systemTheme() : theme
  document.documentElement.classList.toggle('dark', resolved === 'dark')
  document.documentElement.dataset.theme = resolved
  document.documentElement.style.colorScheme = resolved
  return resolved
}

export function initializeTheme() {
  if (typeof window === 'undefined') return 'dark'
  return applyTheme(storedTheme())
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeMode>(storedTheme)
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>(() => applyTheme(storedTheme()))

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const sync = () => setResolvedTheme(applyTheme(theme))
    sync()
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [theme])

  const value = useMemo<ThemeContextValue>(() => ({
    theme,
    resolvedTheme,
    setTheme(next) {
      localStorage.setItem(THEME_KEY, next)
      setThemeState(next)
      setResolvedTheme(applyTheme(next))
    },
    toggleTheme() {
      const next = resolvedTheme === 'dark' ? 'light' : 'dark'
      localStorage.setItem(THEME_KEY, next)
      setThemeState(next)
      setResolvedTheme(applyTheme(next))
    },
  }), [resolvedTheme, theme])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const value = useContext(ThemeContext)
  if (!value) {
    const resolved = document.documentElement.classList.contains('dark') ? 'dark' : 'light'
    return {
      theme: resolved,
      resolvedTheme: resolved,
      setTheme: applyTheme,
      toggleTheme: () => applyTheme(resolved === 'dark' ? 'light' : 'dark'),
    } satisfies ThemeContextValue
  }
  return value
}
