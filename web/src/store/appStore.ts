import { create } from 'zustand'
import type { EventMessage } from '../api/types'

interface AppState {
  darkMode: boolean
  mockMode: boolean
  sseConnected: boolean
  paletteOpen: boolean
  recentEvents: EventMessage[]
  toggleDarkMode: () => void
  setMockMode: (value: boolean) => void
  setSseConnected: (value: boolean) => void
  setPaletteOpen: (value: boolean) => void
  pushEvent: (event: EventMessage) => void
}

const darkDefault = localStorage.getItem('meshpool.theme') !== 'light'

document.documentElement.classList.toggle('dark', darkDefault)

export const useAppStore = create<AppState>((set) => ({
  darkMode: darkDefault,
  mockMode: import.meta.env.VITE_MOCK === 'true',
  sseConnected: false,
  paletteOpen: false,
  recentEvents: [],
  toggleDarkMode: () => set((state) => {
    const next = !state.darkMode
    localStorage.setItem('meshpool.theme', next ? 'dark' : 'light')
    document.documentElement.classList.toggle('dark', next)
    return { darkMode: next }
  }),
  setMockMode: (value) => set({ mockMode: value }),
  setSseConnected: (value) => set({ sseConnected: value }),
  setPaletteOpen: (value) => set({ paletteOpen: value }),
  pushEvent: (event) => set((state) => ({ recentEvents: [event, ...state.recentEvents].slice(0, 20) })),
}))
