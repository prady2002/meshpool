import { create } from 'zustand'
import type { EventMessage } from '../api/types'

interface AppState {
  mockMode: boolean
  sseConnected: boolean
  paletteOpen: boolean
  recentEvents: EventMessage[]
  setMockMode: (value: boolean) => void
  setSseConnected: (value: boolean) => void
  setPaletteOpen: (value: boolean) => void
  pushEvent: (event: EventMessage) => void
}

export const useAppStore = create<AppState>((set) => ({
  mockMode: import.meta.env.VITE_MOCK === 'true',
  sseConnected: false,
  paletteOpen: false,
  recentEvents: [],
  setMockMode: (value) => set({ mockMode: value }),
  setSseConnected: (value) => set({ sseConnected: value }),
  setPaletteOpen: (value) => set({ paletteOpen: value }),
  pushEvent: (event) => set((state) => ({ recentEvents: [event, ...state.recentEvents].slice(0, 20) })),
}))
