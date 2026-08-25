import { defineConfig, type UserConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // Served by the control plane under /console (see meshpool/controlplane/app.py),
  // so built asset URLs must be prefixed to match.
  base: '/console/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/v1': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  // Vitest reads this block. It is asserted rather than typed because vitest v3
  // vendors its own older copy of Vite, and importing `vitest/config` here drags
  // that second, incompatible plugin type-tree into the same file as Vite 8's.
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
  },
} as UserConfig)
