import { defineConfig, devices } from '@playwright/test'

// The console is a client-side-routed SPA served by the Python control plane at
// /console (see meshpool/controlplane/app.py). These tests drive a REAL browser
// against a REAL control plane, because the vitest suite runs in jsdom, where
// HTMLCanvasElement.getContext() is unimplemented and fetch/SSE are mocked --
// so a jsdom test cannot observe whether the app boots, routes, paints, or
// streams at all. Everything asserted here is invisible to that suite.

const PORT = Number(process.env.MESHPOOL_E2E_PORT ?? 8099)

export default defineConfig({
  testDir: './e2e',
  // A flaky end-to-end test is worse than none: it trains people to re-run.
  // Failures here must mean something, so retries stay at zero.
  retries: 0,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'edge',
      // Uses the Microsoft Edge already installed on the machine rather than a
      // separately downloaded Chrome for Testing build. Same Chromium engine,
      // no multi-hundred-MB download in CI, and it is a browser real users
      // actually run. Override with MESHPOOL_E2E_CHANNEL to test another.
      use: { ...devices['Desktop Edge'], channel: process.env.MESHPOOL_E2E_CHANNEL ?? 'msedge' },
    },
  ],
  webServer: {
    // Boots the actual FastAPI control plane, not a static file server, so the
    // API calls the console makes are answered by the real thing.
    command: `python -m uvicorn meshpool.controlplane.app:app --host 127.0.0.1 --port ${PORT}`,
    cwd: '..',
    url: `http://127.0.0.1:${PORT}/api/v1/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
})
