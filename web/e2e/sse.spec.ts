import { test, expect } from '@playwright/test'

// ---------------------------------------------------------------------------
// The control plane streams live updates over SSE (GET /api/v1/events). A client
// that reconnects in a tight loop still *works* -- every screen renders, every
// jsdom test passes, and nothing errors -- while silently generating one new
// connection per cycle. On a single dev machine that is invisible. Against a
// control plane serving many browsers it is a self-inflicted denial of service.
//
// This is only observable by counting connections over time in a real browser.
// No component test can see it, because the reconnect is driven by the browser's
// own EventSource retry behaviour, which jsdom does not implement.
// ---------------------------------------------------------------------------

test('the console does not reconnect to the event stream in a tight loop', async ({ page }) => {
  const eventConnections: number[] = []
  const started = Date.now()

  page.on('request', (r) => {
    if (new URL(r.url()).pathname === '/api/v1/events') {
      eventConnections.push(Date.now() - started)
    }
  })

  await page.goto('/console/mesh', { waitUntil: 'domcontentloaded' })

  // Watch a fixed, generous window. Long enough that a healthy client has
  // settled into one long-lived stream, and any retry loop has had time to show.
  const WINDOW_MS = 10_000
  await page.waitForTimeout(WINDOW_MS)

  // Non-vacuity: if the console never opened a stream at all, a low count would
  // be trivially satisfied and this test would prove nothing.
  expect(
    eventConnections.length,
    'the console never opened an SSE connection, so this measures nothing',
  ).toBeGreaterThan(0)

  // A healthy client opens roughly one stream (plus a small allowance for React
  // strict-mode double-mounting and at most a couple of legitimate retries).
  // A retry loop produces tens to hundreds.
  const perSecond = eventConnections.length / (WINDOW_MS / 1000)
  expect(
    eventConnections.length,
    `opened ${eventConnections.length} SSE connections in ${WINDOW_MS / 1000}s ` +
      `(${perSecond.toFixed(1)}/s) at offsets ${JSON.stringify(eventConnections)} -- ` +
      'this is a reconnect loop, and it scales with the number of connected browsers',
  ).toBeLessThanOrEqual(8)
})
