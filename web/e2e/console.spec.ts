import { test, expect, type Page, type ConsoleMessage } from '@playwright/test'

// ---------------------------------------------------------------------------
// Why this file exists
//
// web/tests/*.test.tsx runs 119 assertions in jsdom. jsdom does not implement
// HTMLCanvasElement.getContext(), does not lay out or paint anything, does not
// execute the real bundle, and does not perform real network I/O. So that suite
// stays green even if the console fails to boot, every route 404s, the Mesh Map
// draws nothing, and no API call ever succeeds. It asserts on component
// internals; it cannot assert that the product works.
//
// These tests drive a real Chromium against the real FastAPI control plane.
// Every assertion below is one the jsdom suite is structurally incapable of
// making.
// ---------------------------------------------------------------------------

/** Records anything the browser reports as broken, for assertion at test end. */
function collectPageFailures(page: Page) {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
  page.on('console', (m: ConsoleMessage) => {
    if (m.type() === 'error') errors.push(`console.error: ${m.text()}`)
  })
  page.on('requestfailed', (r) => {
    // Aborted navigations during teardown are noise, not defects.
    const f = r.failure()?.errorText ?? ''
    if (f.includes('ERR_ABORTED')) return
    errors.push(`requestfailed: ${r.url()} (${f})`)
  })
  return errors
}

test('the console boots in a real browser with no errors and real assets', async ({ page }) => {
  const errors = collectPageFailures(page)

  // Track asset responses so a silently-404ing bundle cannot pass as "booted".
  const bad: string[] = []
  page.on('response', (r) => {
    if (r.status() >= 400) bad.push(`${r.status()} ${r.url()}`)
  })

  await page.goto('/console/inference', { waitUntil: 'networkidle' })

  // Non-vacuity: React must have actually mounted. An empty #root that throws
  // nothing would otherwise satisfy "no errors".
  const rootChildren = await page.evaluate(
    () => document.getElementById('root')?.childElementCount ?? 0,
  )
  expect(rootChildren, 'React never mounted into #root').toBeGreaterThan(0)

  expect(bad, `HTTP failures while loading the console:\n${bad.join('\n')}`).toEqual([])
  expect(errors, `browser reported errors:\n${errors.join('\n')}`).toEqual([])
})

test('deep-linking a client-side route renders that page, not a 404', async ({ page }) => {
  // The SPA is client-side routed but server-served: app.py returns index.html
  // for every /console/* path. A misconfigured mount breaks only deep links --
  // never the in-app navigation that jsdom exercises.
  for (const path of ['/console/mesh', '/console/store', '/console/usage']) {
    const resp = await page.goto(path, { waitUntil: 'domcontentloaded' })
    expect(resp?.status(), `${path} did not return 200`).toBe(200)
    const mounted = await page.evaluate(
      () => document.getElementById('root')?.childElementCount ?? 0,
    )
    expect(mounted, `${path} served a shell that never mounted`).toBeGreaterThan(0)
  }
})

test('the Mesh Map canvas actually paints pixels', async ({ page }) => {
  await page.goto('/console/mesh', { waitUntil: 'networkidle' })

  const canvas = page.locator('canvas[role="img"]').first()
  await expect(canvas).toBeVisible()

  // Give the render loop a frame or two to run.
  await page.waitForTimeout(1200)

  const probe = await canvas.evaluate((el) => {
    const c = el as HTMLCanvasElement
    const ctx = c.getContext('2d')
    if (!ctx) return { ok: false as const, reason: 'no 2d context' }
    if (c.width === 0 || c.height === 0) {
      return { ok: false as const, reason: `zero-sized canvas ${c.width}x${c.height}` }
    }
    const { data } = ctx.getImageData(0, 0, c.width, c.height)

    // Count distinct RGBA values. A canvas that was never drawn to has exactly
    // one (fully transparent black). Anything painted has more.
    const distinctOf = (buf: Uint8ClampedArray) => {
      const seen = new Set<number>()
      for (let i = 0; i < buf.length; i += 4) {
        seen.add((buf[i] << 24) | (buf[i + 1] << 16) | (buf[i + 2] << 8) | buf[i + 3])
        if (seen.size > 64) break
      }
      return seen.size
    }

    const distinct = distinctOf(data)

    // POSITIVE CONTROL: wipe the canvas and re-measure with the identical
    // procedure. If the wiped reading is not "blank", the measurement cannot
    // tell painted from unpainted and every assertion here is worthless.
    ctx.clearRect(0, 0, c.width, c.height)
    const distinctAfterWipe = distinctOf(ctx.getImageData(0, 0, c.width, c.height).data)

    return { ok: true as const, width: c.width, height: c.height, distinct, distinctAfterWipe }
  })

  expect(probe.ok, probe.ok ? '' : `canvas unusable: ${probe.reason}`).toBe(true)
  if (!probe.ok) return

  expect(
    probe.distinctAfterWipe,
    'positive control failed: a deliberately cleared canvas did not read as blank, ' +
      'so this measurement cannot distinguish painted from unpainted',
  ).toBe(1)

  expect(
    probe.distinct,
    `Mesh Map canvas (${probe.width}x${probe.height}) drew nothing -- ` +
      'the jsdom tests for this screen pass without a single pixel being rendered',
  ).toBeGreaterThan(1)
})

test('the console reaches the real control plane, not an error state', async ({ page }) => {
  // Record what the app genuinely asked the backend for, and how it answered.
  const apiCalls: { url: string; status: number }[] = []
  page.on('response', (r) => {
    const u = new URL(r.url())
    if (u.pathname.startsWith('/api/') || u.pathname.startsWith('/v1/')) {
      apiCalls.push({ url: u.pathname, status: r.status() })
    }
  })

  await page.goto('/console/mesh', { waitUntil: 'networkidle' })
  await page.waitForTimeout(1500)

  // Non-vacuity: if the console made no backend calls at all, "no failures"
  // below would be trivially true and would prove nothing.
  expect(
    apiCalls.length,
    'the console issued zero API requests, so backend integration is untested here',
  ).toBeGreaterThan(0)

  const serverErrors = apiCalls.filter((c) => c.status >= 500)
  expect(
    serverErrors,
    `control plane returned 5xx:\n${serverErrors.map((e) => `${e.status} ${e.url}`).join('\n')}`,
  ).toEqual([])
})

test('the command palette opens and navigates', async ({ page }) => {
  const errors = collectPageFailures(page)
  await page.goto('/console/inference', { waitUntil: 'networkidle' })

  const before = page.url()
  await page.keyboard.press('ControlOrMeta+k')

  const dialog = page.getByRole('dialog')
  await expect(dialog, 'Ctrl/Cmd+K did not open the command palette').toBeVisible()

  await page.keyboard.type('mesh')
  await page.keyboard.press('Enter')

  // Non-vacuity: assert the URL genuinely changed, not merely that no error
  // occurred. A palette that opens but does nothing would otherwise pass.
  await expect
    .poll(() => page.url(), { message: 'palette selection did not navigate' })
    .not.toBe(before)

  expect(errors, `browser reported errors:\n${errors.join('\n')}`).toEqual([])
})
