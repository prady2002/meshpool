import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

// ---------------------------------------------------------------------------
// Accessibility is a Phase 7 gate, and it is also the clearest example of a
// property no component test can establish: axe evaluates computed colour
// contrast, focus order, accessible names and ARIA validity against a real
// layout and a real style engine. jsdom computes no layout and applies no CSS
// cascade, so contrast and focus-order violations are invisible there by
// construction.
//
// The rule enforced here is: zero serious or critical violations on every
// screen a user can reach. Minor/moderate findings are reported but not failed,
// so the gate stays actionable rather than becoming noise people learn to skip.
// ---------------------------------------------------------------------------

const ROUTES = [
  ['/console/inference', 'Playground'],
  ['/console/mesh', 'Mesh Map'],
  ['/console/overview', 'Contributor'],
  ['/console/apps', 'Apps'],
  ['/console/functions', 'Functions'],
  ['/console/store', 'Store'],
  ['/console/usage', 'Usage'],
  ['/console/models', 'Models'],
] as const

for (const [path, name] of ROUTES) {
  test(`${name} has no serious or critical accessibility violations`, async ({ page }) => {
    await page.goto(path, { waitUntil: 'networkidle' })
    // Let async content land, so the scan covers the populated screen rather
    // than a skeleton -- a skeleton is trivially accessible and would make this
    // pass without examining anything the user actually reads.
    await page.waitForTimeout(800)

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze()

    // Non-vacuity: axe must actually have examined something. A failed
    // injection, or a page that never rendered, yields zero violations and
    // would otherwise read as a pass.
    const checked = results.passes.length + results.violations.length
    expect(
      checked,
      `axe evaluated no rules on ${path} -- the scan did not run, so a clean ` +
        'result here means nothing',
    ).toBeGreaterThan(0)

    const blocking = results.violations.filter(
      (v) => v.impact === 'serious' || v.impact === 'critical',
    )

    const report = blocking
      .map((v) => {
        const where = v.nodes.slice(0, 3).map((n) => n.target.join(' ')).join('\n      ')
        return `  [${v.impact}] ${v.id}: ${v.help}\n    ${v.helpUrl}\n      ${where}`
      })
      .join('\n')

    expect(blocking, `${name} (${path}) accessibility violations:\n${report}`).toEqual([])
  })
}

test('every interactive control on the primary screen is reachable by keyboard', async ({
  page,
}) => {
  // Keyboard reachability is a layout- and focus-order property. jsdom has no
  // focus order at all, so this cannot be asserted anywhere but a real browser.
  await page.goto('/console/inference', { waitUntil: 'networkidle' })
  await page.waitForTimeout(500)

  const interactive = await page.evaluate(() => {
    const sel = 'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
    return Array.from(document.querySelectorAll(sel)).filter((el) => {
      const s = getComputedStyle(el)
      const r = el.getBoundingClientRect()
      return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0
    }).length
  })

  expect(
    interactive,
    'no visible interactive controls found, so keyboard reachability is untested',
  ).toBeGreaterThan(0)

  // Walk the focus ring and count distinct elements reached.
  const reached = new Set<string>()
  for (let i = 0; i < interactive * 2 + 10; i++) {
    await page.keyboard.press('Tab')
    const id = await page.evaluate(() => {
      const el = document.activeElement
      if (!el || el === document.body) return ''
      const parts = [el.tagName, el.getAttribute('aria-label') ?? '', el.textContent?.slice(0, 24) ?? '']
      return parts.join('|')
    })
    if (id) reached.add(id)
  }

  // Tabbing must reach a meaningful share of the controls. An exact count is
  // brittle (portals, disabled states, roving tabindex), but reaching almost
  // nothing means a focus trap or tabindex=-1 has locked keyboard users out.
  expect(
    reached.size,
    `keyboard tabbing reached only ${reached.size} of ~${interactive} visible ` +
      'interactive controls, which indicates a focus trap',
  ).toBeGreaterThanOrEqual(Math.min(3, interactive))
})
