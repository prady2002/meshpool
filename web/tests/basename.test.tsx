/**
 * The console must render at the URL the server actually advertises.
 *
 * This file exists because the console shipped rendering a COMPLETELY BLANK PAGE
 * at `/console` -- the exact path `meshpool/controlplane/app.py` declares and
 * serves (`@app.get("/console")`). Every unit test passed, `tsc -b` passed, and
 * `vite build` passed, because nothing ever mounted the router at a real URL.
 * The bug was only visible in a browser.
 *
 * Cause: `vite.config.ts` sets `base: '/console/'` (trailing slash, correct --
 * it prefixes asset URLs), and `main.tsx` passed that same string straight to
 * React Router's `basename`. React Router does not match "/console" against
 * basename "/console/", so it rendered nothing at all.
 *
 * Note the shape of these tests: a pure string assertion on `routerBasename`
 * alone would NOT have caught the original defect, because the original code
 * never called such a function -- it passed BASE_URL through untouched. So the
 * decisive test below mounts the real <BrowserRouter> at the bare "/console"
 * URL and asserts real page content appears.
 */
import { render, screen, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { routerBasename } from '../src/basename'
import { apiClient } from '../src/api/client'
import { mockCluster } from '../src/api/mock'

beforeEach(() => {
  vi.spyOn(apiClient, 'getCluster').mockResolvedValue(mockCluster)
  vi.spyOn(apiClient, 'getModels').mockResolvedValue([])
  vi.spyOn(apiClient, 'getJobs').mockResolvedValue([])
  vi.spyOn(apiClient, 'getBlobs').mockResolvedValue([])
})

function renderAtRealUrl(url: string) {
  window.history.pushState({}, '', url)
  return render(
    <BrowserRouter basename={routerBasename('/console/')}>
      <App />
    </BrowserRouter>,
  )
}

describe('router basename', () => {
  it('strips the trailing slash Vite puts on BASE_URL', () => {
    expect(routerBasename('/console/')).toBe('/console')
  })

  it('leaves an already-clean basename alone', () => {
    expect(routerBasename('/console')).toBe('/console')
  })

  it('collapses a bare root to "/" rather than an empty string', () => {
    // basename="" makes React Router throw; "/" is the valid root form.
    expect(routerBasename('/')).toBe('/')
  })
})

describe('the console renders at the URL the server serves', () => {
  it('renders real content at the bare /console path (regression: blank page)', async () => {
    renderAtRealUrl('/console')
    expect(
      await screen.findByRole('heading', { name: /community compute console/i }),
    ).toBeInTheDocument()
    // Navigation must be present too: a shell with no nav would still be broken.
    expect(await screen.findByRole('navigation', { name: /primary/i })).toBeInTheDocument()
  })

  it('also renders at /console/ with the trailing slash', async () => {
    renderAtRealUrl('/console/')
    expect(
      await screen.findByRole('heading', { name: /community compute console/i }),
    ).toBeInTheDocument()
  })

  it('renders a deep link such as /console/jobs', async () => {
    renderAtRealUrl('/console/jobs')
    expect(
      await screen.findByRole('heading', { name: /community compute console/i }),
    ).toBeInTheDocument()
  })

  it('NON-VACUITY: the old trailing-slash basename really did render nothing', async () => {
    // Reproduce the shipped bug exactly. If this ever starts passing, the two
    // tests above have stopped proving anything and this file is worthless.
    window.history.pushState({}, '', '/console')
    const { container } = render(
      <BrowserRouter basename="/console/">
        <App />
      </BrowserRouter>,
    )
    await waitFor(() => {
      expect(
        screen.queryByRole('heading', { name: /community compute console/i }),
      ).not.toBeInTheDocument()
    })
    expect(container.textContent?.trim()).toBe('')
  })
})
