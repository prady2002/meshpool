import { test, expect, type Page } from '@playwright/test'
import { spawn, type ChildProcess } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// ---------------------------------------------------------------------------
// The Playground is the flagship screen -- the plan's "default landing state for
// a new user is a working chat box". Nothing tested it end to end.
//
// The jsdom suite mocks fetch, so it renders whatever shape the mock invents and
// can never disagree with the real server. web/e2e/console.spec.ts drives a real
// browser but against a bare control plane with no nodes, where the model list is
// empty and the Send button is disabled -- so the streaming path never runs and
// every assertion there passes regardless of whether it works.
//
// This spec boots a real mesh (control plane + 3 agent processes), deploys a real
// model, and asserts on what a user would actually see in the chat bubble.
// ---------------------------------------------------------------------------

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO = path.resolve(HERE, '..', '..')

let stack: ChildProcess | undefined
let base = ''

test.describe.configure({ mode: 'serial' })

test.beforeAll(async () => {
  test.setTimeout(420_000)
  base = await new Promise<string>((resolve, reject) => {
    const proc = spawn(
      process.env.PYTHON ?? 'python',
      [path.join('sim', 'e2e_stack.py'), '--agents', '3'],
      { cwd: REPO, stdio: ['pipe', 'pipe', 'pipe'] },
    )
    stack = proc

    const noise: string[] = []
    let settled = false
    const fail = (why: string) => {
      if (settled) return
      settled = true
      reject(new Error(`${why}\n--- stack output ---\n${noise.join('')}`))
    }

    let buf = ''
    proc.stdout?.on('data', (d: Buffer) => {
      const s = d.toString()
      noise.push(s)
      buf += s
      const line = buf.split(/\r?\n/).find((l) => l.startsWith('MESHPOOL_E2E_READY'))
      if (line && !settled) {
        settled = true
        resolve(line.split(' ')[1])
      }
    })
    proc.stderr?.on('data', (d: Buffer) => noise.push(d.toString()))
    proc.on('exit', (code) => fail(`e2e stack exited early with code ${code}`))
    proc.on('error', (e) => fail(`could not spawn the e2e stack: ${e.message}`))
    setTimeout(() => fail('e2e stack never reported ready within 360s'), 360_000)
  })
})

test.afterAll(async () => {
  if (!stack) return
  // Closing stdin is the shutdown protocol; see sim/e2e_stack.py for why kill()
  // is not good enough on Windows.
  stack.stdin?.end()
  await new Promise<void>((resolve) => {
    const t = setTimeout(() => {
      stack?.kill()
      resolve()
    }, 30_000)
    stack?.on('exit', () => {
      clearTimeout(t)
      resolve()
    })
  })
})

/** The assistant's latest bubble, which is what a user reads. */
function assistantBubble(page: Page) {
  return page.getByTestId('chat-assistant').last()
}

/**
 * Just the answer text -- NOT the bubble.
 *
 * The bubble renders the literal placeholder "Streaming..." while the answer is
 * empty, so measuring the bubble reports ~10 characters for an answer that has
 * not started. That made a single final paint look like two growth steps and
 * let a mutation that drops every incremental delta pass this test.
 */
async function answerLength(page: Page) {
  const text = page.getByTestId('chat-assistant-text').last()
  return (await text.count()) === 0 ? 0 : (await text.innerText()).trim().length
}

async function openPlaygroundAndSend(page: Page, prompt: string) {
  await page.goto(`${base}/console/inference`, { waitUntil: 'networkidle' })

  // Non-vacuity: with no deployed model the select is empty and Send is disabled,
  // in which case everything below would pass without exercising anything.
  const options = page.locator('select option')
  await expect
    .poll(() => options.count(), {
      message: 'no model appeared in the Playground, so the streaming path cannot run',
      timeout: 30_000,
    })
    .toBeGreaterThan(0)

  const send = page.getByRole('button', { name: /send to mesh/i })
  await expect(send, 'the Send button never became enabled').toBeEnabled()

  await page.locator('#playground-prompt').fill(prompt)
  await send.click()
  return send
}

test('the playground shows the answer, not the JSON envelope the API returns', async ({ page }) => {
  test.setTimeout(180_000)
  await openPlaygroundAndSend(page, 'hello mesh')

  const bubble = assistantBubble(page)
  await expect(bubble).toBeVisible()

  // Wait for generation to finish, then read what the user is left looking at.
  await expect
    .poll(async () => (await bubble.innerText()).trim().length, {
      message: 'the assistant bubble never received any text at all',
      timeout: 120_000,
    })
    .toBeGreaterThan(0)
  await expect(page.getByRole('button', { name: /send to mesh/i })).toBeEnabled({
    timeout: 120_000,
  })

  const shown = (await bubble.innerText()).trim()

  // POST /api/v1/inference/chat answers with a JSON object carrying the text plus
  // deployment_id, tokens_per_sec, pipeline stats and the echoed prompt. The
  // console used to read that response body as a byte stream and append the raw
  // bytes to the bubble, so the user was shown the envelope instead of the answer.
  for (const leak of ['deployment_id', 'tokens_per_sec', 'finish_reason', 'pipeline']) {
    expect(
      shown,
      `the chat bubble is showing the raw API envelope, not the model's answer:\n${shown.slice(0, 400)}`,
    ).not.toContain(leak)
  }
  expect(shown, 'the bubble looks like raw JSON').not.toMatch(/^\s*[{[]/)
  expect(
    shown,
    'the console fell back to its generic failure text, so nothing streamed',
  ).not.toContain('could not complete this response')
})

test('tokens arrive progressively rather than in one lump at the end', async ({ page }) => {
  test.setTimeout(180_000)
  await openPlaygroundAndSend(page, 'stream me something long enough to observe')

  // While generating, the Send button is RELABELLED to "Streaming…", so a locator
  // for /send to mesh/i matches nothing at all during the request. isEnabled() on
  // a locator that does not currently match auto-waits for it to appear, so
  // polling it would block for the whole response and then report a single
  // sample -- the sampler would observe nothing and blame the product. count()
  // resolves immediately and is the only safe way to poll a transient state.
  const busy = page.getByRole('button', { name: /streaming/i })

  // Sample what the user can see, while they can see it. A non-streaming client
  // produces one jump from empty to final; a streaming one produces a staircase.
  const lengths: number[] = []
  // Samples taken while the request was still in flight. This is the whole test:
  // a partial answer visible DURING generation is the only thing that separates
  // streaming from one paint at the end.
  const partials: number[] = []
  const started = Date.now()
  let firstTextMs = -1
  let sawBusy = false
  for (;;) {
    const busyNow = (await busy.count()) > 0
    sawBusy = sawBusy || busyNow
    const n = await answerLength(page)
    if (busyNow && n > 0) partials.push(n)
    if (n > 0 && firstTextMs < 0) firstTextMs = Date.now() - started
    if (lengths[lengths.length - 1] !== n) lengths.push(n)
    if (!busyNow && n > 0) break
    if (Date.now() - started > 120_000) break
    await page.waitForTimeout(60)
  }
  const totalMs = Date.now() - started

  // Non-vacuity: if the run finished before the first sample we never observed
  // the in-flight state, and "it arrived in one lump" would be a statement about
  // the sampler rather than about the transport.
  expect(sawBusy, 'the request finished before sampling began; nothing was observed').toBe(true)
  expect(firstTextMs, 'no text ever appeared').toBeGreaterThanOrEqual(0)
  expect(lengths[lengths.length - 1], 'the final answer was empty').toBeGreaterThan(0)

  // Distinct growing lengths, excluding the initial empty reading. Two or more
  // means the DOM was updated with a partial answer at least twice before the
  // request finished -- which is only possible if the transport is incremental.
  const growth = lengths.filter((n) => n > 0)
  expect(
    growth.length,
    `the answer appeared in ${growth.length} step(s) (${JSON.stringify(lengths)}) ` +
      `after ${totalMs}ms -- the whole response landed at once, so this is not streaming`,
  ).toBeGreaterThanOrEqual(2)

  // The decisive check. `growth.length >= 2` can still be satisfied by a
  // placeholder followed by the finished answer; only an INCOMPLETE answer seen
  // while the button still said "Streaming" proves tokens reached the user early.
  const final = lengths[lengths.length - 1]
  expect(
    partials.filter((n) => n < final).length,
    `no partial answer was ever visible while generating (in-flight samples ` +
      `${JSON.stringify(partials)}, final length ${final}) -- the response was ` +
      `painted in one go, so time-to-first-token equals total generation time`,
  ).toBeGreaterThan(0)

  // Time-to-first-token must be a fraction of total time. If the client waits for
  // the complete body it cannot paint anything until the very end.
  expect(
    firstTextMs,
    `first text at ${firstTextMs}ms of a ${totalMs}ms request -- the client is ` +
      'waiting for the whole response before rendering',
  ).toBeLessThan(totalMs * 0.9)
})

test('the run reports real measured throughput after it finishes', async ({ page }) => {
  test.setTimeout(180_000)
  await openPlaygroundAndSend(page, 'report your speed')

  // The final NDJSON frame carries the engine's own measurements. Surfacing them
  // proves the client parsed the stream rather than concatenating bytes, and it
  // is the number the plan requires to be labelled as measured.
  const stats = page.getByTestId('chat-run-stats')
  await expect(stats, 'the completed run published no statistics').toBeVisible({
    timeout: 120_000,
  })
  const text = await stats.innerText()
  expect(text, `run stats were rendered but empty: ${text}`).toMatch(/\d/)
  const tps = Number(/([\d.]+)\s*tok\/s/i.exec(text)?.[1] ?? '0')
  expect(tps, `throughput reported as ${tps} tok/s`).toBeGreaterThan(0)
})
