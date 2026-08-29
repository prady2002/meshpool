/**
 * Mesh Functions is the feature that turns MeshPool from "a pool that runs OUR six
 * functions" into "a pool that runs YOUR code", and until now it existed only over
 * raw HTTP. These tests pin the console surface for it.
 *
 * They are written to fail if the page merely *looks* right: each one asserts on the
 * API call that was actually made, with the actual bytes / function name.
 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { FunctionsPage } from '../src/pages/FunctionsPage'
import { apiClient, ApiError } from '../src/api/client'
import type { BatchJob, FunctionUpload, JobSubmitResponse } from '../src/api/types'

const SHA = '3'.repeat(64)

const upload: FunctionUpload = {
  function: `wasm:${SHA}`,
  sha256: SHA,
  size_bytes: 42,
  abi: 'wasi_snapshot_preview1',
  imports: ['wasi_snapshot_preview1.fd_write'],
  exports: ['_start', 'memory'],
  name: 'triple.wasm',
  owner: 'default',
}

const submitted: JobSubmitResponse = { job_id: 'job-1', units: 2, owner: 'default', trace_id: null }

function completedJob(func: string): BatchJob {
  return {
    active_nodes: [],
    units_total: 2,
    id: 'job-1',
    name: func.slice(0, 40),
    func,
    owner: 'default',
    status: 'completed',
    created_ms: Date.now(),
    progress: { pending: 0, assigned: 0, running: 0, done: 2, failed: 0, total: 2 },
    units: [
      { id: 'u0', status: 'done', assigned_node: 'node-a', result: { tripled: 0 } },
      { id: 'u1', status: 'done', assigned_node: 'node-b', result: { tripled: 3 } },
    ],
  }
}

function renderPage() {
  return render(<MemoryRouter><FunctionsPage /></MemoryRouter>)
}

async function uploadFile(name = 'triple.wasm') {
  const file = new File([new Uint8Array([0x00, 0x61, 0x73, 0x6d])], name, {
    type: 'application/wasm',
  })
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  // userEvent.upload, not fireEvent: a FileList is readonly in jsdom, so assigning
  // through `target` silently leaves input.files empty and the handler never runs.
  await userEvent.upload(input, file)
  return file
}

beforeEach(() => {
  localStorage.clear()
  vi.restoreAllMocks()
})

describe('uploading a module', () => {
  it('sends the raw bytes and shows the content-addressed name', async () => {
    const spy = vi.spyOn(apiClient, 'uploadFunction').mockResolvedValue(upload)
    renderPage()
    await uploadFile()

    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
    const [bytes, name] = spy.mock.calls[0]
    // Must be the file's bytes, not a FormData wrapper: the endpoint reads the
    // request body as the module itself.
    expect(bytes).toBeInstanceOf(ArrayBuffer)
    expect(new Uint8Array(bytes as ArrayBuffer)).toEqual(new Uint8Array([0x00, 0x61, 0x73, 0x6d]))
    expect(name).toBe('triple.wasm')
    expect((await screen.findAllByText(`wasm:${SHA}`)).length).toBeGreaterThan(0)
    expect(screen.getByText(/module accepted/i)).toBeInTheDocument()
  })

  it('surfaces the pool\'s rejection reason verbatim', async () => {
    vi.spyOn(apiClient, 'uploadFunction').mockRejectedValue(
      new ApiError('module imports host functions outside WASI: env.evil', 400),
    )
    renderPage()
    await uploadFile('evil.wasm')

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/imports host functions outside WASI/i)
    expect(alert).toHaveTextContent(/env\.evil/)
    // A generic "upload failed" would hide the one thing the user needs.
    expect(alert).not.toHaveTextContent(/^Upload failed$/)
  })

  it('remembers uploads across a remount', async () => {
    vi.spyOn(apiClient, 'uploadFunction').mockResolvedValue(upload)
    const first = renderPage()
    await uploadFile()
    await screen.findAllByText(`wasm:${SHA}`)
    first.unmount()

    renderPage()
    expect((await screen.findAllByText(`wasm:${SHA}`)).length).toBeGreaterThan(0)
  })
})

describe('running a module', () => {
  it('submits one unit per requested index, naming the uploaded function', async () => {
    vi.spyOn(apiClient, 'uploadFunction').mockResolvedValue(upload)
    const submit = vi.spyOn(apiClient, 'submitJob')
      .mockResolvedValue(submitted)
    vi.spyOn(apiClient, 'getJob').mockResolvedValue(completedJob(`wasm:${SHA}`))

    renderPage()
    await uploadFile()
    await screen.findAllByText(`wasm:${SHA}`)

    const units = screen.getByRole('spinbutton')
    await userEvent.clear(units)
    await userEvent.type(units, '2')
    await userEvent.click(screen.getByRole('button', { name: /run across the pool/i }))

    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1))
    const body = submit.mock.calls[0][0]
    expect(body.func).toBe(`wasm:${SHA}`)
    expect(body.payloads).toHaveLength(2)
    // Each unit must be distinguishable, or a fan-out is just the same work twice.
    expect(body.payloads.map((p) => (p as { index: number }).index)).toEqual([0, 1])
    expect((body.payloads[0] as { x: number }).x).toBe(7)
  })

  it('refuses to submit a payload that is not JSON', async () => {
    vi.spyOn(apiClient, 'uploadFunction').mockResolvedValue(upload)
    const submit = vi.spyOn(apiClient, 'submitJob')
    renderPage()
    await uploadFile()
    await screen.findAllByText(`wasm:${SHA}`)

    const box = screen.getByRole('textbox')
    await userEvent.clear(box)
    await userEvent.type(box, 'not json')
    await userEvent.click(screen.getByRole('button', { name: /run across the pool/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/must be JSON/i)
    expect(submit).not.toHaveBeenCalled()
  })

  it('shows each unit\'s result and which machine produced it', async () => {
    vi.spyOn(apiClient, 'uploadFunction').mockResolvedValue(upload)
    vi.spyOn(apiClient, 'submitJob').mockResolvedValue(submitted)
    vi.spyOn(apiClient, 'getJob').mockResolvedValue(completedJob(`wasm:${SHA}`))

    renderPage()
    await uploadFile()
    await screen.findAllByText(`wasm:${SHA}`)
    await userEvent.click(screen.getByRole('button', { name: /run across the pool/i }))

    expect(await screen.findByText('{"tripled":3}', {}, { timeout: 4000 })).toBeInTheDocument()
    expect(screen.getByText('node-a')).toBeInTheDocument()
    expect(screen.getByText('node-b')).toBeInTheDocument()
    // The fan-out claim on screen must reflect the distinct nodes actually used.
    expect(screen.getByText(/spread over 2 machines/i)).toBeInTheDocument()
  })
})
