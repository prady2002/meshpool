import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import type { CapabilityToken, CapabilityTokenMinted, SpendCapStatus, WalletResponse } from '../api/types'
import { Metric } from '../components/ui'

const TOKEN_SCOPES = [
  ['inference:invoke', 'Run inference'],
  ['deployments:read', 'View apps'],
  ['deployments:write', 'Manage apps'],
  ['functions:read', 'View functions'],
  ['functions:write', 'Upload functions'],
  ['jobs:read', 'Read batch jobs'],
  ['jobs:write', 'Submit batch jobs'],
  ['blobs:read', 'Read object storage'],
  ['blobs:write', 'Write object storage'],
  ['storage:read', 'S3/object read'],
  ['storage:write', 'S3/object write'],
  ['nodes:read', 'View mesh nodes'],
  ['wallet:read', 'Read usage and billing'],
  ['ledger:read', 'Read ledger'],
  ['tokens:read', 'List tokens'],
  ['tokens:write', 'Create/revoke tokens'],
  ['nodes:admin', 'Administer nodes'],
] as const

function fmtMs(ms: number | null) {
  if (!ms) return 'never'
  return new Date(ms).toLocaleString()
}

function fmtCredits(n: number) {
  return n.toLocaleString(undefined, { maximumFractionDigits: 1 })
}

export function AccountPage() {
  const [wallet, setWallet] = useState<WalletResponse | null>(null)
  const [secrets, setSecrets] = useState<string[]>([])
  const [secretName, setSecretName] = useState('')
  const [secretValue, setSecretValue] = useState('')
  const [cap, setCap] = useState<SpendCapStatus | null>(null)
  const [capDraft, setCapDraft] = useState('')
  const [tokens, setTokens] = useState<CapabilityToken[]>([])
  const [tokenLabel, setTokenLabel] = useState('')
  const [tokenTtlDays, setTokenTtlDays] = useState('30')
  const [tokenSpendCap, setTokenSpendCap] = useState('')
  const [tokenScopes, setTokenScopes] = useState<string[]>(['inference:invoke'])
  const [minted, setMinted] = useState<CapabilityTokenMinted | null>(null)

  const load = () => {
    apiClient.getWallet().then(setWallet).catch(() => setWallet(null))
    apiClient.getSecrets().then(setSecrets).catch(() => setSecrets([]))
    apiClient.getSpendCap().then(setCap).catch(() => setCap(null))
    apiClient.getCapabilityTokens().then(setTokens).catch(() => setTokens([]))
  }

  useEffect(load, [])

  const applyCap = (value: number | null) => {
    apiClient.setSpendCap(value).then(() => { setCapDraft(''); load() }).catch(() => undefined)
  }

  const toggleScope = (scope: string) => {
    setTokenScopes((prev) => prev.includes(scope)
      ? prev.filter((s) => s !== scope)
      : [...prev, scope])
  }

  const mintToken = () => {
    const days = Number.parseFloat(tokenTtlDays)
    const spend = Number.parseFloat(tokenSpendCap)
    if (!tokenLabel.trim() || tokenScopes.length === 0 || !Number.isFinite(days) || days <= 0) return
    apiClient.createCapabilityToken({
      label: tokenLabel.trim(),
      scopes: tokenScopes,
      expires_in_seconds: days * 24 * 3600,
      ...(Number.isFinite(spend) && spend >= 0 ? { spend_cap: spend } : {}),
    }).then((token) => {
      setMinted(token)
      setTokenLabel('')
      setTokenSpendCap('')
      load()
    }).catch(() => undefined)
  }

  return (
    <section className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.28em] text-indigo-600 dark:text-indigo-400">Account & Billing</p>
        <h2 className="mt-2 text-3xl font-semibold">Usage, spend caps, and API keys (capability tokens)</h2>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <Metric label="Current balance" value={typeof wallet?.balance === 'number' ? wallet.balance.toFixed(1) : '-'} unit="credits" tier="measured" className="rounded-[28px] bg-white/90 dark:bg-slate-950/75" />
        <Metric label="Spend this month" value={typeof wallet?.consumed === 'number' ? wallet.consumed.toFixed(1) : '-'} unit="credits" tier="measured" className="rounded-[28px] bg-white/90 dark:bg-slate-950/75" />
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <Metric label="Hard spend cap" value={cap?.enforced && typeof cap.cap === 'number' ? cap.cap.toFixed(1) : '-'} unit={cap?.enforced ? 'credits' : undefined} tier="measured" className="border-0 bg-transparent p-0 shadow-none" />
          {cap?.enforced ? (
            <p className={`mt-1 text-xs ${cap.exceeded ? 'text-rose-600 dark:text-rose-400' : 'text-slate-600 dark:text-slate-400'}`}>
              {cap.exceeded
                ? 'Cap reached - work is stopped, not just flagged.'
                : 'Stops work when reached.'}
            </p>
          ) : (
            <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">No cap set - spending is unlimited.</p>
          )}
          {cap?.enforced && !cap.exceeded ? (
            <Metric label="Remaining cap" value={(cap.remaining ?? 0).toFixed(1)} unit="remaining credits" tier="measured" density="compact" className="mt-3" />
          ) : null}
          <div className="mt-3 flex gap-2">
            <input
              value={capDraft}
              onChange={(event) => setCapDraft(event.target.value)}
              inputMode="decimal"
              aria-label="Spend cap"
              placeholder="Set cap"
              className="w-full rounded-2xl border border-slate-200 bg-transparent px-3 py-2 text-sm dark:border-slate-800"
            />
            <button
              type="button"
              onClick={() => { const n = Number.parseFloat(capDraft); if (Number.isFinite(n) && n >= 0) applyCap(n) }}
              className="rounded-2xl bg-indigo-500 px-3 py-2 text-sm font-semibold text-white"
            >Set</button>
            {cap?.enforced ? (
              <button type="button" onClick={() => applyCap(null)} className="rounded-2xl border border-slate-200 px-3 py-2 text-sm font-semibold dark:border-slate-800">Clear</button>
            ) : null}
          </div>
        </div>
      </div>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-xl font-semibold">Secrets</h3>
          <div className="mt-4 grid gap-3 md:grid-cols-[1fr_1fr_auto]">
            <input value={secretName} onChange={(event) => setSecretName(event.target.value)} placeholder="Name" className="rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-800" />
            <input value={secretValue} onChange={(event) => setSecretValue(event.target.value)} placeholder="Value" className="rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-800" />
            <button type="button" onClick={() => apiClient.putSecret(secretName, secretValue).then(() => { setSecretName(''); setSecretValue(''); load() })} className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white">Save</button>
          </div>
          <ul className="mt-5 space-y-3">
            {secrets.map((name) => (
              <li key={name} className="flex items-center justify-between rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-800 dark:border-slate-800 dark:text-slate-200">
                <span>{name}</span>
                <button type="button" onClick={() => apiClient.deleteSecret(name).then(load)} className="text-rose-700 dark:text-rose-300">Delete</button>
              </li>
            ))}
            {secrets.length === 0 && (
              <li className="rounded-2xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-slate-600 dark:border-slate-700 dark:text-slate-400">
                No secrets yet. Add one above to make it available to your jobs - values are never returned by the API, only names.
              </li>
            )}
          </ul>
        </div>
        <aside className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-xl font-semibold">Usage notes</h3>
          <ul className="mt-4 space-y-3 text-sm text-slate-600 dark:text-slate-400">
            <li>? Billing uses exact wallet balances from the control plane.</li>
            <li>? Caps stop new work server-side when reached.</li>
            <li>? Capability tokens are narrow API keys: readable scopes, expiry, optional spend cap, and immediate revocation.</li>
          </ul>
        </aside>
      </div>
      <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="text-xl font-semibold">Capability tokens</h3>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
              Mint readable, narrow API keys for CI and automation. The secret is shown exactly once.
            </p>
          </div>
          {minted ? (
            <div className="rounded-2xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-100">
              <p className="font-semibold">Copy this token now - it cannot be shown again.</p>
              <code className="mt-2 block break-all rounded-xl bg-white/70 p-2 dark:bg-black/30">{minted.token}</code>
              <button type="button" onClick={() => setMinted(null)} className="mt-2 text-xs font-semibold underline">I copied it</button>
            </div>
          ) : null}
        </div>
        <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(220px,1fr)_160px_160px]">
          <input value={tokenLabel} onChange={(event) => setTokenLabel(event.target.value)} placeholder="Label, e.g. GitHub Actions deploy" className="rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-800" />
          <input value={tokenTtlDays} onChange={(event) => setTokenTtlDays(event.target.value)} inputMode="decimal" aria-label="Token lifetime in days" placeholder="Days" className="rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-800" />
          <input value={tokenSpendCap} onChange={(event) => setTokenSpendCap(event.target.value)} inputMode="decimal" aria-label="Optional token spend cap" placeholder="Spend cap (optional)" className="rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-800" />
        </div>
        <fieldset className="mt-4">
          <legend className="text-sm font-semibold">Choose scopes</legend>
          <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {TOKEN_SCOPES.map(([scope, label]) => (
              <label key={scope} className="flex items-center gap-3 rounded-2xl border border-slate-200 px-3 py-2 text-sm dark:border-slate-800">
                <input type="checkbox" checked={tokenScopes.includes(scope)} onChange={() => toggleScope(scope)} />
                <span><span className="font-medium">{label}</span> <span className="text-slate-500">({scope})</span></span>
              </label>
            ))}
          </div>
        </fieldset>
        <button type="button" onClick={mintToken} disabled={!tokenLabel.trim() || tokenScopes.length === 0} className="mt-4 rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50">Create token</button>
        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-slate-500">
              <tr><th className="py-2 pr-4">Label</th><th className="py-2 pr-4">Scopes</th><th className="py-2 pr-4">Expires</th><th className="py-2 pr-4">Last used</th><th className="py-2 pr-4">Spend</th><th className="py-2">Action</th></tr>
            </thead>
            <tbody>
              {tokens.map((token) => (
                <tr key={token.id} className="border-t border-slate-200 dark:border-slate-800">
                  <td className="py-3 pr-4"><div className="font-medium">{token.label}</div><div className="text-xs text-slate-500">{token.id}</div></td>
                  <td className="py-3 pr-4">{token.scopes.join(', ')}</td>
                  <td className="py-3 pr-4">{fmtMs(token.expires_at_ms)}</td>
                  <td className="py-3 pr-4">{fmtMs(token.last_used_ms)}</td>
                  <td className="py-3 pr-4">{fmtCredits(token.spent)}{typeof token.spend_cap === 'number' ? ` / ${fmtCredits(token.spend_cap)}` : ''}</td>
                  <td className="py-3"><button type="button" onClick={() => apiClient.revokeCapabilityToken(token.id).then(load)} className="text-rose-700 dark:text-rose-300">Revoke</button></td>
                </tr>
              ))}
              {tokens.length === 0 ? (
                <tr><td colSpan={6} className="py-6 text-center text-slate-600 dark:text-slate-400">No capability tokens yet.</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
