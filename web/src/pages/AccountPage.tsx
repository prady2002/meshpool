import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import type { SpendCapStatus, WalletResponse } from '../api/types'

export function AccountPage() {
  const [wallet, setWallet] = useState<WalletResponse | null>(null)
  const [secrets, setSecrets] = useState<string[]>([])
  const [secretName, setSecretName] = useState('')
  const [secretValue, setSecretValue] = useState('')
  const [cap, setCap] = useState<SpendCapStatus | null>(null)
  const [capDraft, setCapDraft] = useState('')

  const load = () => {
    apiClient.getWallet().then(setWallet)
    apiClient.getSecrets().then(setSecrets)
    apiClient.getSpendCap().then(setCap).catch(() => setCap(null))
  }

  useEffect(load, [])

  const applyCap = (value: number | null) => {
    apiClient.setSpendCap(value).then(() => { setCapDraft(''); load() }).catch(() => undefined)
  }

  return (
    <section className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Account & Billing</p>
        <h2 className="mt-2 text-3xl font-semibold">Usage, spend caps, and API keys</h2>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75"><p className="text-sm text-slate-500 dark:text-slate-400">Current balance</p><p className="mt-2 text-3xl font-semibold">{typeof wallet?.balance === 'number' ? wallet.balance.toFixed(1) : '—'}</p></div>
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75"><p className="text-sm text-slate-500 dark:text-slate-400">Spend this month</p><p className="mt-2 text-3xl font-semibold">{typeof wallet?.consumed === 'number' ? wallet.consumed.toFixed(1) : '—'}</p></div>
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <p className="text-sm text-slate-500 dark:text-slate-400">Hard spend cap</p>
          <p className="mt-2 text-3xl font-semibold">{cap?.enforced && typeof cap.cap === 'number' ? cap.cap.toFixed(1) : '—'}</p>
          {cap?.enforced ? (
            <p className={`mt-1 text-xs ${cap.exceeded ? 'text-rose-500' : 'text-slate-500 dark:text-slate-400'}`}>
              {cap.exceeded
                ? 'Cap reached — work is stopped, not just flagged.'
                : `${(cap.remaining ?? 0).toFixed(1)} remaining · stops work when reached.`}
            </p>
          ) : (
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">No cap set — spending is unlimited.</p>
          )}
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
          <h3 className="text-xl font-semibold">API keys / secrets</h3>
          <div className="mt-4 grid gap-3 md:grid-cols-[1fr_1fr_auto]">
            <input value={secretName} onChange={(event) => setSecretName(event.target.value)} placeholder="Name" className="rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-800" />
            <input value={secretValue} onChange={(event) => setSecretValue(event.target.value)} placeholder="Value" className="rounded-2xl border border-slate-200 bg-transparent px-4 py-3 dark:border-slate-800" />
            <button type="button" onClick={() => apiClient.putSecret(secretName, secretValue).then(() => { setSecretName(''); setSecretValue(''); load() })} className="rounded-2xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white">Save</button>
          </div>
          <ul className="mt-5 space-y-3">
            {secrets.map((name) => (
              <li key={name} className="flex items-center justify-between rounded-2xl border border-slate-800 px-4 py-3 text-sm text-slate-200">
                <span>{name}</span>
                <button type="button" onClick={() => apiClient.deleteSecret(name).then(load)} className="text-rose-300">Delete</button>
              </li>
            ))}
            {secrets.length === 0 && (
              <li className="rounded-2xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
                No secrets yet. Add one above to make it available to your jobs — values are
                never returned by the API, only names.
              </li>
            )}
          </ul>
        </div>
        <aside className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-xl font-semibold">Usage notes</h3>
          <ul className="mt-4 space-y-3 text-sm text-slate-400">
            <li>• Billing uses exact wallet balances from the control plane.</li>
            <li>• Caps are client-visible and should be enforced server-side for real charging.</li>
            <li>• Keys are currently mapped to the secrets API surface; dedicated API-key endpoints are still needed.</li>
          </ul>
        </aside>
      </div>
    </section>
  )
}
