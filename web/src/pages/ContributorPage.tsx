import { useEffect, useState } from 'react'
import { apiClient } from '../api/client'
import type { LedgerResponse, Node, OnboardingResponse, WalletResponse } from '../api/types'
import { NodeCard } from '../components/NodeCard'

export function ContributorPage() {
  const [wallet, setWallet] = useState<WalletResponse | null>(null)
  const [ledger, setLedger] = useState<LedgerResponse | null>(null)
  const [nodes, setNodes] = useState<Node[]>([])
  const [onboarding, setOnboarding] = useState<OnboardingResponse | null>(null)

  const load = () => {
    apiClient.getWallet().then(setWallet)
    apiClient.getLedger().then(setLedger)
    apiClient.getNodes().then(setNodes)
    apiClient.getOnboarding().then(setOnboarding)
  }

  useEffect(load, [])

  return (
    <section className="space-y-6">
      <div className="grid gap-4 xl:grid-cols-4">
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75"><p className="text-sm text-slate-500 dark:text-slate-400">Balance</p><p className="mt-2 text-3xl font-semibold">{wallet ? `${wallet.balance.toFixed(1)} credits` : '—'}</p></div>
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75"><p className="text-sm text-slate-500 dark:text-slate-400">Donated</p><p className="mt-2 text-3xl font-semibold">{wallet ? wallet.donated.toFixed(1) : '—'}</p></div>
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75"><p className="text-sm text-slate-500 dark:text-slate-400">Consumed</p><p className="mt-2 text-3xl font-semibold">{wallet ? wallet.consumed.toFixed(1) : '—'}</p></div>
        <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75"><p className="text-sm text-slate-500 dark:text-slate-400">Reputation</p><p className="mt-2 text-3xl font-semibold">{wallet ? `${Math.round(wallet.reputation * 100)}%` : '—'}</p></div>
      </div>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_360px]">
        <div className="space-y-4">
          <div>
            <p className="text-sm uppercase tracking-[0.28em] text-indigo-400">Contributor View</p>
            <h2 className="mt-2 text-3xl font-semibold">My machines and earnings</h2>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            {nodes.map((node) => (
              <div key={node.id} className="space-y-3 rounded-[28px] border border-slate-200 bg-white/90 p-4 dark:border-slate-800 dark:bg-slate-950/75">
                <NodeCard node={node} />
                <div className="flex gap-3">
                  <button type="button" onClick={() => apiClient.reclaimNode(node.id).then(load)} className="flex-1 rounded-2xl bg-amber-500 px-4 py-3 text-sm font-semibold text-slate-950">Pause donation</button>
                  <button type="button" onClick={() => apiClient.resumeNode(node.id).then(load)} className="flex-1 rounded-2xl border border-emerald-500/40 px-4 py-3 text-sm font-semibold text-emerald-300">Resume</button>
                </div>
              </div>
            ))}
          </div>
          <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
            <h3 className="text-xl font-semibold">Earnings chart</h3>
            <div className="mt-5 flex h-48 items-end gap-3 rounded-3xl border border-slate-800 p-4">
              {ledger?.by_node.map((item) => <div key={item.node_id} className="flex flex-1 flex-col items-center gap-2"><div className="w-full rounded-t-2xl bg-gradient-to-t from-indigo-500 to-violet-500" style={{ height: `${Math.max(22, item.earned * 3)}px` }} /><span className="text-xs text-slate-500">{item.node_id.slice(0, 8)}</span></div>)}
            </div>
            <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Estimated CPU/RAM hours inferred from current ledger exports.</p>
          </div>
        </div>
        <aside className="space-y-4 rounded-[28px] border border-slate-200 bg-white/90 p-5 dark:border-slate-800 dark:bg-slate-950/75">
          <h3 className="text-xl font-semibold">Join more machines</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">Invite token and join commands for your fleet.</p>
          <div className="rounded-3xl border border-slate-800 p-4"><p className="text-sm text-slate-500">Invite token</p><p className="mt-2 font-mono text-lg text-slate-100">{onboarding?.invite_token ?? '—'}</p></div>
          <div className="rounded-3xl border border-slate-800 p-4"><p className="text-sm text-slate-500">Join command</p><pre className="mt-2 overflow-x-auto text-xs text-slate-200">{onboarding?.join_command ?? '—'}</pre></div>
          <div className="rounded-3xl border border-dashed border-slate-700 p-4 text-center">
            <div className="mx-auto grid h-36 w-36 place-items-center rounded-3xl border border-slate-700 font-mono text-[10px] leading-none text-slate-400">## ##<br /># ## #<br />### ##<br /># ## #<br />## ###</div>
            <p className="mt-3 text-xs text-slate-500">QR-like quick join placeholder for native apps.</p>
          </div>
        </aside>
      </div>
    </section>
  )
}
