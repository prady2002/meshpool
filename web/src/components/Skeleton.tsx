import { cn } from '../utils'

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-2xl bg-slate-800/80', className)} aria-hidden="true" />
}
