import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as TabsPrimitive from '@radix-ui/react-tabs'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { type ComponentPropsWithoutRef, type JSX, type ReactNode } from 'react'
import { cn } from '../../utils'

export type MetricTier = 'measured' | 'calibrated projection' | 'cited'

type PrimitiveProps<T extends keyof JSX.IntrinsicElements> = ComponentPropsWithoutRef<T>

export function Button({ className, ...props }: PrimitiveProps<'button'>) {
  return <button className={cn('inline-flex items-center justify-center gap-2 rounded-2xl bg-[var(--color-accent)] px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-[var(--color-accent-strong)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-bg)] disabled:cursor-not-allowed disabled:opacity-50 motion-reduce:transition-none', className)} {...props} />
}

export function Input({ className, ...props }: PrimitiveProps<'input'>) {
  return <input className={cn('w-full rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2.5 text-sm text-[var(--color-text)] outline-none transition placeholder:text-[var(--color-muted)] focus-visible:ring-2 focus-visible:ring-[var(--color-focus)] motion-reduce:transition-none', className)} {...props} />
}

export function Select({ className, children, ...props }: PrimitiveProps<'select'>) {
  return <select className={cn('w-full rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2.5 text-sm text-[var(--color-text)] outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)]', className)} {...props}>{children}</select>
}

export const Dialog = DialogPrimitive.Root
export const DialogTrigger = DialogPrimitive.Trigger
export function DialogContent({ className, children, title, description, ...props }: DialogPrimitive.DialogContentProps & { title: string; description?: string }) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/55 backdrop-blur-sm" />
      <DialogPrimitive.Content className={cn('fixed left-1/2 top-[15vh] z-50 w-[min(720px,92vw)] -translate-x-1/2 rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-panel)] p-5 text-[var(--color-text)] shadow-[var(--shadow-elevated)] outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-focus)]', className)} {...props}>
        <DialogPrimitive.Title className="text-lg font-semibold">{title}</DialogPrimitive.Title>
        {description ? <DialogPrimitive.Description className="mt-1 text-sm text-[var(--color-muted)]">{description}</DialogPrimitive.Description> : null}
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export const Sheet = Dialog
export function SheetContent(props: ComponentPropsWithoutRef<typeof DialogContent>) {
  return <DialogContent {...props} className={cn('left-auto right-3 top-3 h-[calc(100vh-1.5rem)] w-[min(420px,92vw)] translate-x-0', props.className)} />
}

export const TooltipProvider = TooltipPrimitive.Provider
export function Tooltip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content className="z-50 rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] px-3 py-2 text-xs text-[var(--color-text)] shadow-[var(--shadow-elevated)]" sideOffset={8}>
          {label}<TooltipPrimitive.Arrow className="fill-[var(--color-panel)]" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  )
}

export function Toast({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'success' | 'danger' }) {
  const toneClass = tone === 'success' ? 'border-emerald-500/40 text-emerald-200' : tone === 'danger' ? 'border-rose-500/40 text-rose-200' : 'border-[var(--color-border)] text-[var(--color-text)]'
  return <div role="status" className={cn('rounded-2xl border bg-[var(--color-panel)] px-4 py-3 text-sm shadow-[var(--shadow-soft)]', toneClass)}>{children}</div>
}

export const Tabs = TabsPrimitive.Root
export const TabsList = TabsPrimitive.List
export const TabsTrigger = TabsPrimitive.Trigger
export const TabsContent = TabsPrimitive.Content

export function Table({ className, ...props }: PrimitiveProps<'table'>) {
  return <table className={cn('w-full border-collapse text-left text-sm', className)} {...props} />
}

export function Badge({ className, ...props }: PrimitiveProps<'span'>) {
  return <span className={cn('inline-flex items-center rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1 text-xs font-semibold text-[var(--color-muted)]', className)} {...props} />
}

export function Card({ className, ...props }: PrimitiveProps<'section'>) {
  return <section className={cn('rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-panel)] p-5 shadow-[var(--shadow-soft)]', className)} {...props} />
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cn('animate-pulse rounded-2xl bg-[var(--color-skeleton)] motion-reduce:animate-none', className)} />
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return <div className="rounded-[var(--radius-xl)] border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] px-6 py-10 text-center"><h3 className="text-lg font-semibold text-[var(--color-text)]">{title}</h3><p className="mx-auto mt-2 max-w-md text-sm text-[var(--color-muted)]">{body}</p>{action ? <div className="mt-4">{action}</div> : null}</div>
}

export function CodeBlock({ code, className }: { code: string; className?: string }) {
  return <pre className={cn('overflow-x-auto rounded-2xl border border-[var(--color-border)] bg-slate-950 p-4 text-xs text-slate-100', className)}><code>{code}</code></pre>
}

export function Toolbar({ className, ...props }: PrimitiveProps<'div'>) {
  return <div role="toolbar" className={cn('flex flex-wrap items-center gap-2 rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-panel)] p-2', className)} {...props} />
}

export function Metric({ label, value, unit, tier, className }: { label: string; value: number | string; unit?: string; tier: MetricTier; className?: string }) {
  if (tier == null) throw new Error('Metric requires a measurement tier label')
  return <div className={cn('rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4', className)}><p className="text-xs font-medium uppercase tracking-[0.16em] text-[var(--color-muted)]">{label}</p><p className="mt-2 text-2xl font-semibold text-[var(--color-text)]">{value}{unit ? <span className="ml-1 text-sm text-[var(--color-muted)]">{unit}</span> : null}</p><p className="mt-1 text-xs text-[var(--color-muted)]">{tier}</p></div>
}
