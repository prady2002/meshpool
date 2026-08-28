import * as Progress from '@radix-ui/react-progress'

export function ProgressBar({ value }: { value: number }) {
  return (
    <Progress.Root className="relative h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800" value={value}>
      <Progress.Indicator className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-transform" style={{ transform: `translateX(-${100 - value}%)` }} />
    </Progress.Root>
  )
}
