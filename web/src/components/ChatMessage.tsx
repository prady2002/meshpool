export function ChatMessage({ role, content }: { role: 'user' | 'assistant'; content: string }) {
  const isUser = role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-3xl px-4 py-3 text-sm leading-6 ${isUser ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-slate-900 dark:bg-slate-900 dark:text-slate-100'}`}>
        {content || <span className="text-slate-600 dark:text-slate-400">Streaming…</span>}
      </div>
    </div>
  )
}
