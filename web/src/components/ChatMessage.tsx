export function ChatMessage({ role, content }: { role: 'user' | 'assistant'; content: string }) {
  const isUser = role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-3xl px-4 py-3 text-sm leading-6 ${isUser ? 'bg-indigo-500 text-white' : 'bg-slate-900 text-slate-100'}`}>
        {content || <span className="text-slate-400">Streaming…</span>}
      </div>
    </div>
  )
}
