import { useState } from 'react'
import { apiClient } from '../api/client'

interface ChatOptions {
  modelId: string
  temperature: number
  maxNewTokens: number
}

export function useChat() {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const sendMessage = async (prompt: string, options: ChatOptions) => {
    setMessages((current) => [...current, { role: 'user', content: prompt }, { role: 'assistant', content: '' }])
    setStreaming(true)
    setError(null)

    try {
      const reader = await apiClient.chat(prompt, {
        model_id: options.modelId,
        temperature: options.temperature,
        max_new_tokens: options.maxNewTokens,
      })

      let complete = false
      while (!complete) {
        const { done, value } = await reader.read()
        if (done) {
          complete = true
        } else if (value) {
          setMessages((current) => {
            const next = [...current]
            const last = next[next.length - 1]
            if (last?.role === 'assistant') {
              next[next.length - 1] = { ...last, content: `${last.content}${value}` }
            }
            return next
          })
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Streaming failed'
      setError(message)
      setMessages((current) => {
        const next = [...current]
        const last = next[next.length - 1]
        if (last?.role === 'assistant') {
          next[next.length - 1] = { ...last, content: 'MeshPool could not complete this response.' }
        }
        return next
      })
    } finally {
      setStreaming(false)
    }
  }

  return { messages, streaming, error, sendMessage }
}
