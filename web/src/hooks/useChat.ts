import { useCallback, useState } from 'react'
import { apiClient } from '../api/client'
import type { ChatRunResult } from '../api/types'

interface ChatOptions {
  modelId: string
  temperature: number
  maxNewTokens: number
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export function useChat() {
  const [messages, setMessages] = useState<ChatTurn[]>([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastRun, setLastRun] = useState<ChatRunResult | null>(null)

  const appendToAssistant = useCallback((text: string) => {
    setMessages((current) => {
      const next = [...current]
      const last = next[next.length - 1]
      if (last?.role === 'assistant') {
        next[next.length - 1] = { ...last, content: `${last.content}${text}` }
      }
      return next
    })
  }, [])

  const sendMessage = async (prompt: string, options: ChatOptions) => {
    setMessages((current) => [
      ...current,
      { role: 'user', content: prompt },
      { role: 'assistant', content: '' },
    ])
    setStreaming(true)
    setError(null)
    setLastRun(null)

    try {
      let sawDelta = false
      for await (const event of apiClient.chat(prompt, {
        model_id: options.modelId,
        temperature: options.temperature,
        max_new_tokens: options.maxNewTokens,
      })) {
        // An error can arrive mid-stream, after the 200 has already been sent.
        // Surfacing it is what turns a silently truncated answer into a visible
        // failure.
        if (event.error) throw new Error(event.error)
        if (typeof event.delta === 'string' && event.delta) {
          sawDelta = true
          appendToAssistant(event.delta)
        } else if (event.done && event.result) {
          setLastRun(event.result)
          // Backends that generate without emitting incremental tokens (a
          // whole-model replica behind Ollama) report their text only in this
          // final frame. Render it rather than leaving the bubble empty -- and
          // report the absence of streaming honestly instead of faking a drip.
          if (!sawDelta && event.result.text) appendToAssistant(event.result.text)
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Streaming failed'
      setError(message)
      setMessages((current) => {
        const next = [...current]
        const last = next[next.length - 1]
        if (last?.role === 'assistant' && !last.content) {
          next[next.length - 1] = { ...last, content: 'MeshPool could not complete this response.' }
        }
        return next
      })
    } finally {
      setStreaming(false)
    }
  }

  return { messages, streaming, error, lastRun, sendMessage }
}
