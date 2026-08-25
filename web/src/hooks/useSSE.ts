import { useEffect, useRef, useState } from 'react'

interface SSEHandlers<T> {
  onMessage?: (event: T) => void
  onError?: () => void
}

export function useSSE<T>(url: string, handlers: SSEHandlers<T> = {}) {
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<T | null>(null)
  const retryRef = useRef(0)

  useEffect(() => {
    let source: EventSource | null = null
    let reconnectTimer: number | undefined
    let cancelled = false

    const connect = () => {
      source = new EventSource(url)
      source.onopen = () => {
        retryRef.current = 0
        setConnected(true)
      }
      source.onmessage = (event) => {
        const parsed = JSON.parse(event.data) as T
        setLastEvent(parsed)
        handlers.onMessage?.(parsed)
      }
      source.onerror = () => {
        setConnected(false)
        handlers.onError?.()
        source?.close()
        if (!cancelled) {
          const delay = Math.min(1000 * 2 ** retryRef.current, 15000)
          retryRef.current += 1
          reconnectTimer = window.setTimeout(connect, delay)
        }
      }
    }

    connect()

    return () => {
      cancelled = true
      source?.close()
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
    }
  }, [handlers, url])

  return { connected, lastEvent }
}
