import { useEffect, useRef, useState } from 'react'

interface SSEHandlers<T> {
  onMessage?: (event: T) => void
  onError?: () => void
}

export function useSSE<T>(url: string, handlers: SSEHandlers<T> = {}) {
  const [connected, setConnected] = useState(false)
  const [lastEvent, setLastEvent] = useState<T | null>(null)
  const retryRef = useRef(0)

  // Callers pass a fresh object literal for `handlers` on every render, so an
  // effect that depended on it tore down and reopened the EventSource on every
  // render. Because onMessage sets state, each delivered message caused a
  // render, which caused a reconnect, which delivered a message: a
  // self-sustaining loop measured at ~73 connections per second per tab, which
  // scales with the number of open browsers and is invisible to component
  // tests. Hold the handlers in a ref so the stream's lifetime depends only on
  // the URL, while calls always see the newest callbacks.
  const handlersRef = useRef(handlers)
  useEffect(() => {
    handlersRef.current = handlers
  }, [handlers])

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
        handlersRef.current.onMessage?.(parsed)
      }
      source.onerror = () => {
        setConnected(false)
        handlersRef.current.onError?.()
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
  }, [url])

  return { connected, lastEvent }
}
