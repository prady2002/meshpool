import { useEffect, useMemo, useState } from 'react'
import { apiClient } from '../api/client'
import type { ClusterResponse, Node } from '../api/types'

export function useCluster() {
  const [cluster, setCluster] = useState<ClusterResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true

    const load = async () => {
      try {
        const data = await apiClient.getCluster()
        if (!mounted) return
        setCluster(data)
        setError(null)
      } catch (err) {
        if (!mounted) return
        setError(err instanceof Error ? err.message : 'Unable to load cluster')
      } finally {
        if (mounted) setLoading(false)
      }
    }

    load()
    const timer = window.setInterval(load, 10000)
    return () => {
      mounted = false
      window.clearInterval(timer)
    }
  }, [])

  const onlineNodes = useMemo<Node[]>(() => cluster?.nodes.filter((node) => node.status === 'online') ?? [], [cluster])
  return { cluster, onlineNodes, loading, error, refresh: () => apiClient.getCluster().then(setCluster) }
}
