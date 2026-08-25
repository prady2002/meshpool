import { useCallback } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { useCluster } from './hooks/useCluster'
import { useSSE } from './hooks/useSSE'
import { useAppStore } from './store/appStore'
import { apiClient } from './api/client'
import { AccountPage } from './pages/AccountPage'
import { ContributorPage } from './pages/ContributorPage'
import { FunctionsPage } from './pages/FunctionsPage'
import { JobTimelinePage } from './pages/JobTimelinePage'
import { JobsPage } from './pages/JobsPage'
import { MeshMapPage } from './pages/MeshMapPage'
import { ModelsPage } from './pages/ModelsPage'
import { Playground } from './pages/Playground'
import { StoragePage } from './pages/StoragePage'
import type { EventMessage } from './api/types'

export default function App() {
  const { onlineNodes } = useCluster()
  const setSseConnected = useAppStore((s) => s.setSseConnected)
  const pushEvent = useAppStore((s) => s.pushEvent)

  // Stable identities: useSSE re-subscribes whenever `handlers` changes, so an
  // inline object literal here would tear the stream down on every render.
  const onMessage = useCallback((event: EventMessage) => {
    pushEvent(event)
    setSseConnected(true)
  }, [pushEvent, setSseConnected])
  const onError = useCallback(() => setSseConnected(false), [setSseConnected])
  useSSE<EventMessage>(`${apiClient.baseUrl}/api/v1/events`, { onMessage, onError })

  return (
    <Routes>
      <Route element={<Layout nodes={onlineNodes} />}>
        {/* Playground first: a new user's landing state is a working prompt box
            against a real pooled model, not a dashboard and not a wizard. */}
        <Route index element={<Navigate to="/playground" replace />} />
        <Route path="/playground" element={<Playground />} />
        <Route path="/mesh" element={<MeshMapPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/functions" element={<FunctionsPage />} />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="/jobs/:jobId/timeline" element={<JobTimelinePage />} />
        <Route path="/storage" element={<StoragePage />} />
        <Route path="/contribute" element={<ContributorPage />} />
        <Route path="/account" element={<AccountPage />} />
        <Route path="*" element={<Navigate to="/playground" replace />} />
      </Route>
    </Routes>
  )
}
