import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { BatchJob, ConstellationGroup, ModelDeployment, Node, NodeClass, NodeStatus } from '../api/types'
import { aggregateNodes, groupConstellations, inferNodeClass } from '../mesh'
import { Metric } from './ui'

// ── canvas dimensions ──────────────────────────────────────────────────────────
const W = 1000
const H = 560

// ── visual palette ─────────────────────────────────────────────────────────────
const STATUS_COLOR: Record<NodeStatus, string> = {
  online:   '#22c55e',
  draining: '#f59e0b',
  offline:  '#6b7280',
}
const STATUS_GLOW: Record<NodeStatus, string> = {
  online:   'rgba(34,197,94,0.28)',
  draining: 'rgba(245,158,11,0.2)',
  offline:  'rgba(107,114,128,0.1)',
}
const PIPELINE_COLORS = ['#818cf8', '#a78bfa', '#c084fc']

// ── simulation node ────────────────────────────────────────────────────────────
interface SNode {
  id: string
  x: number; y: number
  vx: number; vy: number
  r: number
  status: NodeStatus
  count: number
  label: string
  raw: Node[]
  nodeClass: NodeClass | null   // non-null only when raw.length === 1
}

interface SEdge {
  a: number
  b: number
  color: string
}

interface Particle {
  sx: number; sy: number
  tx: number; ty: number
  t: number; speed: number
}

// Token traversing model pipeline stages (ingress → relay → egress)
interface PipelineTokenParticle {
  modelIdx: number
  stageIdx: number
  progress: number  // 0..1 within the current stage
  speed: number
}

// Failover event: a node went offline while another took over
interface FailoverAnim {
  failedSNodeId: string
  spareSNodeId: string | null
  startMs: number  // performance.now() at detection
}

/**
 * Cap on rows rendered in the accessible table. Each row is 9 cells, so an
 * uncapped table on a 5000-node mesh is 45,000 DOM elements -- enough to make
 * the page janky purely from layout. The caption states plainly how many rows
 * are hidden rather than pretending the list is complete.
 */
const TABLE_ROW_LIMIT = 80

// ── node class canvas marker ───────────────────────────────────────────────────

function drawNodeClassMarker(
  ctx: CanvasRenderingContext2D,
  nodeClass: NodeClass,
  cx: number, cy: number, r: number,
): void {
  switch (nodeClass) {
    case 'anchor': {
      // 4-point gold star — stable, dedicated hardware
      ctx.save()
      ctx.translate(cx, cy)
      const sr = r * 0.42
      ctx.beginPath()
      for (let p = 0; p < 4; p++) {
        const a = (p / 4) * Math.PI * 2 - Math.PI / 4
        const b = a + Math.PI / 4
        if (p === 0) ctx.moveTo(Math.cos(a) * sr, Math.sin(a) * sr)
        else ctx.lineTo(Math.cos(a) * sr, Math.sin(a) * sr)
        ctx.lineTo(Math.cos(b) * sr * 0.38, Math.sin(b) * sr * 0.38)
      }
      ctx.closePath()
      ctx.fillStyle = '#fbbf24'; ctx.fill()
      ctx.restore()
      break
    }
    case 'relay': {
      // Forward-pointing cyan triangle — routing specialist
      ctx.save()
      ctx.translate(cx, cy)
      const tr = r * 0.38
      ctx.beginPath()
      ctx.moveTo(tr * 0.9, 0)
      ctx.lineTo(-tr * 0.55, -tr * 0.8)
      ctx.lineTo(-tr * 0.55, tr * 0.8)
      ctx.closePath()
      ctx.fillStyle = '#38bdf8'; ctx.fill()
      ctx.restore()
      break
    }
    case 'drifter': {
      // Dashed outer ring — transient / uncertain node
      ctx.save()
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.arc(cx, cy, r + 5, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(156,163,175,0.55)'; ctx.lineWidth = 1.5; ctx.stroke()
      ctx.setLineDash([])
      ctx.restore()
      break
    }
    // 'constellation' — default circle, no extra marker
  }
}

// ── physics helpers ────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v
}

function buildSNodes(
  items: ReturnType<typeof aggregateNodes>['items'],
): SNode[] {
  const cx = W / 2
  const cy = H / 2
  const n = items.length || 1
  return items.map((item, i) => {
    const angle = (i / n) * Math.PI * 2 - Math.PI / 2
    const ring = Math.floor(i / 8)
    const spread = 140 + ring * 80
    const r = clamp(10 + Math.cbrt(item.totalCompute) + Math.log1p(item.count) * 4, 14, 40)
    return {
      id: item.id,
      x: cx + Math.cos(angle) * spread,
      y: cy + Math.sin(angle) * spread,
      vx: 0, vy: 0,
      r,
      status: item.status,
      count: item.count,
      label: item.label,
      raw: item.nodes,
      nodeClass: item.nodes.length === 1 ? inferNodeClass(item.nodes[0]) : null,
    }
  })
}

function buildSEdges(nodes: SNode[], models: ModelDeployment[]): SEdge[] {
  const edges: SEdge[] = []
  models.forEach((m, mi) => {
    const color = PIPELINE_COLORS[mi % PIPELINE_COLORS.length]
    const sorted = [...m.partition].sort((a, b) => a.start_layer - b.start_layer)
    for (let k = 0; k + 1 < sorted.length; k++) {
      const ai = nodes.findIndex((sn) => sn.raw.some((r) => r.id === sorted[k].node_id))
      const bi = nodes.findIndex((sn) => sn.raw.some((r) => r.id === sorted[k + 1].node_id))
      if (ai >= 0 && bi >= 0 && ai !== bi) edges.push({ a: ai, b: bi, color })
    }
  })
  return edges
}

function tickSim(nodes: SNode[], edges: SEdge[], alpha: number) {
  const cx = W / 2; const cy = H / 2

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].x - nodes[i].x; const dy = nodes[j].y - nodes[i].y
      const d2 = dx * dx + dy * dy
      if (d2 < 0.01) { nodes[i].x += 1; continue }
      const d = Math.sqrt(d2)
      const minD = (nodes[i].r + nodes[j].r) * 1.8 + 18
      if (d < minD) {
        const k = ((minD - d) / d) * 0.5 * alpha
        nodes[i].vx -= dx * k; nodes[i].vy -= dy * k
        nodes[j].vx += dx * k; nodes[j].vy += dy * k
      }
    }
  }

  for (const e of edges) {
    if (e.a >= nodes.length || e.b >= nodes.length) continue
    const na = nodes[e.a]; const nb = nodes[e.b]
    const dx = nb.x - na.x; const dy = nb.y - na.y
    const d = Math.sqrt(dx * dx + dy * dy) || 1
    const rest = na.r + nb.r + 80
    const k = ((d - rest) / d) * 0.06 * alpha
    na.vx += dx * k; na.vy += dy * k
    nb.vx -= dx * k; nb.vy -= dy * k
  }

  for (const n of nodes) {
    n.vx += (cx - n.x) * 0.005 * alpha
    n.vy += (cy - n.y) * 0.005 * alpha
    n.vx *= 0.82; n.vy *= 0.82
    n.x += n.vx; n.y += n.vy
    n.x = clamp(n.x, n.r + 8, W - n.r - 8)
    n.y = clamp(n.y, n.r + 8, H - n.r - 8)
  }
}

// ── component ──────────────────────────────────────────────────────────────────

export interface MeshMapProps {
  nodes: Node[]
  jobs?: BatchJob[]
  models?: ModelDeployment[]
}

export function MeshMap({ nodes, jobs = [], models = [] }: MeshMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  // Sim state lives in refs so physics updates never trigger React re-renders.
  const sNodesRef   = useRef<SNode[]>([])
  const sEdgesRef   = useRef<SEdge[]>([])
  const alphaRef    = useRef(1.0)
  const particlesRef = useRef<Particle[]>([])
  const rafRef      = useRef(0)
  const frameRef    = useRef(0)
  const prevKeyRef  = useRef('')
  const modelsRef              = useRef<ModelDeployment[]>([])
  const constellationGroupsRef = useRef<ConstellationGroup[]>([])
  const pipelineTokensRef      = useRef<PipelineTokenParticle[]>([])
  const failoverAnimsRef       = useRef<FailoverAnim[]>([])
  const prevStatusRef          = useRef<Map<string, NodeStatus>>(new Map())

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [showTable, setShowTable]   = useState(false)

  // Mirror reactive values into refs so the RAF closure stays fresh without
  // restarting the loop on every render.
  const selectedIdRef      = useRef<string | null>(null)
  selectedIdRef.current    = selectedId
  const activeNodeIdsRef   = useRef(new Set<string>())
  const aggregateRef       = useRef(aggregateNodes([]))

  const aggregate = useMemo(() => aggregateNodes(nodes), [nodes])
  aggregateRef.current = aggregate

  const constellations = useMemo(() => groupConstellations(nodes), [nodes])
  constellationGroupsRef.current = constellations
  modelsRef.current = models

  const activeNodeIds = useMemo<Set<string>>(() => {
    const s = new Set<string>()
    for (const job of jobs) {
      for (const unit of (job.units ?? [])) {
        if (unit.status === 'running' && unit.assigned_node) s.add(unit.assigned_node)
      }
    }
    return s
  }, [jobs])
  activeNodeIdsRef.current = activeNodeIds

  // reducedMotion is stable for the session; browsers require page reload to apply it
  const reducedMotion = typeof window !== 'undefined'
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : false

  // ── Effect 1: (re)init sim when the node SET changes ──────────────────────
  useEffect(() => {
    const key = aggregate.items.map((i) => i.id).sort().join('\x00')
    if (key !== prevKeyRef.current) {
      prevKeyRef.current = key
      sNodesRef.current  = buildSNodes(aggregate.items)
      alphaRef.current   = 1.0
    } else {
      // Same nodes: update mutable fields without resetting positions/velocities.
      for (const item of aggregate.items) {
        const sn = sNodesRef.current.find((n) => n.id === item.id)
        if (sn) { sn.status = item.status; sn.count = item.count; sn.raw = item.nodes }
      }
    }
    sEdgesRef.current = buildSEdges(sNodesRef.current, models)

    // Detect failover transitions (online → draining/offline) and schedule
    // a short animation so the reliability story is rendered visually.
    const snapshot = new Map<string, NodeStatus>()
    for (const item of aggregate.items) {
      for (const raw of item.nodes) {
        snapshot.set(raw.id, raw.status)
        const prev = prevStatusRef.current.get(raw.id)
        if (prev === 'online' && raw.status !== 'online') {
          const failedSNode = sNodesRef.current.find((sn) => sn.raw.some((r) => r.id === raw.id))
          if (failedSNode) {
            const spare = [...sNodesRef.current]
              .filter((sn) => sn.status === 'online' && sn.id !== failedSNode.id)
              .sort((a, b) =>
                Math.hypot(a.x - failedSNode.x, a.y - failedSNode.y) -
                Math.hypot(b.x - failedSNode.x, b.y - failedSNode.y),
              )[0]
            failoverAnimsRef.current.push({
              failedSNodeId: failedSNode.id,
              spareSNodeId: spare?.id ?? null,
              startMs: performance.now(),
            })
          }
        }
      }
    }
    prevStatusRef.current = snapshot
  }, [aggregate, models])

  // ── Effect 2: RAF animation loop (started once; restarts only if
  //   reducedMotion changes, which effectively never happens) ──────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    frameRef.current = 0

    const draw = () => {
      // Physics
      if (alphaRef.current > 0.001) {
        tickSim(sNodesRef.current, sEdgesRef.current, alphaRef.current)
        alphaRef.current = Math.max(0, alphaRef.current * 0.985)
      }

      // Spawn work particles (one random spawn per active node per frame)
      if (!reducedMotion) {
        for (const sn of sNodesRef.current) {
          if (sn.status !== 'online') continue
          if (!sn.raw.some((r) => activeNodeIdsRef.current.has(r.id))) continue
          if (Math.random() < 0.04) {
            particlesRef.current.push({
              sx: W / 2, sy: H / 2, tx: sn.x, ty: sn.y, t: 0,
              speed: 0.008 + Math.random() * 0.01,
            })
          }
        }
        particlesRef.current = particlesRef.current
          .map((p) => ({ ...p, t: p.t + p.speed }))
          .filter((p) => p.t < 1)
        if (particlesRef.current.length > 100) {
          particlesRef.current = particlesRef.current.slice(-100)
        }
      }

      // Canvas draw
      if (ctx) {
        ctx.clearRect(0, 0, W, H)
        ctx.fillStyle = '#020617'
        ctx.fillRect(0, 0, W, H)

        if (sNodesRef.current.length === 0) {
          ctx.fillStyle  = 'rgba(148,163,184,0.6)'
          ctx.font       = '500 15px sans-serif'
          ctx.textAlign  = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText('No nodes registered — join the mesh to see topology here', W / 2, H / 2 - 14)
          ctx.fillStyle = 'rgba(100,116,139,0.45)'
          ctx.font      = '13px sans-serif'
          ctx.fillText('Use the Contribute page to add your first machine', W / 2, H / 2 + 14)
        } else {
          // Constellation halos — drawn before nodes so they appear behind
          for (const cg of constellationGroupsRef.current) {
            if (cg.nodeIds.length < 2) continue
            const cgSet = new Set(cg.nodeIds)
            const members = sNodesRef.current.filter((sn) => sn.raw.some((r) => cgSet.has(r.id)))
            if (members.length < 2) continue
            const hcx = members.reduce((s, n) => s + n.x, 0) / members.length
            const hcy = members.reduce((s, n) => s + n.y, 0) / members.length
            const maxR = members.reduce((mx, n) =>
              Math.max(mx, Math.hypot(n.x - hcx, n.y - hcy) + n.r), 0) + 28
            if (maxR < 44) continue
            const grad = ctx.createRadialGradient(hcx, hcy, Math.max(maxR - 36, 0), hcx, hcy, maxR)
            grad.addColorStop(0, 'rgba(129,140,248,0.07)')
            grad.addColorStop(1, 'rgba(129,140,248,0)')
            ctx.beginPath(); ctx.arc(hcx, hcy, maxR, 0, Math.PI * 2); ctx.fillStyle = grad; ctx.fill()
            ctx.beginPath(); ctx.arc(hcx, hcy, maxR - 3, 0, Math.PI * 2)
            ctx.strokeStyle = 'rgba(129,140,248,0.18)'; ctx.lineWidth = 1
            ctx.setLineDash([4, 6]); ctx.stroke(); ctx.setLineDash([])
          }

          // Pipeline edges
          for (const e of sEdgesRef.current) {
            const na = sNodesRef.current[e.a]; const nb = sNodesRef.current[e.b]
            if (!na || !nb) continue
            ctx.beginPath(); ctx.strokeStyle = e.color; ctx.lineWidth = 2
            ctx.moveTo(na.x, na.y); ctx.lineTo(nb.x, nb.y); ctx.stroke()
          }

          // Proximity edges (full-mesh for ≤5 nodes; proximity-gated otherwise)
          if (sNodesRef.current.length > 1 && sNodesRef.current.length <= 30) {
            ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(100,116,139,0.15)'
            for (let i = 0; i < sNodesRef.current.length; i++) {
              for (let j = i + 1; j < sNodesRef.current.length; j++) {
                const na = sNodesRef.current[i]; const nb = sNodesRef.current[j]
                const dist = Math.hypot(nb.x - na.x, nb.y - na.y)
                if (sNodesRef.current.length <= 5 || dist < 220) {
                  ctx.beginPath(); ctx.moveTo(na.x, na.y); ctx.lineTo(nb.x, nb.y); ctx.stroke()
                }
              }
            }
          }

          // Work particles
          for (const p of particlesRef.current) {
            const px = p.sx + (p.tx - p.sx) * p.t
            const py = p.sy + (p.ty - p.sy) * p.t
            const a  = (Math.sin(p.t * Math.PI) * 0.9).toFixed(2)
            ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2)
            ctx.fillStyle = `rgba(99,102,241,${a})`; ctx.fill()
          }

          // Pipeline token particles — bright amber dots following the pipeline
          // stage→stage to visualise the t_token ≈ Σ compute_i + (N-1)×RTT path
          if (!reducedMotion) {
            const nextTokens: PipelineTokenParticle[] = []
            // Advance existing tokens
            for (const pt of pipelineTokensRef.current) {
              const m = modelsRef.current[pt.modelIdx]
              if (!m) continue
              const sorted = [...m.partition].sort((a, b) => a.start_layer - b.start_layer)
              let stageIdx = pt.stageIdx
              let progress = pt.progress + pt.speed
              if (progress >= 1) { progress = 0; stageIdx++ }
              if (stageIdx >= sorted.length - 1) continue
              const aId = sorted[stageIdx].node_id
              const bId = sorted[stageIdx + 1].node_id
              const aSNode = sNodesRef.current.find((sn) => sn.raw.some((r) => r.id === aId))
              const bSNode = sNodesRef.current.find((sn) => sn.raw.some((r) => r.id === bId))
              if (aSNode && bSNode) {
                const px = aSNode.x + (bSNode.x - aSNode.x) * progress
                const py = aSNode.y + (bSNode.y - aSNode.y) * progress
                ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2)
                ctx.fillStyle = '#fcd34d'
                ctx.shadowColor = '#f59e0b'; ctx.shadowBlur = 6; ctx.fill(); ctx.shadowBlur = 0
              }
              nextTokens.push({ modelIdx: pt.modelIdx, stageIdx, progress, speed: pt.speed })
            }
            // Spawn new tokens for running models
            for (let mi = 0; mi < modelsRef.current.length; mi++) {
              const m = modelsRef.current[mi]
              if (m.status !== 'running' || !m.tokens_per_sec || m.partition.length < 2) continue
              const spawnP = Math.min(0.12, (m.tokens_per_sec) / 60)
              if (Math.random() < spawnP) {
                nextTokens.push({ modelIdx: mi, stageIdx: 0, progress: 0, speed: 0.018 + Math.random() * 0.012 })
              }
            }
            if (nextTokens.length > 60) nextTokens.splice(0, nextTokens.length - 60)
            pipelineTokensRef.current = nextTokens
          }

          // Nodes
          const frame = frameRef.current
          for (const sn of sNodesRef.current) {
            const isSelected = sn.id === selectedIdRef.current
            const isActive   = sn.raw.some((r) => activeNodeIdsRef.current.has(r.id))

            if (!reducedMotion && sn.status === 'online') {
              const pulse = Math.sin(frame / 20 + sn.x * 0.01) * 4
              ctx.beginPath(); ctx.arc(sn.x, sn.y, sn.r + 8 + pulse, 0, Math.PI * 2)
              ctx.strokeStyle = STATUS_GLOW[sn.status]; ctx.lineWidth = 7; ctx.stroke()
            }
            if (isActive) {
              ctx.beginPath(); ctx.arc(sn.x, sn.y, sn.r + 5, 0, Math.PI * 2)
              ctx.strokeStyle = 'rgba(99,102,241,0.7)'; ctx.lineWidth = 2.5; ctx.stroke()
            }
            if (isSelected) {
              ctx.beginPath(); ctx.arc(sn.x, sn.y, sn.r + 7, 0, Math.PI * 2)
              ctx.strokeStyle = '#e0e7ff'; ctx.lineWidth = 3; ctx.stroke()
            }

            ctx.beginPath(); ctx.arc(sn.x, sn.y, sn.r, 0, Math.PI * 2)
            ctx.globalAlpha = sn.status === 'offline' ? 0.45 : 0.92
            ctx.fillStyle   = STATUS_COLOR[sn.status]; ctx.fill()
            ctx.globalAlpha = 1

            // Node class marker (only for individual, non-aggregated nodes)
            if (sn.nodeClass && sn.raw.length === 1) {
              drawNodeClassMarker(ctx, sn.nodeClass, sn.x, sn.y, sn.r)
            }

            if (sn.count > 1) {
              const fs = clamp(sn.r * 0.65, 9, 14)
              ctx.font = `bold ${fs}px sans-serif`; ctx.fillStyle = '#0f172a'
              ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
              ctx.fillText(`${sn.count}`, sn.x, sn.y)
            }

            ctx.font = '11px sans-serif'; ctx.fillStyle = '#94a3b8'
            ctx.textAlign = 'center'; ctx.textBaseline = 'top'
            const lbl = sn.label.length > 20 ? `${sn.label.slice(0, 18)}…` : sn.label
            ctx.fillText(lbl, sn.x, sn.y + sn.r + 5)
          }

          // Failover animations — drawn after nodes so rings appear on top
          const nowMs = performance.now()
          failoverAnimsRef.current = failoverAnimsRef.current.filter((anim) => {
            const elapsed = nowMs - anim.startMs
            if (elapsed >= 3000) return false
            const intensity = 1 - elapsed / 3000
            const failedSNode = sNodesRef.current.find((sn) => sn.id === anim.failedSNodeId)
            if (failedSNode) {
              ctx.beginPath(); ctx.arc(failedSNode.x, failedSNode.y, failedSNode.r + 14 * intensity, 0, Math.PI * 2)
              ctx.strokeStyle = `rgba(239,68,68,${0.7 * intensity})`; ctx.lineWidth = 2.5; ctx.stroke()
            }
            if (anim.spareSNodeId) {
              const spareSNode = sNodesRef.current.find((sn) => sn.id === anim.spareSNodeId)
              if (spareSNode) {
                ctx.beginPath(); ctx.arc(spareSNode.x, spareSNode.y, spareSNode.r + 12 * intensity, 0, Math.PI * 2)
                ctx.strokeStyle = `rgba(245,158,11,${0.8 * intensity})`; ctx.lineWidth = 2; ctx.stroke()
                if (failedSNode && intensity > 0.2) {
                  ctx.beginPath(); ctx.moveTo(failedSNode.x, failedSNode.y); ctx.lineTo(spareSNode.x, spareSNode.y)
                  ctx.strokeStyle = `rgba(245,158,11,${0.3 * intensity})`; ctx.lineWidth = 1
                  ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([])
                }
              }
            }
            return true
          })

          ctx.font = '11px sans-serif'; ctx.fillStyle = 'rgba(100,116,139,0.55)'
          ctx.textAlign = 'left'; ctx.textBaseline = 'top'
          ctx.fillText(
            `Topology view (${aggregateRef.current.mode}) · positions are synthetic`,
            14, 10,
          )
        }
      }

      frameRef.current++
      if (!reducedMotion) rafRef.current = requestAnimationFrame(draw)
    }

    draw()
    if (reducedMotion) return undefined

    const onVis = () => {
      if (document.hidden) cancelAnimationFrame(rafRef.current)
      else rafRef.current = requestAnimationFrame(draw)
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      cancelAnimationFrame(rafRef.current)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [reducedMotion])
  // sNodesRef / sEdgesRef / alphaRef are mutated by Effect 1;
  // selectedIdRef / activeNodeIdsRef are updated each render.
  // None of those need to be deps — they're refs or module-level constants.

  // ── click / keyboard ──────────────────────────────────────────────────────

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = (e.clientX - rect.left) * (W / rect.width)
    const my = (e.clientY - rect.top)  * (H / rect.height)
    const hit = sNodesRef.current.find((n) => {
      const dx = n.x - mx; const dy = n.y - my
      return dx * dx + dy * dy <= (n.r + 6) * (n.r + 6)
    })
    setSelectedId(hit?.id ?? null)
  }, [])

  const handleCanvasKeyDown = useCallback((e: React.KeyboardEvent<HTMLCanvasElement>) => {
    const ns = sNodesRef.current
    if (!ns.length) return
    const idx = ns.findIndex((n) => n.id === selectedId)
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault(); setSelectedId(ns[(idx + 1) % ns.length]?.id ?? null)
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault(); setSelectedId(ns[(idx - 1 + ns.length) % ns.length]?.id ?? null)
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      setSelectedId(idx < 0 && ns[0] ? ns[0].id : null)
    } else if (e.key === 'Escape') {
      e.preventDefault(); setSelectedId(null)
    }
  }, [selectedId])

  // ── derived render data ───────────────────────────────────────────────────

  const selectedSNode = sNodesRef.current.find((n) => n.id === selectedId) ?? null
  const selectedNode  = selectedSNode?.raw[0] ?? null

  const nodeModels = useMemo(
    () => models.filter((m) =>
      selectedNode && m.partition.some((p) => p.node_id === selectedNode.id)
    ),
    [models, selectedNode],
  )

  const sortedNodes = useMemo(
    () => [...nodes].sort((a, b) => {
      const ord: Record<string, number> = { online: 0, draining: 1, offline: 2 }
      return (ord[a.status] ?? 3) - (ord[b.status] ?? 3)
    }),
    [nodes],
  )

  // node id -> simulated-item id, built once per aggregate.
  // The table previously did `sNodes.find(n => n.raw.some(r => r.id === node.id))`
  // INSIDE its row loop, which is O(items x nodes-per-item) per row and becomes
  // tens of millions of operations per render on a large mesh.
  const itemIdByNodeId = useMemo(() => {
    const m = new Map<string, string>()
    for (const item of aggregate.items) {
      for (const raw of item.nodes) m.set(raw.id, item.id)
    }
    return m
  }, [aggregate])

  const visibleRows = sortedNodes.slice(0, TABLE_ROW_LIMIT)

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">
      {/* ── canvas + detail panel ────────────────────────────────────────── */}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-950/70">
          <canvas
            ref={canvasRef}
            width={W}
            height={H}
            className="h-[560px] w-full cursor-pointer"
            role="img"
            aria-label={
              nodes.length === 0
                ? 'Mesh map: no nodes registered'
                : `Mesh map: ${nodes.length} node${nodes.length !== 1 ? 's' : ''}, ${aggregate.mode} view`
            }
            tabIndex={0}
            onClick={handleCanvasClick}
            onKeyDown={handleCanvasKeyDown}
          />
          {/* Legend */}
          <div className="pointer-events-none absolute bottom-3 left-3 space-y-1.5">
            <div className="flex gap-2">
              {(['online', 'draining', 'offline'] as const).map((s) => (
                <span key={s} className="flex items-center gap-1.5 rounded-full bg-slate-950/80 px-2.5 py-1 text-xs text-slate-300">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: STATUS_COLOR[s] }} aria-hidden="true" />
                  {s}
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              {([
                { cls: 'anchor', icon: '★', color: '#fbbf24' },
                { cls: 'relay', icon: '▶', color: '#38bdf8' },
                { cls: 'drifter', icon: '⊙', color: '#9ca3af' },
              ] as const).map(({ cls, icon, color }) => (
                <span key={cls} className="flex items-center gap-1 rounded-full bg-slate-950/80 px-2.5 py-1 text-xs text-slate-300">
                  <span style={{ color }} aria-hidden="true">{icon}</span>
                  {cls}
                </span>
              ))}
            </div>
          </div>
          {activeNodeIds.size > 0 && (
            <Metric label="Nodes running work" value={activeNodeIds.size} tier="measured" density="compact" className="pointer-events-none absolute right-3 top-3 border-indigo-500/30 bg-indigo-500/20 text-indigo-300" />
          )}
        </div>

        {/* Detail panel */}
        <aside className="rounded-3xl border border-slate-800 bg-slate-950/70 p-4">
          {selectedSNode && selectedNode ? (
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-500">
                    {selectedSNode.raw.length > 1 ? 'Group detail' : 'Node detail'}
                  </p>
                  <h3 className="mt-1 font-semibold text-slate-100">
                    {selectedSNode.raw.length > 1 ? selectedSNode.label : selectedNode.name}
                  </h3>
                  <p className="mt-0.5 text-xs text-slate-500">{selectedNode.owner}</p>
                  {selectedSNode.raw.length > 1 ? (
                    <p className="mt-1 text-xs text-amber-400/80">
                      This marker represents {selectedSNode.raw.length} nodes. The figures
                      below are for {selectedNode.name} only, not a group total.
                    </p>
                  ) : null}
                </div>
                <span className="rounded-full px-2 py-0.5 text-xs font-semibold"
                  style={{ backgroundColor: `${STATUS_COLOR[selectedNode.status]}22`, color: STATUS_COLOR[selectedNode.status] }}>
                  {selectedNode.status}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Metric label="CPU cores" value={selectedNode.capability.cpu_cores} unit={selectedNode.capability.cpu_arch} tier="measured" density="compact" />
                <Metric label="CPU load" value={selectedNode.load.cpu_load_pct} unit="%" tier="measured" density="compact" />
                <Metric label="RAM used" value={`${selectedNode.load.ram_used_mb}/${selectedNode.capability.ram_mb}`} unit="MB" tier="measured" density="compact" />
                <Metric label="GPU VRAM" value={selectedNode.capability.gpu ? selectedNode.capability.vram_mb : '—'} unit={selectedNode.capability.gpu ? 'MB' : undefined} tier="measured" density="compact" />
                <Metric label="Trust" value={Math.round(selectedNode.trust_score * 100)} unit="%" tier="measured" density="compact" />
                <Metric label="Running units" value={selectedNode.load.running_units} tier="measured" density="compact" />
                <Metric label="Latency" value={selectedNode.link.latency_ms} unit="ms" tier="measured" density="compact" />
                <Metric label="Bandwidth" value={selectedNode.link.bandwidth_mbps} unit="Mbps" tier="measured" density="compact" />
              </div>
              {nodeModels.length > 0 && (
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-500">Model layers</p>
                  <ul className="mt-2 space-y-1.5">
                    {nodeModels.map((m) => {
                      const slice = m.partition.find((p) => p.node_id === selectedNode.id)
                      return slice ? (
                        <li key={m.id} className="rounded-2xl border border-indigo-500/30 bg-indigo-500/10 px-3 py-2 text-xs">
                          <span className="font-medium text-indigo-300">{m.model}</span>
                          <span className="ml-2 text-slate-400">layers {slice.start_layer}–{slice.end_layer}</span>
                          <span className="ml-2 text-slate-500">{slice.is_first ? '· Ingress' : slice.is_last ? '· Egress' : '· Relay'}</span>
                        </li>
                      ) : null
                    })}
                  </ul>
                </div>
              )}
              <button type="button" onClick={() => setSelectedId(null)}
                className="mt-2 w-full rounded-2xl border border-slate-700 py-1.5 text-xs text-slate-400 hover:bg-slate-800">
                Deselect
              </button>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-3 py-8 text-center">
              <p className="text-xs uppercase tracking-widest text-slate-500">Topology</p>
              <p className="text-sm text-slate-400">
                {nodes.length === 0
                  ? 'No nodes in the mesh yet'
                  : `${aggregate.mode} view · ${aggregate.items.length} item${aggregate.items.length !== 1 ? 's' : ''}`}
              </p>
              {nodes.length > 0 && (
                <p className="text-xs text-slate-500">
                  Click a node or use ↑↓←→ keys to inspect
                </p>
              )}
              {nodes.length > 0 && (
                <div className="mt-1 w-full text-left">
                  <p className="text-xs uppercase tracking-widest text-slate-500">Node classes</p>
                  <div className="mt-2 space-y-1.5">
                    {([
                      { cls: 'anchor', icon: '★', color: '#fbbf24', desc: 'High-trust · stable' },
                      { cls: 'relay', icon: '▶', color: '#38bdf8', desc: 'High-bandwidth · routing' },
                      { cls: 'constellation', icon: '●', color: '#818cf8', desc: 'General pool member' },
                      { cls: 'drifter', icon: '⊙', color: '#9ca3af', desc: 'New or high-latency' },
                    ] as const).map(({ cls, icon, color, desc }) => (
                      <div key={cls} className="flex items-center gap-2">
                        <span style={{ color }} aria-hidden="true" className="w-4 text-center text-sm">{icon}</span>
                        <span className="text-xs text-slate-400"><span className="font-medium text-slate-300">{cls}</span> — {desc}</span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-slate-500/70">
                    Classes are calibrated projections from latency, uptime, and bandwidth.
                  </p>
                </div>
              )}
            </div>
          )}
        </aside>
      </div>

      {/* ── model shard pipeline view ─────────────────────────────────────── */}
      {models.length > 0 && (
        <section aria-label="Deployed model pipelines"
          className="rounded-3xl border border-slate-800 bg-slate-950/70 p-4">
          <p className="text-xs uppercase tracking-widest text-slate-500">Model shard pipelines</p>
          <p className="mt-1 text-xs text-slate-500">
            Token path for each deployed model — each segment shows the layer range on one node.
          </p>
          <div className="mt-4 space-y-4">
            {models.map((m) => {
              const sorted = [...m.partition].sort((a, b) => a.start_layer - b.start_layer)
              return (
                <div key={m.id} className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-slate-200">{m.model}</span>
                    <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-400">{m.backend}</span>
                    {m.tokens_per_sec != null ? (
                      <Metric label="Tokens/sec" value={m.tokens_per_sec.toFixed(1)} unit="tok/s" tier="measured" density="compact" className="border-emerald-500/30" />
                    ) : null}
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${m.status === 'running' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'}`}>
                      {m.status}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {sorted.map((slice, idx) => {
                      const nodeName = nodes.find((nd) => nd.id === slice.node_id)?.name ?? slice.node_id.slice(0, 12)
                      const isSliceSelected =
                        selectedId != null &&
                        sNodesRef.current.find((sn) => sn.id === selectedId)
                          ?.raw.some((r) => r.id === slice.node_id) === true
                      return (
                        <div key={`${m.id}-${slice.node_id}-${idx}`} className="flex items-center gap-2">
                          <button type="button"
                            className={`rounded-2xl border px-3 py-1.5 text-xs transition ${isSliceSelected ? 'border-indigo-500/60 bg-indigo-500/15 text-indigo-200' : 'border-slate-700 bg-slate-900/50 text-slate-300 hover:border-indigo-500/40'}`}
                            onClick={() => {
                              const sn = sNodesRef.current.find((n) => n.raw.some((r) => r.id === slice.node_id))
                              setSelectedId(sn?.id ?? null)
                            }}>
                            <span className="font-medium">{nodeName}</span>
                            <span className="ml-1.5 text-slate-500">{slice.start_layer}–{slice.end_layer}</span>
                            <span className="ml-1 text-slate-500">{slice.is_first ? '· in' : slice.is_last ? '· out' : '· relay'}</span>
                          </button>
                          {idx < sorted.length - 1 && <span className="text-slate-600" aria-hidden="true">→</span>}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* ── accessible table controls ─────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <button type="button" onClick={() => setShowTable((t) => !t)}
          className="rounded-2xl border border-slate-700 px-3 py-2 text-xs text-slate-400 hover:bg-slate-800">
          {showTable ? 'Hide accessible table' : 'Show as accessible table'}
        </button>
        <div className="flex gap-2">
          <Metric label="Nodes" value={nodes.length} tier="measured" density="compact" />
          <Metric label="Online" value={nodes.filter((n) => n.status === 'online').length} tier="measured" density="compact" />
        </div>
      </div>

      {/* ── accessible table (sr-only until toggled; always in DOM for
            screen readers; keyboard-navigable rows ──────────────────────── */}
      <div className={showTable ? undefined : 'sr-only'}>
        {nodes.length === 0 ? (
          <p role="status" className="text-sm text-slate-400">
            No nodes registered in this mesh. Use the Contribute page to add your machine.
          </p>
        ) : (
          <div className="overflow-x-auto rounded-3xl border border-slate-800">
            <table className="min-w-full text-sm">
              <caption className="py-3 text-xs text-slate-500">
                Mesh nodes — {nodes.length} total, {nodes.filter((n) => n.status === 'online').length} online.
                {sortedNodes.length > TABLE_ROW_LIMIT
                  ? ` Showing the first ${TABLE_ROW_LIMIT} rows; ${sortedNodes.length - TABLE_ROW_LIMIT} more are not listed.`
                  : ''}
                {' '}Positions in the canvas above are synthesised from mesh topology, not geography.
              </caption>
              <thead className="border-b border-slate-800 bg-slate-950/70">
                <tr>
                  {['Node', 'Status', 'Arch', 'CPU cores', 'CPU load', 'RAM MB', 'GPU', 'Trust', 'Running units'].map((h) => (
                    <th key={h} scope="col" className="px-4 py-3 text-left font-medium text-slate-400">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {visibleRows.map((node) => {
                  const snId = itemIdByNodeId.get(node.id)
                  const isRowSelected = selectedId != null && snId === selectedId
                  return (
                    <tr key={node.id}
                      className={`cursor-pointer transition hover:bg-slate-900/50 ${isRowSelected ? 'bg-indigo-950/30' : ''}`}
                      onClick={() => setSelectedId(snId ?? node.id)}
                      role="row">
                      <td className="px-4 py-3 font-medium text-slate-200">{node.name}</td>
                      <td className="px-4 py-3">
                        <span className="rounded-full px-2 py-0.5 text-xs font-semibold"
                          style={{ backgroundColor: `${STATUS_COLOR[node.status]}22`, color: STATUS_COLOR[node.status] }}>
                          {node.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-400">{node.capability.cpu_arch}</td>
                      <td className="px-4 py-3 text-slate-300"><Metric label="CPU cores" value={node.capability.cpu_cores} tier="measured" density="compact" /></td>
                      <td className="px-4 py-3 text-slate-300"><Metric label="CPU load" value={node.load.cpu_load_pct} unit="%" tier="measured" density="compact" /></td>
                      <td className="px-4 py-3 text-slate-300"><Metric label="RAM" value={node.capability.ram_mb} unit="MB" tier="measured" density="compact" /></td>
                      <td className="px-4 py-3 text-slate-400">{node.capability.gpu ? <Metric label="GPU VRAM" value={node.capability.vram_mb} unit="MB" tier="measured" density="compact" /> : 'No'}</td>
                      <td className="px-4 py-3 text-slate-300"><Metric label="Trust" value={Math.round(node.trust_score * 100)} unit="%" tier="measured" density="compact" /></td>
                      <td className="px-4 py-3 text-slate-300"><Metric label="Running units" value={node.load.running_units} tier="measured" density="compact" /></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
