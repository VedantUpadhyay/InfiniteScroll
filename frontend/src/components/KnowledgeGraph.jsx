import { useEffect, useRef, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const ANALYSIS_FLASH_MS = 2200

function safeArray(value) {
  return Array.isArray(value) ? value : []
}

function edgeStyle(edgeType) {
  if (edgeType === 'SYNTHESIS') {
    return {
      stroke: 'rgba(114, 189, 255, 0.78)',
      strokeDasharray: '8 6',
    }
  }

  if (edgeType === 'EXTENSION') {
    return {
      stroke: 'rgba(255, 206, 122, 0.72)',
      strokeDasharray: '0',
    }
  }

  return {
    stroke: 'rgba(124, 214, 151, 0.8)',
    strokeDasharray: '0',
  }
}

function buttonVariant(nodeType) {
  if (nodeType === 'SYNTHESIS') return 'concept'
  if (nodeType === 'BRANCH') return 'deflection'
  if (nodeType === 'EXTENSION') return 'message'
  return 'topic'
}

function nodePrefix(nodeType) {
  if (nodeType === 'SYNTHESIS') return '🧩'
  if (nodeType === 'BRANCH') return '⚡'
  if (nodeType === 'EXTENSION') return '↳'
  return '🌱'
}

function emptyGraph(sessionId = '') {
  return {
    session_id: sessionId,
    source: 'neo4j',
    tree: {
      nodes: [],
      edges: [],
      root: 'General',
    },
    message_count: 0,
    deflection_count: 0,
  }
}

function buildTreeView(graphData) {
  const tree = graphData?.tree ?? {}
  const nodes = safeArray(tree.nodes)
    .map((node) => ({
      id: String(node?.id ?? '').trim(),
      label: String(node?.label ?? node?.id ?? '').trim(),
      depth: Number(node?.depth ?? 0),
      type: String(node?.type ?? 'BRANCH').toUpperCase(),
    }))
    .filter((node) => node.id && node.label)

  const edges = safeArray(tree.edges)
    .map((edge) => ({
      from: String(edge?.from ?? '').trim(),
      to: String(edge?.to ?? '').trim(),
      type: String(edge?.type ?? 'BRANCH').toUpperCase(),
      score: Number(edge?.score ?? 0),
    }))
    .filter((edge) => edge.from && edge.to)

  const incomingEdgeByNode = new Map()
  for (const edge of edges) {
    if (!incomingEdgeByNode.has(edge.to)) {
      incomingEdgeByNode.set(edge.to, edge)
    }
  }

  const groupedDepths = new Map()
  for (const node of nodes) {
    if (!groupedDepths.has(node.depth)) {
      groupedDepths.set(node.depth, [])
    }
    groupedDepths.get(node.depth).push(node)
  }

  const depthLevels = [...groupedDepths.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([depth, levelNodes]) => ({
      depth,
      nodes: levelNodes.slice().sort((left, right) => left.label.localeCompare(right.label)),
    }))

  return {
    nodes,
    edges,
    depthLevels,
    root: String(tree.root ?? 'General'),
    activeNodeId: edges.length > 0 ? edges[edges.length - 1].to : (nodes.at(-1)?.id ?? null),
    incomingEdgeByNode,
  }
}

function TreeConnectorOverlay({ containerRef, nodeRefs, edges, visible }) {
  const [paths, setPaths] = useState([])
  const [viewBox, setViewBox] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!visible) {
      setPaths([])
      return undefined
    }

    let rafId = 0
    let observer = null

    const measure = () => {
      cancelAnimationFrame(rafId)
      rafId = window.requestAnimationFrame(() => {
        const container = containerRef.current
        if (!container) return

        const containerRect = container.getBoundingClientRect()
        const nextPaths = []

        for (const edge of edges) {
          const fromNode = nodeRefs.current.get(edge.from)
          const toNode = nodeRefs.current.get(edge.to)
          if (!fromNode || !toNode) continue

          const fromRect = fromNode.getBoundingClientRect()
          const toRect = toNode.getBoundingClientRect()
          const startX = fromRect.left - containerRect.left + fromRect.width / 2
          const startY = fromRect.top - containerRect.top + fromRect.height
          const endX = toRect.left - containerRect.left + toRect.width / 2
          const endY = toRect.top - containerRect.top
          const midY = startY + (endY - startY) / 2
          const style = edgeStyle(edge.type)

          nextPaths.push({
            id: `${edge.from}-${edge.to}-${edge.type}`,
            d: `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`,
            stroke: style.stroke,
            strokeDasharray: style.strokeDasharray,
          })
        }

        setPaths(nextPaths)
        setViewBox({
          width: Math.max(container.scrollWidth, container.clientWidth, 1),
          height: Math.max(container.scrollHeight, container.clientHeight, 1),
        })
      })
    }

    measure()

    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(measure)
      if (containerRef.current) {
        observer.observe(containerRef.current)
      }
      for (const node of nodeRefs.current.values()) {
        if (node) observer.observe(node)
      }
    }

    window.addEventListener('resize', measure)
    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', measure)
      observer?.disconnect()
    }
  }, [containerRef, edges, nodeRefs, visible])

  if (!visible) return null

  return (
    <svg
      aria-hidden="true"
      viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}
      preserveAspectRatio="none"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        overflow: 'visible',
        zIndex: 0,
      }}
    >
      {paths.map((path) => (
        <path
          key={path.id}
          d={path.d}
          fill="none"
          stroke={path.stroke}
          strokeDasharray={path.strokeDasharray}
          strokeLinecap="round"
          strokeWidth="2.25"
          opacity="0.9"
        />
      ))}
    </svg>
  )
}

export default function KnowledgeGraph() {
  const [graphData, setGraphData] = useState(emptyGraph())
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [activeSessionId, setActiveSessionId] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const analysisTimerRef = useRef(null)
  const treeContainerRef = useRef(null)
  const nodeRefs = useRef(new Map())

  const view = buildTreeView(graphData)

  useEffect(() => {
    fetchGraph()
    return () => {
      if (analysisTimerRef.current) {
        window.clearTimeout(analysisTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    function handleChatUpdated(event) {
      const sessionId = String(event?.detail?.sessionId ?? '').trim()
      const totalMessages = Number(event?.detail?.totalMessages ?? 0)
      const consolidationRan = Boolean(event?.detail?.consolidationRan)

      if (sessionId) {
        setActiveSessionId(sessionId)
      }

      if (consolidationRan || (totalMessages > 0 && totalMessages % 10 === 0)) {
        setIsAnalyzing(true)
        if (analysisTimerRef.current) {
          window.clearTimeout(analysisTimerRef.current)
        }
        analysisTimerRef.current = window.setTimeout(() => {
          setIsAnalyzing(false)
        }, ANALYSIS_FLASH_MS)
      }

      fetchGraph(sessionId)
    }

    window.addEventListener('mindchat:chat-updated', handleChatUpdated)
    return () => window.removeEventListener('mindchat:chat-updated', handleChatUpdated)
  }, [])

  async function fetchGraph(sessionId = activeSessionId) {
    setIsLoading(true)
    setError('')
    try {
      const params = new URLSearchParams()
      if (sessionId) {
        params.set('session_id', sessionId)
      }
      const url = `${API_BASE_URL}/api/graph${params.toString() ? `?${params}` : ''}`
      const response = await fetch(url)

      if (response.status === 404) {
        setGraphData(emptyGraph(sessionId))
        if (sessionId) {
          setActiveSessionId(sessionId)
        }
        return
      }

      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || `Graph request failed (${response.status})`)
      }

      const data = await response.json()
      setGraphData(data)
      if (data?.session_id) {
        setActiveSessionId(String(data.session_id))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load graph')
    } finally {
      setIsLoading(false)
    }
  }

  const hasTree = view.nodes.length > 0

  return (
    <section className="panel panel-graph" aria-labelledby="graph-heading">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Long-Term Memory</p>
          <h2 id="graph-heading">Knowledge Graph</h2>
        </div>
        <button
          type="button"
          className="graph-refresh-btn"
          onClick={() => {
            setIsAnalyzing(true)
            if (analysisTimerRef.current) window.clearTimeout(analysisTimerRef.current)
            analysisTimerRef.current = window.setTimeout(() => setIsAnalyzing(false), 1200)
            fetchGraph()
          }}
          disabled={isLoading}
        >
          {isLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      <div className="graph-counts" aria-label="Graph counts">
        <span>Messages: {Number(graphData?.message_count ?? 0)}</span>
        <span>Deflections: {Number(graphData?.deflection_count ?? 0)}</span>
        <span>Depths: {view.depthLevels.length}</span>
      </div>

      {graphData?.session_id ? (
        <div className="graph-summary">
          <span>Session: {String(graphData.session_id).slice(0, 8)}…</span>
          <span>Source: {String(graphData.source ?? 'neo4j')}</span>
          <span>Root: {view.root}</span>
        </div>
      ) : null}

      {error ? <p className="graph-error">{error}</p> : null}

      {!error && !hasTree && !isLoading ? (
        <div className="graph-empty">
          <p>Start chatting — your memory map will appear here</p>
        </div>
      ) : null}

      {hasTree ? (
        <div className={`graph-tree ${isAnalyzing ? 'analyzing' : ''}`}>
          <section className="graph-section">
            <h3 className="graph-section-title">Topic Tree</h3>
            <div
              ref={treeContainerRef}
              style={{
                position: 'relative',
                display: 'grid',
                gap: '1rem',
                padding: '0.2rem 0 0.4rem',
              }}
            >
              <TreeConnectorOverlay
                containerRef={treeContainerRef}
                nodeRefs={nodeRefs}
                edges={view.edges}
                visible={hasTree && view.edges.length > 0}
              />

              {view.depthLevels.map((level) => (
                <div
                  key={`depth-${level.depth}`}
                  className="tree-topic"
                  style={{
                    zIndex: 1,
                    display: 'grid',
                    gap: '0.7rem',
                    padding: '0.85rem',
                  }}
                >
                  <div className="tree-topic-row">
                    <div className="tree-meta">Depth {level.depth}</div>
                    <div className="tree-meta" style={{ justifySelf: 'end' }}>
                      {level.nodes.length} topic{level.nodes.length === 1 ? '' : 's'}
                    </div>
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: '0.7rem',
                      alignItems: 'stretch',
                    }}
                  >
                    {level.nodes.map((node) => {
                      const incomingEdge = view.incomingEdgeByNode.get(node.id) ?? null
                      const isActive = node.id === view.activeNodeId
                      const isSelected = selectedNodeId === node.id || isActive
                      const variant = buttonVariant(node.type)

                      return (
                        <button
                          key={node.id}
                          ref={(element) => {
                            if (element) {
                              nodeRefs.current.set(node.id, element)
                              return
                            }
                            nodeRefs.current.delete(node.id)
                          }}
                          type="button"
                          className={`tree-node-btn ${variant} ${isSelected ? 'selected' : ''}`}
                          onClick={() => setSelectedNodeId(node.id)}
                          title={`${node.type}: ${node.label}`}
                          style={{
                            minWidth: '13rem',
                            maxWidth: '100%',
                            flex: '1 1 13rem',
                            display: 'grid',
                            gap: '0.28rem',
                            animation: isActive ? 'deflection-glow 2s ease-in-out infinite' : undefined,
                          }}
                        >
                          <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <span className={node.type === 'BRANCH' ? 'deflection-mark' : 'tree-prefix'}>
                              {node.type === 'BRANCH' ? '⚡' : nodePrefix(node.type)}
                            </span>
                            <span>{node.label}</span>
                          </span>
                          <span className="tree-meta" style={{ whiteSpace: 'normal' }}>
                            {incomingEdge
                              ? `${incomingEdge.type.toLowerCase()} • score ${incomingEdge.score.toFixed(2)}`
                              : 'root topic'}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      ) : null}
    </section>
  )
}
