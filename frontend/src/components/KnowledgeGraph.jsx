import { useEffect, useRef, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const ANALYSIS_FLASH_MS = 2200

function safeArray(value) {
  return Array.isArray(value) ? value : []
}

function compareByLabel(a, b) {
  return String(a?.label ?? '').localeCompare(String(b?.label ?? ''))
}

function previewText(content, maxLength = 86) {
  const normalized = String(content ?? '').replace(/\s+/g, ' ').trim()
  if (!normalized) return '(empty message)'
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength - 1)}…`
}

function dispatchScrollToMessage(messageId) {
  if (!messageId) return
  window.dispatchEvent(new CustomEvent('mindchat:scroll-to-message', { detail: { messageId } }))
}

function buildGraphView(graphData) {
  const nodes = safeArray(graphData?.nodes)
  const edges = safeArray(graphData?.edges)

  const nodeById = new Map()
  const messageNodes = []
  const topicNodes = []
  const deflectionNodes = []

  for (const node of nodes) {
    if (!node || !node.id) continue
    nodeById.set(node.id, node)
    if (node.type === 'Message') messageNodes.push(node)
    if (node.type === 'Topic') topicNodes.push(node)
    if (node.type === 'DeflectionPoint') deflectionNodes.push(node)
  }

  const messageById = new Map(messageNodes.map((node) => [node.id, node]))

  const topicToMessages = new Map()
  const messageToConcepts = new Map()
  const conceptToMessages = new Map()
  const conceptRelations = new Map()
  const deflectionTriggerMessage = new Map()
  const transitionEdges = []

  function mapPush(map, key, value) {
    if (!key || !value) return
    if (!map.has(key)) map.set(key, [])
    map.get(key).push(value)
  }

  for (const edge of edges) {
    if (!edge || !edge.source || !edge.target) continue
    if (edge.type === 'PART_OF') {
      mapPush(topicToMessages, edge.target, edge.source)
    }
    if (edge.type === 'DISCUSSES') {
      mapPush(messageToConcepts, edge.source, edge.target)
      mapPush(conceptToMessages, edge.target, edge.source)
    }
    if (edge.type === 'RELATES_TO') {
      mapPush(conceptRelations, edge.source, {
        conceptId: edge.target,
        type: edge.properties?.type ?? 'related',
      })
      mapPush(conceptRelations, edge.target, {
        conceptId: edge.source,
        type: edge.properties?.type ?? 'related',
      })
    }
    if (edge.type === 'TRIGGERED') {
      deflectionTriggerMessage.set(edge.target, edge.source)
    }
    if (edge.type === 'TRANSITIONS_TO') {
      transitionEdges.push(edge)
    }
  }

  const transitionByDeflectionId = new Map()
  for (const edge of transitionEdges) {
    const viaDeflectionId = edge.properties?.via
    if (!viaDeflectionId) continue
    transitionByDeflectionId.set(viaDeflectionId, edge)
  }

  function normalizeUnique(list) {
    const seen = new Set()
    const result = []
    for (const item of list ?? []) {
      if (!item || seen.has(item)) continue
      seen.add(item)
      result.push(item)
    }
    return result
  }

  function uniqueRelations(relations) {
    const seen = new Set()
    const result = []
    for (const relation of relations ?? []) {
      if (!relation || !relation.id) continue
      const key = `${relation.id}:${relation.type ?? 'related'}`
      if (seen.has(key)) continue
      seen.add(key)
      result.push(relation)
    }
    return result
  }

  const deflections = deflectionNodes
    .map((node) => {
      const props = node.properties ?? {}
      const transitionEdge = transitionByDeflectionId.get(node.id)
      const fromTopicNode = transitionEdge ? nodeById.get(transitionEdge.source) : null
      const toTopicNode = transitionEdge ? nodeById.get(transitionEdge.target) : null
      const signalWords = safeArray(props.signal_words).map((word) => String(word).trim()).filter(Boolean)
      const messageId = String(props.message_id ?? deflectionTriggerMessage.get(node.id) ?? '').trim()
      return {
        id: node.id,
        messageId: messageId || null,
        fromTopic:
          String(props.from_topic ?? fromTopicNode?.label ?? fromTopicNode?.properties?.name ?? '')
            .trim() || 'Unknown topic',
        toTopic:
          String(props.to_topic ?? toTopicNode?.label ?? toTopicNode?.properties?.name ?? '').trim() ||
          'Unknown topic',
        signalWords,
        reason: String(props.reason ?? '').trim(),
      }
    })
    .sort((a, b) => {
      const aMessage = a.messageId ? messageById.get(a.messageId) : null
      const bMessage = b.messageId ? messageById.get(b.messageId) : null
      const aTs = String(aMessage?.properties?.timestamp ?? '')
      const bTs = String(bMessage?.properties?.timestamp ?? '')
      return aTs.localeCompare(bTs)
    })

  const topics = topicNodes
    .slice()
    .sort(compareByLabel)
    .map((topicNode) => {
      const messageIds = normalizeUnique(topicToMessages.get(topicNode.id) ?? [])
      const sortedMessageIds = messageIds
        .slice()
        .sort((leftId, rightId) =>
          String(messageById.get(leftId)?.properties?.timestamp ?? '').localeCompare(
            String(messageById.get(rightId)?.properties?.timestamp ?? ''),
          ),
        )

      const conceptSet = new Set()
      for (const messageId of sortedMessageIds) {
        for (const conceptId of messageToConcepts.get(messageId) ?? []) {
          conceptSet.add(conceptId)
        }
      }

      const concepts = [...conceptSet]
        .map((conceptId) => {
          const conceptNode = nodeById.get(conceptId)
          if (!conceptNode) return null
          const conceptMessages = normalizeUnique(conceptToMessages.get(conceptId) ?? [])
          const representativeMessageId =
            conceptMessages.find((messageId) => sortedMessageIds.includes(messageId)) ??
            conceptMessages[0] ??
            sortedMessageIds[0] ??
            null
          const relatedConcepts = uniqueRelations(
            (conceptRelations.get(conceptId) ?? [])
              .map((relation) => {
                const relatedNode = nodeById.get(relation.conceptId)
                if (!relatedNode) return null
                return {
                  id: relation.conceptId,
                  label: String(relatedNode.label ?? relatedNode.properties?.name ?? 'Concept'),
                  type: String(relation.type ?? 'related'),
                }
              })
              .filter(Boolean),
          ).slice(0, 4)

          return {
            id: conceptId,
            label: String(conceptNode.label ?? conceptNode.properties?.name ?? 'Concept'),
            representativeMessageId,
            relatedConcepts,
          }
        })
        .filter(Boolean)
        .sort((a, b) => a.label.localeCompare(b.label))

      const messages = sortedMessageIds
        .map((messageId) => {
          const messageNode = messageById.get(messageId)
          if (!messageNode) return null
          return {
            id: messageId,
            role: String(messageNode.properties?.role ?? messageNode.label ?? 'message'),
            content: String(messageNode.properties?.content ?? ''),
            timestamp: String(messageNode.properties?.timestamp ?? ''),
          }
        })
        .filter(Boolean)

      const topicName = String(topicNode.label ?? topicNode.properties?.name ?? 'Topic')
      const topicDeflections = deflections.filter(
        (point) => point.fromTopic === topicName || point.toTopic === topicName,
      )

      return {
        id: topicNode.id,
        label: topicName,
        messages,
        concepts,
        deflections: topicDeflections,
      }
    })

  return {
    topics,
    deflections,
    counts: graphData?.counts ?? null,
    analysisAvailable: Boolean(graphData?.analysis_available),
    source: graphData?.source ?? 'unknown',
    notes: safeArray(graphData?.notes),
    hasData: nodes.length > 0 || edges.length > 0,
  }
}

function BranchConnectorOverlay({ rowRefs, rowCount, visible }) {
  const containerRef = useRef(null)
  const [paths, setPaths] = useState([])
  const [viewHeight, setViewHeight] = useState(0)

  useEffect(() => {
    if (!visible) {
      setPaths([])
      return undefined
    }

    let rafId = 0
    let observer = null

    const measure = () => {
      cancelAnimationFrame(rafId)
      rafId = requestAnimationFrame(() => {
        const container = containerRef.current
        if (!container) return
        const containerRect = container.getBoundingClientRect()
        const nextPaths = []

        for (const row of rowRefs.current) {
          if (!row) continue
          const rect = row.getBoundingClientRect()
          const cy = rect.top - containerRect.top + rect.height / 2
          const endX = 28
          nextPaths.push(`M 6 18 C 12 18, 14 ${cy}, ${endX} ${cy}`)
        }

        setPaths(nextPaths)
        setViewHeight(Math.max(container.scrollHeight, container.clientHeight, 36))
      })
    }

    measure()

    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(measure)
      if (containerRef.current) observer.observe(containerRef.current)
      for (const row of rowRefs.current) {
        if (row) observer.observe(row)
      }
    }

    window.addEventListener('resize', measure)
    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', measure)
      observer?.disconnect()
    }
  }, [rowCount, rowRefs, visible])

  return (
    <div ref={containerRef} className="topic-branch-overlay" aria-hidden="true">
      <svg
        className={`topic-branch-svg ${visible ? 'is-visible' : ''}`}
        viewBox={`0 0 36 ${Math.max(viewHeight, 36)}`}
        preserveAspectRatio="none"
      >
        <path className="topic-branch-trunk" d={`M 6 8 L 6 ${Math.max(viewHeight - 8, 8)}`} />
        {paths.map((path, index) => (
          <path key={index} className="topic-branch-path" d={path} />
        ))}
      </svg>
    </div>
  )
}

function ConceptRow({ concept, isSelected, onSelect }) {
  const hasRelations = concept.relatedConcepts.length > 0

  return (
    <div className={`concept-with-relations ${hasRelations ? 'has-relations' : ''}`}>
      <button
        type="button"
        className={`tree-node-btn concept ${isSelected ? 'selected' : ''}`}
        onClick={() => onSelect(concept.representativeMessageId, concept.id)}
        title={
          concept.representativeMessageId
            ? `Jump to a message discussing ${concept.label}`
            : `Concept: ${concept.label}`
        }
      >
        <span className="tree-prefix">🧠</span>
        <span>{concept.label}</span>
      </button>

      {hasRelations ? (
        <>
          <svg className="concept-relation-svg" viewBox="0 0 100 56" preserveAspectRatio="none">
            {concept.relatedConcepts.slice(0, 3).map((_, index) => {
              const endX = 22 + index * 22
              const endY = 48
              const controlX = 30 + index * 10
              return (
                <path
                  key={`${concept.id}-rel-${index}`}
                  className="concept-relation-line"
                  d={`M 12 10 C ${controlX} 16, ${controlX} 36, ${endX} ${endY}`}
                />
              )
            })}
          </svg>

          <div className="concept-relations-chips" aria-label={`Relations for ${concept.label}`}>
            {concept.relatedConcepts.map((related) => (
              <button
                key={`${concept.id}-${related.id}-${related.type}`}
                type="button"
                className="concept-rel-chip"
                onClick={() => onSelect(concept.representativeMessageId, related.id)}
                title={`${related.type}: ${related.label}`}
              >
                <span>{related.label}</span>
                <small>{related.type}</small>
              </button>
            ))}
          </div>
        </>
      ) : null}
    </div>
  )
}

function DeflectionSplit({
  beforeTopic,
  afterTopic,
  signalWord,
  messageId,
  onSelectMessage,
  onSelectTopic,
}) {
  return (
    <div className="deflection-split">
      <div className="topic-before">
        <button
          type="button"
          className="tree-node-btn topic"
          onClick={() => onSelectTopic(beforeTopic)}
          title={`Focus topic: ${beforeTopic}`}
        >
          <span className="tree-prefix">📊</span>
          <span>{beforeTopic}</span>
        </button>
      </div>

      <button
        type="button"
        className="deflection-point-visual"
        onClick={() => onSelectMessage(messageId)}
        title={messageId ? 'Jump to deflection message' : 'Deflection point'}
      >
        <span className="lightning deflection-mark" aria-hidden="true">
          ⚡
        </span>
        {signalWord ? <span className="signal-word">"{signalWord}"</span> : null}
      </button>

      <div className="topic-after">
        <button
          type="button"
          className="tree-node-btn topic"
          onClick={() => onSelectTopic(afterTopic)}
          title={`Focus topic: ${afterTopic}`}
        >
          <span className="tree-prefix">📊</span>
          <span>{afterTopic}</span>
        </button>
      </div>
    </div>
  )
}

function TopicCard({
  topic,
  expanded,
  onToggle,
  onSelectNode,
  selectedNodeId,
}) {
  const rowRefs = useRef([])
  rowRefs.current = []

  const conceptRows = topic.concepts.map((concept) => ({
    key: `concept-${concept.id}`,
    render: (
      <ConceptRow
        concept={concept}
        isSelected={selectedNodeId === concept.id}
        onSelect={(messageId, nodeId) => onSelectNode({ messageId, nodeId })}
      />
    ),
  }))

  const messageRows = topic.messages.slice(-3).map((message) => ({
    key: `message-${message.id}`,
    render: (
      <button
        type="button"
        className={`tree-node-btn message ${selectedNodeId === message.id ? 'selected' : ''}`}
        onClick={() => onSelectNode({ messageId: message.id, nodeId: message.id })}
        title={message.content}
      >
        <span className="tree-prefix">{message.role === 'assistant' ? '🤖' : '💬'}</span>
        <span className="message-preview-label">{previewText(message.content)}</span>
      </button>
    ),
  }))

  const deflectionRows = topic.deflections.slice(0, 2).map((point) => ({
    key: `deflection-${point.id}`,
    render: (
      <button
        type="button"
        className={`tree-node-btn deflection ${selectedNodeId === point.id ? 'selected' : ''}`}
        onClick={() => onSelectNode({ messageId: point.messageId, nodeId: point.id })}
        title={point.reason || `${point.fromTopic} → ${point.toTopic}`}
      >
        <span className="deflection-mark" aria-hidden="true">
          ⚡
        </span>
        <span>
          {point.fromTopic} → {point.toTopic}
          {point.signalWords[0] ? ` (${point.signalWords[0]})` : ''}
        </span>
      </button>
    ),
  }))

  const rows = [...conceptRows, ...deflectionRows, ...messageRows]

  return (
    <section className="tree-topic">
      <div className="tree-topic-row">
        <button
          type="button"
          className="tree-toggle-btn"
          onClick={() => onToggle(topic.id)}
          aria-expanded={expanded}
          aria-controls={`topic-children-${topic.id}`}
          title={expanded ? 'Collapse topic' : 'Expand topic'}
        >
          {expanded ? '−' : '+'}
        </button>

        <button
          type="button"
          className={`tree-node-btn topic ${selectedNodeId === topic.id ? 'selected' : ''}`}
          onClick={() => onSelectNode({ messageId: topic.messages[0]?.id ?? null, nodeId: topic.id })}
          title={`Topic: ${topic.label}`}
        >
          <span className="tree-prefix">📊</span>
          <span>{topic.label}</span>
        </button>

        <span className="tree-meta">
          {topic.messages.length} msg • {topic.concepts.length} concepts
        </span>
      </div>

      {expanded ? (
        <div className="tree-children-shell">
          <BranchConnectorOverlay rowRefs={rowRefs} rowCount={rows.length} visible={rows.length > 0} />
          <div className="tree-children has-svg" id={`topic-children-${topic.id}`}>
            {rows.length === 0 ? (
              <div className="tree-child-row tree-empty-row">
                <span className="tree-prefix">·</span>
                <span className="message-preview-label">No consolidated concepts yet.</span>
              </div>
            ) : (
              rows.map((row) => (
                <div
                  key={row.key}
                  className="tree-child-row"
                  ref={(node) => {
                    if (node) rowRefs.current.push(node)
                  }}
                >
                  {row.render}
                </div>
              ))
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}

export default function KnowledgeGraph() {
  const [graphData, setGraphData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [expandedTopicIds, setExpandedTopicIds] = useState({})
  const [activeSessionId, setActiveSessionId] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const analysisTimerRef = useRef(null)

  const view = buildGraphView(graphData)

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
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || `Graph request failed (${response.status})`)
      }

      const data = await response.json()
      setGraphData(data)

      const topicIds = safeArray(data?.nodes)
        .filter((node) => node?.type === 'Topic' && node?.id)
        .slice(0, 2)
        .map((node) => node.id)
      if (topicIds.length > 0) {
        setExpandedTopicIds((prev) => {
          const next = { ...prev }
          for (const topicId of topicIds) {
            if (!(topicId in next)) next[topicId] = true
          }
          return next
        })
      }
      if (data?.session_id) {
        setActiveSessionId(String(data.session_id))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load graph')
    } finally {
      setIsLoading(false)
    }
  }

  function toggleTopic(topicId) {
    setExpandedTopicIds((prev) => ({ ...prev, [topicId]: !prev[topicId] }))
  }

  function handleSelectNode({ messageId, nodeId }) {
    if (nodeId) {
      setSelectedNodeId(nodeId)
    }
    if (messageId) {
      dispatchScrollToMessage(messageId)
    }
  }

  function focusTopicByName(topicName) {
    if (!topicName) return
    const topic = view.topics.find((candidate) => candidate.label === topicName)
    if (!topic) return
    setExpandedTopicIds((prev) => ({ ...prev, [topic.id]: true }))
    setSelectedNodeId(topic.id)
    if (topic.messages[0]?.id) {
      dispatchScrollToMessage(topic.messages[0].id)
    }
  }

  const counts = view.counts
  const hasGraphData = view.hasData

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

      {counts ? (
        <div className="graph-counts" aria-label="Graph counts">
          <span>Topics: {counts.topics ?? 0}</span>
          <span>Concepts: {counts.concepts ?? 0}</span>
          <span>Messages: {counts.messages ?? 0}</span>
          <span>Deflections: {counts.deflection_points ?? 0}</span>
          <span>Edges: {counts.edges ?? 0}</span>
        </div>
      ) : null}

      {graphData?.session_id ? (
        <div className="graph-summary">
          <span>Session: {String(graphData.session_id).slice(0, 8)}…</span>
          <span>Source: {view.source}</span>
          <span>{view.analysisAvailable ? 'Consolidated memory active' : 'Awaiting consolidation'}</span>
        </div>
      ) : null}

      {error ? <p className="graph-error">{error}</p> : null}

      {!error && !hasGraphData && !isLoading ? (
        <div className="graph-empty">
          <p>Knowledge graph is empty for this session.</p>
          <p>Send messages and reach 10 total messages to trigger consolidation.</p>
        </div>
      ) : null}

      {hasGraphData ? (
        <div className={`graph-tree ${isAnalyzing ? 'analyzing' : ''}`}>
          {view.deflections.length > 0 ? (
            <section className="graph-section">
              <h3 className="graph-section-title">Topic Transitions</h3>
              {view.deflections.map((deflection) => (
                <DeflectionSplit
                  key={deflection.id}
                  beforeTopic={deflection.fromTopic}
                  afterTopic={deflection.toTopic}
                  signalWord={deflection.signalWords[0] ?? ''}
                  messageId={deflection.messageId}
                  onSelectMessage={(messageId) =>
                    handleSelectNode({ messageId, nodeId: deflection.id })
                  }
                  onSelectTopic={focusTopicByName}
                />
              ))}
            </section>
          ) : null}

          <section className="graph-section">
            <h3 className="graph-section-title">Topic Tree</h3>
            {view.topics.length === 0 ? (
              <div className="graph-empty">
                <p>No topics extracted yet. Run analysis or send more messages.</p>
              </div>
            ) : (
              view.topics.map((topic) => (
                <TopicCard
                  key={topic.id}
                  topic={topic}
                  expanded={Boolean(expandedTopicIds[topic.id])}
                  onToggle={toggleTopic}
                  onSelectNode={handleSelectNode}
                  selectedNodeId={selectedNodeId}
                />
              ))
            )}
          </section>

          {view.notes.length > 0 ? (
            <section className="cue-panel">
              <h3>Retrieval Cue Notes</h3>
              <ul>
                {view.notes.map((note, index) => (
                  <li key={`${index}-${note}`}>{note}</li>
                ))}
              </ul>
            </section>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
