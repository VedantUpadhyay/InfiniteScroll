import { useEffect, useRef, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

function createSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `session-${Date.now()}`
}

export default function ChatInterface() {
  const [sessionId] = useState(createSessionId)
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        'Start with fragments. I can work from scenes, concepts, numbers, and related ideas.',
    },
  ])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const [telemetry, setTelemetry] = useState({
    workingMemorySize: 0,
    totalMessages: 0,
    consolidationDue: false,
    consolidationRan: false,
    researchHint: false,
  })
  const [isBuildingGraph, setIsBuildingGraph] = useState(false)
  const [graphStatusMessage, setGraphStatusMessage] = useState('')
  const [recentConcepts, setRecentConcepts] = useState([])
  const logRef = useRef(null)

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    function handleScrollToMessage(event) {
      const messageId = event?.detail?.messageId
      if (!messageId) return

      const escapedId =
        typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(String(messageId)) : String(messageId)
      const target = document.querySelector(`[data-message-id="${escapedId}"]`)
      if (!target) return

      target.scrollIntoView({ behavior: 'smooth', block: 'center' })
      target.classList.add('chat-bubble-highlight')
      window.setTimeout(() => {
        target.classList.remove('chat-bubble-highlight')
      }, 1400)
    }

    window.addEventListener('mindchat:scroll-to-message', handleScrollToMessage)
    return () => window.removeEventListener('mindchat:scroll-to-message', handleScrollToMessage)
  }, [])

  async function handleSubmit(event) {
    event.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || isSending) return

    const optimisticUser = {
      id: `local-${Date.now()}`,
      role: 'user',
      content: trimmed,
    }
    const optimisticUserId = optimisticUser.id

    setMessages((prev) => [...prev, optimisticUser])
    setInput('')
    setError('')
    setIsSending(true)

    const projectedTotalMessages = telemetry.totalMessages + 2
    const willBuildKnowledgeGraph =
      projectedTotalMessages > 0 && projectedTotalMessages % 10 === 0
    if (willBuildKnowledgeGraph) {
      setIsBuildingGraph(true)
      setGraphStatusMessage('Building memory map...')
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: trimmed,
        }),
      })

      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || `Request failed (${response.status})`)
      }

      const data = await response.json()
      const extractedConcepts = Array.isArray(data.concepts_extracted) ? data.concepts_extracted : []
      setMessages((prev) => [
        ...prev.map((message) =>
          message.id === optimisticUserId && data.user_message_id
            ? { ...message, id: data.user_message_id }
            : message,
        ),
        {
          id: data.assistant_message_id ?? `assistant-${Date.now()}`,
          role: 'assistant',
          content: data.reply,
        },
      ])
      setTelemetry({
        workingMemorySize: data.working_memory_size ?? 0,
        totalMessages: data.total_messages ?? 0,
        consolidationDue: Boolean(data.consolidation_due),
        consolidationRan: Boolean(data.consolidation_ran),
        researchHint: Boolean(data.research_hint),
      })
      setRecentConcepts(extractedConcepts.slice(0, 5))

      if (data.consolidation_ran) {
        setIsBuildingGraph(false)
        setGraphStatusMessage('Memory map updated.')
      } else if (data.consolidation_error) {
        setIsBuildingGraph(false)
        setGraphStatusMessage('Memory map build failed.')
      } else if (!willBuildKnowledgeGraph) {
        setGraphStatusMessage('')
      }

      window.dispatchEvent(
        new CustomEvent('mindchat:chat-updated', {
          detail: {
            sessionId,
            totalMessages: data.total_messages ?? 0,
            consolidationRan: Boolean(data.consolidation_ran),
          },
        }),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
      setIsBuildingGraph(false)
    } finally {
      setIsSending(false)
    }
  }

  function handleInputKeyDown(event) {
    if (event.key !== 'Enter') return
    if (event.shiftKey) return
    if (event.nativeEvent?.isComposing) return

    event.preventDefault()
    event.currentTarget.form?.requestSubmit()
  }

  const statusLabel = isBuildingGraph
    ? 'Building memory map'
    : telemetry.researchHint
      ? 'Research mode'
      : 'Recall mode active'

  return (
    <section className="panel panel-chat" aria-labelledby="chat-heading">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Conversation</p>
          <h2 id="chat-heading">Recall Workspace</h2>
        </div>
        <code className="session-pill" title="Session ID used by backend + Neo4j">
          {sessionId.slice(0, 8)}...
        </code>
      </div>

      <div className="composer-guide">
        <div className="status-chip" role="status" aria-live="polite">
          <span className={`status-dot ${isBuildingGraph ? 'busy' : ''}`} aria-hidden="true" />
          <strong>{statusLabel}</strong>
        </div>
        <div className="prompt-suggestions" aria-label="Example prompts">
          <span>Try: "that thing about memory and 7"</span>
          <span>Try: "the blue alien jungle movie"</span>
        </div>
      </div>

      <div className="chat-log" ref={logRef}>
        {messages.map((message) => (
          <article
            key={message.id}
            id={`chat-msg-${message.id}`}
            data-message-id={message.id}
            className={`chat-bubble ${message.role === 'user' ? 'user' : 'assistant'}`}
          >
            <header>{message.role === 'user' ? 'You' : 'MindChat'}</header>
            <p>{message.content}</p>
          </article>
        ))}
        {isSending ? (
          <article className="chat-bubble assistant" aria-live="polite">
            <header>MindChat</header>
            <p>{isBuildingGraph ? 'Thinking... Building memory map...' : 'Thinking...'}</p>
          </article>
        ) : null}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="mindchat-input">
          Message
        </label>
        <textarea
          id="mindchat-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder="Describe fragments, scenes, numbers, or related ideas..."
          rows={3}
          disabled={isSending}
        />
        <div className="chat-actions">
          <div className="form-hints">
            {recentConcepts.length > 0 ? (
              <span>Recent recall cues: {recentConcepts.join(', ')}</span>
            ) : null}
            {graphStatusMessage ? <span>{graphStatusMessage}</span> : null}
            {error ? <span className="error-text">{error}</span> : null}
          </div>
          <button type="submit" disabled={isSending || !input.trim()}>
            {isSending ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </form>
    </section>
  )
}
