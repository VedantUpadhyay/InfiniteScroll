import './App.css'
import ChatInterface from './components/ChatInterface.jsx'
import KnowledgeGraph from './components/KnowledgeGraph.jsx'

function App() {
  return (
    <div className="mindchat-app">
      <header className="app-header">
        <div className="hero-copy">
          <p className="eyebrow">Vague recall, recovered</p>
          <h1>MindChat</h1>
          <p className="subtitle">
            Recover what you meant, even when you cannot remember the exact words.
          </p>
        </div>
        <div className="hero-callout">
          <strong>Built for tip-of-the-tongue moments</strong>
          <p>Use fragments, scenes, numbers, and related ideas instead of exact keywords.</p>
        </div>
      </header>

      <main className="app-grid">
        <ChatInterface />
        <KnowledgeGraph />
      </main>
    </div>
  )
}

export default App
