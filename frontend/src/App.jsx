import './App.css'
import ChatInterface from './components/ChatInterface.jsx'
import KnowledgeGraph from './components/KnowledgeGraph.jsx'

function App() {
  return (
    <div className="mindchat-app">
      <header className="app-header">
        <div>
          <p className="eyebrow">Autonomous Agents Hackathon • 7-Day Sprint</p>
          <h1>MindChat</h1>
          <p className="subtitle">
            A Tip-of-the-Tongue assistant built on cognitive science: retrieval cues, chunking, and
            a Neo4j semantic memory graph.
          </p>
        </div>
        <div className="header-badges" aria-label="Core architecture badges">
          <span>OpenAI = Working Memory</span>
          <span>Neo4j = Long-Term Memory</span>
          <span>Tavily = Research Augmentation</span>
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
