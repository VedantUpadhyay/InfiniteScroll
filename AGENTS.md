# AGENTS.md - MindChat Hackathon Project

## Project Overview
Building MindChat - a chatbot that solves the Tip-of-Tongue (TOT) problem using cognitive science principles.

**Core Innovation:** Users forget exact keywords but remember concepts. Our system uses Neo4j knowledge graph as "long-term memory" and OpenAI API as "working memory" - mimicking human cognition (Atkinson-Shiffrin Memory Model, 1968).

## Cognitive Science Foundation
1. **Chunking (Miller's 7±2):** Store conversations as semantic chunks, not verbatim text
2. **Schema Theory:** Build relationship graph between concepts in Neo4j
3. **Working vs Long-term Memory:** OpenAI context = working memory (20 msgs), Neo4j = long-term (unlimited)
4. **Retrieval Cues:** Semantic search for vague queries (solves TOT problem)
5. **Signal Words:** Detect topic transitions (however, therefore, because)

## Tech Stack (4 Sponsors - REQUIRED)
- **OpenAI API:** Chat responses + conversation analysis
- **Neo4j AuraDB:** Knowledge graph storage (long-term memory)
- **Tavily API:** Research-enhanced responses
- **Render:** Deployment platform
- **Backend:** FastAPI (Python 3.10+)
- **Frontend:** React with Vite

## Repository Structure
```
/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── neo4j_manager.py     # Graph database operations
│   ├── openai_manager.py    # LLM interactions (working memory)
│   ├── analysis_agent.py    # Conversation analysis (consolidation)
│   ├── search.py            # Semantic search (retrieval)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   └── KnowledgeGraph.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── .env                     # API keys (gitignored)
├── AGENTS.md               # This file
└── README.md
```

## Environment Variables
```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
NEO4J_URI=neo4j+s://...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

## Setup Instructions
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt

# Frontend
cd frontend
npm install

# Test connections
python -c "from neo4j import GraphDatabase; print('Neo4j OK')"
python -c "from openai import OpenAI; print('OpenAI OK')"
```

## Testing Instructions
```bash
# Backend tests
cd backend
pytest tests/

# Start development servers
# Terminal 1 - Backend:
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend:
cd frontend && npm run dev
```

## Development Guidelines

### Code Style
- Python: Follow PEP 8, use type hints
- JavaScript: Use ES6+, functional components with hooks
- Neo4j queries: ALWAYS use parameterized Cypher (never string concatenation)
- Error handling: Graceful fallbacks, log errors

### Neo4j Schema (Critical!)
```cypher
// Node types
(Message {id, role, content, timestamp})
(Topic {name, message_ids[]})
(Concept {name})
(DeflectionPoint {message_id, from_topic, to_topic})

// Relationships
(Message)-[:DISCUSSES]->(Concept)
(Message)-[:PART_OF]->(Topic)
(Topic)-[:TRANSITIONS_TO {via: DeflectionPoint}]->(Topic)
(Concept)-[:RELATES_TO {type: string}]->(Concept)
```

### Key Architectural Principles
1. **Separation of Concerns:** Chat responses use sliding window (20 msgs), analysis uses full history from Neo4j
2. **Periodic Consolidation:** Every 10 messages, run analysis agent to update knowledge graph
3. **Semantic Retrieval:** When user searches, query Neo4j by concepts not keywords
4. **Research Enhancement:** Detect research questions, call Tavily, add sources to graph

## MVP Features (7-Day Sprint)

### Day 1-2: Foundation
- Basic FastAPI server with CORS
- Chat endpoint: `/api/chat` (POST)
- Store messages in Neo4j with relationships
- OpenAI integration with sliding window

### Day 3-4: Analysis Agent
- `/api/analyze` endpoint
- Extract topics, concepts, relationships, deflection points
- Build knowledge graph in Neo4j
- Use GPT-4 for deep semantic processing

### Day 5: Semantic Search (TOT Solution)
- `/api/search` endpoint
- Take vague query → extract features → query graph
- Return messages + context explaining WHY found

### Day 6: Tavily Integration
- Detect research questions in chat
- Fetch from Tavily API
- Store research sources as nodes
- Link to relevant concepts

### Day 7: Visualization + Deploy
- React tree view of knowledge graph
- Click node → scroll to message
- Highlight deflection points
- Deploy to Render

## Demo Strategy

### Target Test Conversation
Have a pre-prepared 50-message conversation about cognitive science research. Then demonstrate:

1. **Keyword search FAILS:**
   - User: "What was that thing about... memory... and 7?"
   - Traditional search: No results

2. **Our system SUCCEEDS:**
   - Same query → Retrieves Miller's 7±2 research
   - Shows knowledge graph connections
   - Explains WHY it was found (concept matching)

### Judging Criteria Alignment
- **Innovation:** Cognitive science-based architecture
- **Sponsor Integration:** Uses all 4 sponsors meaningfully
- **Technical Execution:** Working demo, clean code
- **Impact:** Solves real problem (TOT phenomenon)

## Citations & Research
Include these in presentation:
- Atkinson & Shiffrin (1968) - Multi-store memory model
- Miller (1956) - 7±2 chunking theory
- Craik & Lockhart (1972) - Levels of processing
- Brown & McNeill (1966) - Tip-of-tongue phenomenon

## Common Pitfalls to Avoid
- ❌ Don't use string concatenation for Cypher queries (SQL injection risk)
- ❌ Don't re-analyze entire conversation for chat responses (use sliding window)
- ❌ Don't forget to handle Neo4j connection errors gracefully
- ❌ Don't commit .env file
- ❌ Don't skip Tavily integration (it's a sponsor!)

## Render Deployment
```bash
# Backend service
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

# Frontend static site
- Build Command: npm run build
- Publish Directory: dist
```

## Questions? Common Issues?

### "Neo4j connection timeout"
- Check NEO4J_URI has `neo4j+s://` prefix
- Verify password is correct
- Ensure AuraDB instance is running

### "OpenAI rate limit"
- Use gpt-3.5-turbo for chat (cheaper)
- Use gpt-4 only for analysis
- Add retry logic with exponential backoff

### "Graph gets too complex"
- Limit concepts per analysis to top 10
- Prune old relationships periodically
- Focus on meaningful connections only

## Success Criteria
By end of Day 7, must have:
- ✅ Working chat storing to Neo4j
- ✅ Knowledge graph visualization
- ✅ Semantic search demo (vague query → find result)
- ✅ Live deployment on Render
- ✅ 5-minute demo script ready
