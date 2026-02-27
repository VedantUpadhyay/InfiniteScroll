# MindChat

MindChat is a cognitive-science-inspired chatbot for the Tip-of-the-Tongue (TOT) problem.

- OpenAI API = working memory (sliding window, ~20 messages)
- Neo4j AuraDB = long-term memory (semantic graph)
- Tavily (via MCP by default) = research augmentation
- Render = deployment target

## Why This Is Different

MindChat is scaffolded around memory and retrieval, not just chat completion:

- Semantic chunking over verbatim transcript replay
- Knowledge graph schema for concepts/topics/deflection points
- Retrieval cues for vague user queries
- Periodic consolidation (analysis agent) every 10 messages

## Project Structure

```text
backend/   FastAPI API + memory/search/analysis modules
frontend/  React + Vite UI (chat + graph panels)
.env.example
AGENTS.md
```

## Setup

### 1. Add environment variables

Create a root `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Then fill in:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY` (recommended, for MCP URL auto-generation)
- `NEO4J_URI`
- `NEO4J_PASSWORD`
- `NEO4J_USER` (usually `neo4j`)

Tavily is configured MCP-first to minimize local setup:

- Recommended: set `TAVILY_API_KEY` and leave `TAVILY_MCP_SERVER_URL` blank
- Optional: set `TAVILY_MCP_SERVER_URL` directly (remote Tavily MCP URL)
- `TAVILY_PROVIDER` defaults to `mcp`

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and calls the backend at `http://localhost:8000` by default.

## Initial API Endpoints

- `GET /api/health`
- `POST /api/chat`
- `POST /api/analyze`
- `POST /api/search`

## Notes

- The backend includes graceful fallbacks if API keys are missing, so the scaffold can run before integrations are wired.
- Neo4j queries in the scaffold use parameterized Cypher (no string concatenation).
- Tavily is wired as a sponsor integration in MCP-first mode (no local `tavily-python` SDK required for scaffold setup).
