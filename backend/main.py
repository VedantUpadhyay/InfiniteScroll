from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analysis_agent import AnalysisAgent
from neo4j_manager import Neo4jManager
from openai_manager import OpenAIManager
from search import SemanticSearchService


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Conversation/session identifier")
    message: str = Field(..., min_length=1)


class AnalyzeRequest(BaseModel):
    session_id: str


class SearchRequest(BaseModel):
    session_id: str
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


def _resolve_session_id(session_id: str | None) -> str | None:
    if session_id:
        return session_id

    all_messages = neo4j_manager.get_all_messages()
    if not all_messages:
        return None

    # Default to the most recently active session so GET /api/search?q=... works during demos.
    latest_message = all_messages[-1]
    resolved = latest_message.get("session_id")
    return str(resolved) if resolved else None


def _graph_snapshot_from_neo4j(session_id: str, message_limit: int = 500) -> list[dict[str, Any]] | None:
    if not neo4j_manager.driver:
        return None

    query = """
    MATCH (m:Message {session_id: $session_id})
    OPTIONAL MATCH (m)-[:PART_OF]->(t:Topic)
    OPTIONAL MATCH (m)-[:DISCUSSES]->(c:Concept)
    RETURN
      m.id AS id,
      m.session_id AS session_id,
      m.role AS role,
      m.content AS content,
      m.timestamp AS timestamp,
      collect(DISTINCT t.name) AS topics,
      collect(DISTINCT c.name) AS concepts
    ORDER BY m.timestamp ASC
    LIMIT $message_limit
    """
    try:
        with neo4j_manager.driver.session() as session:
            return session.run(
                query,
                session_id=session_id,
                message_limit=message_limit,
            ).data()
    except Exception:
        return None


def _concept_relationships_from_neo4j(session_id: str, edge_limit: int = 500) -> list[dict[str, Any]]:
    if not neo4j_manager.driver:
        return []

    query = """
    MATCH (m:Message {session_id: $session_id})-[:DISCUSSES]->(c:Concept)
    WITH collect(DISTINCT c) AS session_concepts
    UNWIND session_concepts AS c1
    MATCH (c1)-[r:RELATES_TO]-(c2:Concept)
    WHERE c2 IN session_concepts
    RETURN DISTINCT c1.name AS source, c2.name AS target, r.type AS type
    LIMIT $edge_limit
    """
    try:
        with neo4j_manager.driver.session() as session:
            return session.run(
                query,
                session_id=session_id,
                edge_limit=edge_limit,
            ).data()
    except Exception:
        return []


def _build_graph_response(session_id: str, *, message_limit: int = 500) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_seen: set[str] = set()
    edge_seen: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, node_type: str, label: str, properties: dict[str, Any] | None = None) -> None:
        if not node_id or node_id in node_seen:
            return
        node_seen.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "properties": properties or {},
            }
        )

    def add_edge(source: str, target: str, edge_type: str, properties: dict[str, Any] | None = None) -> None:
        if not source or not target:
            return
        key = (source, target, edge_type)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "properties": properties or {},
            }
        )

    graph_rows = _graph_snapshot_from_neo4j(session_id, message_limit=message_limit)
    source = "neo4j-live" if graph_rows is not None else "memory-fallback"

    if graph_rows is not None:
        for row in graph_rows:
            message_id = str(row.get("id", ""))
            add_node(
                message_id,
                "Message",
                row.get("role", "message"),
                {
                    "session_id": row.get("session_id"),
                    "role": row.get("role"),
                    "content": row.get("content"),
                    "timestamp": row.get("timestamp"),
                },
            )

            for topic_name in row.get("topics", []) or []:
                if not topic_name:
                    continue
                topic_node_id = f"topic:{topic_name}"
                add_node(topic_node_id, "Topic", str(topic_name), {"name": topic_name})
                add_edge(message_id, topic_node_id, "PART_OF")

            for concept_name in row.get("concepts", []) or []:
                if not concept_name:
                    continue
                concept_node_id = f"concept:{str(concept_name).lower()}"
                add_node(concept_node_id, "Concept", str(concept_name), {"name": concept_name})
                add_edge(message_id, concept_node_id, "DISCUSSES")

        for rel in _concept_relationships_from_neo4j(session_id):
            source_name = str(rel.get("source", "")).strip()
            target_name = str(rel.get("target", "")).strip()
            rel_type = str(rel.get("type", "related")).strip() or "related"
            if not source_name or not target_name:
                continue
            source_id = f"concept:{source_name.lower()}"
            target_id = f"concept:{target_name.lower()}"
            add_node(source_id, "Concept", source_name, {"name": source_name})
            add_node(target_id, "Concept", target_name, {"name": target_name})
            add_edge(source_id, target_id, "RELATES_TO", {"type": rel_type})
    else:
        messages = neo4j_manager.get_all_messages(session_id)
        for msg in messages:
            message_id = str(msg.get("id", ""))
            add_node(
                message_id,
                "Message",
                str(msg.get("role", "message")),
                {
                    "session_id": session_id,
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "timestamp": msg.get("timestamp"),
                },
            )

            topic_name = msg.get("topic_name")
            if isinstance(topic_name, str) and topic_name:
                topic_node_id = f"topic:{topic_name}"
                add_node(topic_node_id, "Topic", topic_name, {"name": topic_name})
                add_edge(message_id, topic_node_id, "PART_OF")

            for concept_name in msg.get("concepts", []) or []:
                concept_str = str(concept_name).strip()
                if not concept_str:
                    continue
                concept_node_id = f"concept:{concept_str.lower()}"
                add_node(concept_node_id, "Concept", concept_str, {"name": concept_str})
                add_edge(message_id, concept_node_id, "DISCUSSES")

    analysis = neo4j_manager.get_analysis(session_id) or {}

    for concept in analysis.get("concepts", []):
        concept_name = str(concept.get("name", "")).strip()
        if not concept_name:
            continue
        add_node(
            f"concept:{concept_name.lower()}",
            "Concept",
            concept_name,
            {k: v for k, v in concept.items() if k != "name"},
        )

    for topic in analysis.get("topics", []):
        topic_name = str(topic.get("name", "")).strip()
        if not topic_name:
            continue
        topic_node_id = f"topic:{topic_name}"
        add_node(topic_node_id, "Topic", topic_name, {"name": topic_name})
        for message_id in topic.get("message_ids", []) or []:
            add_edge(str(message_id), topic_node_id, "PART_OF")

    for rel in analysis.get("relationships", []):
        source_name = str(rel.get("source", "")).strip()
        target_name = str(rel.get("target", "")).strip()
        rel_type = str(rel.get("type", "related")).strip() or "related"
        if not source_name or not target_name:
            continue
        source_id = f"concept:{source_name.lower()}"
        target_id = f"concept:{target_name.lower()}"
        add_node(source_id, "Concept", source_name, {"name": source_name})
        add_node(target_id, "Concept", target_name, {"name": target_name})
        add_edge(source_id, target_id, "RELATES_TO", {"type": rel_type})

    for point in analysis.get("deflection_points", []) or []:
        message_id = str(point.get("message_id", "")).strip()
        from_topic = str(point.get("from_topic", "unknown")).strip() or "unknown"
        to_topic = str(point.get("to_topic", "unknown")).strip() or "unknown"
        deflection_node_id = f"deflection:{session_id}:{message_id or len(nodes)}"
        add_node(
            deflection_node_id,
            "DeflectionPoint",
            "Deflection",
            {
                "message_id": message_id,
                "from_topic": from_topic,
                "to_topic": to_topic,
                "signal_words": point.get("signal_words", []),
                "reason": point.get("reason"),
            },
        )
        if message_id:
            add_edge(message_id, deflection_node_id, "TRIGGERED")

        from_topic_id = f"topic:{from_topic}"
        to_topic_id = f"topic:{to_topic}"
        add_node(from_topic_id, "Topic", from_topic, {"name": from_topic})
        add_node(to_topic_id, "Topic", to_topic, {"name": to_topic})
        add_edge(from_topic_id, to_topic_id, "TRANSITIONS_TO", {"via": deflection_node_id})

    counts = {
        "messages": sum(1 for node in nodes if node["type"] == "Message"),
        "topics": sum(1 for node in nodes if node["type"] == "Topic"),
        "concepts": sum(1 for node in nodes if node["type"] == "Concept"),
        "deflection_points": sum(1 for node in nodes if node["type"] == "DeflectionPoint"),
        "edges": len(edges),
    }

    return {
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges,
        "counts": counts,
        "analysis_available": bool(analysis),
        "source": source,
        "notes": [
            "Graph combines stored message/topic/concept links with consolidation output (relationships/deflection points).",
            "Run /api/analyze (or reach the 10-message auto-trigger) to enrich RELATES_TO and TRANSITIONS_TO edges.",
        ],
    }


def _extract_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _detect_research_request(
    user_message: str,
    recent_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    detection = openai_manager.detect_research_intent(user_message, recent_messages)
    return {
        "is_research": bool(detection.get("is_research", False)),
        "reason": str(detection.get("reason", "")).strip(),
        "search_query": str(detection.get("search_query", user_message)).strip() or user_message,
        "topic": str(detection.get("topic", "general")).strip().lower() or "general",
        "method": str(detection.get("method", "heuristic-fallback")).strip(),
    }


def _call_tavily_search(
    query: str,
    *,
    topic: str = "general",
    max_results: int = 5,
) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {
            "used": False,
            "provider": "tavily",
            "error": "TAVILY_API_KEY is not configured.",
            "sources": [],
        }

    search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "basic").strip() or "basic"
    max_results = max(1, min(max_results, 10))
    if topic not in {"general", "news", "finance"}:
        topic = "general"

    payload = {
        "query": query,
        "topic": topic,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": "basic",
        "include_raw_content": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post("https://api.tavily.com/search", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return {
            "used": False,
            "provider": "tavily",
            "query": query,
            "error": str(exc),
            "sources": [],
        }

    raw_results = data.get("results", [])
    normalized_sources: list[dict[str, Any]] = []
    if isinstance(raw_results, list):
        for idx, result in enumerate(raw_results, start=1):
            if not isinstance(result, dict):
                continue
            url = str(result.get("url", "")).strip()
            if not url:
                continue
            normalized_sources.append(
                {
                    "id": f"tavily:{idx}:{url}",
                    "url": url,
                    "title": str(result.get("title", "")).strip() or url,
                    "content": str(result.get("content", "")).strip(),
                    "snippet": str(result.get("content", "")).strip(),
                    "domain": str(result.get("domain", "")).strip() or _extract_domain(url),
                    "score": result.get("score"),
                    "published_date": str(result.get("published_date", "")).strip(),
                }
            )

    answer = str(data.get("answer", "")).strip()
    if not answer and normalized_sources:
        answer = " ".join(src["content"] for src in normalized_sources[:2] if src.get("content")).strip()

    return {
        "used": True,
        "provider": "tavily",
        "transport": "api-direct",
        "query": str(data.get("query", query)).strip() or query,
        "answer": answer,
        "sources": normalized_sources,
        "response_time": data.get("response_time"),
        "search_depth": search_depth,
        "topic": topic,
    }


def _format_tavily_context_for_llm(research: dict[str, Any]) -> str:
    sources = research.get("sources", []) or []
    lines = [
        "Tavily research context (external sources). Use this only if relevant and mention uncertainty when needed.",
    ]
    answer = str(research.get("answer", "")).strip()
    if answer:
        lines.append(f"Summary: {answer}")
    if sources:
        lines.append("Sources:")
    for idx, source in enumerate(sources[:4], start=1):
        if not isinstance(source, dict):
            continue
        title = str(source.get("title", "")).strip() or str(source.get("url", "")).strip()
        url = str(source.get("url", "")).strip()
        snippet = str(source.get("content", "")).strip()
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
        lines.append(f"{idx}. {title} ({url})")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
    return "\n".join(lines)


def _append_research_context_to_reply(reply: str, research: dict[str, Any]) -> str:
    sources = research.get("sources", []) or []
    if not research.get("used"):
        return reply

    lines = [reply.strip(), "", "Research Context (Tavily):"]
    answer = str(research.get("answer", "")).strip()
    if answer:
        lines.append(answer)
    if sources:
        lines.append("Sources:")
        for idx, source in enumerate(sources[:3], start=1):
            if not isinstance(source, dict):
                continue
            title = str(source.get("title", "")).strip() or str(source.get("url", "")).strip()
            url = str(source.get("url", "")).strip()
            lines.append(f"{idx}. {title} - {url}")
    return "\n".join(lines).strip()


neo4j_manager = Neo4jManager()
openai_manager = OpenAIManager()
analysis_agent = AnalysisAgent(openai_manager)
semantic_search = SemanticSearchService(neo4j_manager, openai_manager)


app = FastAPI(
    title="MindChat API",
    description=(
        "Cognitive-science-inspired chatbot API for Tip-of-the-Tongue retrieval. "
        "OpenAI = working memory, Neo4j = long-term memory."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "project": "MindChat",
        "architecture": {
            "working_memory": "OpenAI (sliding window, 20 messages)",
            "long_term_memory": "Neo4j knowledge graph",
            "retrieval_mode": "semantic cues for TOT queries",
        },
        "integrations": {
            "openai_configured": openai_manager.is_configured,
            "neo4j_configured": neo4j_manager.is_configured,
            "neo4j_connected": neo4j_manager.verify_connection() if neo4j_manager.is_connected else False,
            "tavily_configured": semantic_search.tavily_configured,
            "tavily_mode": semantic_search.tavily_mode,
        },
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    user_message = neo4j_manager.store_message(request.session_id, "user", request.message.strip())
    recent_messages = neo4j_manager.get_recent_messages(request.session_id, limit=20)
    research_detection = _detect_research_request(request.message.strip(), recent_messages)
    research_context: dict[str, Any] | None = None
    enriched_messages = list(recent_messages)

    if research_detection["is_research"]:
        tavily_payload = _call_tavily_search(
            research_detection["search_query"],
            topic=research_detection["topic"],
            max_results=5,
        )
        if tavily_payload.get("used"):
            research_context = {
                **tavily_payload,
                "detection": research_detection,
            }
            enriched_messages = [
                *recent_messages,
                {
                    "role": "system",
                    "content": _format_tavily_context_for_llm(research_context),
                },
            ]
        else:
            research_context = {
                **tavily_payload,
                "detection": research_detection,
            }

    reply = openai_manager.get_chat_response(enriched_messages)
    if research_context and research_context.get("used"):
        reply = _append_research_context_to_reply(reply, research_context)

    assistant_message = neo4j_manager.store_message(request.session_id, "assistant", reply)

    # Tier 1: quick concept extraction on every exchange (immediate encoding).
    concepts_extracted: list[str] = []
    quick_extraction_error: str | None = None
    try:
        concepts_extracted = openai_manager.extract_concepts_quick(request.message.strip(), reply)
        if concepts_extracted:
            neo4j_manager.add_concepts_to_message(
                message_id=user_message["id"],
                concepts=concepts_extracted,
                session_id=request.session_id,
            )
            neo4j_manager.add_concepts_to_message(
                message_id=assistant_message["id"],
                concepts=concepts_extracted,
                session_id=request.session_id,
            )
            print(f"Extracted {len(concepts_extracted)} quick concepts: {concepts_extracted}")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        quick_extraction_error = str(exc)
        print(f"Quick concept extraction failed: {exc}")

    stored_research_sources: list[dict[str, Any]] = []
    if research_context and research_context.get("used"):
        try:
            extracted = semantic_search._extract_query_concepts(  # noqa: SLF001 - pragmatic reuse in hackathon scaffold
                research_detection["search_query"]
            )
            linked_concepts = extracted.get("concepts", []) if isinstance(extracted, dict) else []
            stored_research_sources = neo4j_manager.store_research_sources(
                request.session_id,
                message_id=assistant_message["id"],
                sources=research_context.get("sources", []) or [],
                concepts=linked_concepts,
                research_query=research_detection["search_query"],
            )
            research_context["linked_concepts"] = linked_concepts
            research_context["stored_source_count"] = len(stored_research_sources)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            research_context["storage_error"] = str(exc)

    total_messages = neo4j_manager.get_message_count(request.session_id)
    consolidation_due = total_messages > 0 and total_messages % 10 == 0
    consolidation_ran = False
    consolidation_summary: dict[str, Any] | None = None
    consolidation_error: str | None = None

    if consolidation_due:
        print(f"Triggering deep analysis at {total_messages} messages...")
        try:
            # Consolidation uses full conversation (long-term memory), not the working-memory window.
            full_conversation = neo4j_manager.get_all_messages(request.session_id)
            analysis = analysis_agent.analyze_conversation(full_conversation, request.session_id)
            neo4j_manager.upsert_analysis(request.session_id, analysis)
            consolidation_ran = True
            consolidation_summary = {
                "analysis_mode": analysis.get("analysis_mode"),
                "model": analysis.get("model"),
                "topic_count": len(analysis.get("topics", [])),
                "concept_count": len(analysis.get("concepts", [])),
                "relationship_count": len(analysis.get("relationships", [])),
                "deflection_point_count": len(analysis.get("deflection_points", [])),
            }
            print(
                "Deep analysis complete: "
                f"{consolidation_summary['topic_count']} topics, "
                f"{consolidation_summary['deflection_point_count']} deflection points"
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            consolidation_error = str(exc)
            print(f"Deep analysis failed: {exc}")

    return {
        "session_id": request.session_id,
        "user_message_id": user_message["id"],
        "message_id": assistant_message["id"],
        "response": reply,
        "reply": reply,
        "assistant_message_id": assistant_message["id"],
        "working_memory_size": min(len(recent_messages), openai_manager.max_working_memory_messages),
        "total_messages": total_messages,
        "message_count": total_messages,
        "concepts_extracted": concepts_extracted,
        "quick_extraction_error": quick_extraction_error,
        "deep_analysis_triggered": consolidation_due,
        "consolidation_due": consolidation_due,
        "consolidation_ran": consolidation_ran,
        "consolidation_summary": consolidation_summary,
        "consolidation_error": consolidation_error,
        "research_hint": bool(research_detection["is_research"]),
        "research_detection": research_detection,
        "research_context": research_context,
        "notes": [
            "Chat path uses working memory (last ~20 messages).",
            "Quick concept extraction runs on every exchange for incremental graph updates.",
            "Long-term memory consolidation runs automatically every 10 messages using the FULL conversation.",
            "You can also call /api/analyze manually to force consolidation.",
            "Research questions can trigger Tavily search; sources are stored as Neo4j Source nodes linked to concepts.",
        ],
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    conversation = neo4j_manager.get_all_messages(request.session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="No conversation found for session.")

    analysis = analysis_agent.analyze_conversation(conversation, request.session_id)
    neo4j_manager.upsert_analysis(request.session_id, analysis)

    return {
        "session_id": request.session_id,
        "analysis_mode": analysis.get("analysis_mode"),
        "topic_count": len(analysis.get("topics", [])),
        "concept_count": len(analysis.get("concepts", [])),
        "relationship_count": len(analysis.get("relationships", [])),
        "deflection_point_count": len(analysis.get("deflection_points", [])),
        "analysis": analysis,
    }


@app.get("/api/search")
def search_get(
    q: str = Query(..., min_length=1, description="Vague TOT query"),
    session_id: str | None = Query(default=None, description="Conversation/session to search"),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    resolved_session_id = _resolve_session_id(session_id)
    if not resolved_session_id:
        raise HTTPException(status_code=404, detail="No conversation found to search.")

    result = semantic_search.semantic_search(resolved_session_id, q, limit=limit)
    if session_id is None:
        result["resolved_session_id"] = resolved_session_id
    return result


@app.post("/api/search")
def search_post(request: SearchRequest) -> dict[str, Any]:
    """Backward-compatible POST search route for early frontend scaffolds."""
    return semantic_search.semantic_search(request.session_id, request.query, limit=request.limit)


@app.get("/api/graph")
def graph(
    session_id: str | None = Query(default=None, description="Conversation/session to visualize"),
    message_limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    resolved_session_id = _resolve_session_id(session_id)
    if not resolved_session_id:
        raise HTTPException(status_code=404, detail="No conversation found for graph visualization.")

    response = _build_graph_response(resolved_session_id, message_limit=message_limit)
    if session_id is None:
        response["resolved_session_id"] = resolved_session_id
    return response


@app.on_event("shutdown")
def _shutdown() -> None:
    neo4j_manager.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
