from __future__ import annotations

import asyncio
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
from deflection_detector import SemanticDeflectionDetector
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


async def _build_concept_payloads(
    concept_names: list[str],
    *,
    lowercase: bool = False,
    embedding_cache: dict[str, list[float] | None] | None = None,
) -> list[dict[str, Any]]:
    normalized_names: list[str] = []
    seen: set[str] = set()
    for concept_name in concept_names:
        normalized = str(concept_name).strip()
        if lowercase:
            normalized = normalized.lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_names.append(normalized)

    if not normalized_names:
        return []

    cache = embedding_cache if embedding_cache is not None else {}
    missing_names = [name for name in normalized_names if name not in cache]
    if missing_names:
        embeddings = await asyncio.gather(
            *(openai_manager.get_embedding(name) for name in missing_names)
        )
        for name, embedding in zip(missing_names, embeddings, strict=False):
            cache[name] = embedding

    return [
        {
            "name": name,
            "embedding": cache.get(name),
        }
        for name in normalized_names
    ]


def _build_concept_payloads_sync(
    concept_names: list[str],
    *,
    lowercase: bool = False,
    embedding_cache: dict[str, list[float] | None] | None = None,
) -> list[dict[str, Any]]:
    return asyncio.run(
        _build_concept_payloads(
            concept_names,
            lowercase=lowercase,
            embedding_cache=embedding_cache,
        )
    )


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


def _deflection_points_from_neo4j(session_id: str) -> list[dict[str, Any]]:
    if not neo4j_manager.driver:
        return []

    query = """
    MATCH (d:DeflectionPoint {session_id: $session_id})
    RETURN
      d.message_id AS message_id,
      d.from_topic AS from_topic,
      d.to_topic AS to_topic,
      d.deflection_type AS deflection_type,
      d.similarity_score AS similarity_score,
      d.created_at AS created_at
    ORDER BY d.created_at ASC
    """
    try:
        with neo4j_manager.driver.session() as session:
            return session.run(query, session_id=session_id).data()
    except Exception:
        return []


def _build_graph_response(session_id: str, *, message_limit: int = 500) -> dict[str, Any]:
    message_count = neo4j_manager.get_message_count(session_id)
    source = "neo4j" if neo4j_manager.driver else "memory-fallback"
    root_topic = "General"

    if not neo4j_manager.driver:
        return {
            "session_id": session_id,
            "source": source,
            "tree": {
                "nodes": [],
                "edges": [],
                "root": root_topic,
            },
            "message_count": message_count,
            "deflection_count": 0,
        }

    query = """
    MATCH (d:DeflectionPoint {session_id: $session_id})
    RETURN d.from_topic AS from_topic,
           d.to_topic AS to_topic,
           d.deflection_type AS type,
           d.similarity_score AS score,
           d.created_at AS created_at
    ORDER BY d.created_at ASC
    """
    try:
        with neo4j_manager.driver.session() as session:
            rows = session.run(query, session_id=session_id).data()
    except Exception:
        rows = []
        source = "memory-fallback"

    if not rows:
        return {
            "session_id": session_id,
            "source": source,
            "tree": {
                "nodes": [],
                "edges": [],
                "root": root_topic,
            },
            "message_count": message_count,
            "deflection_count": 0,
        }

    topic_nodes: dict[str, dict[str, Any]] = {}
    tree_edges: list[dict[str, Any]] = []
    children_by_parent: dict[str, list[str]] = {}
    ordered_topics: list[str] = []
    seen_edges: set[tuple[str, str, str, float]] = set()

    def ensure_topic(topic_name: str) -> None:
        if not topic_name or topic_name in topic_nodes:
            return
        topic_nodes[topic_name] = {
            "id": topic_name,
            "label": topic_name,
            "depth": 0,
            "type": "BRANCH",
        }
        ordered_topics.append(topic_name)

    first_row = rows[0] if rows else {}
    first_from_topic = str(first_row.get("from_topic", "")).strip()
    first_to_topic = str(first_row.get("to_topic", "")).strip()
    first_type = str(first_row.get("type", "")).strip().upper()
    if first_from_topic == "ROOT" or first_type == "ROOT":
        root_topic = first_to_topic or root_topic
    else:
        root_topic = first_from_topic or first_to_topic or root_topic
    ensure_topic(root_topic)
    topic_nodes[root_topic]["type"] = "ROOT"

    for row in rows:
        from_topic = str(row.get("from_topic", "")).strip()
        to_topic = str(row.get("to_topic", "")).strip()
        edge_type = str(row.get("type", "BRANCH")).strip().upper() or "BRANCH"
        if edge_type == "ROOT" or from_topic == "ROOT":
            if to_topic:
                ensure_topic(to_topic)
            continue
        if edge_type not in {"EXTENSION", "BRANCH", "SYNTHESIS"}:
            edge_type = "BRANCH"

        try:
            score = float(row.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        if not from_topic or not to_topic:
            continue

        ensure_topic(from_topic)
        ensure_topic(to_topic)

        if from_topic != root_topic and topic_nodes[from_topic]["type"] == "BRANCH":
            topic_nodes[from_topic]["type"] = "EXTENSION"
        if to_topic != root_topic:
            topic_nodes[to_topic]["type"] = edge_type

        edge_key = (from_topic, to_topic, edge_type, round(score, 6))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        tree_edges.append(
            {
                "from": from_topic,
                "to": to_topic,
                "type": edge_type,
                "score": score,
            }
        )
        children_by_parent.setdefault(from_topic, []).append(to_topic)

    depths: dict[str, int] = {root_topic: 0}
    queue: list[str] = [root_topic]
    while queue:
        parent = queue.pop(0)
        parent_depth = depths[parent]
        for child in children_by_parent.get(parent, []):
            if child in depths:
                continue
            depths[child] = parent_depth + 1
            queue.append(child)

    for topic_name in ordered_topics:
        if topic_name in depths:
            continue
        candidate_depths = [
            depths[edge["from"]] + 1
            for edge in tree_edges
            if edge["to"] == topic_name and edge["from"] in depths
        ]
        if candidate_depths:
            depths[topic_name] = min(candidate_depths)
        else:
            depths[topic_name] = max(depths.values(), default=0) + 1

    tree_nodes = [
        {
            "id": topic_name,
            "label": node["label"],
            "depth": depths.get(topic_name, 0),
            "type": "ROOT" if topic_name == root_topic else node["type"],
        }
        for topic_name, node in topic_nodes.items()
    ]
    tree_nodes.sort(key=lambda node: (int(node.get("depth", 0)), str(node.get("label", "")).lower()))

    return {
        "session_id": session_id,
        "source": source,
        "tree": {
            "nodes": tree_nodes,
            "edges": tree_edges,
            "root": root_topic,
        },
        "message_count": message_count,
        "deflection_count": len(tree_edges),
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
deflection_detector = SemanticDeflectionDetector(neo4j_manager, openai_manager)
_session_current_topic: dict[str, str] = {}


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


@app.on_event("startup")
def _startup() -> None:
    if neo4j_manager.ensure_vector_index():
        print("Vector index ensured.")
    if neo4j_manager.cleanup_noise_concepts():
        print("Cleaned up noise concepts.")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id
    stripped_message = request.message.strip()
    embedding_cache: dict[str, list[float] | None] = {}
    current_topic = _session_current_topic.get(session_id, "General")

    user_concepts = await _build_concept_payloads(
        neo4j_manager._extract_concepts(stripped_message),  # noqa: SLF001 - preserves existing concept extraction path
        embedding_cache=embedding_cache,
    )
    user_message = neo4j_manager.store_message(
        session_id,
        "user",
        stripped_message,
        concepts=user_concepts,
    )
    recent_messages = neo4j_manager.get_recent_messages(session_id, limit=20)
    research_detection = _detect_research_request(stripped_message, recent_messages)
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

    assistant_concepts = await _build_concept_payloads(
        neo4j_manager._extract_concepts(reply),  # noqa: SLF001 - preserves existing concept extraction path
        embedding_cache=embedding_cache,
    )
    assistant_message = neo4j_manager.store_message(
        session_id,
        "assistant",
        reply,
        concepts=assistant_concepts,
    )

    # Tier 1: quick concept extraction on every exchange (immediate encoding).
    concepts_extracted: list[str] = []
    quick_extraction_error: str | None = None
    try:
        concepts_extracted = openai_manager.extract_concepts_quick(stripped_message, reply)
        if concepts_extracted:
            quick_concept_payloads = await _build_concept_payloads(
                concepts_extracted,
                lowercase=True,
                embedding_cache=embedding_cache,
            )
            neo4j_manager.add_concepts_to_message(
                message_id=user_message["id"],
                concepts=quick_concept_payloads,
                session_id=session_id,
            )
            neo4j_manager.add_concepts_to_message(
                message_id=assistant_message["id"],
                concepts=quick_concept_payloads,
                session_id=session_id,
            )
            print(f"Extracted {len(concepts_extracted)} quick concepts: {concepts_extracted}")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        quick_extraction_error = str(exc)
        print(f"Quick concept extraction failed: {exc}")

    msg_count = neo4j_manager.get_message_count(session_id)
    if msg_count <= 2:
        await neo4j_manager.ensure_root_topic(session_id, "General")

    deflection_result = await deflection_detector.detect(
        session_id=session_id,
        new_message_content=stripped_message,
        new_concepts=concepts_extracted,
        current_topic_label=current_topic,
    )
    if deflection_result["type"] in ("BRANCH", "SYNTHESIS"):
        await neo4j_manager.store_deflection_point(
            session_id=session_id,
            message_id=user_message["id"],
            from_topic=deflection_result["parent_topic"],
            to_topic=deflection_result["topic_label"],
            deflection_type=deflection_result["type"],
            similarity_score=deflection_result["similarity_score"],
        )
    current_topic = str(deflection_result.get("topic_label", current_topic)).strip() or current_topic
    _session_current_topic[session_id] = current_topic

    stored_research_sources: list[dict[str, Any]] = []
    if research_context and research_context.get("used"):
        try:
            extracted = semantic_search._extract_query_concepts(  # noqa: SLF001 - pragmatic reuse in hackathon scaffold
                research_detection["search_query"]
            )
            linked_concepts = extracted.get("concepts", []) if isinstance(extracted, dict) else []
            stored_research_sources = neo4j_manager.store_research_sources(
                session_id,
                message_id=assistant_message["id"],
                sources=research_context.get("sources", []) or [],
                concepts=linked_concepts,
                research_query=research_detection["search_query"],
            )
            research_context["linked_concepts"] = linked_concepts
            research_context["stored_source_count"] = len(stored_research_sources)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            research_context["storage_error"] = str(exc)

    total_messages = neo4j_manager.get_message_count(session_id)
    consolidation_due = total_messages > 0 and total_messages % 10 == 0
    consolidation_ran = False
    consolidation_summary: dict[str, Any] | None = None
    consolidation_error: str | None = None

    if consolidation_due:
        print(f"Triggering deep analysis at {total_messages} messages...")
        try:
            # Consolidation uses full conversation (long-term memory), not the working-memory window.
            full_conversation = neo4j_manager.get_all_messages(session_id)
            analysis = analysis_agent.analyze_conversation(full_conversation, session_id)
            neo4j_manager.upsert_analysis(session_id, analysis)
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
        "session_id": session_id,
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
        "deflection": {
            "type": deflection_result["type"],
            "topic_label": deflection_result["topic_label"],
            "similarity_score": deflection_result["similarity_score"],
        },
        "notes": [
            "Chat path uses working memory (last ~20 messages).",
            "Quick concept extraction runs on every exchange for incremental graph updates.",
            "Real-time semantic deflection detection updates the session topic state and persists BRANCH/SYNTHESIS transitions.",
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
