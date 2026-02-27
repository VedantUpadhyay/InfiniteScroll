from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import logging
from typing import Any
from uuid import uuid4
import os
import re

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - dependency not installed yet
    GraphDatabase = None  # type: ignore[assignment]


class Neo4jManager:
    """Long-term memory manager backed by Neo4j, with in-memory fallback for local scaffold use."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.uri = os.getenv("NEO4J_URI", "")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")
        self.driver = None
        self.last_error: str | None = None
        self._messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._analysis: dict[str, dict[str, Any]] = {}
        self._sources: dict[str, list[dict[str, Any]]] = defaultdict(list)

        if self.is_configured and GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                # AuraDB connection issues should not crash local development; fallback remains active.
                self.driver.verify_connectivity()
            except Exception as exc:
                self._record_error("Failed to connect to Neo4j AuraDB", exc)
                self.driver = None
        elif GraphDatabase is None:
            self._record_error("Neo4j driver is not installed", None)

    def _record_error(self, message: str, exc: Exception | None) -> None:
        if exc is None:
            self.last_error = message
            self.logger.debug(message)
            return
        self.last_error = f"{message}: {exc}"
        self.logger.warning("%s: %s", message, exc)

    def _extract_concepts(self, content: str, max_concepts: int = 6) -> list[str]:
        """Heuristic concept extraction for Day 1 storage; analysis agent deepens graph later."""
        tokens = re.findall(r"[A-Za-z0-9]+", content.lower())
        stopwords = {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "have",
            "just",
            "like",
            "what",
            "when",
            "where",
            "which",
            "would",
            "could",
            "should",
            "there",
            "about",
            "into",
            "your",
            "because",
            "however",
            "therefore",
            "thing",
            "was",
            "were",
        }
        concepts: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in stopwords:
                continue
            if not (len(token) > 2 or token.isdigit()):
                continue
            if token in seen:
                continue
            seen.add(token)
            concepts.append(token)
            if len(concepts) >= max_concepts:
                break
        return concepts

    @property
    def is_configured(self) -> bool:
        return bool(self.uri and self.user and self.password)

    @property
    def is_connected(self) -> bool:
        return self.driver is not None

    def verify_connection(self) -> bool:
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as exc:
            self._record_error("Neo4j connectivity check failed", exc)
            return False

    def close(self) -> None:
        if self.driver:
            try:
                self.driver.close()
            except Exception as exc:
                self._record_error("Neo4j driver close failed", exc)

    def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        timestamp: str | None = None,
        topic_name: str | None = None,
        concepts: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_topic = topic_name or f"session:{session_id}"
        concept_names = concepts if concepts is not None else self._extract_concepts(content)
        concept_names = [c.strip() for c in concept_names if c and c.strip()]

        msg = {
            "id": message_id or str(uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "topic_name": normalized_topic,
            "concepts": concept_names,
        }

        self._messages[session_id].append(msg)

        if self.driver:
            message_and_topic_query = """
            CREATE (m:Message {
              id: $id,
              role: $role,
              content: $content,
              timestamp: $timestamp,
              session_id: $session_id
            })
            MERGE (t:Topic {name: $topic_name})
            ON CREATE SET t.message_ids = []
            SET t.message_ids = CASE
              WHEN $id IN coalesce(t.message_ids, []) THEN coalesce(t.message_ids, [])
              ELSE coalesce(t.message_ids, []) + $id
            END
            MERGE (m)-[:PART_OF]->(t)
            RETURN m.id AS id
            """
            message_concepts_query = """
            MATCH (m:Message {id: $message_id})
            UNWIND $concept_names AS concept_name
            MERGE (c:Concept {name: concept_name})
            MERGE (m)-[:DISCUSSES]->(c)
            """
            params = {
                "id": msg["id"],
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "session_id": msg["session_id"],
                "topic_name": msg["topic_name"],
            }
            try:
                with self.driver.session() as session:
                    session.run(message_and_topic_query, **params).consume()
                    if msg["concepts"]:
                        session.run(
                            message_concepts_query,
                            message_id=msg["id"],
                            concept_names=msg["concepts"],
                        ).consume()
            except Exception as exc:
                # Fallback stays available for local development even if AuraDB is offline.
                self._record_error("Failed to store message in Neo4j", exc)

        return msg

    def get_all_messages(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if self.driver:
            query_all = """
            MATCH (m:Message)
            RETURN m.id AS id, m.session_id AS session_id, m.role AS role, m.content AS content, m.timestamp AS timestamp
            ORDER BY m.timestamp ASC
            """
            query_session = """
            MATCH (m:Message {session_id: $session_id})
            RETURN m.id AS id, m.session_id AS session_id, m.role AS role, m.content AS content, m.timestamp AS timestamp
            ORDER BY m.timestamp ASC
            """
            try:
                with self.driver.session() as session:
                    if session_id is None:
                        rows = session.run(query_all).data()
                    else:
                        rows = session.run(query_session, session_id=session_id).data()
                return [
                    {
                        "id": row["id"],
                        "session_id": row.get("session_id"),
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                    }
                    for row in rows
                ]
            except Exception as exc:
                self._record_error("Failed to fetch messages from Neo4j", exc)

        if session_id is None:
            all_messages = [
                msg
                for session_messages in self._messages.values()
                for msg in session_messages
            ]
            return sorted(all_messages, key=lambda item: item["timestamp"])

        return list(self._messages.get(session_id, []))

    def get_recent_messages(
        self,
        session_id: str,
        n: int = 20,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        effective_limit = limit if limit is not None else n
        if self.driver:
            query = """
            MATCH (m:Message {session_id: $session_id})
            RETURN m.id AS id, m.role AS role, m.content AS content, m.timestamp AS timestamp
            ORDER BY m.timestamp DESC
            LIMIT $limit
            """
            try:
                with self.driver.session() as session:
                    rows = session.run(query, session_id=session_id, limit=effective_limit).data()
                rows.reverse()
                return [
                    {
                        "id": row["id"],
                        "session_id": session_id,
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                    }
                    for row in rows
                ]
            except Exception as exc:
                self._record_error("Failed to fetch recent messages from Neo4j", exc)

        return self._messages.get(session_id, [])[-effective_limit:]

    def get_full_conversation(self, session_id: str) -> list[dict[str, Any]]:
        # Backward-compatible alias used by the scaffold.
        return self.get_all_messages(session_id)

    def count_messages(self, session_id: str) -> int:
        if self.driver:
            query = """
            MATCH (m:Message {session_id: $session_id})
            RETURN count(m) AS count
            """
            try:
                with self.driver.session() as session:
                    row = session.run(query, session_id=session_id).single()
                if row is not None:
                    return int(row.get("count", 0))
            except Exception as exc:
                self._record_error("Failed to count messages in Neo4j", exc)

        return len(self._messages.get(session_id, []))

    def get_message_count(self, session_id: str) -> int:
        """Compatibility alias for chat orchestration code."""
        return self.count_messages(session_id)

    def add_concepts_to_message(self, message_id: str, concepts: list[str], session_id: str) -> None:
        """
        Incrementally link extracted concepts to a specific message.
        Called for every chat exchange to avoid graph gaps between deep analyses.
        """
        normalized_concepts: list[str] = []
        seen: set[str] = set()
        for concept in concepts:
            concept_str = str(concept).lower().strip()
            if not concept_str or concept_str in seen:
                continue
            seen.add(concept_str)
            normalized_concepts.append(concept_str)

        if not normalized_concepts:
            return

        # Keep memory fallback consistent even when Neo4j is unavailable.
        session_messages = self._messages.get(session_id, [])
        for message in session_messages:
            if message.get("id") != message_id:
                continue
            existing = [str(item).strip() for item in message.get("concepts", []) if str(item).strip()]
            merged = list(dict.fromkeys([*existing, *normalized_concepts]))
            message["concepts"] = merged
            break

        if not self.driver:
            return

        query = """
        MATCH (m:Message {id: $message_id, session_id: $session_id})
        UNWIND $concepts AS concept_name
        MERGE (c:Concept {name: concept_name})
        ON CREATE SET c.first_seen = timestamp()
        MERGE (m)-[:DISCUSSES]->(c)
        WITH DISTINCT c
        MATCH (c)<-[:DISCUSSES]-(related_message:Message)
        WITH c, count(DISTINCT related_message) AS msg_count
        SET c.mention_count = msg_count
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    message_id=message_id,
                    session_id=session_id,
                    concepts=normalized_concepts,
                ).consume()
        except Exception as exc:
            self._record_error("Failed to incrementally add concepts to message", exc)

    def upsert_analysis(self, session_id: str, analysis: dict[str, Any]) -> None:
        self._analysis[session_id] = analysis

        if not self.driver:
            return

        try:
            with self.driver.session() as session:
                for topic in analysis.get("topics", []):
                    session.run(
                        """
                        MERGE (t:Topic {name: $name})
                        ON CREATE SET t.message_ids = $message_ids
                        ON MATCH SET t.message_ids = coalesce(t.message_ids, []) + $message_ids
                        """,
                        name=topic.get("name", "Unknown Topic"),
                        message_ids=topic.get("message_ids", []),
                    ).consume()

                for concept in analysis.get("concepts", []):
                    session.run(
                        "MERGE (c:Concept {name: $name})",
                        name=concept.get("name", ""),
                    ).consume()

                for relation in analysis.get("relationships", []):
                    session.run(
                        """
                        MERGE (a:Concept {name: $source})
                        MERGE (b:Concept {name: $target})
                        MERGE (a)-[r:RELATES_TO {type: $relation_type}]->(b)
                        """,
                        source=relation.get("source", ""),
                        target=relation.get("target", ""),
                        relation_type=relation.get("type", "related"),
                    ).consume()

                for point in analysis.get("deflection_points", []):
                    session.run(
                        """
                        MERGE (from_t:Topic {name: $from_topic})
                        MERGE (to_t:Topic {name: $to_topic})
                        MERGE (d:DeflectionPoint {
                          message_id: $message_id,
                          from_topic: $from_topic,
                          to_topic: $to_topic
                        })
                        MERGE (from_t)-[:TRANSITIONS_TO {via: $message_id}]->(to_t)
                        """,
                        message_id=point.get("message_id", ""),
                        from_topic=point.get("from_topic", "unknown"),
                        to_topic=point.get("to_topic", "unknown"),
                    ).consume()
        except Exception:
            pass

    def update_graph_structure(self, session_id: str, analysis: dict[str, Any]) -> None:
        """Compatibility alias for orchestration code naming."""
        self.upsert_analysis(session_id, analysis)

    def get_analysis(self, session_id: str) -> dict[str, Any] | None:
        return self._analysis.get(session_id)

    def store_research_sources(
        self,
        session_id: str,
        *,
        message_id: str,
        sources: list[dict[str, Any]],
        concepts: list[str] | None = None,
        research_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Persist Tavily sources as Source nodes and link them to concepts."""
        normalized_concepts = [
            str(concept).strip().lower()
            for concept in (concepts or [])
            if str(concept).strip()
        ]

        normalized_sources: list[dict[str, Any]] = []
        for idx, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                continue
            url = str(source.get("url", "")).strip()
            if not url:
                continue

            content = str(source.get("content", "") or source.get("snippet", "")).strip()
            title = str(source.get("title", "")).strip() or url
            domain = str(source.get("domain", "")).strip()
            normalized_sources.append(
                {
                    "id": str(source.get("id", "")).strip() or f"{message_id}:{idx}",
                    "url": url,
                    "title": title,
                    "content": content,
                    "domain": domain,
                    "score": source.get("score"),
                    "published_date": str(source.get("published_date", "")).strip(),
                    "query": research_query or "",
                    "message_id": message_id,
                    "session_id": session_id,
                }
            )

        if not normalized_sources:
            return []

        self._sources[session_id].extend(normalized_sources)

        if not self.driver:
            return normalized_sources

        query = """
        MATCH (m:Message {id: $message_id})
        UNWIND $sources AS src
        MERGE (s:Source {url: src.url})
        ON CREATE SET s.id = src.id
        SET
          s.title = src.title,
          s.content = src.content,
          s.domain = src.domain,
          s.score = src.score,
          s.published_date = src.published_date,
          s.session_id = src.session_id,
          s.research_query = src.query,
          s.last_retrieved_at = $retrieved_at
        MERGE (m)-[:USES_SOURCE]->(s)
        WITH s
        UNWIND $concepts AS concept_name
        MERGE (c:Concept {name: concept_name})
        MERGE (s)-[:MENTIONS]->(c)
        """

        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    message_id=message_id,
                    sources=normalized_sources,
                    concepts=normalized_concepts,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                ).consume()
        except Exception as exc:
            self._record_error("Failed to store research sources in Neo4j", exc)

        return normalized_sources

    def get_research_sources(self, session_id: str) -> list[dict[str, Any]]:
        if self.driver:
            query = """
            MATCH (:Message {session_id: $session_id})-[:USES_SOURCE]->(s:Source)
            RETURN DISTINCT
              s.id AS id,
              s.url AS url,
              s.title AS title,
              s.content AS content,
              s.domain AS domain,
              s.score AS score,
              s.published_date AS published_date,
              s.research_query AS query
            ORDER BY s.last_retrieved_at DESC
            """
            try:
                with self.driver.session() as session:
                    return session.run(query, session_id=session_id).data()
            except Exception as exc:
                self._record_error("Failed to fetch research sources from Neo4j", exc)

        return list(self._sources.get(session_id, []))

    def semantic_search(
        self,
        session_id: str,
        query: str,
        cues: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        normalized_cues = [c.lower().strip() for c in cues if c.strip()]

        if self.driver:
            neo4j_query = """
            MATCH (m:Message {session_id: $session_id})
            OPTIONAL MATCH (m)-[:DISCUSSES]->(c:Concept)
            WITH m, collect(toLower(c.name)) AS concept_names, $cues AS cues
            WITH m, [cue IN cues WHERE
              toLower(m.content) CONTAINS cue OR
              ANY(concept_name IN concept_names WHERE concept_name CONTAINS cue)
            ] AS matched_cues
            WHERE size(matched_cues) > 0
            RETURN
              m.id AS id,
              m.role AS role,
              m.content AS content,
              m.timestamp AS timestamp,
              matched_cues
            ORDER BY size(matched_cues) DESC, m.timestamp DESC
            LIMIT $limit
            """
            try:
                with self.driver.session() as session:
                    rows = session.run(
                        neo4j_query,
                        session_id=session_id,
                        cues=normalized_cues,
                        limit=limit,
                    ).data()
                return [
                    {
                        "id": row["id"],
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "matched_cues": row.get("matched_cues", []),
                    }
                    for row in rows
                ]
            except Exception as exc:
                self._record_error("Semantic search query failed", exc)

        search_terms = set(normalized_cues)
        search_terms.update(
            token.lower()
            for token in query.split()
            if len(token.strip(".,!?")) > 2
            for token in [token.strip(".,!?")]
        )

        scored: list[dict[str, Any]] = []
        for msg in self._messages.get(session_id, []):
            haystack = msg["content"].lower()
            concept_haystack = " ".join(msg.get("concepts", []))
            matched = sorted(
                [
                    term
                    for term in search_terms
                    if term in haystack or term in concept_haystack
                ]
            )
            if matched:
                scored.append(
                    {
                        **msg,
                        "matched_cues": matched,
                        "score": len(matched),
                    }
                )

        scored.sort(key=lambda item: (item["score"], item["timestamp"]), reverse=True)
        return scored[:limit]
