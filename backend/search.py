from __future__ import annotations

from collections import defaultdict
import json
from typing import Any
import os
import re
import time
from urllib.parse import quote_plus

from neo4j_manager import Neo4jManager
from openai_manager import OpenAIManager


class SemanticSearchService:
    """Semantic concept-graph search service for Tip-of-the-Tongue queries."""

    def __init__(self, neo4j_manager: Neo4jManager, openai_manager: OpenAIManager) -> None:
        self.neo4j_manager = neo4j_manager
        self.openai_manager = openai_manager
        self.tavily_provider = os.getenv("TAVILY_PROVIDER", "mcp").strip().lower() or "mcp"
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.tavily_mcp_server_url = self._resolve_tavily_mcp_server_url()

    def _resolve_tavily_mcp_server_url(self) -> str:
        explicit = os.getenv("TAVILY_MCP_SERVER_URL", "").strip()
        if explicit:
            return explicit
        if self.tavily_api_key:
            # Tavily supports remote MCP; this avoids local MCP server setup during hackathon week.
            return f"https://mcp.tavily.com/mcp/?tavilyApiKey={quote_plus(self.tavily_api_key)}"
        return ""

    @property
    def tavily_configured(self) -> bool:
        if self.tavily_provider == "mcp":
            return bool(self.tavily_mcp_server_url)
        return bool(self.tavily_api_key)

    @property
    def tavily_mode(self) -> str:
        if self.tavily_provider == "mcp":
            return "mcp-remote" if self.tavily_mcp_server_url else "mcp-unconfigured"
        return "api-direct" if self.tavily_api_key else "api-unconfigured"

    def search(self, session_id: str, query: str, limit: int = 5) -> dict[str, Any]:
        """Backward-compatible alias."""
        return self.semantic_search(session_id, query, limit=limit)

    def semantic_search(self, session_id: str, query: str, limit: int = 5) -> dict[str, Any]:
        """Solve TOT queries by extracting concepts and querying the Neo4j concept graph."""
        extraction = self._extract_query_concepts(query)
        query_concepts = extraction["concepts"]
        search_backend = "neo4j-concept-graph" if self.neo4j_manager.is_connected else "local-concept-fallback"

        if self.neo4j_manager.is_connected and query_concepts:
            raw_matches = self._search_neo4j_concept_graph(session_id, query_concepts, limit=limit)
            if not raw_matches:
                # Fallback preserves UX if the concept graph is sparse or query traversal returns no rows.
                fallback_rows = self.neo4j_manager.semantic_search(
                    session_id, query, query_concepts, limit=limit
                )
                raw_matches = [
                    {
                        "id": row.get("id"),
                        "role": row.get("role"),
                        "content": row.get("content"),
                        "timestamp": row.get("timestamp"),
                        "topic_names": [],
                        "direct_concept_matches": row.get("matched_cues", []),
                        "related_concept_matches": [],
                        "relation_types": [],
                        "score": len(row.get("matched_cues", [])),
                    }
                    for row in fallback_rows
                ]
                if raw_matches:
                    search_backend = "neo4j-concept-graph+manager-fallback"
        else:
            raw_matches = self._search_local_concept_graph(session_id, query_concepts, limit=limit)

        # Last-resort fallback if concept extraction yields nothing.
        if not raw_matches and not query_concepts:
            fallback_cues = self.openai_manager.extract_retrieval_cues(query)
            raw_matches = self._search_local_concept_graph(session_id, fallback_cues, limit=limit)
            extraction = {
                **extraction,
                "fallback_concepts_used": fallback_cues,
                "method": f'{extraction.get("method", "unknown")}+heuristic-cues',
            }
            query_concepts = fallback_cues

        matches = []
        for row in raw_matches:
            direct_matches = row.get("direct_concept_matches", [])
            related_matches = row.get("related_concept_matches", [])
            topic_names = row.get("topic_names", [])

            why_parts = []
            if direct_matches:
                why_parts.append(
                    f"Direct concept match via `DISCUSSES`: {', '.join(direct_matches[:4])}"
                )
            if related_matches:
                why_parts.append(
                    f"Related concept match via `RELATES_TO`: {', '.join(related_matches[:4])}"
                )
            if topic_names:
                why_parts.append(f"Topic context: {', '.join(topic_names[:2])}")
            why_found = "; ".join(why_parts) if why_parts else "Matched via concept-graph fallback."

            matches.append(
                {
                    "id": row.get("id"),
                    "role": row.get("role"),
                    "content": row.get("content"),
                    "timestamp": row.get("timestamp"),
                    "score": row.get("score", 0),
                    "matched_cues": direct_matches + related_matches,  # backward compatibility
                    "direct_concept_matches": direct_matches,
                    "related_concept_matches": related_matches,
                    "topic_names": topic_names,
                    "relation_types": row.get("relation_types", []),
                    "why_found": why_found,
                }
            )

        return {
            "query": query,
            "session_id": session_id,
            "query_concepts": query_concepts,
            "retrieval_cues": query_concepts,  # backward compatibility
            "query_extraction": extraction,
            "matches": matches,
            "tot_mode": True,
            "search_strategy": "concept-graph",
            "search_backend": search_backend,
            "explanation": (
                "MindChat extracted semantic concepts from the vague query using OpenAI, then queried "
                "the Neo4j concept graph (`DISCUSSES` and `RELATES_TO`) instead of plain keyword search."
            ),
            "notes": [
                "Designed for Tip-of-the-Tongue retrieval: concept-first search over keyword matching.",
                "If results are weak, run /api/analyze after enough messages to enrich `RELATES_TO` graph edges.",
                "Tavily integration is reserved for research-enhanced responses in Day 6.",
                f"Tavily transport configured for scaffold: {self.tavily_mode}.",
            ],
        }

    def _extract_query_concepts(self, query: str) -> dict[str, Any]:
        heuristic_cues = self._normalize_concepts(self.openai_manager.extract_retrieval_cues(query))
        numeric_hints = self._normalize_concepts(re.findall(r"\d+", query))

        if not self.openai_manager.is_configured or self.openai_manager.client is None:
            concepts = self._merge_concept_lists(heuristic_cues, numeric_hints)
            return {
                "method": "heuristic-fallback",
                "model": None,
                "concepts": concepts,
                "raw_query": query,
            }

        prompt = f"""
You are extracting retrieval concepts for a Tip-of-the-Tongue search query.
The user may not remember exact keywords.

Extract semantic concepts the query likely refers to.
Prioritize:
1) core ideas/topics
2) names of theories/researchers if strongly implied
3) notable numbers (like "7")
4) related memory/cognition concepts if relevant

Return JSON only:
{{
  "concepts": ["concept1", "concept2"],
  "interpretation": "short sentence",
  "confidence": "low|medium|high"
}}

Query: {query}
""".strip()

        attempts = self.openai_manager.max_rate_limit_retries + 1
        for attempt in range(attempts):
            try:
                response = self.openai_manager.client.chat.completions.create(
                    model=self.openai_manager.chat_model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract semantic retrieval concepts for vague TOT queries. "
                                "Return JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                payload = self._parse_json_object(content)
                if not isinstance(payload, dict):
                    raise ValueError("OpenAI concept extraction did not return a JSON object")

                concepts = self._normalize_concepts(payload.get("concepts", []))
                concepts = self._merge_concept_lists(concepts, heuristic_cues, numeric_hints)
                return {
                    "method": "openai-concept-extraction",
                    "model": self.openai_manager.chat_model,
                    "concepts": concepts,
                    "interpretation": str(payload.get("interpretation", "")).strip(),
                    "confidence": str(payload.get("confidence", "")).strip().lower(),
                    "raw_query": query,
                }
            except Exception as exc:
                is_rate_limited = self._is_rate_limit_error(exc)
                is_last_attempt = attempt >= attempts - 1
                if is_rate_limited and not is_last_attempt:
                    time.sleep(self.openai_manager.retry_base_delay_seconds * (2**attempt))
                    continue
                # Fall back gracefully to heuristics for local dev / transient failures.
                concepts = self._merge_concept_lists(heuristic_cues, numeric_hints)
                return {
                    "method": "heuristic-fallback",
                    "model": self.openai_manager.chat_model,
                    "concepts": concepts,
                    "raw_query": query,
                    "error": str(exc),
                }

        concepts = self._merge_concept_lists(heuristic_cues, numeric_hints)
        return {
            "method": "heuristic-fallback",
            "model": self.openai_manager.chat_model,
            "concepts": concepts,
            "raw_query": query,
            "error": "retry_exhausted",
        }

    def _search_neo4j_concept_graph(
        self, session_id: str, query_concepts: list[str], limit: int = 5
    ) -> list[dict[str, Any]]:
        if not self.neo4j_manager.driver or not query_concepts:
            return []

        cypher = """
        MATCH (m:Message {session_id: $session_id})
        OPTIONAL MATCH (m)-[:DISCUSSES]->(direct_c:Concept)
        WITH m, [c IN collect(DISTINCT toLower(direct_c.name)) WHERE c IS NOT NULL] AS direct_concepts
        OPTIONAL MATCH (m)-[:DISCUSSES]->(:Concept)-[rel:RELATES_TO]-(related_c:Concept)
        WITH
          m,
          direct_concepts,
          [c IN collect(DISTINCT toLower(related_c.name)) WHERE c IS NOT NULL] AS related_concepts,
          [rt IN collect(DISTINCT rel.type) WHERE rt IS NOT NULL] AS relation_types,
          $query_concepts AS query_concepts
        WITH
          m,
          direct_concepts,
          related_concepts,
          relation_types,
          [qc IN query_concepts WHERE ANY(dc IN direct_concepts WHERE dc = qc OR dc CONTAINS qc OR qc CONTAINS dc)]
            AS direct_matches,
          [qc IN query_concepts WHERE ANY(rc IN related_concepts WHERE rc = qc OR rc CONTAINS qc OR qc CONTAINS rc)]
            AS related_matches
        WHERE size(direct_matches) > 0 OR size(related_matches) > 0
        OPTIONAL MATCH (m)-[:PART_OF]->(t:Topic)
        RETURN
          m.id AS id,
          m.role AS role,
          m.content AS content,
          m.timestamp AS timestamp,
          collect(DISTINCT t.name) AS topic_names,
          direct_matches AS direct_concept_matches,
          related_matches AS related_concept_matches,
          relation_types,
          (size(direct_matches) * 3 + size(related_matches)) AS score
        ORDER BY score DESC, m.timestamp DESC
        LIMIT $limit
        """
        try:
            with self.neo4j_manager.driver.session() as session:
                rows = session.run(
                    cypher,
                    session_id=session_id,
                    query_concepts=query_concepts,
                    limit=limit,
                ).data()
        except Exception:
            return []

        return [
            {
                "id": row.get("id"),
                "role": row.get("role"),
                "content": row.get("content"),
                "timestamp": row.get("timestamp"),
                "topic_names": [t for t in row.get("topic_names", []) if t],
                "direct_concept_matches": row.get("direct_concept_matches", []),
                "related_concept_matches": row.get("related_concept_matches", []),
                "relation_types": [t for t in row.get("relation_types", []) if t],
                "score": int(row.get("score") or 0),
            }
            for row in rows
        ]

    def _search_local_concept_graph(
        self, session_id: str, query_concepts: list[str], limit: int = 5
    ) -> list[dict[str, Any]]:
        if not query_concepts:
            return []

        messages = self.neo4j_manager.get_all_messages(session_id)
        analysis = self.neo4j_manager.get_analysis(session_id) or {}
        adjacency = self._build_concept_adjacency(analysis.get("relationships", []))

        results: list[dict[str, Any]] = []
        for msg in messages:
            msg_concepts = self._normalize_concepts(msg.get("concepts", []))
            if not msg_concepts:
                continue

            direct_matches = [
                qc for qc in query_concepts if any(self._concept_match(qc, mc) for mc in msg_concepts)
            ]

            related_matches: list[str] = []
            for qc in query_concepts:
                neighbors = adjacency.get(qc, set())
                if not neighbors:
                    continue
                if any(any(self._concept_match(neighbor, mc) for mc in msg_concepts) for neighbor in neighbors):
                    related_matches.append(qc)

            direct_matches = self._normalize_concepts(direct_matches)
            related_matches = self._normalize_concepts(related_matches)

            if not direct_matches and not related_matches:
                continue

            topic_name = msg.get("topic_name")
            results.append(
                {
                    "id": msg.get("id"),
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "timestamp": msg.get("timestamp"),
                    "topic_names": [topic_name] if isinstance(topic_name, str) and topic_name else [],
                    "direct_concept_matches": direct_matches,
                    "related_concept_matches": related_matches,
                    "relation_types": [],
                    "score": (len(direct_matches) * 3) + len(related_matches),
                }
            )

        results.sort(key=lambda row: (row.get("score", 0), row.get("timestamp", "")), reverse=True)
        return results[:limit]

    def _build_concept_adjacency(self, relationships: Any) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        if not isinstance(relationships, list):
            return adjacency
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            source = self._normalize_concepts([rel.get("source")])
            target = self._normalize_concepts([rel.get("target")])
            if not source or not target:
                continue
            s = source[0]
            t = target[0]
            adjacency[s].add(t)
            adjacency[t].add(s)
        return adjacency

    def _parse_json_object(self, text: str) -> Any:
        if not text:
            return None

        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _normalize_concepts(self, concepts: Any) -> list[str]:
        if not isinstance(concepts, list):
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in concepts:
            concept = str(item).strip().lower()
            if not concept:
                continue
            concept = re.sub(r"\s+", " ", concept)
            # Keep short numeric concepts (e.g., "7"), but remove low-signal one-letter text.
            if len(concept) == 1 and not concept.isdigit():
                continue
            if concept in seen:
                continue
            seen.add(concept)
            normalized.append(concept)
        return normalized

    def _merge_concept_lists(self, *concept_lists: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for concept_list in concept_lists:
            for concept in concept_list:
                if concept in seen:
                    continue
                seen.add(concept)
                merged.append(concept)
        return merged[:10]

    def _concept_match(self, query_concept: str, candidate_concept: str) -> bool:
        q = query_concept.strip().lower()
        c = candidate_concept.strip().lower()
        if not q or not c:
            return False
        return q == c or q in c or c in q

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        name = exc.__class__.__name__.lower()
        return "ratelimit" in name or "rate limit" in str(exc).lower()
