from __future__ import annotations

import inspect
import re
from typing import Any


class SemanticDeflectionDetector:
    def __init__(self, neo4j_manager: Any, openai_manager: Any) -> None:
        self.neo4j_manager = neo4j_manager
        self.openai_manager = openai_manager

    def _fallback_result(self, current_topic_label: str, reason: str) -> dict[str, Any]:
        print(f"SemanticDeflectionDetector fallback: {reason}")
        return {
            "type": "EXTENSION",
            "topic_label": current_topic_label,
            "similarity_score": 1.0,
            "parent_topic": None,
            "linked_topic": None,
        }

    async def _cold_start_branch_result(
        self,
        *,
        session_id: str,
        new_message_content: str,
        current_topic_label: str,
        reason: str,
    ) -> dict[str, Any]:
        print(f"SemanticDeflectionDetector fallback: {reason}")
        session_msg_count = self.neo4j_manager.get_message_count(session_id)
        if session_msg_count > 2:
            topic_label = await self._generate_topic_label(new_message_content)
            return {
                "type": "BRANCH",
                "topic_label": topic_label,
                "similarity_score": 0.0,
                "parent_topic": current_topic_label,
                "linked_topic": None,
            }
        return {
            "type": "EXTENSION",
            "topic_label": current_topic_label,
            "similarity_score": 1.0,
            "parent_topic": None,
            "linked_topic": None,
        }

    def _heuristic_topic_label(self, text: str, concepts: list[str], current_topic_label: str) -> str:
        candidates = [concept.strip() for concept in concepts if concept and concept.strip()]
        if len(candidates) >= 3:
            return " ".join(candidates[:5]).title()

        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
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
            "then",
            "than",
        }
        words: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in stopwords or token in seen or len(token) < 3:
                continue
            seen.add(token)
            words.append(token)
            if len(words) >= 5:
                break

        if len(words) >= 3:
            return " ".join(words[:5]).title()
        return current_topic_label

    async def _generate_topic_label(self, content: str) -> str:
        client = getattr(self.openai_manager, "client", None)
        if client is None:
            return "New Topic"

        prompt = (
            "Give a 3-5 word topic label for this message. "
            "Reply with ONLY the label, no punctuation:\n\n"
            f"{content[:300]}"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=15,
            )
            if inspect.isawaitable(response):
                response = await response

            content = getattr(response.choices[0].message, "content", "") or ""
            label = re.sub(r"[\r\n]+", " ", content).strip().strip("\"'`")
            words = [word for word in label.split() if word]
            if 3 <= len(words) <= 5:
                return " ".join(words)
            if len(words) > 5:
                return " ".join(words[:5])
        except Exception:
            pass

        return "New Topic"

    def _choose_primary_topic(self, topic_names: list[str], current_topic_label: str) -> str | None:
        cleaned = [topic.strip() for topic in topic_names if topic and topic.strip()]
        if not cleaned:
            return None
        for topic_name in cleaned:
            if topic_name != current_topic_label:
                return topic_name
        return cleaned[0]

    def _vector_matches_for_concept(
        self,
        session_id: str,
        concept_name: str,
        embedding: list[float],
        exclude_message_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not self.neo4j_manager.driver:
            return []

        query = """
        CALL db.index.vector.queryNodes('concept-embeddings', 5, $embedding)
        YIELD node AS similarConcept, score
        WHERE similarConcept.name <> $concept_name
        MATCH (m:Message {session_id: $session_id})-[:DISCUSSES]->(similarConcept)
        WHERE NOT m.id IN $exclude_message_ids
        OPTIONAL MATCH (m)-[:PART_OF]->(t:Topic)
        RETURN
          similarConcept.name AS name,
          score,
          collect(DISTINCT t.name) AS topic_names
        ORDER BY score DESC
        LIMIT 2
        """

        with self.neo4j_manager.driver.session() as session:
            rows = session.run(
                query,
                embedding=embedding,
                concept_name=concept_name,
                session_id=session_id,
                exclude_message_ids=exclude_message_ids,
            ).data()

        matches: list[dict[str, Any]] = []
        for row in rows:
            match_name = str(row.get("name", "")).strip()
            if not match_name:
                continue
            matches.append(
                {
                    "name": match_name,
                    "score": float(row.get("score", 0.0) or 0.0),
                    "topic_names": [
                        str(topic_name).strip()
                        for topic_name in (row.get("topic_names") or [])
                        if topic_name
                    ],
                }
            )
        return matches

    async def detect(
        self,
        session_id: str,
        new_message_content: str,
        new_concepts: list[str],
        current_topic_label: str,
    ) -> dict[str, Any]:
        normalized_concepts: list[str] = []
        seen: set[str] = set()
        for concept in new_concepts:
            concept_name = str(concept).strip().lower()
            if not concept_name or concept_name in seen:
                continue
            seen.add(concept_name)
            normalized_concepts.append(concept_name)

        if not self.neo4j_manager.driver:
            return self._fallback_result(current_topic_label, "Neo4j driver unavailable.")
        if not normalized_concepts:
            return self._fallback_result(current_topic_label, "No concepts available for deflection detection.")

        embedded_concepts: list[tuple[str, list[float]]] = []
        for concept_name in normalized_concepts:
            embedding = await self.openai_manager.get_embedding(concept_name)
            if embedding is not None:
                embedded_concepts.append((concept_name, embedding))

        if not embedded_concepts:
            return await self._cold_start_branch_result(
                session_id=session_id,
                new_message_content=new_message_content,
                current_topic_label=current_topic_label,
                reason="No concept embeddings available yet.",
            )

        all_matches: list[dict[str, Any]] = []
        recent_message_ids = [
            str(message.get("id", "")).strip()
            for message in self.neo4j_manager.get_recent_messages(session_id, limit=2)
            if str(message.get("id", "")).strip()
        ]
        try:
            for concept_name, embedding in embedded_concepts:
                matches = self._vector_matches_for_concept(
                    session_id,
                    concept_name,
                    embedding,
                    recent_message_ids,
                )
                for match in matches:
                    match["query_concept"] = concept_name
                all_matches.extend(matches)
        except Exception as exc:
            return self._fallback_result(current_topic_label, f"Neo4j vector query failed: {exc}")

        if not all_matches:
            return await self._cold_start_branch_result(
                session_id=session_id,
                new_message_content=new_message_content,
                current_topic_label=current_topic_label,
                reason="No similar embedded concepts found in Neo4j.",
            )

        all_matches.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        similarity_score = float(all_matches[0].get("score", 0.0))

        top_two_matches = all_matches[:2]
        first_topic = self._choose_primary_topic(
            top_two_matches[0].get("topic_names", []),
            current_topic_label,
        )
        second_topic = None
        if len(top_two_matches) > 1:
            second_topic = self._choose_primary_topic(
                top_two_matches[1].get("topic_names", []),
                current_topic_label,
            )

        if first_topic and second_topic and first_topic != second_topic:
            return {
                "type": "SYNTHESIS",
                "topic_label": await self._generate_topic_label(new_message_content),
                "similarity_score": similarity_score,
                "parent_topic": current_topic_label,
                "linked_topic": second_topic,
            }

        if similarity_score >= 0.78:
            return {
                "type": "EXTENSION",
                "topic_label": current_topic_label,
                "similarity_score": similarity_score,
                "parent_topic": None,
                "linked_topic": None,
            }

        return {
            "type": "BRANCH",
            "topic_label": await self._generate_topic_label(new_message_content),
            "similarity_score": similarity_score,
            "parent_topic": current_topic_label,
            "linked_topic": None,
        }
