from __future__ import annotations

from collections import Counter
import json
import re
import time
from typing import Any

from openai_manager import OpenAIManager


class AnalysisAgent:
    """Long-term memory consolidation agent (full conversation, GPT-4 JSON extraction)."""

    def __init__(self, openai_manager: OpenAIManager) -> None:
        self.openai_manager = openai_manager
        self.signal_words = {"however", "therefore", "because", "but", "although", "meanwhile"}
        self.stopwords = {
            "the",
            "and",
            "that",
            "with",
            "from",
            "have",
            "this",
            "there",
            "about",
            "would",
            "could",
            "should",
            "your",
            "into",
            "after",
            "before",
            "while",
            "where",
            "when",
            "which",
            "what",
            "they",
            "them",
            "their",
            "just",
            "like",
            "then",
            "than",
        }
        self.max_analysis_retries = 2

    def analyze_conversation(self, messages: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
        """Consolidate the FULL conversation into long-term memory structures."""
        if not messages:
            return {
                "session_id": session_id,
                "topics": [],
                "concepts": [],
                "relationships": [],
                "deflection_points": [],
                "signal_words_used": [],
                "analysis_mode": "empty",
            }

        if not self.openai_manager.is_configured:
            # Keep the system usable before keys are added.
            return self._heuristic_fallback(messages, session_id, reason="openai-unconfigured")

        raw_payload = self._run_gpt4_consolidation(messages, session_id)
        if raw_payload is None:
            return self._heuristic_fallback(messages, session_id, reason="gpt4-analysis-failed")

        normalized = self._normalize_analysis(raw_payload, messages, session_id)
        normalized.setdefault("analysis_mode", "gpt4-consolidation")
        normalized.setdefault("model", self.openai_manager.analysis_model)
        normalized.setdefault("source", "full-conversation-from-neo4j")
        normalized.setdefault(
            "notes",
            [
                "This analysis uses the full conversation (long-term memory consolidation), not the 20-message chat window."
            ],
        )
        return normalized

    def _run_gpt4_consolidation(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        client = self.openai_manager.client
        if client is None:
            return None

        conversation_text = self._format_conversation(messages)
        prompt = self._analysis_prompt(session_id=session_id, messages=messages, conversation_text=conversation_text)

        attempts = self.max_analysis_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=self.openai_manager.analysis_model,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are MindChat's memory consolidation agent. "
                                "Analyze the FULL conversation as long-term memory consolidation. "
                                "Return strict JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    return None
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                is_rate_limited = self.openai_manager._is_rate_limit_error(exc)
                is_last_attempt = attempt >= attempts - 1
                if is_rate_limited and not is_last_attempt:
                    delay = self.openai_manager.retry_base_delay_seconds * (2**attempt)
                    time.sleep(delay)
                    continue
                break

        if last_error is not None:
            # Non-fatal; caller will fall back to heuristics.
            return None
        return None

    def _analysis_prompt(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        conversation_text: str,
    ) -> str:
        message_ids = [str(msg.get("id", "")) for msg in messages]
        return f"""
Analyze this conversation using cognitive science principles.
Extract: 1) Topics discussed 2) Key concepts 3) Concept relationships 4) Where topics changed (deflection points) 5) Signal words used

This is MindChat memory consolidation:
- Use the FULL conversation (long-term memory), not a sliding window.
- Detect semantic chunks and topic transitions.
- Use signal words such as however, therefore, because when present.

Return JSON: {{
  "topics": [{{"name": "string", "message_ids": ["message-id"], "summary": "string"}}],
  "concepts": [{{"name": "string", "type": "string", "importance": 1-5}}],
  "relationships": [{{"source": "concept", "target": "concept", "type": "string", "evidence_message_ids": ["message-id"]}}],
  "deflections": [{{"message_id": "message-id", "from_topic": "string", "to_topic": "string", "signal_words": ["word"], "reason": "string"}}],
  "signal_words_used": ["word"]
}}

Rules:
- Use message IDs from this list only: {message_ids}
- If unsure, return fewer items and stay precise.
- JSON only. No markdown.

Session ID: {session_id}
Message count: {len(messages)}

Conversation:
{conversation_text}
""".strip()

    def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for idx, msg in enumerate(messages, start=1):
            message_id = str(msg.get("id", ""))
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", "")).replace("\n", " ").strip()
            timestamp = str(msg.get("timestamp", ""))
            lines.append(
                f"[{idx}] id={message_id} role={role} timestamp={timestamp} content={content}"
            )
        return "\n".join(lines)

    def _normalize_analysis(
        self,
        payload: dict[str, Any],
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> dict[str, Any]:
        valid_message_ids = {str(m.get("id", "")) for m in messages if m.get("id")}
        message_order = [str(m.get("id", "")) for m in messages if m.get("id")]

        topics = self._normalize_topics(payload.get("topics", []), message_order)
        concepts = self._normalize_concepts(payload.get("concepts", []))
        relationships = self._normalize_relationships(payload.get("relationships", []))

        raw_deflections = payload.get("deflections")
        if raw_deflections is None:
            raw_deflections = payload.get("deflection_points", [])

        deflection_points = self._normalize_deflections(raw_deflections, valid_message_ids)

        signal_words_used = payload.get("signal_words_used")
        if not isinstance(signal_words_used, list):
            signal_words_used = sorted(
                {
                    word
                    for point in deflection_points
                    for word in point.get("signal_words", [])
                    if isinstance(word, str) and word.strip()
                }
            )

        return {
            "session_id": session_id,
            "topics": topics,
            "concepts": concepts,
            "relationships": relationships,
            "deflection_points": deflection_points,
            "signal_words_used": [
                str(word).strip().lower() for word in signal_words_used if str(word).strip()
            ],
            "raw_model_fields": {
                "deflections_count": len(raw_deflections) if isinstance(raw_deflections, list) else 0
            },
        }

    def _normalize_topics(self, topics: Any, all_message_ids: list[str]) -> list[dict[str, Any]]:
        if not isinstance(topics, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in topics:
            if isinstance(item, str):
                name = item.strip()
                if not name:
                    continue
                normalized.append({"name": name, "message_ids": [], "summary": ""})
                continue

            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            if not name:
                continue

            raw_message_ids = item.get("message_ids", [])
            if not isinstance(raw_message_ids, list):
                raw_message_ids = []
            filtered_message_ids = [
                str(mid)
                for mid in raw_message_ids
                if isinstance(mid, (str, int)) and str(mid).strip()
            ]

            # If model omits message_ids for a topic, keep it empty (analysis still useful).
            summary = str(item.get("summary", "")).strip()
            topic_entry: dict[str, Any] = {"name": name, "message_ids": filtered_message_ids}
            if summary:
                topic_entry["summary"] = summary
            normalized.append(topic_entry)

        if not normalized:
            normalized.append({"name": "General Conversation", "message_ids": all_message_ids})
        return normalized

    def _normalize_concepts(self, concepts: Any) -> list[dict[str, Any]]:
        if not isinstance(concepts, list):
            return []

        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in concepts:
            if isinstance(item, str):
                name = item.strip().lower()
                if not name or name in seen:
                    continue
                seen.add(name)
                normalized.append({"name": name})
                continue

            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)

            concept: dict[str, Any] = {"name": name}
            if item.get("type"):
                concept["type"] = str(item["type"]).strip()
            if isinstance(item.get("importance"), (int, float)):
                concept["importance"] = int(item["importance"])
            normalized.append(concept)
        return normalized

    def _normalize_relationships(self, relationships: Any) -> list[dict[str, Any]]:
        if not isinstance(relationships, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in relationships:
            if not isinstance(item, dict):
                continue

            source = str(item.get("source", "")).strip().lower()
            target = str(item.get("target", "")).strip().lower()
            rel_type = str(item.get("type", "related")).strip() or "related"
            if not source or not target:
                continue

            relation: dict[str, Any] = {"source": source, "target": target, "type": rel_type}
            evidence_ids = item.get("evidence_message_ids", [])
            if isinstance(evidence_ids, list):
                relation["evidence_message_ids"] = [
                    str(mid) for mid in evidence_ids if isinstance(mid, (str, int))
                ]
            normalized.append(relation)
        return normalized

    def _normalize_deflections(
        self, deflections: Any, valid_message_ids: set[str]
    ) -> list[dict[str, Any]]:
        if not isinstance(deflections, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in deflections:
            if not isinstance(item, dict):
                continue

            message_id = str(item.get("message_id", "")).strip()
            if message_id and valid_message_ids and message_id not in valid_message_ids:
                # Skip hallucinated message IDs to protect graph integrity.
                continue

            signal_words = item.get("signal_words", [])
            if isinstance(signal_words, str):
                signal_words = [signal_words]
            if not isinstance(signal_words, list):
                signal_words = []

            point: dict[str, Any] = {
                "message_id": message_id,
                "from_topic": str(item.get("from_topic", "unknown")).strip() or "unknown",
                "to_topic": str(item.get("to_topic", "unknown")).strip() or "unknown",
                "signal_words": [
                    str(word).strip().lower()
                    for word in signal_words
                    if str(word).strip()
                ],
            }
            reason = str(item.get("reason", "")).strip()
            if reason:
                point["reason"] = reason
            normalized.append(point)
        return normalized

    def _heuristic_fallback(
        self, messages: list[dict[str, Any]], session_id: str, *, reason: str
    ) -> dict[str, Any]:
        all_text = " ".join(str(m.get("content", "")) for m in messages)
        top_terms = [term for term, _ in Counter(self._tokens(all_text)).most_common(10)]

        topic_name = self._infer_primary_topic(top_terms)
        topics = [{"name": topic_name, "message_ids": [m.get("id", "") for m in messages]}]
        concepts = [{"name": term} for term in top_terms]
        relationships = [
            {"source": source, "target": target, "type": "co_occurs"}
            for source, target in zip(top_terms, top_terms[1:])
        ]

        deflection_points: list[dict[str, Any]] = []
        signal_words_used: set[str] = set()
        for msg in messages:
            content = str(msg.get("content", "")).lower()
            found = [word for word in self.signal_words if f" {word} " in f" {content} "]
            if not found:
                continue
            signal_words_used.update(found)
            deflection_points.append(
                {
                    "message_id": msg.get("id", ""),
                    "from_topic": topic_name,
                    "to_topic": f"{topic_name} (shift)",
                    "signal_words": sorted(found),
                    "reason": "Heuristic signal-word-based transition",
                }
            )

        return {
            "session_id": session_id,
            "topics": topics,
            "concepts": concepts,
            "relationships": relationships,
            "deflection_points": deflection_points,
            "signal_words_used": sorted(signal_words_used),
            "analysis_mode": "heuristic-fallback",
            "fallback_reason": reason,
            "model": self.openai_manager.analysis_model,
            "source": "full-conversation-from-neo4j",
            "notes": [
                "GPT-4 consolidation unavailable, using heuristic fallback.",
                "This still analyzes the FULL conversation, not the chat sliding window.",
            ],
        }

    def _tokens(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if token not in self.stopwords and (len(token) > 3 or token.isdigit())
        ]

    def _infer_primary_topic(self, top_terms: list[str]) -> str:
        memory_terms = {"memory", "recall", "retrieval", "chunking", "schema", "cognitive"}
        if any(term in memory_terms for term in top_terms):
            return "Cognitive Science & Memory"
        if top_terms:
            return f"Concept Cluster: {top_terms[0]}"
        return "General Conversation"

