from __future__ import annotations

import json
from typing import Any
import os
import re
import time

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency not installed yet
    OpenAI = None  # type: ignore[assignment]


class OpenAIManager:
    """Working-memory manager (short context window) plus lightweight cue extraction helpers."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        # Working-memory chat path defaults to gpt-3.5-turbo for hackathon speed/cost efficiency.
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
        self.analysis_model = os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4o")
        self.max_working_memory_messages = 20
        self.max_rate_limit_retries = int(os.getenv("OPENAI_RATE_LIMIT_RETRIES", "3"))
        self.retry_base_delay_seconds = float(os.getenv("OPENAI_RETRY_BASE_DELAY_SECONDS", "1.0"))
        self.client = None

        if self.api_key and OpenAI is not None:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def build_working_memory(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Atkinson-Shiffrin working memory: keep only the most recent 20 chat turns."""
        trimmed = messages[-self.max_working_memory_messages :]
        return [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            for msg in trimmed
        ]

    def _system_prompt(self) -> str:
        return (
            "You are MindChat, a cognitive-science-inspired assistant that helps with "
            "tip-of-the-tongue recall. Treat the chat history as working memory (recent context only). "
            "Use concepts and retrieval cues, not only exact keywords. Be explicit when inferring. "
            "Keep responses practical and concise."
        )

    def _latest_user_message(self, messages: list[dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        name = exc.__class__.__name__.lower()
        if "ratelimit" in name or "rate_limit" in name:
            return True
        return "rate limit" in str(exc).lower()

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None

        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)

        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _scaffold_response(self, working_memory: list[dict[str, str]], *, session_id: str | None = None) -> str:
        latest_user_message = self._latest_user_message(working_memory)
        session_context = f" in session `{session_id}`" if session_id else ""
        return (
            "MindChat scaffold mode: OpenAI key not configured yet. I stored your message and can "
            "simulate working-memory chat. Once `OPENAI_API_KEY` is added, this response will come "
            "from the OpenAI API using the last 20 messages as working memory. "
            f"Latest prompt{session_context}: {latest_user_message}"
        )

    def get_chat_response(self, messages: list[dict[str, Any]]) -> str:
        """Return a chat response using a sliding 20-message working-memory window."""
        working_memory = self.build_working_memory(messages)

        if not self.client:
            return self._scaffold_response(working_memory)

        attempts = self.max_rate_limit_retries + 1
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "system", "content": self._system_prompt()}, *working_memory],
                    temperature=0.4,
                )
                content = response.choices[0].message.content or ""
                return content.strip() or "I couldn't generate a response."
            except Exception as exc:
                is_rate_limited = self._is_rate_limit_error(exc)
                is_last_attempt = attempt >= attempts - 1
                if is_rate_limited and not is_last_attempt:
                    delay = self.retry_base_delay_seconds * (2**attempt)
                    time.sleep(delay)
                    continue

                if is_rate_limited:
                    return (
                        "OpenAI rate limit hit after retries. "
                        "Please wait a moment and try again. Your message was still stored for later analysis."
                    )

                return (
                    "OpenAI call failed in scaffold mode. "
                    f"Error: {exc}. Your message was still stored for later analysis."
                )

        return "OpenAI request exhausted retry attempts."

    def detect_research_intent(
        self,
        latest_message: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Use OpenAI (when available) to decide whether a query needs Tavily/web research."""
        working_memory = self.build_working_memory(messages or [])
        heuristic_guess = self.is_research_question(latest_message)

        if not self.client:
            return {
                "is_research": heuristic_guess,
                "reason": "OpenAI unavailable, heuristic fallback used.",
                "search_query": latest_message.strip(),
                "topic": "general",
                "method": "heuristic-fallback",
            }

        recent_context = "\n".join(
            f"- {msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in working_memory[-6:]
        )
        prompt = f"""
Classify whether the user's latest message needs external web research (Tavily) to answer well.

Return JSON only:
{{
  "is_research": true or false,
  "reason": "short explanation",
  "search_query": "optimized web search query",
  "topic": "general|news|finance"
}}

Guidelines:
- true if the user asks for facts, evidence, citations, recent info, papers, sources, or real-world data.
- false if it is mostly conversational, brainstorming, or answerable from local conversation context alone.
- Keep search_query concise and explicit.

Recent working-memory context:
{recent_context or "(none)"}

Latest user message:
{latest_message}
""".strip()

        attempts = self.max_rate_limit_retries + 1
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You classify whether a user message needs web research and output JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                payload = self._parse_json_object(content) or {}

                is_research = bool(payload.get("is_research", heuristic_guess))
                topic = str(payload.get("topic", "general")).strip().lower() or "general"
                if topic not in {"general", "news", "finance"}:
                    topic = "general"

                search_query = str(payload.get("search_query", "")).strip() or latest_message.strip()
                reason = str(payload.get("reason", "")).strip() or "OpenAI research-intent classification."

                return {
                    "is_research": is_research,
                    "reason": reason,
                    "search_query": search_query,
                    "topic": topic,
                    "method": "openai-classifier",
                }
            except Exception as exc:
                is_rate_limited = self._is_rate_limit_error(exc)
                is_last_attempt = attempt >= attempts - 1
                if is_rate_limited and not is_last_attempt:
                    delay = self.retry_base_delay_seconds * (2**attempt)
                    time.sleep(delay)
                    continue
                return {
                    "is_research": heuristic_guess,
                    "reason": f"Classifier fallback after error: {exc}",
                    "search_query": latest_message.strip(),
                    "topic": "general",
                    "method": "heuristic-fallback",
                }

        return {
            "is_research": heuristic_guess,
            "reason": "Classifier retries exhausted; heuristic fallback used.",
            "search_query": latest_message.strip(),
            "topic": "general",
            "method": "heuristic-fallback",
        }

    def generate_chat_reply(self, messages: list[dict[str, Any]], session_id: str) -> str:
        """Backward-compatible wrapper used by the existing FastAPI route."""
        working_memory = self.build_working_memory(messages)
        if not self.client:
            return self._scaffold_response(working_memory, session_id=session_id)
        return self.get_chat_response(messages)

    def extract_retrieval_cues(self, vague_query: str) -> list[str]:
        # Heuristic fallback for Day 1 setup; can be replaced with GPT-4 extraction in Day 5.
        normalized = vague_query.lower()
        tokens = re.findall(r"[a-z0-9]+", normalized)

        stopwords = {
            "the",
            "and",
            "that",
            "what",
            "thing",
            "about",
            "with",
            "from",
            "this",
            "there",
            "was",
            "were",
            "have",
            "into",
            "your",
            "just",
            "like",
            "they",
            "them",
        }
        cues = [tok for tok in tokens if tok not in stopwords and (len(tok) > 2 or tok.isdigit())]

        # Preserve order while deduplicating.
        deduped: list[str] = []
        seen: set[str] = set()
        for cue in cues:
            if cue not in seen:
                deduped.append(cue)
                seen.add(cue)

        return deduped[:8]

    def is_research_question(self, text: str) -> bool:
        lowered = text.lower()
        patterns = [
            "research",
            "paper",
            "source",
            "citation",
            "study",
            "latest",
            "evidence",
            "find data",
        ]
        if any(p in lowered for p in patterns):
            return True
        return bool(re.search(r"\b(why|how|what)\b.*\b(study|research|evidence)\b", lowered))
