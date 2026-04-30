from __future__ import annotations

from dataclasses import dataclass

from redisvl.extensions.message_history import MessageHistory

from app.config import settings
from app.redis_client import get_redis_client
from app.utils import estimate_tokens, normalize_text


@dataclass
class MemoryEntry:
    role: str
    content: str
    timestamp: float


class ConversationMemory:
    def __init__(self) -> None:
        self.redis = get_redis_client()
        self.index_name = "demo:memory:index"
        self.key_prefix = "memory"
        self._histories: dict[str, MessageHistory] = {}

    def _history(self, session_id: str) -> MessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = MessageHistory(
                name=self.index_name,
                prefix=self.key_prefix,
                redis_client=self.redis,
                session_tag=session_id,
            )
        return self._histories[session_id]

    @staticmethod
    def _normalize_role(role: str) -> str:
        return "llm" if role == "assistant" else role

    def append(self, session_id: str, role: str, content: str) -> None:
        history = self._history(session_id)
        history.add_message(
            {"role": self._normalize_role(role), "content": normalize_text(content)},
            session_tag=session_id,
        )
        count = history.count(session_tag=session_id)
        if count:
            raw_entries = history.get_recent(top_k=count, raw=True, session_tag=session_id)
            ids = [entry.get("entry_id") or entry.get("id") for entry in raw_entries]
            keys = [history._index.key(entry_id) for entry_id in ids if entry_id]  # type: ignore[attr-defined]
            if keys:
                history._index.expire_keys(keys, settings.session_ttl_seconds)  # type: ignore[attr-defined]

    def get_recent(self, session_id: str, limit: int = 8) -> list[MemoryEntry]:
        history = self._history(session_id)
        raw_entries = history.get_recent(top_k=limit, raw=True, session_tag=session_id)
        entries: list[MemoryEntry] = []
        for raw in raw_entries:
            timestamp = float(raw.get("timestamp", 0.0))
            role = str(raw.get("role", "user"))
            if role == "llm":
                role = "assistant"
            entries.append(
                MemoryEntry(
                    role=role,
                    content=str(raw.get("content", "")),
                    timestamp=timestamp,
                )
            )
        return entries

    def build_context(self, session_id: str, limit: int = 8) -> str:
        entries = self.get_recent(session_id, limit=limit)
        lines = [f"{entry.role.title()}: {entry.content}" for entry in entries]
        return "\n".join(lines)

    def summary(self, session_id: str, limit: int = 8) -> dict[str, int | str]:
        entries = self.get_recent(session_id, limit=limit)
        joined = "\n".join(entry.content for entry in entries)
        return {
            "turns": len(entries),
            "estimated_tokens": estimate_tokens(joined),
            "preview": joined[:280] + ("..." if len(joined) > 280 else ""),
        }

    def clear(self, session_id: str) -> None:
        self._history(session_id).clear()
