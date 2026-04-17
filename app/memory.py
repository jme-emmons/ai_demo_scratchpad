from __future__ import annotations

import json
import time
from dataclasses import dataclass

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

    def _list_key(self, session_id: str) -> str:
        return f"memory:{session_id}:turns"

    def append(self, session_id: str, role: str, content: str) -> None:
        entry = json.dumps({"role": role, "content": normalize_text(content), "timestamp": time.time()})
        key = self._list_key(session_id)
        self.redis.rpush(key, entry)
        self.redis.expire(key, settings.session_ttl_seconds)

    def get_recent(self, session_id: str, limit: int = 6) -> list[MemoryEntry]:
        key = self._list_key(session_id)
        raw_entries = self.redis.lrange(key, -limit, -1)
        entries: list[MemoryEntry] = []
        for raw in raw_entries:
            data = json.loads(raw.decode("utf-8"))
            entries.append(MemoryEntry(**data))
        return entries

    def build_context(self, session_id: str, limit: int = 6) -> str:
        entries = self.get_recent(session_id, limit=limit)
        lines = [f"{entry.role.title()}: {entry.content}" for entry in entries]
        return "\n".join(lines)

    def summary(self, session_id: str, limit: int = 6) -> dict[str, int | str]:
        entries = self.get_recent(session_id, limit=limit)
        joined = "\n".join(entry.content for entry in entries)
        return {
            "turns": len(entries),
            "estimated_tokens": estimate_tokens(joined),
            "preview": joined[:280] + ("..." if len(joined) > 280 else ""),
        }

    def clear(self, session_id: str) -> None:
        self.redis.delete(self._list_key(session_id))
