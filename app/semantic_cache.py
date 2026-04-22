from __future__ import annotations

import json
import time
from dataclasses import dataclass

from redis.exceptions import ResponseError

from app.config import settings
from app.model_clients import EmbeddingClient
from app.utils import estimate_tokens, make_id
from app.vector_store import RedisVectorStore


@dataclass
class CacheResult:
    hit: bool
    answer: str | None = None
    matched_question: str | None = None
    score: float | None = None
    tokens_saved: int = 0
    latency_saved_ms: float = 0.0
    cost_saved: float = 0.0


class SemanticCache:
    def __init__(self) -> None:
        self.embedder = EmbeddingClient()
        self.store = RedisVectorStore()
        self.redis = self.store.redis
        self.index_name = "demo:cache:index"
        self.key_prefix = "cache:"

    def ensure_index(self) -> None:
        self.store.index_name = self.index_name
        self.store.key_prefix = self.key_prefix
        self.store.ensure_index()

    def lookup(self, question: str, session_id: str) -> tuple[CacheResult, list[float], float]:
        embedding = self.embedder.embed(question)
        self.store.index_name = self.index_name
        self.store.key_prefix = self.key_prefix
        try:
            matches = self.store.search(embedding.vector, session_id=session_id, top_k=1)
        except ResponseError as exc:
            if self._is_missing_index_error(exc):
                return CacheResult(hit=False), embedding.vector, embedding.latency_ms
            raise
        if not matches:
            return CacheResult(hit=False), embedding.vector, embedding.latency_ms
        match = matches[0]
        payload = self.redis.hget(match.key, "payload")
        if payload is None:
            return CacheResult(hit=False), embedding.vector, embedding.latency_ms
        data = json.loads(payload.decode("utf-8"))
        if match.score > settings.semantic_cache_distance_threshold:
            return CacheResult(hit=False), embedding.vector, embedding.latency_ms
        total_tokens = data.get("total_tokens") or estimate_tokens(data["answer"])
        cost_saved = round((total_tokens / 1000.0) * settings.token_price_per_1k, 6)
        return (
            CacheResult(
                hit=True,
                answer=data["answer"],
                matched_question=data["question"],
                score=match.score,
                tokens_saved=total_tokens,
                latency_saved_ms=float(data.get("latency_ms", 0.0)),
                cost_saved=cost_saved,
            ),
            embedding.vector,
            embedding.latency_ms,
        )

    def store_answer(
        self,
        session_id: str,
        question: str,
        answer: str,
        vector: list[float],
        latency_ms: float,
        total_tokens: int,
    ) -> None:
        self.store.index_name = self.index_name
        self.store.key_prefix = self.key_prefix
        key = make_id(f"{self.key_prefix}{session_id}")
        payload = json.dumps(
            {
                "question": question,
                "answer": answer,
                "latency_ms": latency_ms,
                "total_tokens": total_tokens,
            }
        )
        self.redis.hset(
            key,
            mapping={
                "text": question,
                "title": "semantic_cache",
                "source": "semantic_cache",
                "session_id": session_id,
                "created_at": int(time.time()),
                "payload": payload,
            },
        )
        self.redis.hset(key, "embedding", self._vector_bytes(vector))

    @staticmethod
    def _vector_bytes(vector: list[float]) -> bytes:
        import numpy as np

        return np.array(vector, dtype=np.float32).tobytes()

    def _is_missing_index_error(self, exc: ResponseError) -> bool:
        message = str(exc)
        return "No such index" in message and self.index_name in message
