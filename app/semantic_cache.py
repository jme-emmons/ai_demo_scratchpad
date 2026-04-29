from __future__ import annotations

from dataclasses import dataclass

from redis.exceptions import ResponseError
from redisvl.extensions.llmcache import SemanticCache as RedisVLSemanticCache
from redisvl.query.filter import Tag

from app.config import settings
from app.model_clients import EmbeddingClient
from app.redis_client import get_redis_client
from app.redisvl_helpers import get_redisvl_vectorizer
from app.utils import estimate_tokens


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
        self.redis = get_redis_client()
        self.index_name = "demo:cache:index"
        self.cache = RedisVLSemanticCache(
            name=self.index_name,
            distance_threshold=settings.semantic_cache_distance_threshold,
            ttl=settings.session_ttl_seconds,
            vectorizer=get_redisvl_vectorizer(),
            filterable_fields=[{"name": "session_id", "type": "tag"}],
            redis_client=self.redis,
        )

    def ensure_index(self) -> None:
        return None

    def lookup(self, question: str, session_id: str) -> tuple[CacheResult, list[float], float]:
        embedding = self.embedder.embed(question)
        try:
            matches = self.cache.check(
                vector=embedding.vector,
                num_results=1,
                return_fields=["prompt", "response", "metadata", "vector_distance", "key"],
                filter_expression=Tag("session_id") == session_id,
            )
        except ResponseError as exc:
            if self._is_missing_index_error(exc):
                return CacheResult(hit=False), embedding.vector, embedding.latency_ms
            raise
        if not matches:
            return CacheResult(hit=False), embedding.vector, embedding.latency_ms
        match = matches[0]
        metadata = match.get("metadata") or {}
        score = float(match.get("vector_distance", 1.0))
        if score > settings.semantic_cache_distance_threshold:
            return CacheResult(hit=False), embedding.vector, embedding.latency_ms
        answer = str(match.get("response", ""))
        total_tokens = metadata.get("total_tokens") or estimate_tokens(answer)
        cost_saved = round((total_tokens / 1000.0) * settings.token_price_per_1k, 6)
        return (
            CacheResult(
                hit=True,
                answer=answer,
                matched_question=str(match.get("prompt", "")),
                score=score,
                tokens_saved=total_tokens,
                latency_saved_ms=float(metadata.get("latency_ms", 0.0)),
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
        self.cache.store(
            prompt=question,
            response=answer,
            vector=vector,
            metadata={
                "latency_ms": latency_ms,
                "total_tokens": total_tokens,
            },
            filters={"session_id": session_id},
        )

    def _is_missing_index_error(self, exc: ResponseError) -> bool:
        message = str(exc)
        return "No such index" in message and self.index_name in message
