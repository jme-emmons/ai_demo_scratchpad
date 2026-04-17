from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.config import settings
from app.redis_client import get_redis_client


@dataclass
class SearchMatch:
    key: str
    score: float
    text: str
    title: str
    source: str
    chunk_id: str


class RedisVectorStore:
    def __init__(self) -> None:
        self.redis = get_redis_client()
        self.index_name = settings.vector_index_name
        self.key_prefix = "doc:"

    def ensure_index(self) -> None:
        try:
            self.redis.ft(self.index_name).info()
            return
        except Exception:
            pass

        schema = (
            TextField("text"),
            TextField("title"),
            TextField("source"),
            TagField("session_id"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": settings.vector_dimension,
                    "DISTANCE_METRIC": settings.vector_distance_metric,
                },
            ),
        )
        definition = IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH)
        self.redis.ft(self.index_name).create_index(schema, definition=definition)

    def upsert_chunk(
        self,
        chunk_key: str,
        text: str,
        title: str,
        source: str,
        session_id: str,
        created_at: int,
        vector: list[float],
    ) -> None:
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        self.redis.hset(
            chunk_key,
            mapping={
                "text": text,
                "title": title,
                "source": source,
                "session_id": session_id,
                "created_at": created_at,
                "embedding": vector_bytes,
            },
        )

    def search(self, vector: list[float], session_id: str | None = None, top_k: int = 4) -> list[SearchMatch]:
        query_vector = np.array(vector, dtype=np.float32).tobytes()
        base_filter = "*"
        if session_id:
            base_filter = f"@session_id:{{{session_id}}}"
        query = (
            Query(f"{base_filter}=>[KNN {top_k} @embedding $vector AS score]")
            .sort_by("score")
            .return_fields("score", "text", "title", "source")
            .paging(0, top_k)
            .dialect(2)
        )
        results = self.redis.ft(self.index_name).search(query, {"vector": query_vector})
        matches: list[SearchMatch] = []
        for doc in results.docs:
            matches.append(
                SearchMatch(
                    key=doc.id,
                    score=float(doc.score),
                    text=doc.text,
                    title=getattr(doc, "title", ""),
                    source=getattr(doc, "source", ""),
                    chunk_id=doc.id,
                )
            )
        return matches

    def get_index_stats(self) -> dict[str, str]:
        try:
            info = self.redis.ft(self.index_name).info()
        except Exception:
            return {"status": "missing"}
        return {
            "status": "ready",
            "num_docs": str(info.get("num_docs", "0")),
            "index_name": self.index_name,
        }

    def flush_session_docs(self, session_id: str) -> int:
        keys = self.redis.keys(f"{self.key_prefix}{session_id}:*")
        if not keys:
            return 0
        return self.redis.delete(*keys)

    def session_has_docs(self, session_id: str) -> bool:
        return next(self.redis.scan_iter(f"{self.key_prefix}{session_id}:*"), None) is not None
