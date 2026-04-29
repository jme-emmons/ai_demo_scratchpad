from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import CountQuery, FilterQuery, VectorQuery
from redisvl.query.filter import Tag

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
    def __init__(self, index_name: str | None = None, key_prefix: str = "doc") -> None:
        self.redis = get_redis_client()
        self.index_name = index_name or settings.vector_index_name
        self.key_prefix = key_prefix
        self.index = SearchIndex.from_dict(
            self._schema(),
            redis_client=self.redis,
            validate_on_load=True,
        )

    def _schema(self) -> dict:
        return {
            "index": {
                "name": self.index_name,
                "prefix": self.key_prefix,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "chunk_id", "type": "tag"},
                {"name": "text", "type": "text"},
                {"name": "title", "type": "text"},
                {"name": "source", "type": "text"},
                {"name": "session_id", "type": "tag"},
                {"name": "created_at", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": settings.vector_dimension,
                        "distance_metric": settings.vector_distance_metric.lower(),
                        "datatype": "float32",
                    },
                },
            ],
        }

    def ensure_index(self) -> None:
        if not self.index.exists():
            self.index.create()

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
        self.index.load(
            [
                {
                    "chunk_id": chunk_key,
                    "text": text,
                    "title": title,
                    "source": source,
                    "session_id": session_id,
                    "created_at": created_at,
                    "embedding": np.array(vector, dtype=np.float32).tobytes(),
                }
            ],
            id_field="chunk_id",
            ttl=settings.session_ttl_seconds,
        )

    def search(self, vector: list[float], session_id: str | None = None, top_k: int = 4) -> list[SearchMatch]:
        session_filter = Tag("session_id") == session_id if session_id else None
        query = VectorQuery(
            vector=vector,
            vector_field_name="embedding",
            return_fields=["chunk_id", "text", "title", "source", "vector_distance"],
            filter_expression=session_filter,
            num_results=top_k,
            dtype="float32",
        )
        results = self.index.query(query)
        matches: list[SearchMatch] = []
        for doc in results:
            chunk_id = str(doc["chunk_id"])
            matches.append(
                SearchMatch(
                    key=self.index.key(chunk_id),
                    score=float(doc.get("vector_distance", 0.0)),
                    text=str(doc.get("text", "")),
                    title=str(doc.get("title", "")),
                    source=str(doc.get("source", "")),
                    chunk_id=chunk_id,
                )
            )
        return matches

    def get_index_stats(self) -> dict[str, str]:
        if not self.index.exists():
            return {"status": "missing"}
        info = self.index.info()
        return {
            "status": "ready",
            "num_docs": str(info.get("num_docs", "0")),
            "index_name": self.index_name,
        }

    def flush_session_docs(self, session_id: str) -> int:
        query = FilterQuery(
            filter_expression=Tag("session_id") == session_id,
            return_fields=["chunk_id"],
            num_results=10000,
        )
        results = self.index.query(query)
        ids = [str(doc["chunk_id"]) for doc in results]
        if not ids:
            return 0
        return int(self.index.drop_documents(ids))

    def session_has_docs(self, session_id: str) -> bool:
        query = CountQuery(filter_expression=Tag("session_id") == session_id)
        return bool(self.index.query(query))
