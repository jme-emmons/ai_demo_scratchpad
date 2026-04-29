from __future__ import annotations

from functools import lru_cache

from redisvl.utils.vectorize import CustomTextVectorizer

from app.model_clients import EmbeddingClient


class _RedisVLEmbeddingAdapter:
    def __init__(self) -> None:
        self._embedder = EmbeddingClient()

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text).vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


@lru_cache(maxsize=1)
def get_redisvl_vectorizer() -> CustomTextVectorizer:
    adapter = _RedisVLEmbeddingAdapter()
    return CustomTextVectorizer(
        embed=adapter.embed,
        embed_many=adapter.embed_many,
        dtype="float32",
    )
