from __future__ import annotations

import io
import time
from dataclasses import dataclass

from pypdf import PdfReader

from app.config import settings
from app.model_clients import EmbeddingClient
from app.utils import chunk_text, make_id, normalize_text
from app.vector_store import RedisVectorStore, SearchMatch


@dataclass
class IngestionResult:
    documents: int
    chunks: int
    embedding_latency_ms: float


class RAGService:
    def __init__(self) -> None:
        self.store = RedisVectorStore()
        self.embedder = EmbeddingClient()

    def ensure_index(self) -> None:
        self.store.ensure_index()

    def ingest_text(self, session_id: str, title: str, source: str, text: str) -> IngestionResult:
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        total_latency = 0.0
        for idx, chunk in enumerate(chunks):
            embedding = self.embedder.embed(chunk)
            total_latency += embedding.latency_ms
            key = f"doc:{session_id}:{idx}:{int(time.time())}"
            self.store.upsert_chunk(
                chunk_key=key,
                text=chunk,
                title=title,
                source=source,
                session_id=session_id,
                created_at=int(time.time()),
                vector=embedding.vector,
            )
        return IngestionResult(documents=1, chunks=len(chunks), embedding_latency_ms=round(total_latency, 2))

    def ingest_uploaded_file(self, session_id: str, uploaded_file) -> IngestionResult:
        filename = uploaded_file.name
        suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        raw_bytes = uploaded_file.read()
        if suffix == "pdf":
            reader = PdfReader(io.BytesIO(raw_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = raw_bytes.decode("utf-8", errors="ignore")
        return self.ingest_text(session_id, filename, f"upload:{filename}", normalize_text(text))

    def search(self, session_id: str, question: str, top_k: int) -> tuple[list[SearchMatch], float, list[float]]:
        embedding = self.embedder.embed(question)
        matches = self.store.search(embedding.vector, session_id=session_id, top_k=top_k)
        return matches, embedding.latency_ms, embedding.vector
