from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import settings
from app.model_clients import LLMClient, ModelClientError
from app.router import RouteDecision
from app.semantic_cache import CacheResult
from app.utils import estimate_tokens
from app.vector_store import SearchMatch

if TYPE_CHECKING:
    from app.memory import ConversationMemory
    from app.rag import IngestionResult, RAGService
    from app.router import SemanticRouter
    from app.semantic_cache import SemanticCache


SYSTEM_PROMPT = """You are a helpful enterprise AI assistant for a Redis + OpenShift AI demonstration.
Answer clearly, concisely, and professionally.
If retrieval context is provided, ground the answer in that context and mention sources.
If the request is unsafe, out of scope, or asks for prohibited assistance, refuse briefly and redirect toward safe, policy-aligned help.
"""


@dataclass(frozen=True)
class FeatureFlags:
    semantic_cache: bool = False
    memory: bool = False
    rag_context: bool = False
    routing: bool = False


@dataclass
class AskResult:
    answer: str
    route: RouteDecision
    cache: CacheResult
    retrieval_matches: list[SearchMatch]
    memory_summary: dict[str, int | str]
    llm_latency_ms: float
    embedding_latency_ms: float
    total_tokens: int
    used_cache: bool


class DemoService:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self._cache: SemanticCache | None = None
        self._router: SemanticRouter | None = None
        self._memory: ConversationMemory | None = None
        self._rag: RAGService | None = None
        self._cache_ready = False
        self._rag_ready = False

    def bootstrap(self) -> None:
        # Baseline chat should work even when Redis is unavailable, so Redis-backed
        # components are initialized lazily only when enhanced features require them.
        return None

    @property
    def cache(self):
        if self._cache is None:
            from app.semantic_cache import SemanticCache

            self._cache = SemanticCache()
        return self._cache

    @property
    def router(self):
        if self._router is None:
            from app.router import SemanticRouter

            self._router = SemanticRouter()
        return self._router

    @property
    def memory(self):
        if self._memory is None:
            from app.memory import ConversationMemory

            self._memory = ConversationMemory()
        return self._memory

    @property
    def rag(self):
        if self._rag is None:
            from app.rag import RAGService

            self._rag = RAGService()
        return self._rag

    def _ensure_cache_ready(self) -> None:
        if self._cache_ready:
            return
        self.cache.ensure_index()
        self._cache_ready = True

    def _ensure_rag_ready(self) -> None:
        if self._rag_ready:
            return
        self.rag.ensure_index()
        self._rag_ready = True

    def ingest_uploaded_file(self, session_id: str, uploaded_file) -> "IngestionResult":
        self._ensure_rag_ready()
        return self.rag.ingest_uploaded_file(session_id, uploaded_file)

    def clear_memory(self, session_id: str) -> None:
        self.memory.clear(session_id)

    def ask(
        self,
        session_id: str,
        question: str,
        top_k: int = settings.default_top_k,
        features: FeatureFlags | None = None,
    ) -> AskResult:
        features = features or FeatureFlags()
        cache_result = CacheResult(hit=False)
        query_vector: list[float] = []
        cache_embedding_latency = 0.0
        embedding_latency_ms = 0.0
        has_docs = False
        if features.semantic_cache:
            self._ensure_cache_ready()
            cache_result, query_vector, cache_embedding_latency = self.cache.lookup(question, session_id)
            embedding_latency_ms += cache_embedding_latency
        if features.rag_context:
            self._ensure_rag_ready()
            has_docs = self.rag.store.session_has_docs(session_id)

        if features.routing:
            route = self.router.decide(question, has_documents=has_docs)
            if route.route == "rag" and not features.rag_context:
                route = RouteDecision(
                    route="general",
                    score=route.score,
                    rationale="Routing selected RAG, but RAG context is disabled, so the request fell back to direct chat.",
                )
        elif features.rag_context and has_docs:
            route = RouteDecision(
                route="rag",
                score=0.0,
                rationale="Routing disabled; using uploaded document context because RAG is enabled and documents are available.",
            )
        else:
            route = RouteDecision(
                route="general",
                score=0.0,
                rationale="Routing disabled; sending the prompt directly to the LLM.",
            )

        retrieval_matches: list[SearchMatch] = []
        memory_summary = {"turns": 0, "estimated_tokens": 0, "preview": ""}
        memory_context = ""
        if features.memory:
            memory_context = self.memory.build_context(session_id)

        if route.route == "guardrail":
            answer = (
                "I can’t help with unsafe or prohibited requests. "
                "I can help explain secure AI application patterns, enterprise guardrails, or safe platform design instead."
            )
            total_tokens = estimate_tokens(question + answer)
            if features.memory:
                self.memory.append(session_id, "user", question)
                self.memory.append(session_id, "assistant", answer)
                memory_summary = self.memory.summary(session_id)
            return AskResult(
                answer=answer,
                route=route,
                cache=CacheResult(hit=False),
                retrieval_matches=[],
                memory_summary=memory_summary,
                llm_latency_ms=0.0,
                embedding_latency_ms=embedding_latency_ms,
                total_tokens=total_tokens,
                used_cache=False,
            )

        if features.semantic_cache and cache_result.hit and route.route == "general":
            if features.memory:
                self.memory.append(session_id, "user", question)
                self.memory.append(session_id, "assistant", cache_result.answer or "")
                memory_summary = self.memory.summary(session_id)
            return AskResult(
                answer=cache_result.answer or "",
                route=route,
                cache=cache_result,
                retrieval_matches=[],
                memory_summary=memory_summary,
                llm_latency_ms=0.0,
                embedding_latency_ms=embedding_latency_ms,
                total_tokens=cache_result.tokens_saved,
                used_cache=True,
            )

        prompt = question
        if route.route == "rag" and features.rag_context:
            retrieval_matches, rag_embedding_latency, _ = self.rag.search(session_id, question, top_k=top_k)
            embedding_latency_ms += rag_embedding_latency
            context = "\n\n".join(
                f"Source: {match.title or match.source}\nSnippet: {match.text}" for match in retrieval_matches
            )
            memory_block = f"Conversation memory:\n{memory_context}\n\n" if features.memory and memory_context else ""
            prompt = (
                f"{memory_block}"
                f"Retrieved context:\n{context}\n\n"
                f"User question:\n{question}\n\n"
                "Answer using the retrieved context first. If the context is insufficient, say so clearly."
            )
        elif features.memory and memory_context:
            prompt = (
                f"Conversation memory:\n{memory_context}\n\n"
                f"User question:\n{question}\n\n"
                "Answer as a Redis and OpenShift AI solution assistant for an enterprise audience."
            )

        try:
            generation = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        except ModelClientError as exc:
            raise RuntimeError(
                "The model endpoint could not be reached or returned an unexpected payload. "
                f"Update the API format settings if needed. Details: {exc}"
            ) from exc

        total_tokens = generation.total_tokens or estimate_tokens(prompt + generation.text)
        if features.memory:
            self.memory.append(session_id, "user", question)
            self.memory.append(session_id, "assistant", generation.text)
            memory_summary = self.memory.summary(session_id)
        if features.semantic_cache and route.route == "general":
            self.cache.store_answer(
                session_id=session_id,
                question=question,
                answer=generation.text,
                vector=query_vector,
                latency_ms=generation.latency_ms,
                total_tokens=total_tokens,
            )
        return AskResult(
            answer=generation.text,
            route=route,
            cache=CacheResult(hit=False),
            retrieval_matches=retrieval_matches,
            memory_summary=memory_summary,
            llm_latency_ms=generation.latency_ms,
            embedding_latency_ms=embedding_latency_ms,
            total_tokens=total_tokens,
            used_cache=False,
        )
