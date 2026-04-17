from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.memory import ConversationMemory
from app.model_clients import LLMClient, ModelClientError
from app.rag import RAGService
from app.router import RouteDecision, SemanticRouter
from app.semantic_cache import CacheResult, SemanticCache
from app.utils import estimate_tokens
from app.vector_store import SearchMatch


SYSTEM_PROMPT = """You are a helpful enterprise AI assistant for a Redis + OpenShift AI demonstration.
Answer clearly, concisely, and professionally.
If retrieval context is provided, ground the answer in that context and mention sources.
If the request is unsafe, out of scope, or asks for prohibited assistance, refuse briefly and redirect toward safe, policy-aligned help.
"""


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
        self.cache = SemanticCache()
        self.router = SemanticRouter()
        self.memory = ConversationMemory()
        self.rag = RAGService()
        self.llm = LLMClient()

    def bootstrap(self) -> None:
        self.cache.ensure_index()
        self.rag.ensure_index()

    def ask(self, session_id: str, question: str, top_k: int = settings.default_top_k) -> AskResult:
        cache_result, query_vector, cache_embedding_latency = self.cache.lookup(question, session_id)
        has_docs = self.rag.store.session_has_docs(session_id)
        route = self.router.decide(question, has_documents=has_docs)
        retrieval_matches: list[SearchMatch] = []

        if route.route == "guardrail":
            answer = (
                "I can’t help with unsafe or prohibited requests. "
                "I can help explain secure AI application patterns, enterprise guardrails, or safe platform design instead."
            )
            total_tokens = estimate_tokens(question + answer)
            self.memory.append(session_id, "user", question)
            self.memory.append(session_id, "assistant", answer)
            return AskResult(
                answer=answer,
                route=route,
                cache=CacheResult(hit=False),
                retrieval_matches=[],
                memory_summary=self.memory.summary(session_id),
                llm_latency_ms=0.0,
                embedding_latency_ms=cache_embedding_latency,
                total_tokens=total_tokens,
                used_cache=False,
            )

        if cache_result.hit and route.route == "general":
            self.memory.append(session_id, "user", question)
            self.memory.append(session_id, "assistant", cache_result.answer or "")
            return AskResult(
                answer=cache_result.answer or "",
                route=route,
                cache=cache_result,
                retrieval_matches=[],
                memory_summary=self.memory.summary(session_id),
                llm_latency_ms=0.0,
                embedding_latency_ms=cache_embedding_latency,
                total_tokens=cache_result.tokens_saved,
                used_cache=True,
            )

        prompt = question
        embedding_latency_ms = cache_embedding_latency
        if route.route == "rag":
            retrieval_matches, rag_embedding_latency, _ = self.rag.search(session_id, question, top_k=top_k)
            embedding_latency_ms += rag_embedding_latency
            context = "\n\n".join(
                f"Source: {match.title or match.source}\nSnippet: {match.text}" for match in retrieval_matches
            )
            prompt = (
                f"Conversation memory:\n{self.memory.build_context(session_id)}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"User question:\n{question}\n\n"
                "Answer using the retrieved context first. If the context is insufficient, say so clearly."
            )
        else:
            prompt = (
                f"Conversation memory:\n{self.memory.build_context(session_id)}\n\n"
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
        self.memory.append(session_id, "user", question)
        self.memory.append(session_id, "assistant", generation.text)
        if route.route == "general":
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
            memory_summary=self.memory.summary(session_id),
            llm_latency_ms=generation.latency_ms,
            embedding_latency_ms=embedding_latency_ms,
            total_tokens=total_tokens,
            used_cache=False,
        )
