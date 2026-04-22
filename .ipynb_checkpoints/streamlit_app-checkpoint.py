from __future__ import annotations

import traceback
import uuid

import streamlit as st

from app.config import settings
from app.demo_service import DemoService
from app.seed_data import DEFENSE_KNOWLEDGE_PACK


st.set_page_config(
    page_title="Redis + OpenShift AI Defense Demo",
    page_icon=":satellite:",
    layout="wide",
)


@st.cache_resource
def get_demo_service() -> DemoService:
    service = DemoService()
    service.bootstrap()
    return service


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {"cache_hits": 0, "tokens_saved": 0, "cost_saved": 0.0}
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def ingest_seed_data(service: DemoService, session_id: str) -> None:
    if st.session_state.get("seeded"):
        return
    for name, text in DEFENSE_KNOWLEDGE_PACK.items():
        service.rag.ingest_text(session_id, title=name, source=f"seed:{name}", text=text)
    st.session_state.seeded = True


def render_sidebar_error(message: str, exc: Exception) -> None:
    st.sidebar.error(message)
    with st.sidebar.expander("Technical details"):
        st.code(str(exc))
        st.code(traceback.format_exc())


def render_sidebar(service: DemoService) -> None:
    st.sidebar.title("Demo Controls")
    st.sidebar.caption("OpenShift AI + Redis reference implementation")
    st.sidebar.text_input("Session ID", key="session_id")
    if st.sidebar.button("Load Defense Knowledge Pack", use_container_width=True):
        try:
            ingest_seed_data(service, st.session_state.session_id)
        except Exception as exc:
            render_sidebar_error(
                "Unable to load the defense knowledge pack. Check the embedding endpoint route and API format.",
                exc,
            )
        else:
            st.sidebar.success("Seed documents loaded into Redis vector search.")

    uploads = st.sidebar.file_uploader(
        "Upload documents for RAG",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploads:
        for upload in uploads:
            try:
                result = service.rag.ingest_uploaded_file(st.session_state.session_id, upload)
            except Exception as exc:
                render_sidebar_error(
                    f"Unable to ingest {upload.name}. Check the embedding endpoint route and API format.",
                    exc,
                )
                break
            else:
                st.sidebar.success(f"Ingested {upload.name}: {result.chunks} chunks")

    if st.sidebar.button("Clear Conversation Memory", use_container_width=True):
        service.memory.clear(st.session_state.session_id)
        st.sidebar.info("Memory cleared for the active session.")

    st.sidebar.markdown("### Environment")
    st.sidebar.write(f"LLM format: `{settings.llm_api_format}`")
    st.sidebar.write(f"Embedding format: `{settings.embedding_api_format}`")
    st.sidebar.write(f"Vector index: `{settings.vector_index_name}`")


def render_header() -> None:
    st.title("Redis + OpenShift AI Mission Assistant")
    st.markdown(
        "Demonstrates semantic caching, semantic routing, Redis-backed memory, vector search, and RAG in one webinar-friendly UI."
    )


def render_metrics() -> None:
    metrics = st.session_state.metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Semantic Cache Hits", metrics["cache_hits"])
    col2.metric("Estimated Tokens Saved", metrics["tokens_saved"])
    col3.metric("Estimated Cost Saved", f"${metrics['cost_saved']:.4f}")
    col4.metric("Active Session", st.session_state.session_id)


def render_status_panels() -> None:
    result = st.session_state.last_result
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Conversation")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    with col2:
        st.subheader("Operational Telemetry")
        if not result:
            st.info("Ask a question to populate routing, cache, retrieval, and memory telemetry.")
            return
        st.write(f"**Route selected:** `{result.route.route}`")
        st.write(f"**Route rationale:** {result.route.rationale}")
        st.write(f"**Cache status:** {'Hit' if result.used_cache else 'Miss'}")
        st.write(f"**Embedding latency:** {result.embedding_latency_ms:.1f} ms")
        st.write(f"**LLM latency:** {result.llm_latency_ms:.1f} ms")
        st.write(f"**Estimated total tokens:** {result.total_tokens}")
        memory = result.memory_summary
        st.write(f"**Memory turns retained:** {memory['turns']}")
        st.write(f"**Memory token estimate:** {memory['estimated_tokens']}")
        st.write("**Memory preview:**")
        st.caption(memory["preview"] or "No memory yet.")

        if result.cache.hit:
            st.success(
                f"Semantic cache hit on a similar question. Saved ~{result.cache.tokens_saved} tokens and "
                f"${result.cache.cost_saved:.4f}."
            )

        st.write("**Retrieved evidence:**")
        if result.retrieval_matches:
            for match in result.retrieval_matches:
                st.markdown(
                    f"- `{match.title or match.source}` | score `{match.score:.4f}`\n\n  {match.text[:180]}..."
                )
        else:
            st.caption("No retrieval context used for the last answer.")


def main() -> None:
    init_session_state()
    service = get_demo_service()
    render_header()
    render_sidebar(service)
    render_metrics()
    render_status_panels()

    user_input = st.chat_input("Ask about Redis, OpenShift AI, or your uploaded documents...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        result = service.ask(st.session_state.session_id, user_input)
    except Exception as exc:
        st.error(str(exc))
        st.code(traceback.format_exc())
        return

    st.session_state.last_result = result
    st.session_state.messages.append({"role": "assistant", "content": result.answer})
    if result.cache.hit:
        st.session_state.metrics["cache_hits"] += 1
        st.session_state.metrics["tokens_saved"] += result.cache.tokens_saved
        st.session_state.metrics["cost_saved"] += result.cache.cost_saved
    st.rerun()


if __name__ == "__main__":
    main()
