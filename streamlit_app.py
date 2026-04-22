from __future__ import annotations

import traceback
import uuid

import streamlit as st

from app.config import settings
from app.demo_service import DemoService, FeatureFlags


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
    defaults = {
        "baseline_messages": [],
        "enhanced_messages": [],
        "baseline_last_result": None,
        "enhanced_last_result": None,
        "baseline_error": None,
        "enhanced_error": None,
        "baseline_input": "",
        "enhanced_input": "",
        "enhanced_session_id": uuid.uuid4().hex[:12],
        "enhanced_metrics": {"cache_hits": 0, "tokens_saved": 0, "cost_saved": 0.0},
        "enhanced_feature_semantic_cache": False,
        "enhanced_feature_memory": False,
        "enhanced_feature_rag": False,
        "enhanced_feature_routing": False,
        "enhanced_ingested_uploads": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_error(container, message: str, details: str) -> None:
    container.error(message)
    with container.expander("Technical details"):
        st.code(details)


def render_sidebar() -> None:
    st.sidebar.title("Environment")
    st.sidebar.caption("OpenShift AI + Redis reference implementation")
    st.sidebar.write(f"LLM format: `{settings.llm_api_format}`")
    st.sidebar.write(f"Embedding format: `{settings.embedding_api_format}`")
    st.sidebar.write(f"Vector index: `{settings.vector_index_name}`")
    st.sidebar.write(f"Enhanced session: `{st.session_state.enhanced_session_id}`")


def render_header() -> None:
    st.title("Redis + OpenShift AI Mission Assistant")
    st.markdown(
        "Compare a direct baseline LLM chat against a configurable Redis-enhanced chat in one demo-friendly UI."
    )


def render_messages(container, messages: list[dict[str, str]], empty_text: str) -> None:
    with container:
        if not messages:
            st.info(empty_text)
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def render_enhanced_telemetry(container, features: FeatureFlags) -> None:
    result = st.session_state.enhanced_last_result
    if not result:
        container.caption("Enhanced telemetry will appear after you send a message.")
        return

    container.markdown("#### Telemetry")
    container.write(f"**LLM latency:** {result.llm_latency_ms:.1f} ms")
    container.write(f"**Estimated total tokens:** {result.total_tokens}")
    if features.routing:
        container.write(f"**Route selected:** `{result.route.route}`")
        container.write(f"**Route rationale:** {result.route.rationale}")
    if features.semantic_cache:
        container.write(f"**Cache status:** {'Hit' if result.used_cache else 'Miss'}")
        if result.cache.hit:
            container.success(
                f"Semantic cache hit. Saved ~{result.cache.tokens_saved} tokens and "
                f"${result.cache.cost_saved:.4f}."
            )
    if features.semantic_cache or features.rag_context:
        container.write(f"**Embedding latency:** {result.embedding_latency_ms:.1f} ms")
    if features.memory:
        memory = result.memory_summary
        container.write(f"**Memory turns retained:** {memory['turns']}")
        container.write(f"**Memory token estimate:** {memory['estimated_tokens']}")
        container.caption(memory["preview"] or "No memory yet.")
    if features.rag_context:
        container.write("**Retrieved evidence:**")
        if result.retrieval_matches:
            for match in result.retrieval_matches:
                container.markdown(
                    f"- `{match.title or match.source}` | score `{match.score:.4f}`\n\n  {match.text[:180]}..."
                )
        else:
            container.caption("No retrieval context used for the last answer.")


def render_baseline_telemetry(container) -> None:
    result = st.session_state.baseline_last_result
    if not result:
        container.caption("Baseline telemetry will appear after you send a message.")
        return

    container.markdown("#### Telemetry")
    container.write(f"**LLM latency:** {result.llm_latency_ms:.1f} ms")
    container.write(f"**Estimated total tokens:** {result.total_tokens}")


def enhanced_feature_flags() -> FeatureFlags:
    return FeatureFlags(
        semantic_cache=st.session_state.enhanced_feature_semantic_cache,
        memory=st.session_state.enhanced_feature_memory,
        rag_context=st.session_state.enhanced_feature_rag,
        routing=st.session_state.enhanced_feature_routing,
    )


def process_baseline_submit(service: DemoService) -> None:
    prompt = st.session_state.baseline_input.strip()
    if not prompt:
        return
    st.session_state.baseline_messages.append({"role": "user", "content": prompt})
    try:
        result = service.ask(session_id="baseline", question=prompt, features=FeatureFlags())
    except Exception as exc:
        st.session_state.baseline_error = (f"Unable to get a baseline response: {exc}", traceback.format_exc())
        return
    st.session_state.baseline_last_result = result
    st.session_state.baseline_messages.append({"role": "assistant", "content": result.answer})
    st.session_state.baseline_error = None


def process_enhanced_submit(service: DemoService) -> None:
    prompt = st.session_state.enhanced_input.strip()
    if not prompt:
        return
    features = enhanced_feature_flags()
    st.session_state.enhanced_messages.append({"role": "user", "content": prompt})
    try:
        result = service.ask(
            session_id=st.session_state.enhanced_session_id,
            question=prompt,
            features=features,
        )
    except Exception as exc:
        st.session_state.enhanced_error = (f"Unable to get an enhanced response: {exc}", traceback.format_exc())
        return
    st.session_state.enhanced_last_result = result
    st.session_state.enhanced_messages.append({"role": "assistant", "content": result.answer})
    st.session_state.enhanced_error = None
    if result.cache.hit:
        st.session_state.enhanced_metrics["cache_hits"] += 1
        st.session_state.enhanced_metrics["tokens_saved"] += result.cache.tokens_saved
        st.session_state.enhanced_metrics["cost_saved"] += result.cache.cost_saved


def handle_enhanced_uploads(service: DemoService, container) -> None:
    uploads = container.file_uploader(
        "Upload files for the enhanced panel",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        key="enhanced_uploads",
    )
    if not uploads:
        return
    known_uploads = set(st.session_state.enhanced_ingested_uploads)
    for upload in uploads:
        upload_id = f"{upload.name}:{upload.size}"
        if upload_id in known_uploads:
            continue
        try:
            result = service.ingest_uploaded_file(st.session_state.enhanced_session_id, upload)
        except Exception as exc:
            render_error(container, f"Unable to ingest {upload.name}.", traceback.format_exc())
            break
        else:
            container.success(f"Ingested {upload.name}: {result.chunks} chunks")
            known_uploads.add(upload_id)
    st.session_state.enhanced_ingested_uploads = sorted(known_uploads)


def main() -> None:
    init_session_state()
    service = get_demo_service()
    render_header()
    render_sidebar()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Baseline Chat")
        st.caption("Direct LLM chat using the same model and system prompt, without Redis-backed features.")
        with st.form("baseline_form", clear_on_submit=True):
            st.text_area(
                "Message",
                key="baseline_input",
                placeholder="Ask the baseline model a question...",
                height=80,
            )
            baseline_submitted = st.form_submit_button("Send to Baseline", use_container_width=True)
        if baseline_submitted:
            with st.spinner("Baseline chat is generating a response..."):
                process_baseline_submit(service)
        baseline_messages = st.container(height=420)
        render_messages(
            baseline_messages,
            st.session_state.baseline_messages,
            "Send a message to test the baseline LLM flow.",
        )
        if st.session_state.baseline_error:
            message, details = st.session_state.baseline_error
            render_error(st, message, details)
        baseline_telemetry = st.container(border=True)
        render_baseline_telemetry(baseline_telemetry)

    with right_col:
        st.subheader("Redis-Enhanced Chat")
        st.caption("Toggle Redis-backed features on or off to compare behavior in the same session.")
        with st.form("enhanced_form", clear_on_submit=True):
            st.text_area(
                "Message",
                key="enhanced_input",
                placeholder="Ask the enhanced model a question...",
                height=80,
            )
            enhanced_submitted = st.form_submit_button("Send to Enhanced", use_container_width=True)
        if enhanced_submitted:
            with st.spinner("Enhanced chat is processing with the selected features..."):
                process_enhanced_submit(service)
        enhanced_messages = st.container(height=420)
        render_messages(
            enhanced_messages,
            st.session_state.enhanced_messages,
            "Send a message or upload a file to test the enhanced flow.",
        )
        feature_box = st.container(border=True)
        with feature_box:
            st.markdown("#### Enhanced Features")
            st.toggle("Semantic caching", key="enhanced_feature_semantic_cache")
            st.toggle("Memory", key="enhanced_feature_memory")
            st.toggle("RAG context", key="enhanced_feature_rag")
            st.toggle("Routing", key="enhanced_feature_routing")
            if st.button("Clear Enhanced Memory", use_container_width=True):
                try:
                    service.clear_memory(st.session_state.enhanced_session_id)
                except Exception as exc:
                    render_error(st, f"Unable to clear enhanced memory: {exc}", traceback.format_exc())
                else:
                    st.success("Enhanced memory cleared.")
            handle_enhanced_uploads(service, st)
            metrics = st.session_state.enhanced_metrics
            st.caption(
                f"Cache hits: {metrics['cache_hits']} | Tokens saved: {metrics['tokens_saved']} | "
                f"Estimated cost saved: ${metrics['cost_saved']:.4f}"
            )
            render_enhanced_telemetry(st, enhanced_feature_flags())
        if st.session_state.enhanced_error:
            message, details = st.session_state.enhanced_error
            render_error(st, message, details)


if __name__ == "__main__":
    main()
