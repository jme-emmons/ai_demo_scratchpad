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
    initial_sidebar_state="expanded",
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
        "baseline_metrics": {"total_tokens": 0},
        "enhanced_session_id": uuid.uuid4().hex[:12],
        "enhanced_metrics": {"cache_hits": 0, "tokens_saved": 0, "cost_saved": 0.0, "total_tokens": 0},
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


def render_success(container, message: str) -> None:
    container.markdown(
        f"""
        <div style="
            border-left: 4px solid var(--redis-primary);
            border: 1px solid rgba(255, 68, 56, 0.18);
            background: linear-gradient(135deg, rgba(255, 68, 56, 0.16), rgba(255, 68, 56, 0.06));
            color: var(--redis-ink);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-top: 0.75rem;
            font-weight: 600;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_branding_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --redis-ink: #f6efe5;
          --redis-ink-muted: rgba(246, 239, 229, 0.72);
          --redis-surface: #16100f;
          --redis-surface-strong: #211817;
          --redis-panel: rgba(33, 24, 23, 0.94);
          --redis-panel-soft: rgba(26, 18, 17, 0.88);
          --redis-stroke: rgba(246, 239, 229, 0.12);
          --redis-primary: #ff4438;
          --redis-primary-dark: #c92d24;
          --redis-accent: #a8251e;
          --redis-accent-soft: rgba(255, 68, 56, 0.22);
          --redhat-accent: #ee0000;
          --redhat-accent-soft: rgba(238, 0, 0, 0.14);
          --success-soft: rgba(50, 115, 82, 0.28);
          --shadow-soft: 0 24px 60px rgba(0, 0, 0, 0.38);
          --radius-xl: 24px;
          --radius-lg: 18px;
          --radius-md: 14px;
        }

        .stApp {
          background:
            radial-gradient(circle at top right, rgba(255, 68, 56, 0.16), transparent 26%),
            radial-gradient(circle at bottom left, rgba(238, 0, 0, 0.08), transparent 24%),
            linear-gradient(180deg, #0f0a09 0%, #181110 100%);
          color: var(--redis-ink);
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(24, 16, 15, 0.98), rgba(18, 12, 11, 0.98));
          border-right: 1px solid var(--redis-stroke);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
          color: var(--redis-ink);
        }

        [data-testid="stSidebarCollapsedControl"] button {
          background: rgba(255, 68, 56, 0.14);
          border-radius: 999px;
        }

        .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
        }

        .brand-shell {
          border: 1px solid var(--redis-stroke);
          background: linear-gradient(135deg, rgba(255, 68, 56, 0.28), rgba(27, 19, 18, 0.96));
          border-radius: var(--radius-xl);
          padding: 1.5rem 1.75rem;
          box-shadow: var(--shadow-soft);
          margin-bottom: 1rem;
        }

        .brand-title {
          font-size: 2.2rem;
          font-weight: 700;
          letter-spacing: -0.03em;
          margin: 0;
          color: var(--redis-ink);
        }

        .brand-subtitle {
          margin: 0.5rem 0 0;
          color: var(--redis-ink-muted);
          font-size: 1rem;
          max-width: 64rem;
        }

        .brand-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.6rem;
          margin-top: 1rem;
        }

        .brand-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.45rem 0.8rem;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid var(--redis-stroke);
          font-size: 0.87rem;
          color: var(--redis-ink);
        }

        .brand-chip strong {
          color: #ff948a;
        }

        .panel-card {
          border: 1px solid var(--redis-stroke);
          background: var(--redis-panel);
          border-radius: var(--radius-xl);
          padding: 1rem 1rem 1.2rem;
          box-shadow: var(--shadow-soft);
          min-height: 100%;
        }

        .panel-card.baseline {
          background: linear-gradient(180deg, rgba(33, 24, 23, 0.96), rgba(21, 15, 14, 0.96));
        }

        .panel-card.enhanced {
          background: linear-gradient(180deg, rgba(52, 22, 20, 0.96), rgba(24, 16, 15, 0.96));
          border-color: rgba(255, 68, 56, 0.24);
        }

        .section-card {
          border: 1px solid var(--redis-stroke);
          background: var(--redis-panel-soft);
          border-radius: var(--radius-lg);
          padding: 0.85rem 1rem;
          margin-top: 0.9rem;
        }

        .section-card.controls {
          border-left: 4px solid var(--redis-primary);
        }

        .section-card.telemetry {
          border-left: 4px solid var(--redhat-accent);
        }

        .stMarkdown, .stCaption, .stText, p, label {
          color: var(--redis-ink);
        }

        div[data-testid="stTextArea"] textarea {
          border-radius: var(--radius-md);
          border: 1px solid var(--redis-stroke);
          background: rgba(255, 255, 255, 0.04);
          color: var(--redis-ink);
        }

        div[data-testid="stTextArea"] textarea:focus {
          border-color: var(--redis-primary);
          box-shadow: 0 0 0 1px var(--redis-primary);
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button {
          background: linear-gradient(135deg, var(--redis-primary), var(--redis-primary-dark));
          color: white;
          border: none;
          border-radius: 999px;
          font-weight: 700;
          padding: 0.55rem 1rem;
          box-shadow: 0 14px 28px rgba(255, 68, 56, 0.25);
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
          background: linear-gradient(135deg, var(--redis-primary-dark), var(--redis-primary));
        }

        div[data-testid="stFileUploader"] section {
          border-radius: var(--radius-md);
          border: 1px dashed rgba(255, 68, 56, 0.34);
          background: rgba(255, 68, 56, 0.05);
        }

        div[data-testid="stMetric"] {
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid var(--redis-stroke);
          border-radius: var(--radius-md);
          padding: 0.6rem 0.8rem;
        }

        div[data-testid="stAlert"] {
          border-radius: var(--radius-md);
        }

        .stInfo, .stSuccess, .stWarning, .stError {
          background: rgba(255, 255, 255, 0.03);
        }

        div[data-testid="stAlert"][data-baseweb="notification"] {
          border: 1px solid var(--redis-stroke);
          box-shadow: none;
        }

        div[data-testid="stAlert"][data-baseweb="notification"]:has([data-testid="stMarkdownContainer"]) {
          color: var(--redis-ink);
        }

        div[data-testid="stAlert"] [data-testid="stNotificationContentSuccess"] {
          background: linear-gradient(135deg, rgba(255, 68, 56, 0.16), rgba(255, 68, 56, 0.07));
          border-left: 4px solid var(--redis-primary);
          color: var(--redis-ink);
        }

        div[data-testid="stAlert"] [data-testid="stNotificationContentSuccess"] svg {
          fill: var(--redis-primary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="brand-shell">
          <p class="brand-title">Redis + OpenShift Mission Assistant</p>
          <p class="brand-subtitle">
            Compare a direct baseline LLM experience with a Redis-enhanced workflow built & hosted on OpenShift AI.
          </p>
          <div class="brand-chip-row">
            <span class="brand-chip"><strong>LLM</strong> {settings.llm_model}</span>
            <span class="brand-chip"><strong>Embeddings</strong> {settings.embedding_model}</span>
            <span class="brand-chip"><strong>Session ID</strong> {st.session_state.enhanced_session_id}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_enhanced_sidebar(service: DemoService) -> None:
    with st.sidebar:
        st.markdown("## Enhanced Controls")
        st.caption("Redis-backed feature toggles and uploads for the enhanced panel.")
        st.toggle("Semantic caching", key="enhanced_feature_semantic_cache")
        st.toggle("Memory", key="enhanced_feature_memory")
        st.toggle("RAG context", key="enhanced_feature_rag")
        st.toggle("Routing", key="enhanced_feature_routing")
        if st.button("Clear Enhanced Memory", use_container_width=True):
            try:
                service.clear_memory(st.session_state.enhanced_session_id)
            except Exception as exc:
                render_error(st.sidebar, f"Unable to clear enhanced memory: {exc}", traceback.format_exc())
            else:
                st.sidebar.success("Enhanced memory cleared.")
        handle_enhanced_uploads(service, st.sidebar)


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

    response_latency_ms = result.llm_latency_ms + result.embedding_latency_ms
    container.markdown("#### Telemetry")
    container.write(f"**Response latency:** {response_latency_ms:.1f} ms")
    container.write(f"**Last reply tokens:** {result.total_tokens}")
    container.write(f"**Total session tokens:** {st.session_state.enhanced_metrics['total_tokens']}")
    if features.semantic_cache:
        container.write(f"**Cache status:** {'Hit' if result.used_cache else 'Miss'}")
        metrics = st.session_state.enhanced_metrics
        container.write(
            f"**Cache summary:** {metrics['cache_hits']} hits | "
            f"{metrics['tokens_saved']} tokens saved | "
            f"${metrics['cost_saved']:.4f} estimated cost saved"
        )
    if features.memory:
        memory = result.memory_summary
        container.write(f"**Memory turns retained:** {memory['turns']}")
    if features.rag_context:
        container.write("**Retrieved evidence:**")
        if result.retrieval_matches:
            for match in result.retrieval_matches:
                container.markdown(
                    f"- `{match.title or match.source}` | score `{match.score:.4f}`\n\n  {match.text[:180]}..."
                )
        else:
            container.caption("No retrieval context used for the last answer.")
    if features.routing:
        container.write(f"**Route selected:** `{result.route.route}`")


def render_baseline_telemetry(container) -> None:
    result = st.session_state.baseline_last_result
    if not result:
        container.caption("Baseline telemetry will appear after you send a message.")
        return

    container.markdown("#### Telemetry")
    container.write(f"**Response latency:** {result.llm_latency_ms:.1f} ms")
    container.write(f"**Last reply tokens:** {result.total_tokens}")
    container.write(f"**Total session tokens:** {st.session_state.baseline_metrics['total_tokens']}")


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
        return False
    st.session_state.baseline_messages.append({"role": "user", "content": prompt})
    try:
        result = service.ask(session_id="baseline", question=prompt, features=FeatureFlags())
    except Exception as exc:
        st.session_state.baseline_error = (f"Unable to get a baseline response: {exc}", traceback.format_exc())
        return True
    st.session_state.baseline_last_result = result
    st.session_state.baseline_messages.append({"role": "assistant", "content": result.answer})
    st.session_state.baseline_error = None
    st.session_state.baseline_metrics["total_tokens"] += result.total_tokens
    return True


def process_enhanced_submit(service: DemoService) -> None:
    prompt = st.session_state.enhanced_input.strip()
    if not prompt:
        return False
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
        return True
    st.session_state.enhanced_last_result = result
    st.session_state.enhanced_messages.append({"role": "assistant", "content": result.answer})
    st.session_state.enhanced_error = None
    st.session_state.enhanced_metrics["total_tokens"] += result.total_tokens
    if result.cache.hit:
        st.session_state.enhanced_metrics["cache_hits"] += 1
        st.session_state.enhanced_metrics["tokens_saved"] += result.cache.tokens_saved
        st.session_state.enhanced_metrics["cost_saved"] += result.cache.cost_saved
    return True


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
            render_success(container, f"Ingested {upload.name}: {result.chunks} chunks")
            known_uploads.add(upload_id)
    st.session_state.enhanced_ingested_uploads = sorted(known_uploads)


def main() -> None:
    init_session_state()
    service = get_demo_service()
    inject_branding_styles()
    render_enhanced_sidebar(service)
    render_header()
    rerun_requested = False

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Baseline LLM")
        st.caption("A neutral baseline path that uses the same model and system prompt without Redis-backed features.")
        baseline_messages = st.container(height=420)
        render_messages(
            baseline_messages,
            st.session_state.baseline_messages,
            "Send a message to test the baseline LLM flow.",
        )
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
                baseline_updated = process_baseline_submit(service)
            if baseline_updated:
                rerun_requested = True
        if st.session_state.baseline_error:
            message, details = st.session_state.baseline_error
            render_error(st, message, details)
        baseline_telemetry = st.container()
        render_baseline_telemetry(baseline_telemetry)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right_col:
        st.subheader("Redis Enhanced")
        st.caption("Enable Redis-backed features selectively to compare caching, memory, routing, and retrieval.")
        enhanced_messages = st.container(height=420)
        render_messages(
            enhanced_messages,
            st.session_state.enhanced_messages,
            "Send a message or upload a file to test the enhanced flow.",
        )
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
                enhanced_updated = process_enhanced_submit(service)
            if enhanced_updated:
                rerun_requested = True
        features = enhanced_feature_flags()
        render_enhanced_telemetry(st.container(), features)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.enhanced_error:
            message, details = st.session_state.enhanced_error
            render_error(st, message, details)
        st.markdown("</div>", unsafe_allow_html=True)

    if rerun_requested:
        st.rerun()


if __name__ == "__main__":
    main()
