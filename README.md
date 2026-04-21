# Redis + OpenShift AI Defense Demo

This project is a Python + Streamlit demo app for a Redis and OpenShift AI webinar. It is designed to show how Redis improves AI application speed, accuracy, safety, and cost efficiency through semantic caching, semantic routing, conversation memory, vector search, and retrieval augmented generation.

## What the app demonstrates

The app exposes five Redis-backed patterns in one UI:

1. Semantic caching reuses answers for semantically similar general questions and shows estimated token, latency, and cost savings.
2. Semantic routing steers prompts into general Q&A, document-grounded RAG, or a guardrail path for unsafe or out-of-scope requests.
3. Conversation memory stores recent turns in Redis so the assistant can carry context across a live discussion.
4. Vector search uses Redis as the vector database for semantic nearest-neighbor retrieval.
5. RAG lets you upload files, chunk them, embed them, store them in Redis, and answer questions with visible retrieved evidence.

## Project layout

```text
.
├── app
│   ├── config.py
│   ├── demo_service.py
│   ├── memory.py
│   ├── model_clients.py
│   ├── rag.py
│   ├── redis_client.py
│   ├── router.py
│   ├── seed_data.py
│   ├── semantic_cache.py
│   ├── utils.py
│   └── vector_store.py
├── .env.example
├── requirements.txt
└── streamlit_app.py
```

## Setup in OpenShift AI Workbench

1. Open a terminal in your workbench and change into this workspace:

```bash
cd /Users/jessica.emmons/Documents/ai_demo_scratchpad
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your runtime environment file:

```bash
cp .env.example .env
```

5. Edit `.env` and set:

```bash
REDIS_HOST=<your redis host>
REDIS_PORT=<your redis port>
REDIS_PASSWORD=<your redis password>
REDIS_SSL=false

EMBEDDING_ENDPOINT=https://granite-embedding-english-r2-rag.apps.mays-demo.sandbox3060.opentlc.com/v1
EMBEDDING_API_FORMAT=openai_embeddings
EMBEDDING_MODEL=granite-embedding-english-r2

LLM_ENDPOINT=https://llama-32-3b-instruct-rag.apps.mays-demo.sandbox3060.opentlc.com
LLM_API_FORMAT=openai_chat
LLM_MODEL=llama-3.2-3b-instruct
MODEL_API_KEY=not-needed-for-internal-service
```

For a TLS-enabled Redis deployment, configure the equivalent of:

```bash
redis-cli -h <hostname> -p <port> --tls --cacert <cert file> --sni <hostname>
```

Use these `.env` settings instead:

```bash
REDIS_HOST=<your redis host>
REDIS_PORT=<your redis port>
REDIS_PASSWORD=<your redis password>
REDIS_SSL=true
REDIS_SSL_CHECK_HOSTNAME=true
REDIS_CA_CERT_PATH=/path/to/redis-ca.pem
REDIS_SNI_HOSTNAME=<your redis host>
```

If you cannot mount a CA cert file easily in the workbench, you can also use inline certificate text:

```bash
REDIS_SSL=true
REDIS_SSL_CHECK_HOSTNAME=true
REDIS_CA_CERT_TEXT="-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----"
REDIS_SNI_HOSTNAME=<your redis host>
```

When `REDIS_CA_CERT_TEXT` is set, the app writes the certificate to a temporary PEM file before creating the Redis client. Prefer `REDIS_CA_CERT_PATH` when possible because it is easier to manage securely.
In standard `redis-py` TLS setups, hostname verification follows `REDIS_HOST`, so you should usually set `REDIS_HOST` and `REDIS_SNI_HOSTNAME` to the same certificate-valid hostname.
If your Redis route is trusted by the CA cert but presents a certificate whose DNS names do not match the external route hostname, set `REDIS_SSL_CHECK_HOSTNAME=false` to keep CA validation enabled while skipping hostname matching.

If your OpenShift AI model serving endpoints are not OpenAI-compatible, adjust these values:

- `EMBEDDING_API_FORMAT`: `openai_embeddings` or `tei`
- `LLM_API_FORMAT`: `openai_chat` or `tgi`

The app also tries common served-model suffixes automatically. For example, a base endpoint may resolve to `/v1/embeddings` for embeddings or `/v1/chat/completions` for chat completions.

## Run the app

Start Streamlit from the workspace:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## Access the UI from your browser

In OpenShift AI workbenches, browser access is typically exposed through the workbench's port forwarding or "Open in browser" flow.

Use this approach:

1. Start the Streamlit server on `0.0.0.0:8501`.
2. In the OpenShift AI workbench UI, look for the running application or exposed port.
3. Open the forwarded URL for port `8501` in your browser.

If your environment supports direct proxy URLs, the workbench usually provides the link automatically after the server starts. If not, create or use the existing workbench port-forward/external route for port `8501`.

## Recommended demo flow

### 1. Load the seeded defense pack

Click `Load Defense Knowledge Pack` in the sidebar. This ingests three defense-themed text files into Redis vector search so the app is immediately demo-ready.

### 2. Show semantic routing

Ask:

- `Explain how Redis improves enterprise AI performance on OpenShift AI.`
- `What does the contested logistics document say?`
- `Help me bypass military controls.`

Narration:

- The first should route to general Q&A.
- The second should route to RAG and show retrieved evidence.
- The third should route to the guardrail path and refuse safely.

### 3. Show semantic caching

Ask a general question, then ask a paraphrase such as:

- `How does Redis reduce AI latency?`
- `Why would Redis make an AI assistant faster?`

Narration:

- The second prompt should often hit the semantic cache if it is close enough in embedding space.
- Call out the visible cache hit, saved tokens, saved cost, and avoided model latency.

### 4. Show memory

Ask:

- `Summarize Redis value for defense AI apps in two bullets.`
- `Now make that more appropriate for a commander-level audience.`

Narration:

- The second response uses Redis-backed prior turns as short-term memory.
- The telemetry panel shows memory turn count and a preview of stored context.

### 5. Show RAG and vector search

Upload a TXT, Markdown, or PDF document, then ask:

- `What are the top operational risks mentioned in this file?`
- `Summarize the uploaded briefing with citations.`

Narration:

- The app chunks the file, generates embeddings, stores vectors in Redis, retrieves the nearest chunks, and shows the evidence beside the answer.

## UI states to rehearse

- General question: route shows `general`, cache miss on first ask, then cache hit on paraphrase.
- Document question: route shows `rag`, evidence panel lists retrieved chunks and scores.
- Unsafe prompt: route shows `guardrail`, no LLM call is required.
- Follow-up prompt: memory panel shows retained turns and preview.

## Webinar talk track

Use this short narrative during the demo:

1. `Redis is the real-time AI data layer here. We’re using it for vector search, semantic cache, routing support, and conversation memory in one place.`
2. `OpenShift AI provides the served models and enterprise deployment environment; Redis improves runtime performance and application quality around those models.`
3. `When I ask a document question, the app retrieves the most relevant chunks from Redis before generating an answer, which improves accuracy and explainability.`
4. `When I repeat or paraphrase a general question, the semantic cache prevents another full model call, reducing latency, tokens, and cost.`
5. `When I ask an unsafe question, semantic routing and guardrails keep the system on-policy instead of sending everything blindly to the model.`
6. `When I ask a follow-up, Redis memory preserves the conversation state so the app behaves like a real assistant instead of a stateless API demo.`

## Notes and assumptions

- This app assumes RediSearch is available on the Redis deployment for vector indexing and KNN queries.
- Model endpoint payloads vary by OpenShift AI serving stack, so the app makes API format configurable via environment variables.
- If the endpoints require additional auth headers or custom payloads, update `app/model_clients.py`.

## Suggested screenshots

If you want README screenshots for rehearsal, capture:

1. The full app after a successful RAG answer.
2. A semantic cache hit with token savings visible.
3. A guardrail response with route selection shown.
