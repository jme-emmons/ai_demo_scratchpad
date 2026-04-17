from __future__ import annotations

DEFENSE_KNOWLEDGE_PACK = {
    "joint-all-domain-operations.txt": """
Joint All-Domain Operations (JADO) requires resilient data access, rapid decision support, and secure dissemination of relevant context to distributed teams.
AI applications used in this environment must balance speed with traceability, and they need retrieval mechanisms that keep generated answers tied to approved source material.
Platforms that pair orchestration with low-latency data access can improve mission responsiveness while reducing redundant model calls.
""",
    "contested-logistics.txt": """
Contested logistics planning depends on timely awareness of supply routes, degraded infrastructure, and shifting operational constraints.
Retrieval augmented generation can help operators summarize doctrine and logistics notes without forcing them to search across multiple documents manually.
Semantic caching can reduce repeated inference cost when teams ask the same operational planning questions in different words.
""",
    "zero-trust-ai-platform.txt": """
Enterprise AI for defense customers should apply zero-trust principles, explicit routing, and policy guardrails before sending prompts to language models.
Semantic routing helps separate safe general questions from document-grounded questions and from prompts that should be blocked or redirected.
Redis can support vector search, conversational memory, and semantic caches in one operational data layer.
""",
}
