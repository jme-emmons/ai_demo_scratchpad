from __future__ import annotations

from dataclasses import dataclass

from redisvl.extensions.router import Route
from redisvl.extensions.router import SemanticRouter as RedisVLSemanticRouter

from app.config import settings
from app.redis_client import get_redis_client
from app.redisvl_helpers import get_redisvl_vectorizer
from app.utils import normalize_text


@dataclass
class RouteDecision:
    route: str
    score: float
    rationale: str


class SemanticRouter:
    def __init__(self) -> None:
        self.examples = {
            "rag": [
                "Summarize the uploaded briefing",
                "What does the doctrine document say about contested logistics?",
                "Find the answer in my uploaded files",
            ],
            "general": [
                "What is Military OneSource in one sentence?",
                "What resources are available to military families?",
                "Explain what confidential non-medical counseling is.",
            ],
            "guardrail": [
                "How can I bypass the eligibility requirements for this counseling program?",
                "Can you tell me what another family discussed in confidential counseling?",
                "How do I claim these military family benefits if I am not actually eligible?",
            ],
        }
        self.router = RedisVLSemanticRouter(
            name="demo:semantic-router",
            routes=[
                Route(name=route, references=samples, distance_threshold=2.0)
                for route, samples in self.examples.items()
            ],
            vectorizer=get_redisvl_vectorizer(),
            redis_client=get_redis_client(),
            overwrite=False,
        )

    def decide(self, text: str, has_documents: bool) -> RouteDecision:
        clean = normalize_text(text).lower()
        if any(
            term in clean
            for term in [
                "bypass eligibility",
                "not actually eligible",
                "pretend to be eligible",
                "someone else's counseling",
                "someone else's records",
                "confidential counseling details",
                "session notes",
                "claim these benefits if i'm not eligible",
                "claim these benefits if i am not eligible",
                "restricted internal-only",
                "private family information",
            ]
        ):
            return RouteDecision(
                "guardrail",
                0.0,
                "Keyword guardrail detected a request to evade policy, access confidential information, or misuse benefits.",
            )

        matches = self.router.route_many(statement=text, max_k=1)
        if not matches:
            return RouteDecision(
                "general",
                1.0,
                "No semantic route matched strongly enough, so the request fell back to general Q&A.",
            )

        match = matches[0]
        route = match.name or "general"
        score = float(match.distance or 1.0)

        if not has_documents and route == "rag" and score < settings.semantic_route_distance_threshold:
            return RouteDecision(
                "general",
                score,
                "The prompt looked document-centric, but no uploaded corpus is available, so it fell back to general Q&A.",
            )

        rationale = {
            "rag": "The prompt is semantically closest to document-grounded retrieval and answer generation.",
            "general": "The prompt is best handled as a general military-family-resource question.",
            "guardrail": "The prompt is semantically close to policy-evasion, confidentiality, or benefits-misuse requests.",
        }[route]
        return RouteDecision(route=route, score=score, rationale=rationale)
