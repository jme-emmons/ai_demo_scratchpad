from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from openai import APIError, OpenAI
from openai import OpenAIError

from app.config import settings

OPENAI_EMBEDDING_ENDPOINT_SUFFIXES = [
    "/v1/embeddings",
    "/v1/openai/v1/embeddings",
    "/embeddings",
]

OPENAI_CHAT_ENDPOINT_SUFFIXES = [
    "/v1/chat/completions",
    "/v1/openai/v1/chat/completions",
    "/chat/completions",
]

TEI_EMBEDDING_SUFFIXES = [
    "/embed",
    "/embeddings",
]


class ModelClientError(RuntimeError):
    """Raised when a model endpoint cannot be called or parsed."""


@dataclass
class EmbeddingResult:
    vector: list[float]
    latency_ms: float


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_response: dict[str, Any] | None = None


class EndpointClient:
    def __init__(self) -> None:
        self.timeout = settings.request_timeout_seconds

    @staticmethod
    def _openai_api_key() -> str:
        return settings.model_api_key or "not-needed-for-internal-service"

    def _headers(self, for_openai: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if for_openai or settings.model_api_key:
            headers["Authorization"] = f"Bearer {self._openai_api_key()}"
        return headers

    def _post(self, url: str, payload: dict[str, Any], for_openai: bool = False) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers=self._headers(for_openai=for_openai),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ModelClientError(f"Request to {url} failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            return response.json(), latency_ms
        except json.JSONDecodeError as exc:
            raise ModelClientError(f"Non-JSON response from {url}: {response.text[:300]}") from exc

    def _post_with_fallbacks(
        self,
        base_url: str,
        payload: dict[str, Any],
        suffixes: list[str],
        for_openai: bool = False,
    ) -> tuple[dict[str, Any], float]:
        attempted_urls: list[str] = []
        last_error: Exception | None = None
        for url in self._candidate_urls(base_url, suffixes):
            attempted_urls.append(url)
            try:
                return self._post(url, payload, for_openai=for_openai)
            except ModelClientError as exc:
                last_error = exc
                if "404" not in str(exc):
                    break
        attempted = ", ".join(attempted_urls)
        raise ModelClientError(
            f"Unable to call endpoint. Tried: {attempted}. Last error: {last_error}"
        ) from last_error

    @staticmethod
    def _candidate_urls(base_url: str, suffixes: list[str]) -> list[str]:
        stripped = base_url.rstrip("/")
        candidates = [stripped]
        parsed = urlsplit(stripped)
        current_path = parsed.path.rstrip("/")
        for suffix in suffixes:
            clean_suffix = suffix if suffix.startswith("/") else f"/{suffix}"
            if current_path.endswith(clean_suffix):
                continue
            merged = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    f"{current_path}{clean_suffix}",
                    parsed.query,
                    parsed.fragment,
                )
            )
            candidates.append(merged)
        return candidates

    @staticmethod
    def _strip_known_openai_request_suffix(endpoint: str, suffixes: list[str]) -> str:
        stripped = endpoint.rstrip("/")
        parsed = urlsplit(stripped)
        path = parsed.path.rstrip("/")
        for suffix in suffixes:
            clean_suffix = suffix if suffix.startswith("/") else f"/{suffix}"
            if path.endswith(clean_suffix):
                path = path[: -len(clean_suffix)]
                break
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)).rstrip("/")

    @staticmethod
    def _openai_base_url_candidates(endpoint: str, suffixes: list[str]) -> list[str]:
        normalized = EndpointClient._strip_known_openai_request_suffix(endpoint, suffixes)
        parsed = urlsplit(normalized)
        path = parsed.path.rstrip("/")

        preferred_paths: list[str] = []
        if path.endswith("/v1/openai/v1"):
            preferred_paths.extend([path, path[: -len("/openai/v1")] or "/v1"])
        elif path.endswith("/v1"):
            preferred_paths.extend([path, f"{path}/openai/v1"])
        elif path:
            preferred_paths.extend([f"{path}/v1", f"{path}/v1/openai/v1", path])
        else:
            preferred_paths.extend(["/v1", "/v1/openai/v1", ""])

        candidates: list[str] = []
        seen: set[str] = set()
        for candidate_path in preferred_paths:
            base_url = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    candidate_path,
                    parsed.query,
                    parsed.fragment,
                )
            ).rstrip("/")
            if base_url and base_url not in seen:
                seen.add(base_url)
                candidates.append(base_url)
        return candidates


class EmbeddingClient(EndpointClient):
    def embed(self, text: str) -> EmbeddingResult:
        fmt = settings.embedding_api_format
        if fmt == "openai_embeddings":
            last_error: Exception | None = None
            attempted_bases: list[str] = []
            api_key = self._openai_api_key()
            for base_url in self._openai_base_url_candidates(
                settings.embedding_endpoint,
                OPENAI_EMBEDDING_ENDPOINT_SUFFIXES,
            ):
                attempted_bases.append(base_url)
                client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)
                started = time.perf_counter()
                try:
                    response = client.embeddings.create(model=settings.embedding_model, input=text)
                    latency_ms = (time.perf_counter() - started) * 1000
                    vector = response.data[0].embedding
                    return EmbeddingResult(vector=vector, latency_ms=latency_ms)
                except (OpenAIError, APIError) as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", None)
                    if status_code != 404:
                        break
            raise ModelClientError(
                "Embedding endpoint could not be reached with the OpenAI client. "
                f"Tried base URLs: {', '.join(attempted_bases)}. "
                f"Requested model: {settings.embedding_model}. "
                "The app prefers a /v1 base URL for OpenAI-compatible embeddings and also checks "
                "/v1/openai/v1 for OpenShift AI Llama Stack-style routes. "
                "If this service is TEI-backed instead, set EMBEDDING_API_FORMAT=tei. "
                f"Last error: {last_error}"
            ) from last_error
        if fmt == "tei":
            payload = {"inputs": text}
            try:
                data, latency_ms = self._post_with_fallbacks(
                    settings.embedding_endpoint,
                    payload,
                    TEI_EMBEDDING_SUFFIXES,
                    for_openai=False,
                )
            except ModelClientError as exc:
                raise ModelClientError(
                    "Embedding endpoint could not be reached with the configured TEI paths. "
                    f"Checked base URL plus: {', '.join(TEI_EMBEDDING_SUFFIXES)}. "
                    "If this model is exposed through an OpenAI-compatible OpenShift AI route instead, "
                    "set EMBEDDING_API_FORMAT=openai_embeddings."
                ) from exc
            if not isinstance(data, list):
                raise ModelClientError(f"Unexpected TEI embedding response: {data}")
            return EmbeddingResult(vector=data[0] if data and isinstance(data[0], list) else data, latency_ms=latency_ms)
        raise ModelClientError(f"Unsupported embedding API format: {fmt}")


class LLMClient(EndpointClient):
    def generate(self, prompt: str, system_prompt: str | None = None) -> GenerationResult:
        fmt = settings.llm_api_format
        if fmt == "openai_chat":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            last_error: Exception | None = None
            attempted_bases: list[str] = []
            api_key = self._openai_api_key()
            for base_url in self._openai_base_url_candidates(
                settings.llm_endpoint,
                OPENAI_CHAT_ENDPOINT_SUFFIXES,
            ):
                attempted_bases.append(base_url)
                client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)
                started = time.perf_counter()
                try:
                    response = client.chat.completions.create(
                        model=settings.llm_model,
                        messages=messages,
                        temperature=0.2,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000
                    text = response.choices[0].message.content or ""
                    usage = response.usage
                    raw_response = response.model_dump()
                    return GenerationResult(
                        text=text.strip(),
                        latency_ms=latency_ms,
                        prompt_tokens=usage.prompt_tokens if usage else None,
                        completion_tokens=usage.completion_tokens if usage else None,
                        total_tokens=usage.total_tokens if usage else None,
                        raw_response=raw_response,
                    )
                except (OpenAIError, APIError) as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", None)
                    if status_code != 404:
                        break
            raise ModelClientError(
                "LLM endpoint could not be reached with the OpenAI client. "
                f"Tried base URLs: {', '.join(attempted_bases)}. "
                "The app prefers a /v1 base URL for OpenAI-compatible chat completions and also checks "
                "/v1/openai/v1 for OpenShift AI Llama Stack-style routes. "
                f"Last error: {last_error}"
            ) from last_error
        if fmt == "tgi":
            payload = {
                "inputs": prompt if not system_prompt else f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
                "parameters": {"temperature": 0.2, "max_new_tokens": 512},
            }
            data, latency_ms = self._post_with_fallbacks(
                settings.llm_endpoint,
                payload,
                ["/generate", "/v1/completions"],
                for_openai=False,
            )
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "")
            else:
                text = data.get("generated_text", "")
            return GenerationResult(text=text.strip(), latency_ms=latency_ms, raw_response=data)
        raise ModelClientError(f"Unsupported LLM API format: {fmt}")
