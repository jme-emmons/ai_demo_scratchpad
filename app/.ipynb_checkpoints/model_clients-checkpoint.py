from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from app.config import settings


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

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if settings.model_api_key:
            headers["Authorization"] = f"Bearer {settings.model_api_key}"
        return headers

    def _post(self, url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers=self._headers(),
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
    ) -> tuple[dict[str, Any], float]:
        attempted_urls: list[str] = []
        last_error: Exception | None = None
        for url in self._candidate_urls(base_url, suffixes):
            attempted_urls.append(url)
            try:
                return self._post(url, payload)
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


class EmbeddingClient(EndpointClient):
    def embed(self, text: str) -> EmbeddingResult:
        fmt = settings.embedding_api_format
        if fmt == "openai_embeddings":
            payload = {"model": settings.embedding_model, "input": text}
            data, latency_ms = self._post_with_fallbacks(
                settings.embedding_endpoint,
                payload,
                ["/v1/embeddings", "/embeddings"],
            )
            try:
                vector = data["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ModelClientError(f"Unexpected embedding response: {data}") from exc
            return EmbeddingResult(vector=vector, latency_ms=latency_ms)
        if fmt == "tei":
            payload = {"inputs": text}
            data, latency_ms = self._post_with_fallbacks(
                settings.embedding_endpoint,
                payload,
                ["/embed", "/embeddings"],
            )
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
            payload = {"model": settings.llm_model, "messages": messages, "temperature": 0.2}
            data, latency_ms = self._post_with_fallbacks(
                settings.llm_endpoint,
                payload,
                ["/v1/chat/completions", "/chat/completions"],
            )
            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ModelClientError(f"Unexpected chat response: {data}") from exc
            usage = data.get("usage", {})
            return GenerationResult(
                text=text.strip(),
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                raw_response=data,
            )
        if fmt == "tgi":
            payload = {
                "inputs": prompt if not system_prompt else f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
                "parameters": {"temperature": 0.2, "max_new_tokens": 512},
            }
            data, latency_ms = self._post_with_fallbacks(
                settings.llm_endpoint,
                payload,
                ["/generate", "/v1/completions"],
            )
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "")
            else:
                text = data.get("generated_text", "")
            return GenerationResult(text=text.strip(), latency_ms=latency_ms, raw_response=data)
        raise ModelClientError(f"Unsupported LLM API format: {fmt}")
