from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: str | None = os.getenv("REDIS_PASSWORD")
    redis_ssl: bool = _as_bool(os.getenv("REDIS_SSL"), False)
    redis_ssl_verify: bool = _as_bool(os.getenv("REDIS_SSL_VERIFY"), True)
    redis_ssl_check_hostname: bool = _as_bool(os.getenv("REDIS_SSL_CHECK_HOSTNAME"), True)
    redis_ca_cert_path: str | None = os.getenv("REDIS_CA_CERT_PATH")
    redis_ca_cert_text: str | None = os.getenv("REDIS_CA_CERT_TEXT")
    redis_sni_hostname: str | None = os.getenv("REDIS_SNI_HOSTNAME")

    embedding_endpoint: str = os.getenv("EMBEDDING_ENDPOINT", "")
    embedding_api_format: str = os.getenv("EMBEDDING_API_FORMAT", "openai_embeddings")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")

    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "")
    llm_api_format: str = os.getenv("LLM_API_FORMAT", "openai_chat")
    llm_model: str = os.getenv("LLM_MODEL", "")
    model_api_key: str | None = os.getenv("MODEL_API_KEY")
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

    vector_index_name: str = os.getenv("VECTOR_INDEX_NAME", "demo:docs:index")
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "768"))
    vector_distance_metric: str = os.getenv("VECTOR_DISTANCE_METRIC", "COSINE")

    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8501"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "4"))
    semantic_cache_distance_threshold: float = float(
        os.getenv("SEMANTIC_CACHE_DISTANCE_THRESHOLD", "0.12")
    )
    semantic_route_distance_threshold: float = float(
        os.getenv("SEMANTIC_ROUTE_DISTANCE_THRESHOLD", "0.16")
    )
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    token_price_per_1k: float = float(os.getenv("TOKEN_PRICE_PER_1K", "0.002"))


settings = Settings()
