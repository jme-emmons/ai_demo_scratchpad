from __future__ import annotations

import tempfile
from pathlib import Path

import redis

from app.config import settings


def _redis_ca_cert_path() -> str | None:
    if settings.redis_ca_cert_path:
        return settings.redis_ca_cert_path
    if not settings.redis_ca_cert_text:
        return None

    cert_text = settings.redis_ca_cert_text.strip()
    if not cert_text:
        return None

    temp_dir = Path(tempfile.gettempdir()) / "ai_demo_scratchpad"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cert_file = temp_dir / "redis_ca_cert.pem"
    cert_file.write_text(f"{cert_text}\n", encoding="utf-8")
    return str(cert_file)


def get_redis_client() -> redis.Redis:
    kwargs = {
        "host": settings.redis_host,
        "port": settings.redis_port,
        "password": settings.redis_password,
        "ssl": settings.redis_ssl,
        "decode_responses": False,
    }
    if settings.redis_ssl:
        ca_cert_path = _redis_ca_cert_path()
        kwargs["ssl_cert_reqs"] = "required"
        if ca_cert_path:
            kwargs["ssl_ca_certs"] = ca_cert_path
        if settings.redis_sni_hostname:
            # redis-py performs hostname verification against the TCP host.
            # Keep REDIS_SNI_HOSTNAME available in config/docs, but prefer REDIS_HOST
            # to match the certificate name for standard deployments.
            kwargs["ssl_check_hostname"] = True
        elif ca_cert_path:
            kwargs["ssl_check_hostname"] = True
        else:
            kwargs["ssl_check_hostname"] = False
            kwargs["ssl_cert_reqs"] = None

    return redis.Redis(
        **kwargs,
    )
