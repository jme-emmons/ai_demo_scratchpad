from __future__ import annotations

import ssl
import tempfile
from pathlib import Path

import redis
from redis.connection import SSLConnection

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


class SNIOverrideSSLConnection(SSLConnection):
    def _wrap_socket_with_ssl(self, sock):
        context = ssl.create_default_context()
        context.check_hostname = settings.redis_ssl_check_hostname and self.check_hostname
        context.verify_mode = self.cert_reqs

        if self.certfile or self.keyfile:
            context.load_cert_chain(
                certfile=self.certfile,
                keyfile=self.keyfile,
                password=self.certificate_password,
            )
        if self.ca_certs is not None or self.ca_path is not None or self.ca_data is not None:
            context.load_verify_locations(
                cafile=self.ca_certs,
                capath=self.ca_path,
                cadata=self.ca_data,
            )
        if self.ssl_min_version is not None:
            context.minimum_version = self.ssl_min_version
        if self.ssl_ciphers:
            context.set_ciphers(self.ssl_ciphers)

        server_hostname = settings.redis_sni_hostname or self.host
        return context.wrap_socket(sock, server_hostname=server_hostname)


def get_redis_client() -> redis.Redis:
    kwargs = {
        "host": settings.redis_host,
        "port": settings.redis_port,
        "password": settings.redis_password,
        "decode_responses": False,
    }
    if settings.redis_ssl:
        ca_cert_path = _redis_ca_cert_path()
        kwargs["ssl_cert_reqs"] = "required"
        if settings.redis_ssl_verify:
            kwargs["ssl_cert_reqs"] = "none"
        if ca_cert_path:
            kwargs["ssl_ca_certs"] = ca_cert_path
        if settings.redis_ssl_check_hostname and (settings.redis_sni_hostname or ca_cert_path):
            kwargs["ssl_check_hostname"] = True
        else:
            kwargs["ssl_check_hostname"] = False
        pool = redis.ConnectionPool(connection_class=SNIOverrideSSLConnection, **kwargs)
        return redis.Redis(connection_pool=pool)

    kwargs["ssl"] = False
    return redis.Redis(**kwargs)
