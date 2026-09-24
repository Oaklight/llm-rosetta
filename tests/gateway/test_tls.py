"""Tests for native TLS support."""

from __future__ import annotations

import asyncio
import ssl
from pathlib import Path

import pytest

from llm_rosetta._vendor.httpserver import App, JSONResponse
from llm_rosetta.gateway.config import GatewayConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_raw(**server_overrides) -> dict:
    raw = {
        "providers": {
            "test": {
                "api_key": "sk-test",
                "base_url": "https://api.example.com",
                "type": "openai",
            }
        },
        "models": {"gpt-test": "test"},
        "server": {},
    }
    raw["server"].update(server_overrides)
    return raw


def _generate_self_signed_cert(tmp: Path) -> tuple[str, str]:
    """Generate a self-signed cert+key pair via the openssl CLI."""
    cert_path = str(tmp / "cert.pem")
    key_path = str(tmp / "key.pem")

    import subprocess

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            key_path,
            "-out",
            cert_path,
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_path, key_path


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------


class TestTlsConfig:
    def test_tls_cert_and_key_both_set(self):
        cfg = GatewayConfig(_minimal_raw(tls_cert="/tmp/c.pem", tls_key="/tmp/k.pem"))
        assert cfg.tls_cert == "/tmp/c.pem"
        assert cfg.tls_key == "/tmp/k.pem"

    def test_tls_neither_set(self):
        cfg = GatewayConfig(_minimal_raw())
        assert cfg.tls_cert is None
        assert cfg.tls_key is None

    def test_tls_cert_only_raises(self):
        with pytest.raises(ValueError, match="tls_cert and tls_key must both"):
            GatewayConfig(_minimal_raw(tls_cert="/tmp/c.pem"))

    def test_tls_key_only_raises(self):
        with pytest.raises(ValueError, match="tls_cert and tls_key must both"):
            GatewayConfig(_minimal_raw(tls_key="/tmp/k.pem"))


# ---------------------------------------------------------------------------
# Live TLS server test
# ---------------------------------------------------------------------------


class TestTlsServer:
    @pytest.fixture()
    def cert_pair(self, tmp_path):
        return _generate_self_signed_cert(tmp_path)

    def test_ssl_context_passed_to_serve(self, cert_pair):
        """Verify App._serve accepts ssl_context and serves HTTPS."""
        cert_path, key_path = cert_pair

        async def _run():
            app = App()

            @app.get("/ping")
            async def ping(request):
                return JSONResponse({"pong": True})

            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_path, key_path)

            serve_task = asyncio.create_task(
                app._serve("127.0.0.1", 0, ssl_context=ctx)
            )
            for _ in range(50):
                if app.port is not None:
                    break
                await asyncio.sleep(0.05)
            assert app.port is not None

            try:
                client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                client_ctx.check_hostname = False
                client_ctx.verify_mode = ssl.CERT_NONE

                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", app.port, ssl=client_ctx
                )
                writer.write(b"GET /ping HTTP/1.1\r\nHost: localhost\r\n\r\n")
                await writer.drain()

                data = await asyncio.wait_for(reader.read(4096), timeout=5)
                assert b"200 OK" in data
                assert b'"pong": true' in data

                writer.close()
                await writer.wait_closed()
            finally:
                app.shutdown()
                await asyncio.wait_for(serve_task, timeout=5)

        asyncio.run(_run())
