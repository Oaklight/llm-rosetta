"""Tests for the embeddings passthrough handler."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.embeddings import handle_embeddings
from llm_rosetta.gateway.transport._base import UpstreamResponse
from llm_rosetta.gateway.transport.http import HttpTransport


# ---------------------------------------------------------------------------
# Fake upstream server that echoes back the received model name
# ---------------------------------------------------------------------------


class _EchoEmbeddingHandler(BaseHTTPRequestHandler):
    """Returns an embedding response that echoes the request model name."""

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        model = body.get("model", "")
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                }
            ],
            "model": model,
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def echo_embedding_server():
    """Start a local server that echoes the model name in embedding responses."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoEmbeddingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(base_url: str, upstream_model: str | None = None) -> GatewayConfig:
    """Build a minimal GatewayConfig for embedding tests."""
    model_entry: dict[str, Any] = {
        "provider": "test-provider",
        "capabilities": ["embedding"],
    }
    if upstream_model:
        model_entry["upstream_model"] = upstream_model

    raw = {
        "providers": {
            "test-provider": {
                "api_key": "test-key",
                "base_url": base_url,
                "type": "openai",
            }
        },
        "models": {
            "my-embed": model_entry,
        },
    }
    return GatewayConfig(raw)


def _make_request(
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock HTTP request with the given JSON body."""
    req = MagicMock()
    req.json.return_value = body
    req.headers = headers or {}

    # Provide app-level attributes that handle_embeddings accesses
    app = MagicMock()
    app.metrics = None
    app.request_log = None
    app.transport = HttpTransport()
    req.app = app
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingUpstreamModel:
    """Tests for upstream_model substitution in embedding requests."""

    @pytest.fixture(autouse=True)
    def _clear_transport(self):
        """Ensure fresh transport state between tests."""
        yield

    def test_upstream_model_substituted(self, echo_embedding_server: str):
        """When upstream_model is configured, the body sent to upstream should
        contain the upstream model name, not the gateway alias."""
        config = _make_config(echo_embedding_server, upstream_model="BAAI/bge-m3")
        request = _make_request({"model": "my-embed", "input": "hello"})

        response = asyncio.run(handle_embeddings(request, config))

        body = json.loads(response.body)
        # The echo server returns the model it received — should be the upstream name
        assert body["model"] == "BAAI/bge-m3"

    def test_no_upstream_model(self, echo_embedding_server: str):
        """When no upstream_model is configured, the original model name is used."""
        config = _make_config(echo_embedding_server, upstream_model=None)
        request = _make_request({"model": "my-embed", "input": "hello"})

        response = asyncio.run(handle_embeddings(request, config))

        body = json.loads(response.body)
        assert body["model"] == "my-embed"

    def test_user_agent_forwarded_to_passthrough_transport(self):
        """Embeddings passthrough should preserve the client's User-Agent."""
        config = _make_config("https://api.example.com/v1")
        request = _make_request(
            {"model": "my-embed", "input": "hello"},
            headers={"user-agent": "codex-cli/1.2.3"},
        )
        request.app.transport = MagicMock()
        request.app.transport.send_passthrough = AsyncMock(
            return_value=UpstreamResponse(
                status_code=200,
                body={"object": "list", "data": [], "model": "my-embed"},
                raw_content=b'{"object":"list","data":[],"model":"my-embed"}',
            )
        )

        response = asyncio.run(handle_embeddings(request, config))

        assert response.status_code == 200
        _, kwargs = request.app.transport.send_passthrough.call_args
        assert kwargs["extra_headers"]["User-Agent"] == "codex-cli/1.2.3"
