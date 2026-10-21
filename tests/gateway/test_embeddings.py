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
from llm_rosetta.gateway.pipelines.embeddings import handle_embeddings
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
        "type": "embedding",
    }
    if upstream_model:
        model_entry["upstream_model"] = upstream_model

    raw = {
        "providers": {
            "test-provider": {
                "api_key": "test-key",
                "base_url": base_url,
                "type": "openai",
                "embedding_format": "openai",
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

    def test_same_format_request_goes_through_the_ir(self) -> None:
        """OpenAI → OpenAI is still converted: the input string becomes a list."""
        config = _make_config("https://api.example.com")
        request = _make_request({"model": "my-embed", "input": "hello"})
        request.app.transport = MagicMock()
        request.app.transport.send = AsyncMock(
            return_value=UpstreamResponse(
                status_code=200,
                body={"object": "list", "data": [], "model": "my-embed"},
                raw_content=b'{"object":"list","data":[],"model":"my-embed"}',
            )
        )

        asyncio.run(handle_embeddings(request, config))

        args, _ = request.app.transport.send.call_args
        assert args[2]["input"] == ["hello"]

    def test_same_format_fidelity_check_runs_both_ways(self, monkeypatch) -> None:
        config = _make_config("https://api.example.com")
        request = _make_request({"model": "my-embed", "input": "hello"})
        request.app.transport = MagicMock()
        request.app.transport.send = AsyncMock(
            return_value=UpstreamResponse(
                status_code=200,
                body={"object": "list", "data": [], "model": "my-embed"},
                raw_content=b'{"object":"list","data":[],"model":"my-embed"}',
            )
        )
        calls = []
        # The fidelity check runs in the shared typed-handler flow.
        monkeypatch.setattr(
            "llm_rosetta.gateway.pipelines.typed.check_same_format_fidelity",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )

        asyncio.run(handle_embeddings(request, config))

        assert [args[5] for args, _ in calls] == ["request", "response"]

    def test_validation_error_includes_generated_request_id(self):
        config = _make_config("https://api.example.com/v1")
        request = _make_request({"input": "hello"})

        response = asyncio.run(handle_embeddings(request, config))

        assert response.status_code == 400
        assert response.headers["x-request-id"]

    @pytest.mark.parametrize("client_request_id", ["req-123", None])
    def test_headers_forwarded_to_passthrough_transport(
        self, client_request_id: str | None
    ):
        """Embeddings passthrough should preserve client headers and assign an ID."""
        config = _make_config("https://api.example.com/v1")
        headers = {"user-agent": "codex-cli/1.2.3"}
        if client_request_id:
            headers["x-request-id"] = client_request_id
        request = _make_request(
            {"model": "my-embed", "input": "hello"},
            headers=headers,
        )
        request.app.transport = MagicMock()
        request.app.transport.send = AsyncMock(
            return_value=UpstreamResponse(
                status_code=200,
                body={"object": "list", "data": [], "model": "my-embed"},
                raw_content=b'{"object":"list","data":[],"model":"my-embed"}',
            )
        )

        response = asyncio.run(handle_embeddings(request, config))

        assert response.status_code == 200
        _, kwargs = request.app.transport.send.call_args
        upstream_headers = kwargs["extra_headers"]
        assert upstream_headers["User-Agent"] == "codex-cli/1.2.3"
        assert upstream_headers["x-request-id"]
        assert response.headers["x-request-id"] == upstream_headers["x-request-id"]
        if client_request_id:
            assert upstream_headers["x-request-id"] == client_request_id


# ============================================================================
# Google GenAI routing tests
# ============================================================================


def _make_google_config(
    base_url: str = "https://generativelanguage.googleapis.com",
    *,
    upstream_model: str | None = None,
) -> GatewayConfig:
    """Build a config whose embedding endpoint is the Google shim."""
    model_entry: dict[str, Any] = {
        "provider": "google-embed",
        "type": "embedding",
    }
    if upstream_model:
        model_entry["upstream_model"] = upstream_model
    return GatewayConfig(
        {
            "providers": {
                "google-embed": {
                    "api_key": "test-key",
                    "base_url": base_url,
                    "embedding_shim": "google--embedding",
                }
            },
            "models": {"gemini-embedding": model_entry},
        }
    )


def _google_transport(response_body: dict[str, Any]):
    transport = MagicMock()
    transport.send = AsyncMock(
        return_value=UpstreamResponse(
            status_code=200,
            body=response_body,
            raw_content=json.dumps(response_body).encode(),
        )
    )
    return transport


class TestGoogleEmbeddingRouting:
    """The Google shim carries the model in the path and has a batch suffix."""

    def test_single_input_uses_embed_content(self) -> None:
        config = _make_google_config()
        request = _make_request({"model": "gemini-embedding", "input": "hello"})
        request.app.transport = _google_transport(
            {
                "embedding": {"values": [0.1, 0.2]},
                "usageMetadata": {"promptTokenCount": 2, "totalTokenCount": 2},
            }
        )

        response = asyncio.run(handle_embeddings(request, config))

        assert response.status_code == 200
        args, _ = request.app.transport.send.call_args
        assert args[1] == (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/gemini-embedding:embedContent"
        )
        assert args[2]["content"] == {"parts": [{"text": "hello"}]}
        body = json.loads(response.body)
        assert body["data"][0]["embedding"] == [0.1, 0.2]
        assert body["model"] == "gemini-embedding"

    def test_batch_input_uses_batch_embed_contents(self) -> None:
        config = _make_google_config()
        request = _make_request({"model": "gemini-embedding", "input": ["a", "b"]})
        request.app.transport = _google_transport(
            {"embeddings": [{"values": [0.1]}, {"values": [0.2]}]}
        )

        response = asyncio.run(handle_embeddings(request, config))

        assert response.status_code == 200
        args, _ = request.app.transport.send.call_args
        assert args[1] == (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/gemini-embedding:batchEmbedContents"
        )
        assert [item["model"] for item in args[2]["requests"]] == [
            "models/gemini-embedding",
            "models/gemini-embedding",
        ]
        body = json.loads(response.body)
        assert len(body["data"]) == 2

    def test_upstream_model_is_used_in_the_path(self) -> None:
        config = _make_google_config(upstream_model="text-embedding-004")
        request = _make_request({"model": "gemini-embedding", "input": "hello"})
        request.app.transport = _google_transport({"embedding": {"values": [0.1]}})

        asyncio.run(handle_embeddings(request, config))

        args, _ = request.app.transport.send.call_args
        assert args[1].endswith("/v1beta/models/text-embedding-004:embedContent")

    def test_format_without_a_shim_uses_the_format_default_path(self) -> None:
        """`embedding_format: google_generate` alone gets Google's path shape."""
        config = GatewayConfig(
            {
                "providers": {
                    "google-embed": {
                        "api_key": "test-key",
                        "base_url": "https://generativelanguage.googleapis.com",
                        "embedding_format": "google_generate",
                    }
                },
                "models": {
                    "gemini-embedding": {
                        "provider": "google-embed",
                        "type": "embedding",
                    }
                },
            }
        )
        request = _make_request({"model": "gemini-embedding", "input": "hello"})
        request.app.transport = _google_transport({"embedding": {"values": [0.1]}})

        asyncio.run(handle_embeddings(request, config))

        args, _ = request.app.transport.send.call_args
        assert args[1] == (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/gemini-embedding:embedContent"
        )


# ============================================================================
# Auto-detection tests
# ============================================================================


class TestDetectEmbeddingSource:
    """Tests for _detect_embedding_source auto-detection logic."""

    @pytest.fixture()
    def config(self) -> GatewayConfig:
        return GatewayConfig(
            {
                "providers": {
                    "openai_chat": {
                        "api_key": "sk-test",
                        "base_url": "https://api.openai.com/v1",
                    }
                },
                "models": {"gpt-4o": "openai_chat"},
            }
        )

    @pytest.mark.parametrize(
        "body, expected",
        [
            ({"texts": ["hello"]}, "cohere"),
            ({"input": "hello", "task": "retrieval.query"}, "jina"),
            ({"input": "hello", "output_dtype": "float"}, "voyage"),
            # input_type is the signal an ordinary Voyage request carries.
            ({"input": "hello", "input_type": "query"}, "voyage"),
            # Cohere also sends input_type, but with texts.
            ({"texts": ["hello"], "input_type": "search_query"}, "cohere"),
            # Google shapes: parts for embedContent, requests for batch.
            ({"content": {"parts": [{"text": "hello"}]}}, "google_generate"),
            (
                {"requests": [{"content": {"parts": [{"text": "hello"}]}}]},
                "google_generate",
            ),
            # An OpenAI-shaped body stays OpenAI even if it carries content.
            (
                {"input": "hello", "content": {"parts": [{"text": "hello"}]}},
                "openai",
            ),
            ({"input": "hello"}, "openai"),  # fallback
            ({"input": "hello", "model": "x"}, "openai"),  # fallback
            # edge: texts + input → not Cohere, falls to default
            ({"texts": ["hello"], "input": "hello"}, "openai"),
        ],
    )
    def test_parametrized(
        self, body: dict, expected: str, config: GatewayConfig
    ) -> None:
        from llm_rosetta.gateway.pipelines.embeddings import _detect_embedding_source

        assert _detect_embedding_source(None, body, config) == expected

    def test_cohere_texts_field(self, config: GatewayConfig) -> None:
        from llm_rosetta.gateway.pipelines.embeddings import _detect_embedding_source

        assert _detect_embedding_source(None, {"texts": ["hi"]}, config) == "cohere"

    def test_cohere_skipped_when_input_present(self, config: GatewayConfig) -> None:
        from llm_rosetta.gateway.pipelines.embeddings import _detect_embedding_source

        assert (
            _detect_embedding_source(None, {"texts": ["hi"], "input": "hi"}, config)
            != "cohere"
        )

    def test_jina_task_field(self, config: GatewayConfig) -> None:
        from llm_rosetta.gateway.pipelines.embeddings import _detect_embedding_source

        assert (
            _detect_embedding_source(
                None, {"input": "hi", "task": "retrieval.query"}, config
            )
            == "jina"
        )

    def test_voyage_output_dtype(self, config: GatewayConfig) -> None:
        from llm_rosetta.gateway.pipelines.embeddings import _detect_embedding_source

        assert (
            _detect_embedding_source(
                None, {"input": "hi", "output_dtype": "float"}, config
            )
            == "voyage"
        )

    def test_fallback_to_default(self, config: GatewayConfig) -> None:
        from llm_rosetta.gateway.pipelines.embeddings import _detect_embedding_source

        assert (
            _detect_embedding_source(None, {"input": "hi"}, config)
            == config.default_embedding_format
        )
