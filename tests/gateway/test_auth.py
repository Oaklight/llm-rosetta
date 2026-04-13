"""Gateway auth middleware unit tests."""

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from llm_rosetta.gateway.auth import GatewayAuthMiddleware


def _ok_handler(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def _make_app(api_key: str | None = None) -> Starlette:
    routes = [
        Route("/health", _ok_handler, methods=["GET"]),
        Route("/v1/chat/completions", _ok_handler, methods=["POST"]),
        Route("/v1/messages", _ok_handler, methods=["POST"]),
        Route("/v1/responses", _ok_handler, methods=["POST"]),
        Route("/v1/models", _ok_handler, methods=["GET"]),
        Route("/v1beta/models", _ok_handler, methods=["GET"]),
        Route("/v1beta/models/{model_path:path}", _ok_handler, methods=["POST"]),
        Route("/admin", _ok_handler, methods=["GET"]),
        Route("/admin/api/config", _ok_handler, methods=["GET"]),
    ]
    from starlette.middleware import Middleware

    app = Starlette(
        routes=routes,
        middleware=[Middleware(GatewayAuthMiddleware, api_key=api_key)],
    )
    return app


class TestNoApiKey:
    """When no api_key is configured, all requests pass through."""

    def test_all_requests_allowed(self):
        client = TestClient(_make_app(api_key=None))
        assert client.get("/health").status_code == 200
        assert client.post("/v1/chat/completions", json={}).status_code == 200
        assert client.post("/v1/messages", json={}).status_code == 200
        assert client.get("/admin/api/config").status_code == 200


class TestWithApiKey:
    """When api_key is configured, requests must provide valid credentials."""

    KEY = "test-gateway-key-123"

    @pytest.fixture()
    def client(self):
        return TestClient(_make_app(api_key=self.KEY))

    # --- Health is always public ---
    def test_health_no_auth(self, client: TestClient):
        assert client.get("/health").status_code == 200

    # --- OpenAI Chat ---
    def test_openai_chat_valid(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": f"Bearer {self.KEY}"},
        )
        assert resp.status_code == 200

    def test_openai_chat_missing(self, client: TestClient):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 401
        assert "invalid_api_key" in resp.json()["error"]["code"]

    def test_openai_chat_wrong(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    # --- OpenAI Responses ---
    def test_openai_responses_valid(self, client: TestClient):
        resp = client.post(
            "/v1/responses",
            json={},
            headers={"Authorization": f"Bearer {self.KEY}"},
        )
        assert resp.status_code == 200

    # --- Anthropic ---
    def test_anthropic_valid(self, client: TestClient):
        resp = client.post(
            "/v1/messages",
            json={},
            headers={"x-api-key": self.KEY},
        )
        assert resp.status_code == 200

    def test_anthropic_missing(self, client: TestClient):
        resp = client.post("/v1/messages", json={})
        assert resp.status_code == 401
        assert resp.json()["type"] == "error"
        assert resp.json()["error"]["type"] == "authentication_error"

    def test_anthropic_wrong(self, client: TestClient):
        resp = client.post(
            "/v1/messages", json={}, headers={"x-api-key": "wrong"}
        )
        assert resp.status_code == 401

    # --- Google GenAI (header) ---
    def test_google_header_valid(self, client: TestClient):
        resp = client.post(
            "/v1beta/models/gemini:generateContent",
            json={},
            headers={"x-goog-api-key": self.KEY},
        )
        assert resp.status_code == 200

    def test_google_query_valid(self, client: TestClient):
        resp = client.post(
            f"/v1beta/models/gemini:generateContent?key={self.KEY}",
            json={},
        )
        assert resp.status_code == 200

    def test_google_missing(self, client: TestClient):
        resp = client.post("/v1beta/models/gemini:generateContent", json={})
        assert resp.status_code == 401
        assert resp.json()["error"]["status"] == "UNAUTHENTICATED"

    # --- Models list ---
    def test_models_list_valid(self, client: TestClient):
        resp = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {self.KEY}"},
        )
        assert resp.status_code == 200

    def test_google_models_list_valid(self, client: TestClient):
        resp = client.get(
            "/v1beta/models",
            headers={"x-goog-api-key": self.KEY},
        )
        assert resp.status_code == 200

    # --- Admin ---
    def test_admin_html_no_auth(self, client: TestClient):
        """Admin HTML page loads without auth (SPA needs to load first)."""
        resp = client.get("/admin")
        assert resp.status_code == 200

    def test_admin_api_valid(self, client: TestClient):
        resp = client.get(
            "/admin/api/config",
            headers={"Authorization": f"Bearer {self.KEY}"},
        )
        assert resp.status_code == 200

    def test_admin_api_missing(self, client: TestClient):
        resp = client.get("/admin/api/config")
        assert resp.status_code == 401
