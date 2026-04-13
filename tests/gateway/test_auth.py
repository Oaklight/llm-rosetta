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


def _label_handler(request: Request) -> JSONResponse:
    """Handler that returns the api_key_label from request state."""
    label = getattr(request.state, "api_key_label", None)
    return JSONResponse({"ok": True, "api_key_label": label})


def _make_app(
    api_key: str | None = None,
    api_key_set: set[str] | None = None,
    api_key_labels: dict[str, str] | None = None,
    internal_token: str | None = None,
    use_label_handler: bool = False,
) -> Starlette:
    handler = _label_handler if use_label_handler else _ok_handler
    routes = [
        Route("/health", _ok_handler, methods=["GET"]),
        Route("/v1/chat/completions", handler, methods=["POST"]),
        Route("/v1/messages", handler, methods=["POST"]),
        Route("/v1/responses", handler, methods=["POST"]),
        Route("/v1/models", handler, methods=["GET"]),
        Route("/v1beta/models", handler, methods=["GET"]),
        Route("/v1beta/models/{model_path:path}", handler, methods=["POST"]),
        Route("/admin", _ok_handler, methods=["GET"]),
        Route("/admin/api/config", _ok_handler, methods=["GET"]),
    ]
    from starlette.middleware import Middleware

    mw_kwargs: dict = {}
    if api_key_set is not None:
        mw_kwargs["api_key_set"] = api_key_set
        if api_key_labels:
            mw_kwargs["api_key_labels"] = api_key_labels
    elif api_key is not None:
        mw_kwargs["api_key"] = api_key
    if internal_token is not None:
        mw_kwargs["internal_token"] = internal_token

    app = Starlette(
        routes=routes,
        middleware=[Middleware(GatewayAuthMiddleware, **mw_kwargs)],  # type: ignore[arg-type]
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
        resp = client.post("/v1/messages", json={}, headers={"x-api-key": "wrong"})
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

    # --- Admin (no gateway-level auth — delegated to reverse proxy) ---
    def test_admin_html_no_auth(self, client: TestClient):
        resp = client.get("/admin")
        assert resp.status_code == 200

    def test_admin_api_no_auth(self, client: TestClient):
        """Admin API endpoints pass through without gateway auth."""
        resp = client.get("/admin/api/config")
        assert resp.status_code == 200


class TestMultiKey:
    """When multiple API keys are configured via api_key_set."""

    KEYS = {"key-alpha", "key-beta", "key-gamma"}

    @pytest.fixture()
    def client(self):
        return TestClient(_make_app(api_key_set=self.KEYS))

    def test_first_key_valid(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer key-alpha"},
        )
        assert resp.status_code == 200

    def test_second_key_valid(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer key-beta"},
        )
        assert resp.status_code == 200

    def test_third_key_valid(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer key-gamma"},
        )
        assert resp.status_code == 200

    def test_invalid_key_rejected(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_missing_key_rejected(self, client: TestClient):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 401

    def test_anthropic_multi_key(self, client: TestClient):
        resp = client.post(
            "/v1/messages",
            json={},
            headers={"x-api-key": "key-beta"},
        )
        assert resp.status_code == 200

    def test_google_multi_key(self, client: TestClient):
        resp = client.post(
            "/v1beta/models/gemini:generateContent",
            json={},
            headers={"x-goog-api-key": "key-gamma"},
        )
        assert resp.status_code == 200


class TestInternalToken:
    """Internal token bypasses API key auth for admin panel test requests."""

    KEY = "real-api-key"
    INTERNAL = "rsk-internal-abc123"

    @pytest.fixture()
    def client(self):
        return TestClient(
            _make_app(
                api_key_set={self.KEY},
                internal_token=self.INTERNAL,
            )
        )

    def test_internal_token_accepted(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": f"Bearer {self.INTERNAL}"},
        )
        assert resp.status_code == 200

    def test_real_key_still_works(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": f"Bearer {self.KEY}"},
        )
        assert resp.status_code == 200

    def test_wrong_key_still_rejected(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401


class TestKeyLabelTracking:
    """API key label is attached to request.state for logging."""

    KEYS = {"key-prod", "key-dev"}
    LABELS = {"key-prod": "Production", "key-dev": "Development"}
    INTERNAL = "rsk-internal-test"

    @pytest.fixture()
    def client(self):
        return TestClient(
            _make_app(
                api_key_set=self.KEYS,
                api_key_labels=self.LABELS,
                internal_token=self.INTERNAL,
                use_label_handler=True,
            )
        )

    def test_label_attached_for_prod_key(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer key-prod"},
        )
        assert resp.status_code == 200
        assert resp.json()["api_key_label"] == "Production"

    def test_label_attached_for_dev_key(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer key-dev"},
        )
        assert resp.status_code == 200
        assert resp.json()["api_key_label"] == "Development"

    def test_internal_token_label(self, client: TestClient):
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": f"Bearer {self.INTERNAL}"},
        )
        assert resp.status_code == 200
        assert resp.json()["api_key_label"] == "internal"

    def test_anthropic_label(self, client: TestClient):
        resp = client.post(
            "/v1/messages",
            json={},
            headers={"x-api-key": "key-prod"},
        )
        assert resp.status_code == 200
        assert resp.json()["api_key_label"] == "Production"
