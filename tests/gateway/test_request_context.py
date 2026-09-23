"""Tests for the gateway request context middleware."""

from __future__ import annotations

import asyncio
from typing import Any

from llm_rosetta.gateway.request_context import (
    RequestContext,
    extract_client_ip,
    request_context_var,
    setup_request_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


class FakeRequest:
    """Minimal request mock matching httpserver.Request interface."""

    def __init__(
        self,
        path: str = "/v1/chat/completions",
        method: str = "POST",
        headers: dict[str, str] | None = None,
        client_addr: tuple[str, int] = ("127.0.0.1", 12345),
        query_params: dict[str, list[str]] | None = None,
    ):
        self.path = path
        self.method = method
        self.headers = headers or {}
        self.client_addr = client_addr
        self.query_params = query_params or {}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _run_hook_and_get_context(
    hook: Any, req: Any
) -> tuple[Any, RequestContext | None]:
    """Run the hook and return (result, context) within the same async context.

    ``asyncio.run()`` copies the calling context into the task, so
    contextvars set inside it are *not* visible to the caller after it
    returns.  We therefore read the contextvar inside the coroutine.
    """
    result = await hook(req)
    ctx = request_context_var.get()
    return result, ctx


# ---------------------------------------------------------------------------
# RequestContext dataclass
# ---------------------------------------------------------------------------


class TestRequestContext:
    def test_frozen(self):
        ctx = RequestContext(
            request_id="abc", client_ip="1.2.3.4", api_format="openai", is_admin=False
        )
        try:
            ctx.request_id = "xyz"  # type: ignore[misc]  # ty: ignore[invalid-assignment]
            raise AssertionError("Should not be able to mutate frozen dataclass")
        except AttributeError:
            pass

    def test_fields(self):
        ctx = RequestContext(
            request_id="req-1",
            client_ip="10.0.0.1",
            api_format="anthropic",
            is_admin=True,
        )
        assert ctx.request_id == "req-1"
        assert ctx.client_ip == "10.0.0.1"
        assert ctx.api_format == "anthropic"
        assert ctx.is_admin is True


# ---------------------------------------------------------------------------
# extract_client_ip
# ---------------------------------------------------------------------------


class TestExtractClientIp:
    def test_trust_proxy_xff(self):
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4"})
        assert extract_client_ip(req, trust_proxy=True) == "1.2.3.4"

    def test_trust_proxy_xff_chain_takes_first(self):
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4, 5.6.7.8, 9.0.0.1"})
        assert extract_client_ip(req, trust_proxy=True) == "1.2.3.4"

    def test_trust_proxy_xff_whitespace(self):
        req = FakeRequest(headers={"x-forwarded-for": "  1.2.3.4 , 5.6.7.8"})
        assert extract_client_ip(req, trust_proxy=True) == "1.2.3.4"

    def test_trust_proxy_x_real_ip(self):
        req = FakeRequest(headers={"x-real-ip": "10.0.0.1"})
        assert extract_client_ip(req, trust_proxy=True) == "10.0.0.1"

    def test_trust_proxy_xff_takes_precedence_over_real_ip(self):
        req = FakeRequest(
            headers={"x-forwarded-for": "1.2.3.4", "x-real-ip": "10.0.0.1"}
        )
        assert extract_client_ip(req, trust_proxy=True) == "1.2.3.4"

    def test_no_trust_ignores_xff(self):
        req = FakeRequest(
            headers={"x-forwarded-for": "1.2.3.4"},
            client_addr=("10.0.0.1", 1),
        )
        assert extract_client_ip(req, trust_proxy=False) == "10.0.0.1"

    def test_no_trust_ignores_real_ip(self):
        req = FakeRequest(
            headers={"x-real-ip": "1.2.3.4"},
            client_addr=("10.0.0.1", 1),
        )
        assert extract_client_ip(req, trust_proxy=False) == "10.0.0.1"

    def test_default_trusts_proxy(self):
        """Default trust_proxy=True should honour proxy headers."""
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4"})
        assert extract_client_ip(req) == "1.2.3.4"

    def test_client_addr_fallback(self):
        req = FakeRequest(client_addr=("192.168.1.1", 9999))
        assert extract_client_ip(req, trust_proxy=True) == "192.168.1.1"

    def test_no_addr_returns_unknown(self):
        req = FakeRequest()
        req.client_addr = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        assert extract_client_ip(req, trust_proxy=False) == "unknown"

    def test_empty_tuple_returns_unknown(self):
        req = FakeRequest()
        req.client_addr = (None, 0)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        assert extract_client_ip(req, trust_proxy=False) == "unknown"


# ---------------------------------------------------------------------------
# setup_request_context hook
# ---------------------------------------------------------------------------


class TestSetupRequestContext:
    def test_populates_context_for_openai_path(self):
        hook = setup_request_context(trust_proxy=True)
        req = FakeRequest(
            path="/v1/chat/completions",
            headers={"x-forwarded-for": "1.2.3.4", "x-request-id": "req-123"},
            client_addr=("10.0.0.1", 1),
        )
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.request_id == "req-123"
        assert ctx.client_ip == "1.2.3.4"
        assert ctx.api_format == "openai"
        assert ctx.is_admin is False

    def test_populates_context_for_anthropic_path(self):
        hook = setup_request_context(trust_proxy=True)
        req = FakeRequest(
            path="/v1/messages",
            headers={"x-request-id": "req-456"},
        )
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.api_format == "anthropic"
        assert ctx.is_admin is False

    def test_populates_context_for_google_path(self):
        hook = setup_request_context(trust_proxy=True)
        req = FakeRequest(
            path="/v1beta/models/gemini:generateContent",
            headers={"x-request-id": "req-789"},
        )
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.api_format == "google"

    def test_admin_path_sets_is_admin_and_none_format(self):
        hook = setup_request_context(trust_proxy=True)
        req = FakeRequest(path="/admin/api/config")
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.is_admin is True
        assert ctx.api_format is None

    def test_generates_request_id_when_missing(self):
        hook = setup_request_context(trust_proxy=True)
        req = FakeRequest(path="/v1/chat/completions")
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert len(ctx.request_id) > 0
        # Should be a UUID string
        assert "-" in ctx.request_id

    def test_trust_proxy_false_uses_client_addr(self):
        hook = setup_request_context(trust_proxy=False)
        req = FakeRequest(
            path="/v1/chat/completions",
            headers={"x-forwarded-for": "1.2.3.4"},
            client_addr=("10.0.0.1", 1),
        )
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.client_ip == "10.0.0.1"

    def test_hook_returns_none(self):
        """The hook should not short-circuit the request."""
        hook = setup_request_context(trust_proxy=True)
        req = FakeRequest(path="/v1/chat/completions")
        result, _ = _run(_run_hook_and_get_context(hook, req))
        assert result is None

    def test_health_path_gets_format(self):
        """Health paths are not admin — they get a format string."""
        hook = setup_request_context(trust_proxy=True)
        req = FakeRequest(path="/health")
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.is_admin is False
        # /health defaults to "openai" per detect_api_format
        assert ctx.api_format == "openai"


# ---------------------------------------------------------------------------
# Contextvar isolation
# ---------------------------------------------------------------------------


class TestContextvarIsolation:
    def test_default_is_none(self):
        """Without the hook, context should be None."""
        # Explicitly reset to ensure clean state (earlier tests in the
        # same process may have set the contextvar in the main thread).
        token = request_context_var.set(None)
        assert request_context_var.get() is None
        request_context_var.reset(token)

    def test_context_overwritten_per_request(self):
        """Within a single async context, second hook call overwrites the first."""
        hook = setup_request_context(trust_proxy=True)

        async def _two_requests():
            req1 = FakeRequest(path="/v1/messages", headers={"x-request-id": "first"})
            await hook(req1)
            ctx1 = request_context_var.get()

            req2 = FakeRequest(
                path="/v1/chat/completions", headers={"x-request-id": "second"}
            )
            await hook(req2)
            ctx2 = request_context_var.get()
            return ctx1, ctx2

        ctx1, ctx2 = _run(_two_requests())
        assert ctx1 is not None
        assert ctx1.request_id == "first"
        assert ctx1.api_format == "anthropic"

        assert ctx2 is not None
        assert ctx2.request_id == "second"
        assert ctx2.api_format == "openai"
