"""Tests for the gateway request context middleware."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from llm_rosetta_gateway.middleware.request_context import (
    RequestContext,
    ProxyPolicy,
    TrustedProxies,
    resolve_client_ip,
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


#: Trusts the proxy the fake requests come from (their default peer).
_LOOPBACK = TrustedProxies.parse(["127.0.0.1"])
_NOBODY = TrustedProxies.parse(None)


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
        assert resolve_client_ip(req, _LOOPBACK) == "1.2.3.4"

    def test_trust_proxy_xff_chain_takes_first(self):
        """The first untrusted address from the right, not the leftmost one."""
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4, 5.6.7.8, 9.0.0.1"})
        assert resolve_client_ip(req, _LOOPBACK) == "9.0.0.1"

    def test_trust_proxy_xff_whitespace(self):
        req = FakeRequest(headers={"x-forwarded-for": "  5.6.7.8 , 1.2.3.4  "})
        assert resolve_client_ip(req, _LOOPBACK) == "1.2.3.4"

    def test_trust_proxy_x_real_ip(self):
        req = FakeRequest(headers={"x-real-ip": "10.0.0.1"})
        assert resolve_client_ip(req, _LOOPBACK) == "10.0.0.1"

    def test_trust_proxy_xff_takes_precedence_over_real_ip(self):
        req = FakeRequest(
            headers={"x-forwarded-for": "1.2.3.4", "x-real-ip": "10.0.0.1"}
        )
        assert resolve_client_ip(req, _LOOPBACK) == "1.2.3.4"

    def test_no_trust_ignores_xff(self):
        req = FakeRequest(
            headers={"x-forwarded-for": "1.2.3.4"},
            client_addr=("10.0.0.1", 1),
        )
        assert resolve_client_ip(req, _NOBODY) == "10.0.0.1"

    def test_no_trust_ignores_real_ip(self):
        req = FakeRequest(
            headers={"x-real-ip": "1.2.3.4"},
            client_addr=("10.0.0.1", 1),
        )
        assert resolve_client_ip(req, _NOBODY) == "10.0.0.1"

    def test_default_trusts_proxy(self):
        """An empty trusted set (the default) trusts no proxy header."""
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4"})
        assert resolve_client_ip(req, TrustedProxies()) == "127.0.0.1"

    def test_client_addr_fallback(self):
        req = FakeRequest(client_addr=("192.168.1.1", 9999))
        assert resolve_client_ip(req, _LOOPBACK) == "192.168.1.1"

    def test_no_addr_returns_unknown(self):
        req = FakeRequest()
        req.client_addr = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        assert resolve_client_ip(req, _NOBODY) == "unknown"

    def test_empty_tuple_returns_unknown(self):
        req = FakeRequest()
        req.client_addr = (None, 0)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        assert resolve_client_ip(req, _NOBODY) == "unknown"


# ---------------------------------------------------------------------------
# setup_request_context hook
# ---------------------------------------------------------------------------


class TestSetupRequestContext:
    def test_populates_context_for_openai_path(self):
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )
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
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )
        req = FakeRequest(
            path="/v1/messages",
            headers={"x-request-id": "req-456"},
        )
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.api_format == "anthropic"
        assert ctx.is_admin is False

    def test_populates_context_for_google_path(self):
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )
        req = FakeRequest(
            path="/v1beta/models/gemini:generateContent",
            headers={"x-request-id": "req-789"},
        )
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.api_format == "google"

    def test_admin_path_sets_is_admin_and_none_format(self):
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )
        req = FakeRequest(path="/admin/api/config")
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.is_admin is True
        assert ctx.api_format is None

    def test_generates_request_id_when_missing(self):
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )
        req = FakeRequest(path="/v1/chat/completions")
        _, ctx = _run(_run_hook_and_get_context(hook, req))
        assert ctx is not None
        assert len(ctx.request_id) > 0
        # Should be a UUID string
        assert "-" in ctx.request_id

    def test_trust_proxy_false_uses_client_addr(self):
        hook = setup_request_context(ProxyPolicy())
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
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )
        req = FakeRequest(path="/v1/chat/completions")
        result, _ = _run(_run_hook_and_get_context(hook, req))
        assert result is None

    def test_health_path_gets_format(self):
        """Health paths are not admin — they get a format string."""
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )
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
        hook = setup_request_context(
            ProxyPolicy(TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1"]))
        )

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


class TestAuditAddresses:
    """``peer_ip`` and ``forwarded_for`` keep what the wire showed (#1830 A12)."""

    def _ctx(self, policy: ProxyPolicy, req: Any) -> RequestContext:
        _, ctx = _run(_run_hook_and_get_context(setup_request_context(policy), req))
        assert ctx is not None
        return ctx

    def test_an_untrusted_peer_keeps_its_header_but_it_does_not_decide(self):
        req = FakeRequest(
            headers={"x-forwarded-for": "203.0.113.9, 10.0.0.2"},
            client_addr=("172.29.0.1", 1),
        )
        ctx = self._ctx(ProxyPolicy(_NOBODY), req)
        assert ctx.client_ip == "172.29.0.1"
        assert ctx.peer_ip == "172.29.0.1"
        assert ctx.forwarded_for == "203.0.113.9, 10.0.0.2"

    def test_a_trusted_peer_resolves_the_client_and_still_keeps_the_raw_chain(self):
        trusted = TrustedProxies.parse(["172.29.0.1/32", "127.0.0.0/8"])
        req = FakeRequest(
            headers={"x-forwarded-for": "198.51.100.7, 127.0.0.1"},
            client_addr=("172.29.0.1", 1),
        )
        ctx = self._ctx(ProxyPolicy(trusted), req)
        assert ctx.client_ip == "198.51.100.7"
        assert ctx.peer_ip == "172.29.0.1"
        assert ctx.forwarded_for == "198.51.100.7, 127.0.0.1"

    def test_a_request_with_no_header_has_none(self):
        ctx = self._ctx(ProxyPolicy(_NOBODY), FakeRequest())
        assert ctx.forwarded_for is None
        assert ctx.peer_ip == "127.0.0.1"

    def test_a_request_with_no_peer_has_none(self):
        req = SimpleNamespace(
            path="/v1/chat/completions",
            method="POST",
            headers={},
            client_addr=(),
            query_params={},
        )
        ctx = self._ctx(ProxyPolicy(_NOBODY), req)
        assert ctx.peer_ip is None
        assert ctx.client_ip == "unknown"

    def test_the_header_is_cut_so_a_client_cannot_grow_a_log_row(self):
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4, " * 400})
        ctx = self._ctx(ProxyPolicy(_NOBODY), req)
        assert ctx.forwarded_for is not None and len(ctx.forwarded_for) == 500

    def test_the_audit_fields_never_change_the_resolved_client(self):
        """A forged chain from an untrusted peer decides nothing."""
        forged = FakeRequest(
            headers={"x-forwarded-for": "6.6.6.6", "x-real-ip": "7.7.7.7"},
            client_addr=("203.0.113.5", 1),
        )
        ctx = self._ctx(ProxyPolicy(_LOOPBACK), forged)
        assert ctx.client_ip == "203.0.113.5"
        assert ctx.forwarded_for == "6.6.6.6"
