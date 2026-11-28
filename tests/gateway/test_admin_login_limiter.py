"""The admin login limiter keys on the real client and holds bounded state (#1830 A3)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_rosetta_gateway.admin.routes.auth import admin_login, change_password
from llm_rosetta_gateway.middleware.auth import AuthState
from _admin_app import fake_app
from llm_rosetta_gateway.middleware.request_context import (
    ProxyPolicy,
    TrustedProxies,
    setup_request_context,
)

PASSWORD = "correct-horse"


class _Request:
    """The slice of a request the login handlers read."""

    def __init__(
        self,
        app: Any,
        *,
        peer: str = "10.0.0.9",
        xff: str | None = None,
        password: str = "wrong",
    ) -> None:
        self.app = app
        self.headers = {"x-forwarded-for": xff} if xff else {}
        self.client_addr = (peer, 5000)
        self.cookies: dict[str, str] = {}
        self.path = "/admin/api/login"
        self.method = "POST"
        self._body = {"password": password, "current_password": password}

    def json(self) -> dict[str, Any]:
        return self._body


def _app(*, trust_proxy: bool) -> Any:
    """An app whose policy trusts the test peer (or nobody)."""
    trusted = TrustedProxies.parse(["10.0.0.0/8"] if trust_proxy else None)
    return fake_app(
        auth_state=AuthState(None, None, admin_password=PASSWORD),
        proxy_policy=ProxyPolicy(trusted),
    )


async def _handle(handler: Any, request: _Request) -> Any:
    """Run the request-context hook, then the handler, as the server does."""
    await setup_request_context(request.app.proxy_policy)(request)
    return await handler(request)


def _login(app: Any, **kwargs: Any) -> int:
    request = _Request(app, **kwargs)
    return asyncio.run(_handle(admin_login, request)).status_code


def test_forged_forwarded_for_cannot_dodge_lockout_when_proxy_untrusted() -> None:
    app = _app(trust_proxy=False)
    statuses = [_login(app, xff=f"203.0.113.{i}") for i in range(8)]
    assert statuses[:5] == [401] * 5
    assert statuses[5:] == [429] * 3


def test_forwarded_for_names_the_client_when_proxy_trusted() -> None:
    app = _app(trust_proxy=True)
    for _ in range(5):
        assert _login(app, xff="203.0.113.7") == 401
    assert _login(app, xff="203.0.113.7") == 429
    # A different client behind the same proxy peer is not locked out.
    assert _login(app, xff="203.0.113.8") == 401


def test_a_request_without_a_context_is_counted_under_one_key() -> None:
    app = fake_app(auth_state=AuthState(None, None, admin_password=PASSWORD))
    statuses = [
        asyncio.run(admin_login(_Request(app, xff=f"203.0.113.{i}"))).status_code
        for i in range(6)
    ]
    assert statuses[-1] == 429


def test_change_password_shares_the_lockout_and_the_untrusted_rule() -> None:
    app = _app(trust_proxy=False)
    for i in range(5):
        assert _login(app, xff=f"203.0.113.{i}") == 401
    response = asyncio.run(_handle(change_password, _Request(app, xff="198.51.100.1")))
    assert response.status_code == 429


def test_a_successful_login_clears_the_failures() -> None:
    app = _app(trust_proxy=False)
    for _ in range(4):
        assert _login(app) == 401
    assert _login(app, password=PASSWORD) == 200
    for _ in range(4):
        assert _login(app) == 401


def test_two_apps_do_not_share_lockout_state() -> None:
    first, second = _app(trust_proxy=False), _app(trust_proxy=False)
    for _ in range(5):
        _login(first)
    assert _login(first) == 429
    assert _login(second) == 401


def test_the_failure_table_is_bounded_and_keeps_locked_clients() -> None:
    app = _app(trust_proxy=True)
    limiter = app.auth_state.login_limiter
    cap = limiter.max_entries
    for _ in range(5):
        _login(app, xff="198.51.100.200")
    assert _login(app, xff="198.51.100.200") == 429
    for i in range(cap + 50):
        _login(app, xff=f"192.0.2.{i % 250}.{i}")
    assert len(limiter) <= cap
    # The locked client survived a flood of one-off addresses.
    assert _login(app, xff="198.51.100.200") == 429


def test_a_lockout_expires(monkeypatch: Any) -> None:
    import llm_rosetta_gateway.middleware.auth as auth_module

    now = [1000.0]
    monkeypatch.setattr(auth_module.time, "monotonic", lambda: now[0])
    app = _app(trust_proxy=False)
    for _ in range(5):
        _login(app)
    assert _login(app) == 429
    now[0] += 301
    assert _login(app) == 401
    assert json.loads(
        asyncio.run(_handle(admin_login, _Request(app, password=PASSWORD))).body
    ) == {"ok": True, "remember": False}
