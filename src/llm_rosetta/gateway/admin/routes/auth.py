"""Admin authentication route handlers."""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ..static import load_admin_html

# Cached HTML — loaded once on first request.
_admin_html: str | None = None


async def serve_admin_html(request: Any) -> Response:
    """Serve the admin panel SPA."""
    global _admin_html
    if _admin_html is None:
        _admin_html = load_admin_html()
    return Response(
        body=_admin_html,
        status_code=200,
        content_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# Login rate limiter
# ---------------------------------------------------------------------------

# Per-IP failure tracking: {ip: {"count": int, "locked_until": float}}
_login_failures: dict[str, dict[str, Any]] = {}
_LOGIN_MAX_ATTEMPTS = 5  # failures before lockout
_LOGIN_LOCKOUT_SECONDS = 300  # 5-minute lockout window


def _get_client_ip(request: Any) -> str:
    """Extract client IP, honouring X-Forwarded-For when present."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    addr = getattr(request, "client_addr", None)
    if addr:
        return str(addr[0])
    return "unknown"


def _check_login_rate_limit(ip: str) -> tuple[bool, float]:
    """Return (is_blocked, retry_after_seconds).

    An IP is blocked for ``_LOGIN_LOCKOUT_SECONDS`` after
    ``_LOGIN_MAX_ATTEMPTS`` consecutive failures.
    """
    rec = _login_failures.get(ip)
    if not rec:
        return False, 0.0
    locked_until = rec.get("locked_until", 0.0)
    if locked_until and time.monotonic() < locked_until:
        return True, locked_until - time.monotonic()
    return False, 0.0


def _record_login_failure(ip: str) -> None:
    """Increment failure counter; lock out the IP after max attempts."""
    rec = _login_failures.setdefault(ip, {"count": 0, "locked_until": 0.0})
    # Reset counter if a previous lockout has expired
    if rec["locked_until"] and time.monotonic() >= rec["locked_until"]:
        rec["count"] = 0
        rec["locked_until"] = 0.0
    rec["count"] += 1
    if rec["count"] >= _LOGIN_MAX_ATTEMPTS:
        rec["locked_until"] = time.monotonic() + _LOGIN_LOCKOUT_SECONDS


def _clear_login_failures(ip: str) -> None:
    """Reset failure counter on successful login."""
    _login_failures.pop(ip, None)


async def admin_login(request: Any) -> Response:
    """Validate admin password and return a session token."""
    auth_state = request.app.auth_state
    if not auth_state.admin_password:
        return JSONResponse({"error": "Admin password not configured"}, status_code=400)

    ip = _get_client_ip(request)
    blocked, retry_after = _check_login_rate_limit(ip)
    if blocked:
        return JSONResponse(
            {
                "error": f"Too many failed attempts. Try again in {int(retry_after) + 1}s."
            },
            status_code=429,
        )

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    password = body.get("password", "")
    if password != auth_state.admin_password:
        _record_login_failure(ip)
        blocked, retry_after = _check_login_rate_limit(ip)
        resp: dict[str, Any] = {"error": "Invalid password"}
        if blocked:
            resp["error"] = (
                f"Too many failed attempts. Locked for {int(retry_after) + 1}s."
            )
        return JSONResponse(resp, status_code=401)

    _clear_login_failures(ip)
    return JSONResponse({"ok": True, "token": auth_state.admin_token})


async def admin_check(request: Any) -> Response:
    """Check whether admin auth is required (before loading config)."""
    auth_state = request.app.auth_state
    requires_auth = bool(auth_state.admin_password)
    return JSONResponse({"requires_auth": requires_auth})
