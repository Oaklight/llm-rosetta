"""Admin authentication route handlers."""

from __future__ import annotations

import hmac
import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ..static import load_admin_html, load_static_file

# Cached HTML — loaded once on first request, per custom_head value.
_admin_html_cache: dict[str, str] = {}

_FAVICON_SVG = b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='30 15 490 500'><path fill='%23666' d='M517.5,415.4c.3-10.7-2.6-21.4-2.2-31.9c.1-2.4,2-4.5,2.1-6.9c.2-5.1-1.1-12.1-2.8-16.2c-2-4.9-1.4-11.1-1.1-16.9c.4-5.9-.3-11.6-.3-17.5c0-5.9-1.4-13.4-.3-18.4c.9-3.9-.3-8.7.2-12.7c.2-1.3,1.3-1.6,1.5-2.7c.6-2.8-1.7-2.8-1.7-4.7c0-1.6,1.7-3.2,1.9-5c1.5-3-2.3-8.2-1.4-10.6c4.9-14.2-8.2-27.4-12.9-37c-.1-.2-.2-.5-.2-.8c-3.8-15.2-8.4-27.1-13.8-41.1c-1.8-4.8-6-8.7-7.6-13.6c-1.6-4.9-5.1-9.4-7.9-13.4c-6.6-9.4-14.5-22.3-13.8-35.9c-6-6.5-6-18.5-9-28.5c-1.3-4.3-2-8.8-2.9-13.4c-.8-4.1-2.3-10.6-2.1-14.3c0-.6.8-1.1.7-1.7c-.1-.6-1.2-1-1.4-1.5c-1-2.7-1-6.6-2.2-10c-1.1-3.1-2.3-6.9-2.7-9.6c-.3-1.8-.1-3.2-1.1-4.2c-.9-.9-2.1-.9-2.3-1.9c-.5-1.8.8-2.5.8-3.8c0-2.6-2.7-6.6-3-9.7c-.1-.6.5-1.3.5-1.9c-.2-2.3-3.3-4.1-4.7-5.6c-1.8-1.8-4.3-7.6-4.9-10.1c-.6-2.2.3-6.1-1-7.2c-.7-.6-6-1-6.7-1.1c-11.2-.2-22.7-2.7-35.3-2.3c-8.5.2-17.3.5-25.6,1.4c-8.4.9-15.5,3.4-23.4,4.3c-4.5.5-7.4,1.9-11.1,2.8c-8,1.8-16.2,2.5-23,4.9c-2.8,1-8,2.4-9.6,4.4c-2,2.5-1.4,1.9-4.4,2c-1.1,0-2.4.9-3.7,1.1c-1.4.2-2.7-.5-4-.5c-1.9.1-5.9,2.2-7.9,3c-2.7,1-5.4,2.2-8.1,2.9c-14.2,3.8-26.2,10.3-38.8,16c-2.2,1-4.8,3.7-6.8,4.2c-6.6,1.8-9.7,6.2-15.4,8.7c-2.7,1.2-7.5,1.5-10,3.2c-3.7,2.6-7.4,5.9-11.7,8.4c-3,1.7-3.8.7-5,3.5c-.5,1.2-3.4.7-4.5,1.4c-5.5,3.7-9.7,9.5-15,14.1c-2.8,2.5-6,3.5-8.7,5.9c-2.9,2.5-5.8,3.9-8.9,5.7c-1.5.9-2.5,3.5-4.1,4.2c-4.1,1.9-7.7,4.2-11,7.2c-.3.2-.5.6-.6.8c-7.2,11.5-19.6,18.1-30.6,25.2c-4.6,1.7-6.6,9.6-7.1,15.3c-.2,2.1,1.1,3.6,1,5.7c-.1,1.8-1.4,2.6-1.3,4.4c.1,2.2.1,2.6-.5,4.4c-.9,2.7.8,6,.7,9.2c-.3,6.5-2.2,12.4-2.7,17.9c-.2,2.8,1.9,6.2,1,8.7c-3.3,9-1.6,21.4-.3,31.4c.7,5.2-.9,10.4-1,15.7c-.1,5.6.7,11.4.4,16.4c-.4,6.1-3.1,12.3-2.8,18.1c.3,6.1,2.2,11.8,3.5,17.2c2,7.7-.1,13.8-2.5,19.1c-2.5,5.5-.7,10.8-1.5,17.1c-.3,2.1-2.1,3.7-1.7,6.8c.3,2.5,1.2,5.1,1.3,7.5c.1,3.5-1.3,5.8-.3,9.5c1,3.8-.1,5.9-.8,9.3c-1.8,9.5-4.1,17.8-2.5,29.2c.6,4.1,1.7,7.4,2.1,11.4c.4,4.3-2.4,6.8-2.6,10.8c-.4,9.7.3,20.4-1.9,29c-1,3.8.7,5.6,1.2,8.7c.4,2.5-.4,6.2-1.3,9c-1,3.3.2,6,1.1,8.5c1.8,5.3.4,12.4-.1,18.2c-.5,6.4-.7,12.4-.5,19.1c.2,6.5-1.3,13-.4,19.7c.8,5.8,2,11.2,2,18.5c0,6.8,0,13.5-.3,19.9c-.3,7.4,3.7,9.9,11.2,9.5c3.3-.2,6.6-1.4,9.2-.2c2.4,1,4.6,2.7,7.7,2.9c1.3.1,2.6-.8,3.5-.8c2.6.2,5.4,2.4,8.7,2.1c4.7-.4,8.5.6,13,1.9c3.7,1.1,6.8,3,10.7,3.4c.7.1,1.2-.8,1.8-.8c2.9,0,5.5,1.6,8.7,1.1c3.1-.4,6.4-.4,9.3-.4c6.4.2,11.4-2.6,15.7-4.7c.3-.2.7-.3,1.1-.4c7.2-1.7,13.7-5.7,20.8-8c2.6-.8,6.4,0,9.3-.1c5-.3,11.2-1.2,16.4.8c5.9,2.2,10.9.6,16.6,1.4c2.8.4,4.9.7,7.8.8c9.8.4,19.8-4.8,29.1-5.5c.1,0,.1,0,.2-.1c5.3-1.9,13-1.6,17.9-1.3c8.2.6,18,2.6,25.4-1.1c3.2-1.6,7.4-3.3,11.1-3.7c4.5-.5,9.4,1.5,13.7.6c.8-.2.6-1.1,1.5-1.2c.6,0,1.2.5,1.9.5c12-.4,25.1-2.8,34.6-7c6.8-3,10.3-7.7,18.8-9.3c3.3-.6,6.6-3.3,9.2-4.9c5.1-3.2,15.8-4.4,18.4-9.8c5.6-11.5,22.2-12.4,28.9-22.5c.1-.1.2-.2.3-.3c4.8-4.1,9.7-8.1,13.8-12.8c2-2.3,5.1-3.6,7.1-6.2c2-2.6,4-5,6.3-7c4.6-4,8.2-9.4,12.6-14c6.8-7.1,12.3-15.3,18.6-22.2c1.6-1.7,7.5-4.8,8-7.1C521.1,437.1,517.2,426,517.5,415.4z'/></svg>"


async def serve_favicon(request: Any) -> Response:
    """Serve an inline SVG favicon (unauthenticated)."""
    return Response(
        body=_FAVICON_SVG,
        status_code=200,
        content_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# Cached static assets (CSS/JS) — populated on first request per path.
_static_cache: dict[str, tuple[bytes, str]] = {}


async def serve_admin_html(request: Any) -> Response:
    """Serve the admin panel SPA with optional custom_head injection."""
    custom_head: str = getattr(request.app, "admin_custom_head", "")
    cache_key = custom_head
    if cache_key not in _admin_html_cache:
        html = load_admin_html()
        if custom_head:
            html = html.replace("</head>", f"{custom_head}\n</head>", 1)
        _admin_html_cache[cache_key] = html
    return Response(
        body=_admin_html_cache[cache_key],
        status_code=200,
        content_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def serve_admin_static(request: Any, **kwargs: Any) -> Response:
    """Serve a static CSS/JS file from the admin package."""
    subpath: str = request.path_params["path"]
    if subpath not in _static_cache:
        try:
            _static_cache[subpath] = load_static_file(subpath)
        except FileNotFoundError:
            return Response(body=b"Not Found", status_code=404)
    data, content_type = _static_cache[subpath]
    return Response(
        body=data,
        status_code=200,
        content_type=content_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
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
    if not hmac.compare_digest(password, auth_state.admin_password):
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


async def change_password(request: Any) -> Response:
    """Change the admin password (requires current password)."""
    from ...config import config_lock
    from ._shared import _get_config_io, _get_config_path, _reload_gateway_config

    auth_state = request.app.auth_state
    if not auth_state.admin_password:
        return JSONResponse({"error": "Admin password not configured"}, status_code=400)

    # Rate-limit password change attempts the same way as login
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

    current = body.get("current_password", "")
    new_pw = body.get("new_password", "")

    if not current or not new_pw:
        return JSONResponse(
            {"error": "Both current_password and new_password are required"},
            status_code=400,
        )

    if len(new_pw) < 4:
        return JSONResponse(
            {"error": "New password must be at least 4 characters"},
            status_code=400,
        )

    if not hmac.compare_digest(current, auth_state.admin_password):
        _record_login_failure(ip)
        return JSONResponse({"error": "Current password is incorrect"}, status_code=401)

    _clear_login_failures(ip)

    # Persist to config file
    config_path = _get_config_path(request)

    with config_lock(config_path):
        try:
            data = _get_config_io(request).load_raw(config_path)
        except Exception as exc:
            return JSONResponse(
                {"error": f"Failed to read config: {exc}"}, status_code=500
            )

        data.setdefault("server", {})["admin_password"] = new_pw

        try:
            _get_config_io(request).save(config_path, data)
        except Exception as exc:
            return JSONResponse(
                {"error": f"Failed to write config: {exc}"}, status_code=500
            )

    # Hot-reload config (syncs auth state via _sync_auth_middleware)
    _reload_gateway_config(request, config_path)

    # Return the new admin token so frontend can swap immediately
    return JSONResponse({"ok": True, "token": auth_state.admin_token})


async def rotate_token(request: Any) -> Response:
    """Rotate the internal proxy token and recalculate admin token.

    The new token is in-memory only — not persisted to config.  A restart
    regenerates a fresh token regardless, so persistence is unnecessary.
    The copied token stops working after restart by design.
    """
    auth_state = request.app.auth_state
    new_admin_token = auth_state.rotate_internal_token()

    # Also update the app-level internal_token reference
    request.app.internal_token = auth_state.internal_token

    return JSONResponse({"ok": True, "token": new_admin_token})
