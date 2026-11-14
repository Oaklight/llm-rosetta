"""Admin authentication route handlers."""

from __future__ import annotations

import hmac
import json
import time
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from llm_rosetta_gateway.middleware.auth import ADMIN_COOKIE_NAME

from ..static import load_admin_html, load_static_file
from ..themes import registry_payload


def _json_for_script(payload: object) -> str:
    """Serialise *payload* for an inline ``<script type="application/json">``.

    The JSON is embedded in a ``<script>`` element, and the HTML parser ends
    that element at the first ``</script`` it sees — wherever it sits in the
    text.  Escaping ``<``, ``>`` and ``&`` as ``\\uXXXX`` keeps any value in
    the payload, today's fixed labels included, from closing the element early
    and leaving the rest of the registry as markup.  The escapes are valid
    JSON, so the page reads the same values back.
    """
    return (
        json.dumps(payload, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _theme_script() -> str:
    """The registry as the inline JSON script the boot script reads."""
    return (
        '<script id="adminThemes" type="application/json">'
        + _json_for_script(registry_payload())
        + "</script>"
    )


# Cached HTML — loaded once on first request, per custom_head value.  The
# registry is a module constant, so it does not need to be part of the key;
# a new theme is picked up when the gateway restarts, like any static file.
_admin_html_cache: dict[str, str] = {}

_FAVICON_SVG = (
    b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 635.22 635.22' "
    b"width='635.22' height='635.22'>"
    b"<style>.mark{fill:#666666}@media(prefers-color-scheme:dark){.mark{fill:#ededed}}</style>"
    b"<g transform='translate(23.648,18.658) scale(1)'>"
    b"<polygon class='mark' points='517.5,415.4 513.6,356.6 514,261.6 486.4,193.2 459.4,145.2 431.7,28.1 421.8,6.8 354.1,4.9 299.2,16 243.7,35 187.9,61.9 119.1,112.3 85,140.6 77.2,219.2 79.3,313.2 70.5,389.4 68.4,533 70.8,578.6 76.5,583.1 143.7,593 191.8,579.5 241.7,582.3 305.2,576.5 360.8,568.5 420.2,542.8 472.7,498.6 519.5,443.5 517.5,417.4'/>"
    b"</g></svg>"
)


async def serve_favicon(request: Any) -> Response:
    """Serve an inline SVG favicon (unauthenticated)."""
    return Response(
        body=_FAVICON_SVG,
        status_code=200,
        content_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# Cached static assets (CSS/JS) — populated on first request per path.
# `(data, content_type, etag, gzipped)`; the compressed copy is built on the
# first request that accepts gzip, not for every file at startup (#1522).
_static_cache: dict[str, tuple[bytes, str, str, bytes | None]] = {}


def _etag_for(data: bytes) -> str:
    """A strong validator for one file's bytes."""
    import hashlib

    return '"' + hashlib.sha256(data).hexdigest()[:20] + '"'


def _accepts_gzip(request: Any) -> bool:
    return "gzip" in (getattr(request, "headers", {}) or {}).get("accept-encoding", "")


def _not_modified(request: Any, etag: str) -> bool:
    """Whether the client already holds this exact body."""
    return etag in (getattr(request, "headers", {}) or {}).get("if-none-match", "")


def _compressible(content_type: str) -> bool:
    return content_type.startswith(("text/", "application/javascript", "image/svg"))


async def serve_admin_html(request: Any) -> Response:
    """Serve the admin panel SPA with optional custom_head injection."""
    custom_head: str = getattr(request.app, "admin_custom_head", "")
    cache_key = custom_head
    if cache_key not in _admin_html_cache:
        html = load_admin_html()
        # The boot script needs the registry before it can pick a theme, and
        # the page ships no build step, so the server injects it here.
        html = html.replace("<!--admin-themes-->", _theme_script(), 1)
        if custom_head:
            html = html.replace("</head>", f"{custom_head}\n</head>", 1)
        _admin_html_cache[cache_key] = html
    html = _admin_html_cache[cache_key]
    data = html.encode("utf-8")
    etag = _etag_for(data)
    headers = {
        "Cache-Control": "no-cache",
        "ETag": etag,
        "Vary": "Accept-Encoding",
        "X-Content-Type-Options": "nosniff",
    }
    if _not_modified(request, etag):
        return Response(body=b"", status_code=304, headers=headers)
    if _accepts_gzip(request):
        import gzip

        data = gzip.compress(data, compresslevel=6)
        headers["Content-Encoding"] = "gzip"
    return Response(
        body=data,
        status_code=200,
        content_type="text/html; charset=utf-8",
        headers=headers,
    )


async def serve_admin_static(request: Any, **kwargs: Any) -> Response:
    """Serve a static CSS/JS file from the admin package.

    The panel's own file is the only place these bytes live, so a reload can
    revalidate instead of downloading everything again: the response carries an
    ETag, a conditional request gets a 304, and text is compressed when the
    client accepts it.  The in-process cache means the bytes only change on a
    restart, which is exactly what an ETag says (#1522).
    """
    subpath: str = request.path_params["path"]
    if subpath not in _static_cache:
        try:
            data, content_type = load_static_file(subpath)
        except FileNotFoundError:
            return Response(body=b"Not Found", status_code=404)
        _static_cache[subpath] = (data, content_type, _etag_for(data), None)
    data, content_type, etag, gzipped = _static_cache[subpath]
    headers = {
        # Revalidate every time: the operator restarts the gateway to change a
        # file, and a stale panel is worse than one conditional request.
        "Cache-Control": "no-cache",
        "ETag": etag,
        "Vary": "Accept-Encoding",
        "X-Content-Type-Options": "nosniff",
    }
    if _not_modified(request, etag):
        return Response(body=b"", status_code=304, headers=headers)
    body = data
    if _compressible(content_type) and _accepts_gzip(request):
        if gzipped is None:
            import gzip

            gzipped = gzip.compress(data, compresslevel=6)
            _static_cache[subpath] = (data, content_type, etag, gzipped)
        body = gzipped
        headers["Content-Encoding"] = "gzip"
    return Response(
        body=body,
        status_code=200,
        content_type=content_type,
        headers=headers,
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


def _is_secure_request(request: Any) -> bool:
    """Check if the request arrived over HTTPS (direct or via reverse proxy)."""
    proto = request.headers.get("x-forwarded-proto", "")
    return proto.lower() == "https"


def _set_session_cookie(
    resp: Response, token: str, *, secure: bool = False, max_age: int | None = None
) -> None:
    """Set the admin session cookie on a response.

    ``max_age`` makes the cookie persistent — that is what "stay signed in"
    means.  Without it the browser drops the cookie when it closes, and the
    server-side idle window is the only limit on the session.
    """
    resp.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        path="/admin",
        httponly=True,
        samesite="Lax",
        secure=secure,
        max_age=max_age,
    )


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
    remember = bool(body.get("remember")) and auth_state.remember_seconds > 0
    session_id = auth_state.create_session(ip=ip, remember=remember)
    resp = JSONResponse({"ok": True, "remember": remember})
    _set_session_cookie(
        resp,
        session_id,
        secure=_is_secure_request(request),
        max_age=auth_state.remember_seconds if remember else None,
    )
    return resp


async def admin_check(request: Any) -> Response:
    """Check whether admin auth is required (before loading config).

    Also reports the session lifetimes, because the login page has to know
    whether "stay signed in" is on offer and how long an idle session lasts.

    ``remembered`` answers for the caller's own cookie: a remembered session is
    exempt from the idle window on the server, so the panel must not arm its
    inactivity timer for one — that timer signs the session out, which is
    exactly what "stay signed in" promised not to do.
    """
    auth_state = request.app.auth_state
    requires_auth = bool(auth_state.admin_password)
    cookie_token = request.cookies.get(ADMIN_COOKIE_NAME, "")
    session = auth_state.get_session(cookie_token) if cookie_token else None
    return JSONResponse(
        {
            "requires_auth": requires_auth,
            "idle_minutes": auth_state.session_idle_minutes,
            "remember_days": auth_state.session_remember_days,
            "remembered": bool(session and session.remember),
        }
    )


async def admin_logout(request: Any) -> Response:
    """Clear the admin session cookie and invalidate the session."""
    auth_state = request.app.auth_state
    cookie_token = request.cookies.get(ADMIN_COOKIE_NAME, "")
    if cookie_token:
        auth_state.invalidate_session(cookie_token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(ADMIN_COOKIE_NAME, path="/admin")
    return resp


async def change_password(request: Any) -> Response:
    """Change the admin password (requires current password)."""
    from ...config import async_config_lock
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

    async with async_config_lock(config_path):
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

    # Hot-reload config (syncs auth state via _sync_auth_middleware).
    # This calls change_password() on AuthState, which invalidates all sessions.
    await _reload_gateway_config(request, config_path)

    # Audit log
    from llm_rosetta_gateway.ops.security import OpsPasswordChange

    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsPasswordChange(_ops_ctx).execute()

    # Create a new session for the current user so they stay logged in, with
    # the lifetime they chose at login: changing a password must not quietly
    # turn a remembered session into one that dies with the browser.
    old_session = auth_state.get_session(request.cookies.get(ADMIN_COOKIE_NAME, ""))
    was_remembered = bool(old_session and old_session.remember)
    session_id = auth_state.create_session(ip=ip, remember=was_remembered)
    resp = JSONResponse({"ok": True})
    _set_session_cookie(
        resp,
        session_id,
        secure=_is_secure_request(request),
        max_age=auth_state.remember_seconds if was_remembered else None,
    )
    return resp


async def rotate_token(request: Any) -> Response:
    """Rotate the internal proxy token.

    The new token is in-memory only — not persisted to config.  A restart
    regenerates a fresh token regardless, so persistence is unnecessary.
    Sessions are NOT affected by rotation.
    """
    auth_state = request.app.auth_state
    auth_state.rotate_internal_token()
    request.app.internal_token = auth_state.internal_token

    from llm_rosetta_gateway.ops.security import OpsTokenRotate

    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsTokenRotate(_ops_ctx).execute()

    return JSONResponse({"ok": True})


async def logout_all_sessions(request: Any) -> Response:
    """Invalidate all admin sessions."""
    auth_state = request.app.auth_state
    count = auth_state.invalidate_all_sessions()

    from llm_rosetta_gateway.ops.security import OpsSessionLogoutAll

    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsSessionLogoutAll(_ops_ctx, count=count).execute()

    resp = JSONResponse({"ok": True, "sessions_cleared": count})
    resp.delete_cookie(ADMIN_COOKIE_NAME, path="/admin")
    return resp
