"""Admin authentication route handlers."""

from __future__ import annotations

import hmac
import json
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from llm_rosetta_gateway.admin_api import admin_context, catalogue

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
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 634.78 634.78" '
    b'width="634.78" height="634.78">\n'
    b"<style>.mark{fill:#666666}@media(prefers-color-scheme:dark){.mark{fill:#ededed}}</style>\n"
    b'<g transform="translate(22.369,18.638) scale(1)">\n'
    b'<polygon class="mark" points="515.98,348.14 515.55,328.91 '
    b"514.63,310.81 513.2,293.83 511.28,277.96 508.87,263.22 "
    b"505.96,249.61 502.55,237.11 498.64,225.74 492.04,210.36 "
    b"484.94,194.14 479.55,182.21 475.87,174.57 474.38,172.06 "
    b"473.37,170.33 472.32,168.56 471.25,166.75 470.15,164.89 "
    b"469.03,162.98 467.87,161.02 466.69,159.02 465.48,156.98 "
    b"464.36,154.88 463.33,152.74 462.38,150.54 461.52,148.3 "
    b"460.74,146.01 460.05,143.67 459.45,141.28 458.94,138.85 "
    b"458.4,136.52 457.84,134.28 457.25,132.16 456.64,130.13 "
    b"456.01,128.21 455.35,126.38 454.67,124.66 453.97,123.05 "
    b"453.22,120.96 452.43,118.39 451.6,115.35 450.73,111.84 "
    b"449.81,107.85 448.84,103.39 447.58,97.11 445.22,84.93 438.03,59.1 "
    b"422.77,14.06 422.04,8.64 404.88,6.34 354.13,4.87 294.81,20.72 "
    b"264.64,29.57 243.87,36.7 225.89,43.87 214.5,49.28 206.24,53.67 "
    b"197.33,58.83 187.76,64.78 177.53,71.5 165.03,80.23 150.26,90.97 "
    b"128.4,107.4 84.97,140.6 82.57,213.4 81.29,249.59 80.31,273.44 "
    b"79.39,293.02 78.51,308.35 77.59,322.12 76.62,334.32 75.35,347.24 "
    b"72.99,367.68 70.82,427.89 70.67,511.97 72.08,561.28 76.51,583.12 "
    b"94.23,584.92 101.87,585.27 110.61,587.38 117.21,587.69 "
    b"123.64,589.27 131.67,591.81 135.85,592.63 138.87,592.58 "
    b"143.85,592.56 149.04,592.57 152.68,592.25 156.51,591.62 "
    b"160.51,590.66 163.64,589.7 165.8,589.01 168.02,588.28 170.3,587.51 "
    b"172.64,586.69 175.05,585.82 177.52,584.92 180.06,583.96 "
    b"182.66,582.96 187.02,581.95 193.15,580.93 201.04,579.89 "
    b"210.69,578.84 222.11,577.78 235.29,576.7 254.41,575.34 "
    b"292.01,572.86 334.26,570.11 351.93,569.01 375.62,562.04 "
    b"420.2,542.79 461.88,502.54 481.96,483.04 494.32,470.87 "
    b"503.61,461.56 509.82,455.1 514.5,450.02 517.65,446.3 519.26,443.95 "
    b"519.33,442.97 519.37,441.85 519.37,440.62 519.34,439.25 "
    b"519.27,437.75 519.16,436.13 519.02,434.37 518.84,432.49 "
    b'518.63,430.48"/>\n'
    b"</g>\n"
    b"</svg>\n"
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
    custom_head: str = admin_context(request).admin_custom_head()
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


def _set_session_cookie(
    resp: Response, token: str, *, secure: bool = False, max_age: int | None = None
) -> None:
    """Set the admin session cookie on a response.

    ``max_age`` makes the cookie persistent — that is what "stay signed in"
    means.  Without it the browser drops the cookie when it closes, and the
    server-side idle window is the only limit on the session.
    """
    resp.set_cookie(
        catalogue().admin_cookie_name(),
        token,
        path="/admin",
        httponly=True,
        samesite="Lax",
        secure=secure,
        max_age=max_age,
    )


async def admin_login(request: Any) -> Response:
    """Validate admin password and return a session token."""
    auth_state = admin_context(request).auth_state()
    if not auth_state.admin_password:
        return JSONResponse({"error": "Admin password not configured"}, status_code=400)

    ip = auth_state.login_limiter.client_of()
    blocked, retry_after = auth_state.login_limiter.check(ip)
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
        auth_state.login_limiter.record_failure(ip)
        blocked, retry_after = auth_state.login_limiter.check(ip)
        resp: dict[str, Any] = {"error": "Invalid password"}
        if blocked:
            resp["error"] = (
                f"Too many failed attempts. Locked for {int(retry_after) + 1}s."
            )
        return JSONResponse(resp, status_code=401)

    auth_state.login_limiter.clear(ip)
    remember = bool(body.get("remember")) and auth_state.remember_seconds > 0
    session_id = auth_state.create_session(ip=ip, remember=remember)
    resp = JSONResponse({"ok": True, "remember": remember})
    _set_session_cookie(
        resp,
        session_id,
        secure=catalogue().request_is_https(request),
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
    auth_state = admin_context(request).auth_state()
    requires_auth = bool(auth_state.admin_password)
    cookie_token = request.cookies.get(catalogue().admin_cookie_name(), "")
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
    auth_state = admin_context(request).auth_state()
    cookie_token = request.cookies.get(catalogue().admin_cookie_name(), "")
    if cookie_token:
        auth_state.invalidate_session(cookie_token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(catalogue().admin_cookie_name(), path="/admin")
    return resp


async def change_password(request: Any) -> Response:
    """Change the admin password (requires current password)."""
    from ._shared import _get_config_io, _get_config_path, _reload_gateway_config

    auth_state = admin_context(request).auth_state()
    if not auth_state.admin_password:
        return JSONResponse({"error": "Admin password not configured"}, status_code=400)

    # Rate-limit password change attempts the same way as login
    ip = auth_state.login_limiter.client_of()
    blocked, retry_after = auth_state.login_limiter.check(ip)
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
        auth_state.login_limiter.record_failure(ip)
        return JSONResponse({"error": "Current password is incorrect"}, status_code=401)

    auth_state.login_limiter.clear(ip)

    # Persist to config file
    config_path = _get_config_path(request)

    async with catalogue().config_lock(config_path):
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

    # Hot-reload config (syncs auth state, see AdminContext.reload_config).
    # This calls change_password() on AuthState, which invalidates all sessions.
    await _reload_gateway_config(request, config_path)

    await admin_context(request).record_password_change()

    # Create a new session for the current user so they stay logged in, with
    # the lifetime they chose at login: changing a password must not quietly
    # turn a remembered session into one that dies with the browser.
    old_session = auth_state.get_session(
        request.cookies.get(catalogue().admin_cookie_name(), "")
    )
    was_remembered = bool(old_session and old_session.remember)
    session_id = auth_state.create_session(ip=ip, remember=was_remembered)
    resp = JSONResponse({"ok": True})
    _set_session_cookie(
        resp,
        session_id,
        secure=catalogue().request_is_https(request),
        max_age=auth_state.remember_seconds if was_remembered else None,
    )
    return resp


async def rotate_token(request: Any) -> Response:
    """Rotate the internal proxy token.

    The new token is in-memory only — not persisted to config.  A restart
    regenerates a fresh token regardless, so persistence is unnecessary.
    Sessions are NOT affected by rotation.
    """
    auth_state = admin_context(request).auth_state()
    auth_state.rotate_internal_token()
    admin_context(request).set_internal_token(auth_state.internal_token)

    await admin_context(request).record_token_rotation()

    return JSONResponse({"ok": True})


async def logout_all_sessions(request: Any) -> Response:
    """Invalidate all admin sessions."""
    auth_state = admin_context(request).auth_state()
    count = auth_state.invalidate_all_sessions()

    await admin_context(request).record_sessions_cleared(count)

    resp = JSONResponse({"ok": True, "sessions_cleared": count})
    resp.delete_cookie(catalogue().admin_cookie_name(), path="/admin")
    return resp
