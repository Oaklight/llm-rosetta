"""Shared error response formatting for gateway middleware.

Provides a single source of truth for:

- Detecting the API format from a request path or provider type.
- Building format-specific error envelopes (OpenAI, Anthropic, Google).
- Attaching CORS headers for non-admin error responses.

This module is used by ``proxy.py``, ``auth.py``, ``ratelimit.py``, and
``app.py`` to eliminate duplicated error formatting logic.
"""

from __future__ import annotations

from llm_rosetta._vendor.httpserver import JSONResponse, Response

# ---------------------------------------------------------------------------
# Path → API format detection (single mapping table)
# ---------------------------------------------------------------------------

_PATH_FORMAT_TABLE: list[tuple[str, str]] = [
    ("/v1beta/models", "google"),
    ("/v1/messages", "anthropic"),
    ("/v1/", "openai"),
]

_PROVIDER_FORMAT_MAP: dict[str, str] = {
    "openai_chat": "openai",
    "openai_responses": "openai",
    "open_responses": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "google_generate": "google",
    "google_interactions": "google",
}


def detect_api_format(path: str) -> str:
    """Detect API format from a request path prefix.

    Returns:
        One of ``"openai"``, ``"anthropic"``, or ``"google"``.
        Defaults to ``"openai"`` for unrecognised paths.
    """
    for prefix, fmt in _PATH_FORMAT_TABLE:
        if path.startswith(prefix):
            return fmt
    return "openai"


def detect_api_format_from_provider(source_provider: str) -> str:
    """Detect API format from a :data:`ProviderType` string.

    Returns:
        One of ``"openai"``, ``"anthropic"``, or ``"google"``.
        Defaults to ``"openai"`` for unknown provider types.
    """
    return _PROVIDER_FORMAT_MAP.get(source_provider, "openai")


# ---------------------------------------------------------------------------
# Admin path check
# ---------------------------------------------------------------------------


def is_admin_path(path: str) -> bool:
    """Return *True* if *path* is an admin panel path."""
    return path.startswith("/admin/") or path == "/admin"


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

_CORS_HEADERS: dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
}


def apply_cors_headers(response: Response) -> None:
    """Attach wildcard CORS headers to *response* (mutates in place)."""
    for key, value in _CORS_HEADERS.items():
        response.headers[key] = value


# ---------------------------------------------------------------------------
# Error envelope builders
# ---------------------------------------------------------------------------


def _build_openai_error(
    message: str,
    error_type: str,
    error_code: str | None,
) -> dict:
    """Build an OpenAI-style error body."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": error_code,
        }
    }


def _build_anthropic_error(
    message: str,
    error_type: str,
) -> dict:
    """Build an Anthropic-style error body."""
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }


def _build_google_error(
    message: str,
    status_code: int,
    google_status: str,
) -> dict:
    """Build a Google-style error body."""
    return {
        "error": {
            "code": status_code,
            "message": message,
            "status": google_status,
        }
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def format_error_response(
    api_format: str,
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    error_code: str | None = None,
    google_status: str = "INVALID_ARGUMENT",
    extra_headers: dict[str, str] | None = None,
    cors: bool = False,
) -> Response:
    """Build a format-aware JSON error response.

    Args:
        api_format: One of ``"openai"``, ``"anthropic"``, or ``"google"``.
        status_code: HTTP status code for the response.
        message: Human-readable error message.
        error_type: Error type string for OpenAI ``error.type`` and
            Anthropic ``error.type`` fields.
        error_code: OpenAI-specific ``error.code`` field.  Ignored by
            Anthropic and Google formats.
        google_status: Google-specific ``error.status`` field (e.g.
            ``"INVALID_ARGUMENT"``, ``"UNAUTHENTICATED"``).  Ignored by
            OpenAI and Anthropic formats.
        extra_headers: Additional headers to set on the response (e.g.
            ``Retry-After``, ``X-RateLimit-*``).
        cors: When *True*, attach wildcard CORS headers.

    Returns:
        A :class:`JSONResponse` with the appropriate error envelope.
    """
    if api_format == "anthropic":
        body = _build_anthropic_error(message, error_type)
    elif api_format == "google":
        body = _build_google_error(message, status_code, google_status)
    else:
        body = _build_openai_error(message, error_type, error_code)

    resp = JSONResponse(body, status_code=status_code)

    if extra_headers:
        for key, value in extra_headers.items():
            resp.headers[key] = value

    if cors:
        apply_cors_headers(resp)

    return resp
