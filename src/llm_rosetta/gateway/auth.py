"""Gateway API key authentication — before-request hook.

Validates incoming requests against the gateway's configured API keys,
extracting credentials in the format native to each API standard:

- OpenAI Chat/Responses: ``Authorization: Bearer <key>``
- Anthropic: ``x-api-key: <key>``
- Google GenAI: ``x-goog-api-key: <key>`` or ``?key=<key>`` query param

Supports multiple API keys with labels for tracking. If no keys are
configured, all requests pass through (backward compatible).
"""

from __future__ import annotations

import contextvars
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

# Per-request API key label — set by auth hook, read by proxy handler.
api_key_label_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "api_key_label", default=None
)

# Paths that never require authentication
_PUBLIC_PATHS = frozenset({"/health"})

# Route prefix → key extraction strategy
_ROUTE_EXTRACTORS: list[tuple[str, str]] = [
    # Order matters: more specific prefixes first
    ("/v1beta/models", "google"),
    ("/v1/messages", "anthropic"),
    ("/v1/", "openai"),  # chat/completions, responses, models
]


def _extract_key(request: Any) -> str | None:
    """Extract API key from the request using the appropriate strategy."""
    path = request.path

    strategy = "openai"  # default fallback
    for prefix, strat in _ROUTE_EXTRACTORS:
        if path.startswith(prefix):
            strategy = strat
            break

    if strategy == "anthropic":
        return request.headers.get("x-api-key")
    elif strategy == "google":
        # Google uses x-goog-api-key header or ?key= query param
        google_key = request.headers.get("x-goog-api-key")
        if google_key:
            return google_key
        vals = request.query_params.get("key")
        return vals[0] if vals else None
    else:
        # OpenAI-style Bearer token
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None


def _error_for_path(path: str, status: int, message: str) -> Response:
    """Return an error response in the format matching the API standard."""
    for prefix, strategy in _ROUTE_EXTRACTORS:
        if path.startswith(prefix):
            break
    else:
        strategy = "openai"

    if strategy == "anthropic":
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "authentication_error", "message": message},
            },
            status_code=status,
        )
    elif strategy == "google":
        return JSONResponse(
            {
                "error": {
                    "code": status,
                    "message": message,
                    "status": "UNAUTHENTICATED",
                }
            },
            status_code=status,
        )
    else:
        return JSONResponse(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            status_code=status,
        )


class AuthState:
    """Mutable state container for auth hook — allows hot-reload from admin."""

    def __init__(
        self,
        key_set: frozenset[str],
        labels: dict[str, str],
        internal_token: str | None,
    ) -> None:
        self.key_set = key_set
        self.labels = labels
        self.internal_token = internal_token


def create_auth_hook(auth_state: AuthState) -> Any:
    """Return a before-request hook that validates API keys.

    The hook reads from ``auth_state`` which can be mutated by the admin
    panel's hot-reload logic.
    """

    async def auth_hook(request: Any) -> Response | None:
        # Reset per-request label
        api_key_label_var.set(None)

        if not auth_state.key_set:
            return None  # no keys configured → pass through

        path = request.path

        # Public paths skip auth
        if path in _PUBLIC_PATHS:
            return None

        # Admin panel: no gateway-level auth — delegate to reverse proxy
        if path.startswith("/admin"):
            return None

        # API paths: extract key using format-appropriate strategy
        key = _extract_key(request)

        # Check internal token first (admin panel test requests)
        if key and auth_state.internal_token and key == auth_state.internal_token:
            api_key_label_var.set("internal")
            return None

        if key not in auth_state.key_set:
            return _error_for_path(path, 401, "Invalid or missing API key")

        # Attach key label for request logging
        api_key_label_var.set(auth_state.labels.get(key, ""))
        return None

    return auth_hook
