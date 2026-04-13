"""Gateway API key authentication middleware.

Validates incoming requests against the gateway's configured API keys,
extracting credentials in the format native to each API standard:

- OpenAI Chat/Responses: ``Authorization: Bearer <key>``
- Anthropic: ``x-api-key: <key>``
- Google GenAI: ``x-goog-api-key: <key>`` or ``?key=<key>`` query param

Supports multiple API keys with labels for tracking. If no keys are
configured, all requests pass through (backward compatible).
"""

from __future__ import annotations

from collections.abc import Iterable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send


# Paths that never require authentication
_PUBLIC_PATHS = frozenset({"/health"})

# Route prefix → key extraction strategy
_ROUTE_EXTRACTORS: list[tuple[str, str]] = [
    # Order matters: more specific prefixes first
    ("/v1beta/models", "google"),
    ("/v1/messages", "anthropic"),
    ("/v1/", "openai"),  # chat/completions, responses, models
]


def _extract_key(request: Request) -> str | None:
    """Extract API key from the request using the appropriate strategy."""
    path = request.url.path

    strategy = "openai"  # default fallback
    for prefix, strat in _ROUTE_EXTRACTORS:
        if path.startswith(prefix):
            strategy = strat
            break

    if strategy == "anthropic":
        return request.headers.get("x-api-key")
    elif strategy == "google":
        # Google uses x-goog-api-key header or ?key= query param
        return request.headers.get("x-goog-api-key") or request.query_params.get("key")
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


class GatewayAuthMiddleware:
    """Starlette ASGI middleware for gateway API key authentication."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: str | None = None,
        api_key_set: Iterable[str] | None = None,
        api_key_labels: dict[str, str] | None = None,
        internal_token: str | None = None,
    ) -> None:
        self.app = app
        # Accept either multi-key set or legacy single key
        if api_key_set is not None:
            self._key_set: frozenset[str] = frozenset(api_key_set)
        elif api_key:
            self._key_set = frozenset({api_key})
        else:
            self._key_set = frozenset()
        self._labels: dict[str, str] = dict(api_key_labels or {})
        self._internal_token: str | None = internal_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._key_set:
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        path = request.url.path

        # Public paths skip auth
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Admin panel: no gateway-level auth — delegate to reverse proxy
        if path.startswith("/admin"):
            await self.app(scope, receive, send)
            return

        # API paths: extract key using format-appropriate strategy
        key = _extract_key(request)

        # Check internal token first (admin panel test requests)
        if key and self._internal_token and key == self._internal_token:
            scope.setdefault("state", {})["api_key_label"] = "internal"
            await self.app(scope, receive, send)
            return

        if key not in self._key_set:
            response = _error_for_path(path, 401, "Invalid or missing API key")
            await response(scope, receive, send)
            return

        # Attach key label for request logging
        scope.setdefault("state", {})["api_key_label"] = self._labels.get(key, "")
        await self.app(scope, receive, send)
