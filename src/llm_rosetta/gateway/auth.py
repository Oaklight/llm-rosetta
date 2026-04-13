"""Gateway API key authentication middleware.

Validates incoming requests against the gateway's configured API key,
extracting credentials in the format native to each API standard:

- OpenAI Chat/Responses: ``Authorization: Bearer <key>``
- Anthropic: ``x-api-key: <key>``
- Google GenAI: ``x-goog-api-key: <key>`` or ``?key=<key>`` query param

If no ``api_key`` is configured in ``server`` config, all requests pass
through (backward compatible).
"""

from __future__ import annotations

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
        return (
            request.headers.get("x-goog-api-key")
            or request.query_params.get("key")
        )
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
            {"type": "error", "error": {"type": "authentication_error", "message": message}},
            status_code=status,
        )
    elif strategy == "google":
        return JSONResponse(
            {"error": {"code": status, "message": message, "status": "UNAUTHENTICATED"}},
            status_code=status,
        )
    else:
        return JSONResponse(
            {"error": {"message": message, "type": "invalid_request_error", "code": "invalid_api_key"}},
            status_code=status,
        )


class GatewayAuthMiddleware:
    """Starlette ASGI middleware for gateway API key authentication."""

    def __init__(self, app: ASGIApp, api_key: str | None = None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.api_key:
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        path = request.url.path

        # Public paths skip auth
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Admin panel paths: use Bearer token (same gateway key)
        if path.startswith("/admin"):
            auth = request.headers.get("authorization", "")
            key = auth[7:] if auth.startswith("Bearer ") else None
            # Allow admin HTML pages without auth (SPA loads first, then JS adds headers)
            if path in ("/admin", "/admin/") and request.method == "GET":
                await self.app(scope, receive, send)
                return
            if key != self.api_key:
                response = JSONResponse(
                    {"error": "Invalid or missing API key"}, status_code=401
                )
                await response(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        # API paths: extract key using format-appropriate strategy
        key = _extract_key(request)
        if key != self.api_key:
            response = _error_for_path(path, 401, "Invalid or missing API key")
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
