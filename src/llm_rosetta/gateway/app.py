"""llm-rosetta Gateway — core ASGI application and CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any
from collections.abc import AsyncIterator

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_rosetta import get_converter_for_provider
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.converters.base.stream_context import StreamContext

from .config import GatewayConfig, load_config

logger = logging.getLogger("llm-rosetta-gateway")

# ---------------------------------------------------------------------------
# Upstream request building
# ---------------------------------------------------------------------------

_UPSTREAM_URL_TEMPLATES = {
    "openai_chat": "{base_url}/chat/completions",
    "openai_responses": "{base_url}/responses",
    "anthropic": "{base_url}/v1/messages",
    "google": "{base_url}/v1beta/models/{model}:generateContent",
    "google_stream": "{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse",
}


def _build_auth_headers(provider_type: ProviderType, api_key: str) -> dict[str, str]:
    """Return provider-specific authentication headers."""
    if provider_type in ("openai_chat", "openai_responses"):
        return {"Authorization": f"Bearer {api_key}"}
    elif provider_type == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    elif provider_type == "google":
        return {"x-goog-api-key": api_key}
    return {}


def _build_upstream_url(
    provider_type: ProviderType,
    provider_cfg: dict[str, str],
    model: str,
    *,
    stream: bool,
) -> str:
    """Construct the full upstream URL for a provider."""
    base_url = provider_cfg["base_url"].rstrip("/")
    if provider_type == "google" and stream:
        template = _UPSTREAM_URL_TEMPLATES["google_stream"]
    else:
        template = _UPSTREAM_URL_TEMPLATES[provider_type]
    return template.format(base_url=base_url, model=model)


def _fixup_google_body(provider_request: dict[str, Any]) -> dict[str, Any]:
    """Flatten Google SDK-style nested config to REST API top-level keys.

    ``request_to_provider()`` nests tools/tool_config/generation params
    inside a ``config`` dict (designed for the SDK).  The REST API expects
    them at the top level.

    Reference: examples/rest_based/cross_oc_gg_stream.py:170-183
    """
    body: dict[str, Any] = {"contents": provider_request["contents"]}
    config = provider_request.get("config", {})

    # Lift specific keys from config to top level
    for key in ("tools", "tool_config", "response_mime_type", "response_schema"):
        if config.get(key):
            body[key] = config[key]

    # Lift generation config fields (temperature, top_p, etc.)
    generation_keys = (
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
        "candidate_count",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "response_logprobs",
    )
    generation_config = {}
    for key in generation_keys:
        if key in config:
            generation_config[key] = config[key]
    if generation_config:
        body["generationConfig"] = generation_config

    # system_instruction is already at top level from the converter
    if "system_instruction" in provider_request:
        body["system_instruction"] = provider_request["system_instruction"]

    return body


def _prepare_upstream(
    target_provider: ProviderType,
    provider_cfg: dict[str, str],
    provider_request: dict[str, Any],
    model: str,
    *,
    stream: bool,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Return (url, headers, body) ready for the upstream HTTP call."""
    url = _build_upstream_url(target_provider, provider_cfg, model, stream=stream)
    headers = {
        "Content-Type": "application/json",
        **_build_auth_headers(target_provider, provider_cfg["api_key"]),
    }

    # Provider-specific body fixups
    if target_provider == "google":
        body = _fixup_google_body(provider_request)
    else:
        body = dict(provider_request)

    # Inject stream flag into the body for providers that use it
    if stream:
        if target_provider in ("openai_chat",):
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        elif target_provider in ("openai_responses", "anthropic"):
            body["stream"] = True
        # Google streaming is signaled via URL, not body

    return url, headers, body


# ---------------------------------------------------------------------------
# SSE parsing (upstream → IR events)
# ---------------------------------------------------------------------------


def _iter_sse_lines(line: str) -> tuple[str | None, str | None] | None:
    """Parse a single SSE line into (field, value) or None if not relevant.

    Returns:
        ("data", <value>)  for data lines
        ("event", <value>) for event lines
        None               for empty/irrelevant lines
    """
    if not line:
        return None
    if line.startswith("data: "):
        return ("data", line[6:])
    if line.startswith("event: "):
        return ("event", line[7:])
    return None


def _is_openai_done(data: str) -> bool:
    """Check if the SSE data payload signals end-of-stream (OpenAI [DONE])."""
    return data.strip() == "[DONE]"


# ---------------------------------------------------------------------------
# SSE emission (IR events → source-format SSE text)
# ---------------------------------------------------------------------------


def _format_sse_openai_chat(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Chat SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_openai_chat_done() -> str:
    return "data: [DONE]\n\n"


def _format_sse_anthropic(chunk: dict[str, Any]) -> str:
    """Format a chunk as Anthropic SSE (event: type\\ndata: json)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_openai_responses(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Responses SSE (event: type\\ndata: json)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_google(chunk: dict[str, Any]) -> str:
    """Format a chunk as Google SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


_SSE_FORMATTERS = {
    "openai_chat": _format_sse_openai_chat,
    "openai_responses": _format_sse_openai_responses,
    "anthropic": _format_sse_anthropic,
    "google": _format_sse_google,
}


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _error_response_for_source(
    source_provider: ProviderType, status_code: int, message: str
) -> Response:
    """Return an error response formatted for the source provider's envelope."""
    if source_provider == "openai_chat":
        body = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": None,
            }
        }
    elif source_provider == "openai_responses":
        body = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": None,
            }
        }
    elif source_provider == "anthropic":
        body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
    elif source_provider == "google":
        body = {
            "error": {
                "code": status_code,
                "message": message,
                "status": "INVALID_ARGUMENT",
            }
        }
    else:
        body = {"error": {"message": message}}

    return JSONResponse(body, status_code=status_code)


# ---------------------------------------------------------------------------
# Request body helpers
# ---------------------------------------------------------------------------


def _detect_stream_request(source_provider: ProviderType, body: dict[str, Any]) -> bool:
    """Detect if the incoming request asks for streaming."""
    if source_provider in ("openai_chat", "openai_responses", "anthropic"):
        return bool(body.get("stream", False))
    # Google streaming is determined by the endpoint path, not the body
    return False


def _extract_model(source_provider: ProviderType, body: dict[str, Any]) -> str | None:
    """Extract the model name from a source-format request body."""
    return body.get("model")


# ---------------------------------------------------------------------------
# Core proxy logic
# ---------------------------------------------------------------------------

# Shared httpx client (created once, reused across requests)
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=300.0)
    return _http_client


async def _handle_non_streaming(
    source_provider: ProviderType,
    target_provider: ProviderType,
    provider_cfg: dict[str, str],
    body: dict[str, Any],
    model: str,
) -> Response:
    """Non-streaming proxy: convert -> forward -> convert back -> respond."""
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # 1. Source -> IR
    try:
        ir_request = source_converter.request_from_provider(body)
    except Exception as exc:
        return _error_response_for_source(
            source_provider, 400, f"Failed to parse request: {exc}"
        )

    # 2. IR -> Target
    try:
        target_body, warnings = target_converter.request_to_provider(ir_request)
    except Exception as exc:
        return _error_response_for_source(
            source_provider, 400, f"Conversion error: {exc}"
        )
    if warnings:
        logger.warning("Conversion warnings: %s", warnings)

    # 3. Build upstream request
    url, headers, upstream_body = _prepare_upstream(
        target_provider, provider_cfg, target_body, model, stream=False
    )

    # 4. Forward to upstream
    client = _get_client()
    try:
        upstream_resp = await client.post(url, json=upstream_body, headers=headers)
    except httpx.HTTPError as exc:
        return _error_response_for_source(
            source_provider, 502, f"Upstream request failed: {exc}"
        )

    # 5. Pass through upstream errors
    if upstream_resp.status_code >= 400:
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )

    # 6. Target response -> IR
    try:
        upstream_json = upstream_resp.json()
        ir_response = target_converter.response_from_provider(upstream_json)
    except Exception as exc:
        return _error_response_for_source(
            source_provider, 502, f"Failed to parse upstream response: {exc}"
        )

    # 7. IR -> Source response
    try:
        source_response = source_converter.response_to_provider(ir_response)
    except Exception as exc:
        return _error_response_for_source(
            source_provider, 500, f"Failed to convert response: {exc}"
        )

    return JSONResponse(source_response)


async def _handle_streaming(
    source_provider: ProviderType,
    target_provider: ProviderType,
    provider_cfg: dict[str, str],
    body: dict[str, Any],
    model: str,
) -> Response:
    """Streaming proxy: convert -> forward -> stream-convert back -> SSE."""
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # 1. Source -> IR
    try:
        ir_request = source_converter.request_from_provider(body)
    except Exception as exc:
        return _error_response_for_source(
            source_provider, 400, f"Failed to parse request: {exc}"
        )

    # 2. IR -> Target
    try:
        target_body, warnings = target_converter.request_to_provider(ir_request)
    except Exception as exc:
        return _error_response_for_source(
            source_provider, 400, f"Conversion error: {exc}"
        )
    if warnings:
        logger.warning("Conversion warnings: %s", warnings)

    # 3. Build upstream request (with stream=True)
    url, headers, upstream_body = _prepare_upstream(
        target_provider, provider_cfg, target_body, model, stream=True
    )

    format_sse = _SSE_FORMATTERS[source_provider]

    async def event_generator() -> AsyncIterator[str]:
        """Stream SSE events from upstream, converting each chunk."""
        from_ctx = StreamContext()  # upstream -> IR
        to_ctx = StreamContext()  # IR -> source

        client = _get_client()
        async with client.stream(
            "POST", url, json=upstream_body, headers=headers
        ) as upstream_resp:
            if upstream_resp.status_code >= 400:
                # Read error body and yield as error
                await upstream_resp.aread()
                error_text = upstream_resp.text
                try:
                    error_body = json.loads(error_text)
                    error_msg = json.dumps(error_body)
                except json.JSONDecodeError:
                    error_msg = error_text
                yield f"data: {error_msg}\n\n"
                return

            async for line in upstream_resp.aiter_lines():
                parsed = _iter_sse_lines(line)
                if parsed is None:
                    continue
                field, value = parsed

                # Skip event-type lines (type info is inside the data JSON)
                if field == "event":
                    continue

                if field != "data" or value is None:
                    continue

                # OpenAI [DONE] signal
                if _is_openai_done(value):
                    break

                try:
                    chunk = json.loads(value)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed SSE data: %s", value[:200])
                    continue

                # Upstream chunk -> IR events
                ir_events = target_converter.stream_response_from_provider(
                    chunk, context=from_ctx
                )

                # IR events -> source-format chunks
                for ir_event in ir_events:
                    source_chunks = source_converter.stream_response_to_provider(
                        ir_event, context=to_ctx
                    )
                    if isinstance(source_chunks, list):
                        for sc in source_chunks:
                            if sc:
                                yield format_sse(sc)
                    elif source_chunks:
                        yield format_sse(source_chunks)

        # Emit end-of-stream marker for OpenAI Chat format
        if source_provider == "openai_chat":
            yield _format_sse_openai_chat_done()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

# Global config — set at startup
_config: GatewayConfig | None = None


async def _proxy_handler(
    request: Request,
    source_provider: ProviderType,
    model_override: str | None = None,
    force_stream: bool = False,
) -> Response:
    """Shared handler for all proxy endpoints."""
    assert _config is not None

    try:
        body = await request.json()
    except Exception:
        return _error_response_for_source(source_provider, 400, "Invalid JSON body")

    # Determine model
    model = model_override or _extract_model(source_provider, body)
    if not model:
        return _error_response_for_source(
            source_provider, 400, "Missing 'model' in request body"
        )

    # If model came from URL (Google), inject it into body for the converter
    if model_override and "model" not in body:
        body["model"] = model_override

    # Resolve target provider
    try:
        target_provider, provider_cfg = _config.resolve_model(model)
    except KeyError:
        configured = ", ".join(sorted(_config.models.keys()))
        return _error_response_for_source(
            source_provider,
            404,
            f"Unknown model: '{model}'. Configured models: {configured}",
        )

    # Determine streaming
    is_stream = force_stream or _detect_stream_request(source_provider, body)

    logger.info(
        "%s -> %s | model=%s stream=%s",
        source_provider,
        target_provider,
        model,
        is_stream,
    )

    if is_stream:
        return await _handle_streaming(
            source_provider, target_provider, provider_cfg, body, model
        )
    else:
        return await _handle_non_streaming(
            source_provider, target_provider, provider_cfg, body, model
        )


# --- Endpoint handlers ---


async def handle_openai_chat(request: Request) -> Response:
    return await _proxy_handler(request, source_provider="openai_chat")


async def handle_anthropic(request: Request) -> Response:
    return await _proxy_handler(request, source_provider="anthropic")


async def handle_openai_responses(request: Request) -> Response:
    return await _proxy_handler(request, source_provider="openai_responses")


async def handle_google(request: Request) -> Response:
    model = request.path_params["model"]
    return await _proxy_handler(request, source_provider="google", model_override=model)


async def handle_google_stream(request: Request) -> Response:
    model = request.path_params["model"]
    return await _proxy_handler(
        request,
        source_provider="google",
        model_override=model,
        force_stream=True,
    )


async def handle_health(request: Request) -> Response:
    assert _config is not None
    return JSONResponse(
        {
            "status": "ok",
            "providers": list(_config.providers.keys()),
            "models": list(_config.models.keys()),
        }
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: GatewayConfig) -> Starlette:
    """Create the Starlette ASGI application."""
    global _config
    _config = config

    routes = [
        Route("/v1/chat/completions", handle_openai_chat, methods=["POST"]),
        Route("/v1/messages", handle_anthropic, methods=["POST"]),
        Route("/v1/responses", handle_openai_responses, methods=["POST"]),
        Route(
            "/v1beta/models/{model}:generateContent",
            handle_google,
            methods=["POST"],
        ),
        Route(
            "/v1beta/models/{model}:streamGenerateContent",
            handle_google_stream,
            methods=["POST"],
        ),
        Route("/health", handle_health, methods=["GET"]),
    ]

    async def on_shutdown() -> None:
        global _http_client
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None

    return Starlette(routes=routes, on_shutdown=[on_shutdown])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="llm-rosetta Gateway")
    parser.add_argument(
        "--config",
        default="config.jsonc",
        help="Path to JSONC config file (default: config.jsonc)",
    )
    parser.add_argument("--host", default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    raw_config = load_config(args.config)
    config = GatewayConfig(raw_config)

    host = args.host or config.host
    port = args.port or config.port

    logger.info("Starting llm-rosetta gateway on %s:%d", host, port)
    logger.info("Configured providers: %s", list(config.providers.keys()))
    logger.info("Configured models: %s", list(config.models.keys()))

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
