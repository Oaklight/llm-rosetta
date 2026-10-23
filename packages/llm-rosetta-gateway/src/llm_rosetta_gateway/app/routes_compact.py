"""``POST /v1/responses/compact`` — context compaction, forwarded unchanged.

Compaction is the one Responses-family operation with no neutral form.  The
request is a subset of a Responses request, and the answer is an opaque,
encrypted ``compaction`` item that only the upstream which minted it can read
back: converting it would either mangle it or invent one.  So this route
forwards the body verbatim and relays the answer, marking both as unconverted
(``x-rosetta-conversion: passthrough``).

A target that does not speak the Responses API cannot serve it at all — there
is nothing to convert the request *to*, and no envelope to carry the item — so
the route answers 501 instead of forwarding a body the upstream would reject.

The upstream URL is derived from the target's own Responses endpoint (the
``/compact`` sibling of it) rather than from a second per-format path, so a
provider with a custom base path keeps working.
"""

from __future__ import annotations

import json
import time
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import (
    JSONResponse,
    Response,
    StreamingResponse,
)

from ..config import GatewayConfig
from ..lifecycle import (
    begin_typed,
    current_lifecycle,
    finish_typed,
    instrumented_handler,
)
from ..logging import get_logger
from ..middleware.auth import api_key_context_var
from ..middleware.headers import build_upstream_extra_headers, get_request_id
from ..transport import UpstreamConnectionError, UpstreamTimeoutError

from . import _state
from .routes_llm import _resolve_or_error

logger = get_logger()

SOURCE_FORMAT = "open_responses"


def _error(request_id: str, status: int, message: str, error_type: str) -> Response:
    """An error body in the Responses envelope, carrying the request id."""
    response = JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "code": None,
                "param": None,
            }
        },
        status_code=status,
    )
    response.headers["x-request-id"] = request_id
    return response


def _relay(resp: Any, request_id: str) -> Response:
    """Return the upstream answer as the client's, unchanged."""
    # The transport hands back the body, not its headers, so a relayed answer
    # gets a content type inferred from what it actually is: an error body the
    # transport could not parse as JSON (an HTML 502 from a proxy, say) is not
    # relabelled as JSON.
    content_type = "application/json"
    if resp.body is None:
        try:
            json.loads(resp.raw_content or b"")
        except ValueError:
            content_type = "text/plain; charset=utf-8"
    response = Response(
        body=resp.raw_content,
        status_code=resp.status_code,
        content_type=content_type,
    )
    response.headers["x-request-id"] = request_id
    response.headers["x-rosetta-conversion"] = "passthrough"
    return response


def _usage(body: dict[str, Any] | None) -> dict[str, Any]:
    """The counters a compaction answer reports, in IR usage shape.

    The route has no response conversion to produce IR usage, so the mapping
    the Responses reader applies to a normal answer is applied here instead.
    It is four fields, and the ledger reads IR names (``prompt_tokens`` /
    ``completion_tokens``): a relayed answer that recorded only its total
    would understate what the call cost.
    """
    usage = (body or {}).get("usage")
    if not isinstance(usage, dict):
        return {}
    converted: dict[str, Any] = {
        "prompt_tokens": usage.get("input_tokens") or 0,
        "completion_tokens": usage.get("output_tokens") or 0,
        "total_tokens": usage.get("total_tokens") or 0,
    }
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict) and "cached_tokens" in input_details:
        converted["cache_read_tokens"] = input_details["cached_tokens"]
    output_details = usage.get("output_tokens_details")
    if isinstance(output_details, dict) and "reasoning_tokens" in output_details:
        converted["reasoning_tokens"] = output_details["reasoning_tokens"]
    return converted


async def handle_responses_compact(request: Any, config: GatewayConfig) -> Response:
    """Serve one compaction request."""
    request_id = get_request_id(request)

    try:
        body = request.json()
    except Exception:
        return _error(request_id, 400, "Invalid JSON body", "invalid_request_error")
    if not isinstance(body, dict):
        return _error(request_id, 400, "Invalid JSON body", "invalid_request_error")

    model = body.get("model")
    if not model:
        return _error(
            request_id,
            400,
            "Missing 'model' in request body",
            "invalid_request_error",
        )

    try:
        selection = config.selection_context(model, SOURCE_FORMAT)
    except KeyError:
        # Same wording as the chat route's 404 (``_resolve_or_error``): the
        # model is resolved here rather than in ``config.resolve``.
        configured = ", ".join(sorted(config.models.keys()))
        return _error(
            request_id,
            404,
            f"Unknown model: '{model}'. Configured models: {configured}",
            "model_not_found",
        )
    key_ctx = api_key_context_var.get()
    selection.with_client(key_ctx.key_hash if key_ctx else None)

    resolved = _resolve_or_error(config, selection, request_id)
    if isinstance(resolved, Response):
        return resolved
    route, provider_info = resolved

    ctx = current_lifecycle()

    # The format check comes *before* the breaker is acquired: a 501 here is
    # this route refusing, not the provider failing, and the lifecycle counts
    # any 5xx that passed the breaker as a provider failure.  Charging two
    # refusals to a provider that never saw a request would open its breaker
    # and take its other endpoints out with it.
    if route.target_provider != SOURCE_FORMAT:
        if ctx is not None:
            ctx.note_route(
                model=model,
                target_format=route.target_provider,
                provider_name=route.provider_name,
                is_stream=False,
            )
        return _error(
            request_id,
            501,
            "context compaction needs an open_responses upstream; "
            f"provider '{route.provider_name}' speaks {route.target_provider}",
            "not_implemented",
        )

    rejection = begin_typed(
        request,
        source_format=SOURCE_FORMAT,
        target_format=route.target_provider,
        provider_name=route.provider_name,
    )
    if rejection is not None:
        return rejection

    upstream_model = route.upstream_model or model
    upstream_body = dict(body)
    upstream_body["model"] = upstream_model

    # The `/compact` sibling of the target's own Responses endpoint.
    upstream_url = provider_info.upstream_url(upstream_model).rstrip("/") + "/compact"
    extra_headers = build_upstream_extra_headers(
        request, request_id, session_header=provider_info.session_header
    )

    if ctx is not None:
        ctx.note_route(model=model, is_stream=False)

    started = time.monotonic()
    status_code = 500
    error_detail: str | None = None
    usage: dict[str, Any] = {}
    try:
        resp = await request.app.transport.send(
            provider_info,
            upstream_url,
            upstream_body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code
        usage = _usage(resp.body)
        if resp.is_error:
            error_detail = resp.error_text
            logger.info(
                "[%s] compact upstream error %s: %s",
                request_id,
                status_code,
                (error_detail or "")[:200],
            )
        return _relay(resp, request_id)
    except UpstreamTimeoutError as exc:
        status_code = 504
        error_detail = f"Upstream timeout: {exc}"
        return _error(request_id, 504, error_detail, "upstream_error")
    except UpstreamConnectionError as exc:
        status_code = 502
        error_detail = f"Upstream connection failed: {exc}"
        return _error(request_id, 502, error_detail, "upstream_error")
    finally:
        await finish_typed(
            request,
            model=model,
            source_provider=SOURCE_FORMAT,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=status_code,
            duration_ms=(time.monotonic() - started) * 1000,
            error_detail=error_detail,
            model_type="llm",
            usage=usage or None,
        )


async def route_responses_compact(request: Any) -> Response | StreamingResponse:
    """The router entry point: the same handler under the request lifecycle."""
    assert _state._config is not None

    async def _handle(request: Any, config: GatewayConfig) -> Response:
        return await handle_responses_compact(request, config)

    return await instrumented_handler(
        request,
        _state._config,
        _handle,
        source_format=SOURCE_FORMAT,
        error_phase="conversion",
        error_format=SOURCE_FORMAT,
    )
