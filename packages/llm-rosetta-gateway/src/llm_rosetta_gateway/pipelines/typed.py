"""The shared request flow for the typed (non-LLM) model types.

Embedding, rerank and decision each own an IR conversion pipeline, but the
handler around it is the same shape: parse the body, resolve the provider,
convert the request, forward it, convert the response back, and record the
request in the usage ledger.  A model type supplies the hooks in
:class:`TypedHandler`; nothing else differs, so a newly registered type gets a
handler by filling one in rather than by copying 280 lines.

Requests rejected before the upstream call — bad JSON, an unknown model, a
failed request conversion — are recorded by the shared request lifecycle with
no token usage: nothing was billed.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ..config import GatewayConfig, TypedRoute
from ..lifecycle import begin_typed, finish_typed
from ..logging import get_logger
from ..middleware.headers import build_upstream_extra_headers, get_request_id
from ..transport import UpstreamConnectionError, UpstreamTimeoutError, UpstreamTransport
from ._fidelity import check_same_format_fidelity

logger = get_logger()


def detect_source_format(
    request: Any, body: dict[str, Any], config: GatewayConfig, type_name: str
) -> str:
    """The request's source format, from the type descriptor's detector.

    Falls back to the type's configured default (``default_<type>_format``)
    when the registry has no detector for it — a test double, or a type that
    has not declared one.
    """
    from ..model_types import get_model_type

    descriptor = get_model_type(type_name)
    if descriptor is not None and descriptor.detect_source_format is not None:
        return descriptor.detect_source_format(request, body, config)
    return getattr(config, f"default_{type_name}_format")


@dataclass(frozen=True)
class TypedHandler:
    """What one typed model type supplies to the shared request flow.

    Attributes:
        model_type: Registry category (``embedding``, ``rerank``, …); it names
            the type in logs and in the usage ledger.
        build_pipeline: ``(source_format, target_format) -> pipeline``.
        resolve: ``(config, model) -> route``, or ``None`` when the model is
            not configured for this type.
        configured_models: The model names to list in the 404 message.
        response_usage: ``(format, body) -> usage dict`` for the ledger.
        not_found_label: The 404 message reads ``Unknown <label>: '…'``.
        prepare_body: Mutate the parsed body before conversion (e.g. rewrite
            ``model`` to the provider's upstream name).
        upstream_url: The upstream URL, computed after conversion — Google
            rewrites its path from the converted body.  Defaults to the
            connection's own URL.
        response_post: Final touch on the converted response (e.g. echo the
            client's model name).
        connection_error_message: Wording of the 502 body.
        empty_body_is_passthrough: An empty upstream body becomes a
            passthrough response carrying ``x-rosetta-conversion``; rerank
            does this, embedding and decision return the raw body instead.
        log_conversion: Log the per-request line and any conversion warnings.
    """

    model_type: str
    build_pipeline: Callable[[str, str], Any]
    resolve: Callable[[GatewayConfig, str], TypedRoute | None]
    configured_models: Callable[[GatewayConfig], Iterable[str]]
    response_usage: Callable[[str, dict[str, Any] | None], dict[str, Any]]
    not_found_label: str = "model"
    prepare_body: Callable[[dict[str, Any], str, GatewayConfig], None] | None = None
    upstream_url: (
        Callable[[TypedRoute, dict[str, Any], str, GatewayConfig], str] | None
    ) = None
    response_post: Callable[[dict[str, Any], str], None] | None = None
    connection_error_message: str = "Upstream connection failed"
    empty_body_is_passthrough: bool = False
    log_conversion: bool = True


def _error(message: str, error_type: str, status_code: int) -> JSONResponse:
    """An error body in the gateway's own envelope."""
    return JSONResponse(
        {"error": {"message": message, "type": error_type}},
        status_code=status_code,
    )


def _passthrough(response: Response) -> Response:
    """Mark *response* as forwarded unconverted."""
    response.headers["x-rosetta-conversion"] = "passthrough"
    return response


def _parse_and_resolve(
    request: Any, config: GatewayConfig, handler: TypedHandler
) -> tuple[dict[str, Any], str, TypedRoute] | JSONResponse:
    """Parse the body and resolve the model, or return the error response."""
    try:
        body: dict[str, Any] = request.json()
    except Exception:
        return _error("Invalid JSON body", "invalid_request_error", 400)

    model = body.get("model")
    if not model:
        return _error("Missing 'model' in request body", "invalid_request_error", 400)

    route = handler.resolve(config, model)
    if route is None:
        configured = ", ".join(sorted(handler.configured_models(config)))
        return _error(
            f"Unknown {handler.not_found_label}: '{model}'. "
            f"Configured: {configured or '(none)'}",
            "model_not_found",
            404,
        )
    return body, model, route


def _converted_response(
    resp: Any,
    pipeline: Any,
    handler: TypedHandler,
    *,
    same_format: bool,
    source_format: str,
    target_format: str,
    model: str,
) -> Response | None:
    """Convert a successful upstream body, or ``None`` to pass it through.

    A conversion failure is not an error: the raw upstream body is returned to
    the client with ``x-rosetta-conversion: passthrough``, the same contract the
    chat routes use.
    """
    try:
        original_response = copy.deepcopy(resp.body) if same_format else None
        source_body = pipeline.convert_response(resp.body)
    except Exception as exc:
        logger.warning("%s: response conversion failed: %s", handler.model_type, exc)
        return None
    check_same_format_fidelity(
        handler.model_type,
        source_format,
        target_format,
        original_response,
        source_body,
        "response",
        model=model,
    )
    if handler.response_post is not None:
        handler.response_post(source_body, model)
    return JSONResponse(source_body, status_code=200)


async def handle_typed_request(
    request: Any, config: GatewayConfig, handler: TypedHandler
) -> Response:
    """Run one typed (non-LLM) request through the shared flow."""
    request_id = get_request_id(request)

    def with_request_id(response: Response) -> Response:
        response.headers["x-request-id"] = request_id
        return response

    parsed = _parse_and_resolve(request, config, handler)
    if isinstance(parsed, JSONResponse):
        return with_request_id(parsed)
    body, model, route = parsed

    source_format = detect_source_format(request, body, config, handler.model_type)
    logger.debug("%s: detected source format: %s", handler.model_type, source_format)
    rejection = begin_typed(
        request,
        source_format=source_format,
        target_format=route.format,
        provider_name=route.provider_name,
    )
    if rejection is not None:
        return rejection

    if handler.prepare_body is not None:
        handler.prepare_body(body, model, config)
    # The model the upstream URL is built from; a rename map or the conversion
    # itself may have changed it.
    upstream_model = body.get("model") or model

    pipeline = handler.build_pipeline(source_format, route.format)
    same_format = source_format == route.format
    original_body = copy.deepcopy(body) if same_format else None
    try:
        target_body = pipeline.convert_request(body)
    except Exception as exc:
        logger.warning("%s: request conversion failed: %s", handler.model_type, exc)
        return with_request_id(
            _error(f"Request conversion failed: {exc}", "conversion_error", 400)
        )
    check_same_format_fidelity(
        handler.model_type,
        source_format,
        route.format,
        original_body,
        target_body,
        "request",
        model=model,
    )

    if handler.upstream_url is not None:
        upstream_url = handler.upstream_url(route, target_body, upstream_model, config)
    else:
        upstream_url = route.provider_info.upstream_url("")

    transport: UpstreamTransport = request.app.transport
    extra_headers = build_upstream_extra_headers(request, request_id)

    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None
    usage: dict[str, Any] = {}

    try:
        resp = await transport.send(
            route.provider_info,
            upstream_url,
            target_body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code
        # Read even on an error status: a 4xx can still carry the tokens the
        # upstream counted before it refused.
        usage = handler.response_usage(route.format, resp.body)

        if resp.is_error:
            error_detail = resp.error_text
            return with_request_id(
                Response(
                    body=resp.raw_content,
                    status_code=resp.status_code,
                    content_type="application/json",
                )
            )

        if resp.body is None:
            raw = with_request_id(
                Response(
                    body=resp.raw_content,
                    status_code=200,
                    content_type="application/json",
                )
            )
            return _passthrough(raw) if handler.empty_body_is_passthrough else raw

        converted = _converted_response(
            resp,
            pipeline,
            handler,
            same_format=same_format,
            source_format=source_format,
            target_format=route.format,
            model=model,
        )
        if converted is not None:
            return with_request_id(converted)
        return with_request_id(
            _passthrough(
                Response(
                    body=resp.raw_content,
                    status_code=200,
                    content_type="application/json",
                )
            )
        )

    except UpstreamTimeoutError as exc:
        status_code = 504
        error_detail = f"Upstream timeout: {exc}"
        return with_request_id(
            _error(f"Upstream timeout: {exc}", "upstream_error", 504)
        )
    except UpstreamConnectionError as exc:
        status_code = 502
        error_detail = f"{handler.connection_error_message}: {exc}"
        return with_request_id(_error(error_detail, "upstream_error", 502))
    finally:
        duration_ms = (time.monotonic() - t0) * 1000
        if handler.log_conversion:
            if pipeline.warnings:
                logger.info(
                    "%s: conversion warnings for %s: %s",
                    handler.model_type,
                    model,
                    "; ".join(pipeline.warnings),
                )
            logger.info(
                "%s: %s → %s (%s→%s) %dms status=%d",
                handler.model_type,
                model,
                route.provider_name,
                source_format,
                route.format,
                duration_ms,
                status_code,
            )
        await finish_typed(
            request,
            model=model,
            source_provider=source_format,
            target_provider=route.format,
            provider_name=route.provider_name,
            status_code=status_code,
            duration_ms=duration_ms,
            error_detail=error_detail,
            model_type=handler.model_type,
            usage=usage,
        )


__all__ = ["TypedHandler", "detect_source_format", "handle_typed_request"]
