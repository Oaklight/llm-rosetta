"""``POST /v1/messages/count_tokens`` — exact counting, delegated upstream.

The Anthropic Messages API defines a counting endpoint beside the completion
one, and clients written against it (Claude Code among them) call it before
sending a request.  There is no IR form of a token count, so the request is
forwarded unchanged — bar the model name — and the answer is relayed verbatim,
marked ``x-rosetta-conversion: passthrough``.

The body must not be massaged on the way: Anthropic's endpoint accepts exactly
the Messages request fields minus the generation-only ones, and rejects the
rest (``max_tokens: Extra inputs are not permitted``, probed live 2026-11-06).
Dropping those fields here would diverge from the contract the client is
coding against, so a body the upstream refuses is relayed with its own 400.

A target that does not speak Messages has no counting endpoint to delegate to
and nothing to convert a count into, so the route answers 501 rather than
forwarding a body the upstream would reject.  The upstream URL is the
``/count_tokens`` sibling of the target's own Messages endpoint, so a provider
with a custom base path or a per-model URL template keeps working.
"""

from __future__ import annotations

import json
import time
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import (
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
from ..proxy import error_response_for_source
from ..transport import UpstreamConnectionError, UpstreamTimeoutError

from . import _state
from .routes_llm import _resolve_or_error

logger = get_logger()

SOURCE_FORMAT = "messages"

#: Client-facing route.  Registered beside ``/v1/responses/compact`` in the
#: app factory rather than in the format catalog: the catalog carries one
#: client route per ``(model_type, format)`` and the llm registration selects a
#: handler by ``source_format``, so a second route for ``messages`` cannot be
#: expressed there without breaking that invariant.
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"


def _error(request_id: str, status: int, message: str) -> Response:
    """An error body in the Messages envelope, carrying the request id."""
    response = error_response_for_source(SOURCE_FORMAT, status, message)
    response.headers["x-request-id"] = request_id
    return response


def _relay(resp: Any, request_id: str) -> Response:
    """Return the upstream answer as the client's, unchanged."""
    # The transport hands back the body, not its headers.  An error body it
    # could not parse as JSON (an HTML 502 from a proxy, say) must not be
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


async def handle_messages_count_tokens(request: Any, config: GatewayConfig) -> Response:
    """Serve one token-count request."""
    request_id = get_request_id(request)

    try:
        body = request.json()
    except Exception:
        return _error(request_id, 400, "Invalid JSON body")
    if not isinstance(body, dict):
        return _error(request_id, 400, "Invalid JSON body")

    model = body.get("model")
    if not model:
        return _error(request_id, 400, "Missing 'model' in request body")

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
        )
    key_ctx = api_key_context_var.get()
    selection.with_client(key_ctx.key_hash if key_ctx else None)
    # A model with rules needs the request in IR form before the chain runs;
    # one without them never pays for the extra read leg.
    from .routes_llm import _entry_capabilities, _model_rules, _rule_facts

    if _model_rules(config, model) or _entry_capabilities(config, model):
        from ..routing import required_capabilities

        facts = _rule_facts(SOURCE_FORMAT, body, request)
        selection.with_rule_facts(facts)
        selection.with_required_capabilities(required_capabilities(facts))

    resolved = _resolve_or_error(config, selection, request_id)
    if isinstance(resolved, Response):
        return resolved
    route, provider_info = resolved

    ctx = current_lifecycle()

    # The format check comes before the breaker is acquired: a 501 here is
    # this route refusing, not the provider failing, and the lifecycle counts
    # any 5xx that passed the breaker as a provider failure.
    if route.target_provider != SOURCE_FORMAT:
        if ctx is not None:
            ctx.note_route(
                model=model,
                target_format=route.target_provider,
                provider_name=route.provider_name,
                upstream_model=route.upstream_model,
                is_stream=False,
            )
        return _error(
            request_id,
            501,
            "token counting needs a messages upstream; "
            f"provider '{route.provider_name}' speaks {route.target_provider}",
        )

    rejection = begin_typed(
        request,
        source_format=SOURCE_FORMAT,
        target_format=route.target_provider,
        provider_name=route.provider_name,
        upstream_model=route.upstream_model,
    )
    if rejection is not None:
        return rejection

    upstream_model = route.upstream_model or model
    upstream_body = dict(body)
    upstream_body["model"] = upstream_model

    # The ``/count_tokens`` sibling of the target's own Messages endpoint.
    upstream_url = (
        provider_info.upstream_url(upstream_model).rstrip("/") + "/count_tokens"
    )
    extra_headers = build_upstream_extra_headers(
        request, request_id, session_header=provider_info.session_header
    )

    if ctx is not None:
        # The lane is already recorded: ``begin_typed`` notes it with the
        # provider, before the breaker, so a rejected request carries it too.
        ctx.note_route(model=model, is_stream=False)

    started = time.monotonic()
    status_code = 500
    error_detail: str | None = None
    try:
        resp = await request.app.transport.send(
            provider_info,
            upstream_url,
            upstream_body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code
        if resp.is_error:
            error_detail = resp.error_text
            logger.info(
                "[%s] count_tokens upstream error %s: %s",
                request_id,
                status_code,
                (error_detail or "")[:200],
            )
        return _relay(resp, request_id)
    except UpstreamTimeoutError as exc:
        status_code = 504
        error_detail = f"Upstream timeout: {exc}"
        return _error(request_id, 504, error_detail)
    except UpstreamConnectionError as exc:
        status_code = 502
        error_detail = f"Upstream connection failed: {exc}"
        return _error(request_id, 502, error_detail)
    finally:
        # A count is not a completion: nothing was spent, so the ledger records
        # the request and its status but no usage.
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
            usage=None,
        )


async def route_messages_count_tokens(request: Any) -> Response | StreamingResponse:
    """The router entry point: the same handler under the request lifecycle."""
    assert _state._config is not None

    async def _handle(request: Any, config: GatewayConfig) -> Response:
        return await handle_messages_count_tokens(request, config)

    return await instrumented_handler(
        request,
        _state._config,
        _handle,
        source_format=SOURCE_FORMAT,
        error_phase="conversion",
        error_format=SOURCE_FORMAT,
    )
