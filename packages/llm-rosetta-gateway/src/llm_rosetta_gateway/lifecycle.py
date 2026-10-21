"""One instrumented request lifecycle for chat and typed routes.

Every proxy route — the five chat formats and the typed endpoints
(embedding, rerank, decision) — runs through :func:`instrumented_handler`.
It owns what is identical for all of them and easy to forget in one copy:
the request id, timing, the request-log/metrics record, deep profiling,
error dumps (queued until the log row exists, #964) and the circuit breaker.
A route supplies only its own work in ``handler`` and reports what it learned
about the route on the :class:`LifecycleContext`.

The context is also published on the ``gateway_lifecycle`` context variable for
the duration of the handler.  A typed pipeline resolves its provider itself, so
it calls :func:`begin_typed` after resolution to record the route and consult
the breaker, and :func:`finish_typed` to hand its usage to the ledger — all
without changing its public ``(request, config)`` signature.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from llm_rosetta._vendor.httpserver import Response, StreamingResponse
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.observability.error_dump import dump_error
from llm_rosetta.observability.usage.normalize import DEFAULT_MODEL_TYPE

from .logging import get_logger
from .maintenance.dumps import DeferredDumpStore
from .middleware.circuit_breaker import CircuitBreaker
from .middleware.error_format import detect_api_format, format_error_response
from .middleware.headers import get_request_id
from .middleware.request_context import request_context_var
from .ops.proxy import record_proxy_request

if TYPE_CHECKING:
    from .config import GatewayConfig

logger = get_logger()

_lifecycle_var: ContextVar[LifecycleContext | None] = ContextVar(
    "llm_rosetta_gateway_lifecycle", default=None
)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class LifecycleContext:
    """What the lifecycle and the route handler tell each other.

    The lifecycle fills the request-level fields; the handler updates the
    routing fields as it learns them and hands over its usage at the end.
    """

    config: GatewayConfig
    app: Any
    request_id: str
    entry_id: str
    model_type: str
    body: dict[str, Any] | None
    model: str = ""
    source_format: str = ""
    target_format: str = ""
    provider_name: str = ""
    is_stream: bool = False
    status_code: int = 500
    error_detail: str | None = None
    usage: Mapping[str, Any] | None = None
    profile: dict[str, Any] | None = None
    dump_store: DeferredDumpStore | None = None
    _breaker: CircuitBreaker | None = field(default=None, repr=False)
    _breaker_rejected: bool = field(default=False, repr=False)

    def note_route(
        self,
        *,
        source_format: str | None = None,
        target_format: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        is_stream: bool | None = None,
    ) -> None:
        """Record routing facts the handler has resolved.

        Only the arguments that are not ``None`` are applied, so a caller can
        fill in the model first and the target provider later.
        """
        if source_format is not None:
            self.source_format = source_format
        if target_format is not None:
            self.target_format = target_format
        if provider_name is not None:
            self.provider_name = provider_name
        if model is not None:
            self.model = model
        if is_stream is not None:
            self.is_stream = is_stream

    def note_result(
        self,
        *,
        status_code: int | None = None,
        error_detail: str | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the outcome facts the lifecycle cannot derive itself."""
        if status_code is not None:
            self.status_code = status_code
        if error_detail is not None:
            self.error_detail = error_detail
        if usage is not None:
            self.usage = usage

    def check_circuit_breaker(self, request: Any) -> Response | None:
        """Return a 503 when this route's breaker is open, else ``None``.

        A rejected request must not be counted as a failure again — that is
        what keeps a failing upstream's cooldown from growing while the
        breaker is doing its job — so the lifecycle records an outcome only
        when the request was allowed through.
        """
        if not self.provider_name:
            return None
        registry = self.config.circuit_breaker_registry
        self._breaker = registry.get_or_create(self.provider_name)
        if self._breaker.allow_request():
            return None

        self._breaker_rejected = True
        remaining = self._breaker.cooldown_remaining()
        rctx = request_context_var.get()
        api_format = (
            rctx.api_format
            if rctx and rctx.api_format
            else detect_api_format(request.path)
        )
        resp = format_error_response(
            api_format,
            503,
            (
                f"Provider '{self.provider_name}' is temporarily unavailable "
                f"(circuit breaker open, cooldown {remaining:.0f}s remaining)"
            ),
            error_type="service_unavailable",
            google_status="UNAVAILABLE",
            cors=True,
        )
        resp.headers["x-request-id"] = self.request_id
        resp.headers["Retry-After"] = str(max(1, int(remaining)))
        return resp


def current_lifecycle() -> LifecycleContext | None:
    """Return the lifecycle context for the current request, if any.

    ``None`` when a handler is called directly (tests, embedding the handler
    in another server) rather than through :func:`instrumented_handler`.
    """
    return _lifecycle_var.get()


# ---------------------------------------------------------------------------
# Typed-route helpers
# ---------------------------------------------------------------------------


def begin_typed(
    request: Any,
    *,
    source_format: str,
    target_format: str,
    provider_name: str,
) -> Response | None:
    """Record a typed route's resolved provider and apply the breaker.

    Returns the 503 response when the breaker is open, else ``None``.  A
    handler called outside the lifecycle (no context) is unaffected.
    """
    ctx = current_lifecycle()
    if ctx is None:
        return None
    ctx.note_route(
        source_format=source_format,
        target_format=target_format,
        provider_name=provider_name,
    )
    return ctx.check_circuit_breaker(request)


async def finish_typed(
    request: Any,
    *,
    model: str,
    source_provider: str,
    target_provider: str,
    provider_name: str,
    status_code: int,
    duration_ms: float,
    error_detail: str | None,
    model_type: str,
    usage: Mapping[str, Any] | None,
) -> None:
    """Hand a typed handler's result to the one ledger writer.

    Through the lifecycle the result is attached to the context and written
    once, by the lifecycle, after the request-log row's id is known.  Called
    directly, the handler writes its own row as before.
    """
    ctx = current_lifecycle()
    if ctx is not None:
        ctx.note_result(status_code=status_code, error_detail=error_detail, usage=usage)
        return
    await record_proxy_request(
        request.app,
        model=model,
        source_provider=source_provider,
        target_provider=target_provider,
        provider_name=provider_name,
        is_stream=False,
        status_code=status_code,
        duration_ms=duration_ms,
        error_detail=error_detail,
        model_type=model_type,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def try_start_profiler(app: Any) -> Any | None:
    """Start a per-request deep profiler if profiling is enabled."""
    state = getattr(app, "profiler_state", None)
    if state is None or not state.should_profile():
        return None
    try:
        profiler = state.create_profiler()
        profiler.start()
        return profiler
    except Exception:
        # cProfile on Python 3.12+ raises ValueError when a second profiler
        # is started concurrently (tool-ID collision).  Restore the consumed
        # slot so the remaining count stays accurate.
        state.remaining += 1
        if not state.enabled:
            state.enabled = True
        state.release()
        return None


def try_stop_profiler(
    profiler: Any,
    app: Any,
    *,
    request_id: str,
    model: str,
    source: str,
    target: str,
    is_stream: bool,
    duration_ms: float,
) -> None:
    """Stop a running deep profiler and store the result."""
    if profiler is None:
        return
    state = getattr(app, "profiler_state", None)
    try:
        profiler.stop()
        if state is not None:
            state.store_result(
                profiler,
                request_id=request_id,
                model=model,
                source=source,
                target=target,
                is_stream=is_stream,
                duration_ms=duration_ms,
            )
    except Exception:
        logger.debug("Failed to store profiling result")
    finally:
        if state is not None:
            state.release()


def _stop_profiler(
    profiler: Any,
    ctx: LifecycleContext,
    response: Response | StreamingResponse | None,
    duration_ms: float,
) -> None:
    """Stop the profiler, or defer it to the end of a streamed response."""
    if profiler is None:
        return

    def _stop(is_stream: bool) -> None:
        try_stop_profiler(
            profiler,
            ctx.app,
            request_id=ctx.request_id,
            model=ctx.model,
            source=ctx.source_format,
            target=ctx.target_format,
            is_stream=is_stream,
            duration_ms=duration_ms,
        )

    if isinstance(response, StreamingResponse):
        # The body has not been sent yet: stop only when the generator ends.
        response.background = lambda: _stop(True)
        return
    _stop(ctx.is_stream)


# ---------------------------------------------------------------------------
# The lifecycle
# ---------------------------------------------------------------------------


async def instrumented_handler(
    request: Any,
    config: GatewayConfig,
    handler: Callable[[Any, GatewayConfig], Awaitable[Response | StreamingResponse]],
    *,
    source_format: str,
    model_type: str = DEFAULT_MODEL_TYPE,
    error_phase: str = "handler",
    error_format: ProviderType | None = None,
    provider_name: str = "",
) -> Response | StreamingResponse:
    """Run *handler* inside the shared instrumentation.

    Args:
        request: The incoming request.
        config: The served gateway config.
        handler: The route's own work, ``(request, config) -> Response``.
        source_format: The source format to record before the handler resolves
            the real one (the chat route's format, or the typed type's name).
        model_type: The ledger's model type (``llm``, ``embedding``, ...).
        error_phase: ``error_phase`` for an error dump of an unhandled
            exception.
        error_format: Format of the 500 response; defaults to the context's
            source format.
        provider_name: Provider to record before the handler resolves one.
    """
    app = request.app
    rctx = request_context_var.get()
    request_id = rctx.request_id if rctx else get_request_id(request)
    body = _request_body(request)
    ctx = LifecycleContext(
        config=config,
        app=app,
        request_id=request_id,
        entry_id=uuid.uuid4().hex,
        model_type=model_type,
        body=body,
        model=_body_model(body),
        source_format=source_format,
        target_format=source_format,
        provider_name=provider_name,
    )
    ctx.dump_store = _deferred_dumps_for(
        getattr(app, "persistence", None) if config.error_dumps_enabled else None
    )
    request.state.log_entry_id = ctx.entry_id

    token = _lifecycle_var.set(ctx)
    profiler = try_start_profiler(app)
    t0 = time.monotonic()
    response: Response | StreamingResponse | None = None
    try:
        response = await handler(request, config)
        ctx.status_code = response.status_code
        if response.status_code >= 400:
            ctx.error_detail = ctx.error_detail or _response_error_text(response)
        response.headers["x-request-id"] = request_id
        logger.info("[%s] response status=%s", request_id, ctx.status_code)
        return response
    except Exception as exc:
        return await _handle_unexpected(
            ctx, exc, error_format=error_format, error_phase=error_phase
        )
    finally:
        _lifecycle_var.reset(token)
        duration_ms = (time.monotonic() - t0) * 1000
        _stop_profiler(profiler, ctx, response, duration_ms)
        _record_breaker_outcome(ctx)
        try:
            await record_proxy_request(
                app,
                model=ctx.model,
                source_provider=ctx.source_format,
                target_provider=ctx.target_format,
                provider_name=ctx.provider_name,
                is_stream=ctx.is_stream,
                status_code=ctx.status_code,
                duration_ms=duration_ms,
                error_detail=ctx.error_detail,
                model_type=ctx.model_type,
                profile=ctx.profile,
                usage=ctx.usage,
                entry_id_override=ctx.entry_id,
            )
        finally:
            # The log row exists now, so dumps queued during handling can be
            # written with their request_log_id intact (#964).  In a finally
            # so a failing log write cannot swallow them.
            if ctx.dump_store is not None:
                await ctx.dump_store.flush()


async def _handle_unexpected(
    ctx: LifecycleContext,
    exc: BaseException,
    *,
    error_format: ProviderType | None,
    error_phase: str,
) -> Response:
    """Log, dump and answer an unhandled exception from a route handler."""
    from .proxy import error_response_for_source

    ctx.status_code = 500
    ctx.error_detail = str(exc)
    logger.exception(
        "[%s] unhandled error in %s handler", ctx.request_id, ctx.model_type
    )
    await dump_error(
        ctx.dump_store,
        request_body=ctx.body,
        response_text=ctx.error_detail,
        model=ctx.model or None,
        source_provider=ctx.source_format or None,
        target_provider=ctx.target_format or None,
        provider_name=ctx.provider_name or None,
        status_code=500,
        error_phase=error_phase,
        request_log_id=ctx.entry_id,
    )
    error_source = error_format or cast(ProviderType, ctx.source_format)
    resp = error_response_for_source(
        error_source or "chat_completions",
        500,
        f"Internal server error: {exc}",
    )
    resp.headers["x-request-id"] = ctx.request_id
    return resp


def _record_breaker_outcome(ctx: LifecycleContext) -> None:
    """Record success/failure once per request the breaker let through."""
    if ctx._breaker is None or ctx._breaker_rejected:
        return
    if ctx.status_code >= 500:
        ctx._breaker.record_failure()
    else:
        ctx._breaker.record_success()


def _request_body(request: Any) -> dict[str, Any] | None:
    """Parse the request body, returning ``None`` when it is not a dict."""
    try:
        body = request.json()
    except Exception:
        return None
    return body if isinstance(body, dict) else None


def _body_model(body: dict[str, Any] | None) -> str:
    """Best-effort model name before the handler resolves the real one."""
    if body is None:
        return ""
    model = body.get("model")
    return model if isinstance(model, str) else ""


def _response_error_text(response: Response | StreamingResponse) -> str | None:
    """Decode an error response's body for the request log."""
    body_bytes = getattr(response, "body", None)
    if isinstance(body_bytes, bytes):
        return body_bytes.decode("utf-8", errors="replace")
    return None


def _deferred_dumps_for(persistence: Any | None) -> DeferredDumpStore | None:
    """Wrap *persistence* so dump writes wait for the request-log row (#964)."""
    if persistence is None:
        return None
    return DeferredDumpStore(persistence)
