"""Requests refused before they reached a route handler.

A request the auth or rate-limit middleware rejects, that matches no route, or
that the server refuses while still reading it, never enters
:func:`~llm_rosetta_gateway.lifecycle.instrumented_handler`, so nothing wrote
its ``request_log`` row (#1217).  :class:`OpsRejectedRequest` records one,
through the same row builder as
:class:`~llm_rosetta_gateway.ops.proxy.OpsProxyRequest`.

It writes the row only: a rejection has no provider, and feeding a blank one
to the provider-keyed metrics would add a nameless entry to every breakdown.
"""

from __future__ import annotations

import re
import time
from functools import cache
from types import SimpleNamespace
from typing import Any, ClassVar

from ..logging import get_logger
from ..middleware.error_format import is_admin_path
from ..middleware.headers import get_request_id
from ..middleware.request_context import extract_client_ip, request_context_var
from ..model_types import get_model_type
from ..observability.sanitize import scrub_credential_patterns
from .base import OpsContext
from .proxy import OpsProxyRequest

logger = get_logger()

_UNRECORDED_PATHS = frozenset({"/", "/favicon.ico", "/.well-known/change-password"})
_MAX_DETAIL_CHARS = 1000
_warned_without_ops_context = False


def _ops_context(app: Any) -> OpsContext | None:
    """The app's ops context, warning once when there is none.

    An embedder that skips the admin setup has no ops context, and every
    rejection this module exists to record would vanish without a trace.
    """
    global _warned_without_ops_context
    ctx = getattr(app, "ops_ctx", None)
    if ctx is None and not _warned_without_ops_context:
        _warned_without_ops_context = True
        logger.warning(
            "Rejected requests are not being recorded: the app has no ops "
            "context (was the admin setup skipped?)"
        )
    return ctx


class OpsRejectedRequest(OpsProxyRequest):
    """Record a request that was answered without reaching a route handler."""

    event_type: ClassVar[str] = "rejected_request"

    __slots__ = ()

    def __init__(
        self,
        ctx: OpsContext,
        *,
        status_code: int,
        duration_ms: float,
        error_detail: str,
        request_id: str,
        client_ip: str | None,
        source_provider: str = "",
    ) -> None:
        super().__init__(
            ctx,
            model="",
            source_provider=source_provider,
            target_provider="",
            provider_name="",
            is_stream=False,
            status_code=status_code,
            duration_ms=duration_ms,
            error_detail=scrub_credential_patterns(error_detail),
            request_id=request_id,
            client_ip=client_ip,
        )

    async def _record(self, result: Any, *, error: BaseException | None = None) -> None:
        await self._write_log_row({})


@cache
def _route_pattern(template: str) -> re.Pattern[str]:
    """A route template such as ``/v1beta/models/<path:model>`` as a regex."""
    parts = re.split(r"(<[^>]+>)", template)
    body = "".join(
        (".+" if part.startswith("<path:") else "[^/]+")
        if part.startswith("<")
        else re.escape(part)
        for part in parts
    )
    return re.compile(f"^{body}$")


def route_source_format(path: str) -> str:
    """The chat format whose route serves *path*, or ``""`` when none does.

    A request that reached a route records that route's format in
    ``source_provider``; a refusal has to say the same thing, or anything that
    groups by the column sees one route as two sources.
    """
    descriptor = get_model_type("llm")
    for route in descriptor.routes if descriptor else ():
        if route.source_format and _route_pattern(route.path).match(path):
            return route.source_format
    return ""


def is_recorded_path(path: str) -> bool:
    """Whether a rejection on *path* belongs in the request log.

    The admin panel, health probes and the few public static paths are not
    proxy traffic: recording them would bury the proxy rows in polling.
    """
    return not (
        is_admin_path(path)
        or path in _UNRECORDED_PATHS
        or path == "/health"
        or path.startswith("/health/")
    )


def _response_text(response: Any) -> str:
    body = getattr(response, "body", None)
    return body.decode("utf-8", errors="replace") if isinstance(body, bytes) else ""


async def record_unrecorded_rejection(request: Any, response: Any) -> None:
    """Record *response* when it is an error no route handler logged.

    Runs as a response signal, so it sees every response the server writes.
    A request the lifecycle handled has ``request.state.log_entry_id`` and is
    skipped, as is any success or any path that is not proxy traffic.  The
    response also gains an ``X-Request-Id`` when it has none, so the id the
    client reads resolves to the row written here.
    """
    if response.status_code < 400 or not is_recorded_path(request.path):
        return
    if getattr(request.state, "log_entry_id", None) is not None:
        return

    rctx = request_context_var.get()
    request_id = rctx.request_id if rctx else get_request_id(request)
    client_ip = rctx.client_ip if rctx else extract_client_ip(request)
    response.headers.setdefault("x-request-id", request_id)

    ctx = _ops_context(request.app)
    if ctx is None:
        return
    start = rctx.request_start if rctx else None
    duration_ms = (time.monotonic() - start) * 1000 if start is not None else 0.0
    detail = (
        f"{request.method} {request.path} rejected before routing: "
        f"{_response_text(response)}"
    )[:_MAX_DETAIL_CHARS]
    await OpsRejectedRequest(
        ctx,
        status_code=response.status_code,
        duration_ms=duration_ms,
        error_detail=detail,
        request_id=request_id,
        client_ip=client_ip,
        source_provider=route_source_format(request.path),
    ).execute()


async def record_read_rejection(
    app: Any, rejection: Any, *, trust_proxy: bool = True
) -> None:
    """Record a request the server refused while still reading it.

    These never become a ``Request``, so no hook or lifecycle saw them: a body
    shorter than its ``Content-Length``, an oversized or timed-out read, a
    malformed request line, or a read error.  The server has already chosen the
    ``X-Request-Id`` it answers with; that id is the one stored.

    Args:
        app: The application holding ``ops_ctx``.
        rejection: The server's ``RejectedRequest``.
        trust_proxy: Whether forwarding headers may name the client, as for a
            served request.
    """
    if rejection.path is not None and not is_recorded_path(rejection.path):
        return
    ctx = _ops_context(app)
    if ctx is None:
        return

    headers = rejection.headers or {}
    peer = SimpleNamespace(headers=headers, client_addr=rejection.client_addr)
    cause = rejection.exception
    cause_text = f"{type(cause).__name__}: {cause}" if cause is not None else ""
    detail = (
        f"{rejection.method or '?'} {rejection.path or '?'} rejected while reading "
        f"the request: {rejection.reason} ({cause_text}); "
        f"Content-Length={rejection.content_length}, "
        f"Transfer-Encoding={headers.get('transfer-encoding')}, "
        f"received={rejection.bytes_received} bytes, "
        f"Content-Type={headers.get('content-type')}, "
        f"Content-Encoding={headers.get('content-encoding')}"
    )[:_MAX_DETAIL_CHARS]
    await OpsRejectedRequest(
        ctx,
        status_code=rejection.status_code,
        duration_ms=0.0,
        error_detail=detail,
        request_id=rejection.request_id,
        client_ip=extract_client_ip(peer, trust_proxy=trust_proxy),
        source_provider=route_source_format(rejection.path or ""),
    ).execute()
