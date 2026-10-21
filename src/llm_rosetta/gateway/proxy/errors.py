"""Upstream error mapping: passthrough, connection errors, 401, soft errors, stream errors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_rosetta._vendor.httpserver import Response

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.routing import ResolvedRoute
from llm_rosetta.shims.provider_shim import SoftErrorPattern

from llm_rosetta.observability.capture import CapturedRequest, CaptureState
from llm_rosetta.observability.error_dump import dump_error

from ..middleware.error_annotate import annotate_tool_collisions
from ..middleware.error_format import (
    detect_api_format_from_provider,
    format_error_response,
)
from ..logging import (
    get_logger,
    log_upstream_error,
)
from llm_rosetta.observability.sanitize import sanitize_upstream_error
from ..transport import (
    ProviderInfo,
    UpstreamConnectionError,
)
from ..transport.auth_errors import (
    AuthErrorKind,
    classify_auth_error,
    rewrite_session_policy_error,
)
from ..transport.token_refresh import force_refresh
from ..transport.sse_format import (
    extract_upstream_error_message,
    is_upstream_error_chunk,
)

logger = get_logger()

_COLLISION_HINT_STATUSES = frozenset({400, 422})


async def _upstream_error_passthrough(
    resp_status: int,
    resp_error_text: str,
    resp_raw_content: str | bytes,
    resp_body: Any,
    *,
    body: dict[str, Any],
    target_body: dict[str, Any],
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    persistence: Any | None,
    entry_id: str | None,
    extra_headers: dict[str, str] | None,
    capture_state: CaptureState | None,
    error_phase: str = "upstream",
    is_streaming: bool = False,
    tool_name_collisions: dict[str, int] | None = None,
) -> Response:
    """Log, dump, capture, and return a sanitized upstream error response.

    Args:
        tool_name_collisions: Conflicting tool names from the request leg, when
            the client's own tool list is a plausible cause of the rejection.
            Recorded in the error dump, and appended to the client's error
            message on a request-body rejection (see
            :data:`_COLLISION_HINT_STATUSES`) — not on an auth, not-found or
            rate-limit status, where the collision is incidental.
    """
    log_upstream_error(
        resp_status,
        resp_error_text,
        endpoint=str(route.target_provider),
        is_streaming=is_streaming,
    )
    await dump_error(
        persistence,
        request_body=body,
        response_text=resp_error_text,
        converted_body=target_body,
        model=body.get("model", ""),
        source_provider=route.source_provider,
        target_provider=route.target_provider,
        provider_name=route.provider_name,
        status_code=resp_status,
        error_phase=error_phase,
        upstream_url=str(provider_info.base_url),
        request_log_id=entry_id,
        tool_name_collisions=tool_name_collisions,
    )
    if capture_state is not None:
        capture_state.record(
            CapturedRequest(
                original_request=body,
                converted_body=target_body,
                upstream_response=resp_body,
                request_id=(
                    extra_headers.get("x-request-id", "") if extra_headers else ""
                ),
                model=body.get("model", ""),
                source_provider=route.source_provider,
                target_provider=route.target_provider,
                is_stream=False,
                status_code=resp_status,
            )
        )
    return Response(
        body=annotate_tool_collisions(
            sanitize_upstream_error(resp_raw_content),
            tool_name_collisions if resp_status in _COLLISION_HINT_STATUSES else None,
        ),
        status_code=resp_status,
        content_type="application/json",
    )


async def _connection_error_response(
    exc: UpstreamConnectionError,
    *,
    body: dict[str, Any],
    target_body: dict[str, Any],
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    persistence: Any | None,
    entry_id: str | None,
    extra_headers: dict[str, str] | None,
    capture_state: CaptureState | None,
    error_phase: str = "upstream",
) -> Response:
    """Build a 502 response for a connection-level upstream failure."""
    await dump_error(
        persistence,
        request_body=body,
        response_text=str(exc),
        converted_body=target_body,
        model=body.get("model", ""),
        source_provider=route.source_provider,
        target_provider=route.target_provider,
        provider_name=route.provider_name,
        status_code=502,
        error_phase=error_phase,
        upstream_url=str(provider_info.base_url),
        request_log_id=entry_id,
    )
    if capture_state is not None:
        capture_state.record(
            CapturedRequest(
                original_request=body,
                converted_body=target_body,
                request_id=(
                    extra_headers.get("x-request-id", "") if extra_headers else ""
                ),
                model=body.get("model", ""),
                source_provider=route.source_provider,
                target_provider=route.target_provider,
                is_stream=False,
                status_code=502,
                upstream_response={"error": str(exc)},
            )
        )
    return error_response_for_source(
        route.source_provider, 502, f"Upstream request failed: {exc}"
    )


async def _handle_401(
    provider_info: ProviderInfo,
    status_code: int,
    error_text: str,
    source_provider: ProviderType,
) -> tuple[bool, Response | None]:
    """Classify an upstream error and decide whether to retry.

    Returns ``(should_retry, error_response)``:
    - ``(True, None)`` — token refreshed successfully, caller should retry.
    - ``(False, Response)`` — session-policy error, return this to client.
    - ``(False, None)`` — not a 401, no token_command, or refresh failed.
    """
    if status_code != 401 or not provider_info.token_command:
        return False, None
    kind = classify_auth_error(status_code, error_text)
    if kind == AuthErrorKind.SESSION_POLICY_401:
        logger.warning(
            "ALCF session policy 401 for '%s' — user must re-login",
            provider_info.name,
        )
        return False, error_response_for_source(
            source_provider, 401, rewrite_session_policy_error()
        )
    if await force_refresh(provider_info):
        logger.info(
            "Retrying request after reactive token refresh for '%s'",
            provider_info.name,
        )
        return True, None
    return False, None


def _retry_after_quota_signal(
    provider_info: ProviderInfo,
    status_code: int,
    error_text: str,
) -> bool:
    """Demote the serving credential after a quota refusal and retry, if useful.

    The signal parser belongs to the vendor probe, so a provider that names the
    exhausted window is demoted immediately rather than at the next poll.  A
    retry is only worth it when the ring holds another usable credential; the
    already-demoted key is skipped by :meth:`KeyRing.next`.
    """
    from ..quota.service import has_usable_alternative, note_upstream_quota_error

    signal = note_upstream_quota_error(provider_info, status_code, error_text)
    if signal is None:
        return False
    return has_usable_alternative(provider_info)


def _check_soft_errors(
    patterns: tuple[SoftErrorPattern, ...],
    body: Any,
) -> SoftErrorPattern | None:
    """Check a response body against soft-error patterns.

    Returns the first matching pattern, or ``None``.
    """
    if not patterns:
        return None
    text = json.dumps(body) if not isinstance(body, str) else body
    for p in patterns:
        if p.compiled.search(text):
            return p
    return None


async def _soft_error_response(
    route: ResolvedRoute,
    resp_body: Any,
    body: dict[str, Any],
    target_body: dict[str, Any],
    *,
    provider_info: Any,
    persistence: Any | None,
    entry_id: str | None,
    extra_headers: dict[str, str] | None = None,
    capture_state: Any | None = None,
    tool_name_collisions: dict[str, int] | None = None,
) -> Response | None:
    """Return an error response if *resp_body* matches a soft-error pattern."""
    matched = _check_soft_errors(route.soft_error_patterns, resp_body)
    if matched is None:
        return None
    logger.warning("Soft-error detected in upstream 200 response: %s", matched.pattern)
    return await _upstream_error_passthrough(
        matched.status_code,
        matched.message,
        json.dumps(resp_body).encode() if resp_body else b"",
        resp_body,
        body=body,
        target_body=target_body,
        route=route,
        provider_info=provider_info,
        persistence=persistence,
        entry_id=entry_id,
        extra_headers=extra_headers,
        capture_state=capture_state,
        error_phase="soft_error",
        tool_name_collisions=tool_name_collisions,
    )


async def _detect_stream_chunk_error(
    chunk: dict[str, Any],
    *,
    soft_error_patterns: tuple[SoftErrorPattern, ...],
    stream_status: int,
    model: str,
    dump_ctx: DumpContext | None,
    entry_id: str | None,
) -> str | None:
    """Check a stream chunk for soft-errors and upstream error events.

    Returns the error message string if an error is detected, ``None`` otherwise.
    """
    soft_match = _check_soft_errors(soft_error_patterns, chunk)
    if soft_match:
        logger.warning("Soft-error detected in stream chunk: %s", soft_match.pattern)
        error_msg = soft_match.message
        error_status = soft_match.status_code
        error_phase = "soft_error"
    elif is_upstream_error_chunk(chunk):
        error_msg = extract_upstream_error_message(chunk)
        error_status = stream_status
        error_phase = "stream_chunk"
        log_upstream_error(
            stream_status,
            json.dumps(chunk)[:2000],
            endpoint=model,
            is_streaming=True,
        )
    else:
        return None

    _dc = dump_ctx or DumpContext()
    await dump_error(
        _dc.persistence,
        request_body=_dc.request_body,
        response_text=json.dumps(chunk)[:2000],
        converted_body=_dc.converted_body,
        model=model,
        source_provider=_dc.source_provider,
        target_provider=_dc.target_provider,
        provider_name=_dc.provider_name,
        status_code=error_status,
        error_phase=error_phase,
        upstream_url=_dc.upstream_url,
        request_log_id=entry_id,
        tool_name_collisions=_dc.tool_name_collisions,
    )
    return error_msg


@dataclass
class DumpContext:
    """Bundles error-dump parameters passed through streaming pipelines."""

    persistence: Any | None = None
    request_body: dict | None = None
    converted_body: dict | None = None
    source_provider: str | None = None
    target_provider: str | None = None
    provider_name: str | None = None
    upstream_url: str | None = None
    tool_name_collisions: dict[str, int] | None = None


def error_response_for_source(
    source_provider: ProviderType, status_code: int, message: str
) -> Response:
    """Return an error response formatted for the source provider's envelope."""
    api_format = detect_api_format_from_provider(source_provider)
    return format_error_response(api_format, status_code, message)
