"""Upstream error mapping: passthrough, connection errors, 401, soft errors, stream errors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import Response

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.routing import ResolvedRoute
from llm_rosetta.shims.provider_shim import SoftErrorPattern

from llm_rosetta_gateway.observability.capture import CapturedRequest, CaptureState
from llm_rosetta_gateway.observability.error_dump import dump_error

from ..middleware.error_annotate import annotate_tool_collisions
from ..middleware.error_format import (
    detect_api_format_from_provider,
    format_error_response,
)
from ..logging import (
    get_logger,
    log_upstream_error,
)
from llm_rosetta_gateway.observability.sanitize import sanitize_upstream_error
from ..transport.error_class import classify_status
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
    classify_upstream_error,
    UpstreamStreamError,
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


async def _first_token_timeout_response(
    *,
    body: dict[str, Any],
    target_body: dict[str, Any],
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    persistence: Any | None,
    entry_id: str | None,
    capture_state: CaptureState | None,
    budget_ms: int,
) -> Response:
    """Build the 504 for a lane that accepted the request and then went quiet.

    Nothing has been written to the client when this is called, so the answer
    can still be an honest status rather than a stream that fails to start.

    Args:
        body: The client's request, for the dump and the capture record.
        target_body: The converted request that was sent upstream.
        route: The route whose provider timed out.
        provider_info: The provider that was waiting.
        persistence: The error-dump store, when one exists.
        entry_id: The request-log row this request owns.
        capture_state: Content capture, when it is on.
        budget_ms: The model's first-token budget, for the message.

    Returns:
        A 504 in the client's own error envelope.
    """
    message = (
        f"Upstream '{route.provider_name}' sent no first token within {budget_ms} ms"
    )
    await dump_error(
        persistence,
        request_body=body,
        response_text=message,
        converted_body=target_body,
        model=body.get("model", ""),
        source_provider=route.source_provider,
        target_provider=route.target_provider,
        provider_name=route.provider_name,
        status_code=504,
        error_phase="stream_first_token",
        upstream_url=str(provider_info.base_url),
        request_log_id=entry_id,
    )
    if capture_state is not None:
        capture_state.record(
            CapturedRequest(
                original_request=body,
                converted_body=target_body,
                request_id="",
                model=body.get("model", ""),
                source_provider=route.source_provider,
                target_provider=route.target_provider,
                is_stream=True,
                status_code=504,
                upstream_response={"error": message},
            )
        )
    return error_response_for_source(route.source_provider, 504, message)


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
    - ``(False, None)`` — not a 401, the provider mints no credential, or the
      refresh failed.
    """
    if status_code != 401 or not provider_info.mints_credential:
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


class QuotaRefusal(str, Enum):
    """What an in-band quota refusal leaves the retry budget.

    ``RETRY`` means this provider still has a usable credential, so the same
    upstream may be tried again.  ``EXHAUSTED`` means it has none, and nothing
    this provider does for the rest of the request can succeed.
    """

    RETRY = "retry"
    EXHAUSTED = "exhausted"


#: Upstream statuses that mean "not now" rather than "not this request".
#:
#: They are capacity or transport failures: the same body may succeed on
#: another provider, and none of them says the credential is invalid, so no
#: key is demoted.  ``408`` is a timeout the upstream reported instead of
#: dropping the connection, ``529`` is Anthropic's "overloaded", and
#: ``520``-``524`` are the Cloudflare origin errors a proxied vendor surfaces
#: unchanged.
#:
#: ``501``, ``505``-``507``, ``510`` and ``511`` are deliberately absent: they
#: describe an endpoint that is missing, unimplemented or misconfigured, not a
#: provider that is out of capacity, and treating them as a health signal
#: would route around a provider whose configuration is the actual problem.
TRANSIENT_UPSTREAM_STATUSES = frozenset(
    {408, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
)

#: The transient statuses that mean the provider is down, not the request.
#:
#: These count toward the cross-request ladder that blocks a persistently
#: failing provider.  ``500``/``502``/``504`` may be request-specific and stay
#: out of it: they still fail over for this request, but one of them must not
#: take a provider out of rotation on its own.
CAPACITY_UPSTREAM_STATUSES = frozenset({503, 529, 520, 521, 522, 523, 524})

#: Statuses that mean the serving account cannot pay for this request.
#:
#: A quota probe may explain a ``429`` and demote the credential.  Without a
#: probe neither status names the key at fault, but another provider may still
#: serve.
REFUSAL_STATUSES = frozenset({402, 429})


@dataclass(frozen=True)
class UpstreamRefusal:
    """A refusal no quota probe explained, carrying what to report."""

    status_code: int
    reason: str


def _quota_refusal(
    provider_info: ProviderInfo,
    status_code: int,
    error_text: str,
) -> tuple[QuotaRefusal | None, Any]:
    """Demote the serving credential after a quota refusal and classify it.

    The signal parser belongs to the vendor probe, so a provider that names the
    exhausted window is demoted immediately rather than at the next poll.  A
    same-provider retry is only worth it when the ring holds another usable
    credential; the already-demoted key is skipped by :meth:`KeyRing.next`.

    Returns:
        ``(refusal, signal)`` — ``(None, None)`` when the error was not a
        quota refusal this gateway can classify, otherwise what the refusal
        leaves this provider able to do, and the signal to explain it.
    """
    from ..quota.service import has_usable_alternative, note_upstream_quota_error

    signal = note_upstream_quota_error(provider_info, status_code, error_text)
    if signal is None:
        return None, None
    if has_usable_alternative(provider_info):
        return QuotaRefusal.RETRY, signal
    return QuotaRefusal.EXHAUSTED, signal


def _count_capacity_refusal(
    provider_info: ProviderInfo,
    status_code: int,
    policy: Any | None = None,
) -> None:
    """Feed one capacity refusal into the cross-request ladder.

    Bookkeeping must not change the answer: this request still gets the
    failover it earned, so a failure here is logged and swallowed.
    """
    from ..quota.service import note_transient_refusal

    try:
        note_transient_refusal(provider_info, status_code, policy)
    except Exception:  # noqa: BLE001 - the refusal path always wins
        logger.warning("Could not record the transient refusal", exc_info=True)


def note_upstream_success(provider_info: ProviderInfo) -> None:
    """Clear the capacity-refusal ladder after a served request."""
    from ..quota.service import note_upstream_success as _note

    try:
        _note(provider_info)
    except Exception:  # noqa: BLE001 - bookkeeping must not fail a served request
        logger.warning("Could not clear the transient refusals", exc_info=True)


class RefusalAction(str, Enum):
    """What a failed upstream attempt means for the request.

    The two handlers send differently — one a JSON request, one an SSE
    stream — but a refusal means the same thing to both, so the decision is
    taken once and each handler acts on it in its own transport.
    """

    #: The attempt failed in a way that answers the client directly.
    RETURN = "return"
    #: The same upstream may be tried again (a refreshed credential, or
    #: another key in the ring).
    RETRY = "retry"
    #: This provider has no usable credential left; another may serve.
    FAILOVER = "failover"
    #: Nothing to retry and nothing to fail over to; the caller reports the
    #: upstream's own refusal.
    BREAK = "break"


class RefusalScope(str, Enum):
    """Whose failure a refusal reports, and so what a failover may reuse.

    The distinction decides how much of a route the retry may still use, so it
    is made here, where the status is classified, rather than in the retry
    path: a credential-level refusal takes the provider's siblings down with
    it, a lane-level one leaves them standing.
    """

    #: One key's state — a ``429``, a ``402``, a spent allowance.  Every lane
    #: of the provider shares the key, so all of them are out for the retry.
    CREDENTIAL = "credential"
    #: One upstream model's failure — a transient status, a first token that
    #: never came.  The provider's other lanes may still serve.
    LANE = "lane"


@dataclass(frozen=True)
class RefusalDecision:
    """The outcome of :func:`decide_on_refusal`."""

    action: RefusalAction
    #: The answer to return, when ``action`` is ``RETURN``.
    response: Response | None = None
    #: The quota signal that explains a ``FAILOVER``.
    signal: Any = None
    #: What the refusal was about, when ``action`` is ``FAILOVER``.
    scope: RefusalScope | None = None


async def decide_on_refusal(
    provider_info: ProviderInfo,
    status_code: int,
    error_text: str,
    source_provider: ProviderType,
    *,
    rate_limit: Any | None = None,
) -> RefusalDecision:
    """Classify a failed upstream attempt into one action.

    The order matters: a ``401`` that a credential refresh can fix is a retry,
    not a refusal, and a session-policy ``401`` answers the client before
    either is considered.  A transient status never reaches the quota probe:
    it says the provider could not serve, not that the key is spent.

    A ``429`` is counted against the lane first — every lane, whatever its
    quota declaration — so a provider that no probe can read still sinks and
    finally disables instead of being retried as if nothing happened.  A
    capacity status is counted the same way, against the provider itself,
    before the request fails over: the first refusal sinks it and a run of
    them takes it out of rotation for later requests.

    Args:
        provider_info: The provider that just refused.
        status_code: The upstream's HTTP status.
        error_text: The upstream's error body, as text.
        source_provider: The client's dialect, for a direct answer.
        rate_limit: The continuous-429 rule in force, or ``None`` for the
            default.

    Returns:
        The action to take, with the response for ``RETURN`` and the signal
        for ``FAILOVER``.
    """
    if status_code == 429:
        from ..quota.service import note_upstream_rate_limit

        note_upstream_rate_limit(provider_info, rate_limit)

    retry, err = await _handle_401(
        provider_info, status_code, error_text, source_provider
    )
    if err is not None:
        return RefusalDecision(RefusalAction.RETURN, response=err)
    if retry:
        return RefusalDecision(RefusalAction.RETRY)
    if status_code in TRANSIENT_UPSTREAM_STATUSES:
        if status_code in CAPACITY_UPSTREAM_STATUSES:
            _count_capacity_refusal(provider_info, status_code, rate_limit)
        return RefusalDecision(
            RefusalAction.FAILOVER,
            signal=UpstreamRefusal(status_code, f"upstream {status_code}"),
            scope=RefusalScope.LANE,
        )
    refusal, signal = _quota_refusal(provider_info, status_code, error_text)
    if refusal is None and status_code == 429:
        # No probe could classify it, but the status is unambiguous: this lane
        # is rate limited.  The refusals were counted above, so the provider
        # sinks — or is disabled — for the next request; for this one it has
        # nothing left to give, and another provider may serve.
        from llm_rosetta.shims.quota_probe import InBandSignal

        refusal = QuotaRefusal.EXHAUSTED
        signal = InBandSignal(cooldown_seconds=None, reason="upstream answered 429")
    if refusal is QuotaRefusal.RETRY:
        return RefusalDecision(RefusalAction.RETRY)
    if refusal is QuotaRefusal.EXHAUSTED:
        return RefusalDecision(
            RefusalAction.FAILOVER, signal=signal, scope=RefusalScope.CREDENTIAL
        )
    if status_code in REFUSAL_STATUSES:
        # No probe explained it, but the status alone is a refusal: count it
        # toward the same ladder, so a provider that keeps answering 429 or
        # 402 is taken out of rotation after a few attempts.
        _count_capacity_refusal(provider_info, status_code, rate_limit)
        return RefusalDecision(
            RefusalAction.FAILOVER,
            signal=UpstreamRefusal(status_code, f"upstream {status_code}"),
            scope=RefusalScope.CREDENTIAL,
        )
    return RefusalDecision(RefusalAction.BREAK)


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
) -> UpstreamStreamError | None:
    """Check a stream chunk for soft-errors and upstream error events.

    Returns the classified error if there is one, ``None`` otherwise.  The
    classification keeps the upstream's code and request id, which the
    terminal event needs: a generic message makes a plan-usage failure
    indistinguishable from a transient one.
    """
    soft_match = _check_soft_errors(soft_error_patterns, chunk)
    if soft_match:
        logger.warning("Soft-error detected in stream chunk: %s", soft_match.pattern)
        stream_error = UpstreamStreamError(
            message=soft_match.message, status=soft_match.status_code
        )
        error_phase = "soft_error"
    else:
        classified = classify_upstream_error(chunk)
        if classified is None:
            return None
        stream_error = UpstreamStreamError(
            message=classified.message,
            code=classified.code,
            request_id=classified.request_id,
            status=stream_status,
        )
        error_phase = "stream_chunk"
        log_upstream_error(
            stream_status,
            json.dumps(chunk)[:2000],
            endpoint=model,
            is_streaming=True,
        )

    error_status = stream_error.status or stream_status

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
    return stream_error


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
    source_provider: ProviderType,
    status_code: int,
    message: str,
    *,
    error_type: str | None = None,
    google_status: str | None = None,
) -> Response:
    """Return an error response formatted for the source provider's envelope.

    The type follows the same split the streaming envelope uses, so a client
    gets one verdict whether it asked with ``stream: true`` or not:

    - ``>= 502``: the upstream produced something this gateway could not turn
      into a reply — ``api_error`` (Google: ``UNAVAILABLE``).  The client's
      request was fine, so it must not be told to go fix it.
    - ``500``: the gateway itself broke — ``server_error`` (Google:
      ``INTERNAL``), the word both vendors use for their own fault.
    - ``4xx``: ``invalid_request_error`` (Google: ``INVALID_ARGUMENT``).

    Args:
        source_provider: The client's format, which picks the envelope.
        status_code: HTTP status code.
        message: Human-readable message.
        error_type: Override for the OpenAI / Anthropic ``type`` field.
        google_status: Override for the Google ``status`` field.

    Returns:
        The formatted error response.
    """
    api_format = detect_api_format_from_provider(source_provider)
    terminal = classify_status(status_code)
    if error_type is None:
        # OpenAI distinguishes its own fault from the upstream's; Anthropic
        # has no `server_error` at all — every 5xx is `api_error`, "an
        # unexpected error has occurred internal to Anthropic's systems" —
        # so Messages keeps that word at 500 too.  The table owns the words.
        error_type = (
            terminal.openai_type if api_format == "openai" else terminal.anthropic_type
        )
    if google_status is None:
        google_status = terminal.google_status
    return format_error_response(
        api_format,
        status_code,
        message,
        error_type=error_type,
        google_status=google_status,
    )
