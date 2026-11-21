"""The request stages the streaming and the non-streaming paths share.

Both handlers start a request the same way: read the stream flag and the
model, build the conversion pipeline and convert the body (a conversion
failure is a 400 in the client's format), check same-format fidelity, and run
the routing stages that pick the conversation identity.  :func:`prepare_request`
does that once and hands either handler a :class:`PreparedRequest`, or raises
:class:`RequestRefused` with the response to send.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_rosetta.api import (
    ConversionError,
    ConversionPipeline,
    ProviderType,
)
from llm_rosetta.contracts.routing import ResolvedRoute

from llm_rosetta_gateway._vendor.httpserver import Response
from llm_rosetta_gateway.observability.error_dump import dump_error

from ..observability.logging import (
    log_converted_request,
    log_original_request,
)
from ..routing import (
    ConversationIdentity,
    ProviderKeyAffinity,
    RoutingChain,
    SelectionContext,
    SessionHeaderAffinity,
)
from ..routing.conversation import note_conversation
from ..transport import (
    ProviderInfo,
)
from .errors import (
    error_response_for_source,
)
from .fidelity import (
    _fidelity_check_and_dump,
)
from .image_fetch import fetch_image
from .metadata import (
    ProviderMetadataStore,
)


def _inject_stream_flags(
    target_provider: ProviderType, body: dict[str, Any], *, stream: bool
) -> dict[str, Any]:
    """Inject provider-specific stream flags into an outbound request body.

    Returns the body unchanged when *stream* is False. When True, returns
    a shallow copy with stream flags set for the target provider.
    """
    if not stream:
        return body
    body = dict(body)
    if target_provider == "chat_completions":
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    elif target_provider in ("open_responses", "messages"):
        body["stream"] = True
    elif target_provider == "google_interactions":
        body["stream"] = True
    # Google (generateContent) streaming is signaled via URL, not body
    return body


def detect_stream_request(source_provider: ProviderType, body: dict[str, Any]) -> bool:
    """Detect if the incoming request asks for streaming."""
    if source_provider in (
        "chat_completions",
        "open_responses",
        "messages",
        "google_interactions",
    ):
        return bool(body.get("stream", False))
    # Google (generateContent) streaming is determined by the endpoint path, not the body
    return False


def extract_model(source_provider: ProviderType, body: dict[str, Any]) -> str | None:
    """Extract the model name from a source-format request body."""
    return body.get("model")


_REQUEST_STAGES = RoutingChain(
    [ConversationIdentity(), ProviderKeyAffinity(), SessionHeaderAffinity()]
)


def _apply_request_stages(ctx: SelectionContext) -> ProviderInfo:
    """Fill in the stage-2 decisions and return the transport config to use.

    The caller sets ``ctx.provider_info`` and ``ctx.ir_request`` first; what
    comes back is the transport config to send with.  The conversation id the
    first stage derived is published for the request log, which is written
    after the handler returns and has no other way to reach it.
    """
    _REQUEST_STAGES.run(ctx)
    note_conversation(ctx.identity.conversation)
    provider_info = ctx.provider_info
    assert provider_info is not None  # the caller sets it before running these
    return provider_info


@dataclass
class PreparedRequest:
    """What both request handlers need before their transports diverge.

    Built by :func:`prepare_request`; the two handlers read it and then take
    their own path — one opens a JSON request, the other an SSE stream.
    """

    pipeline: ConversionPipeline
    target_body: dict[str, Any]
    provider_info: ProviderInfo
    model: str


class RequestRefused(Exception):
    """The prelude answered the request itself; the caller returns the response.

    A conversion failure is the source dialect's own ``400``, and it is
    decided in one place instead of once per handler.
    """

    def __init__(self, response: Response, profile: dict[str, Any]) -> None:
        super().__init__("the request was refused before it reached an upstream")
        self.response = response
        self.profile = profile


async def prepare_request(
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    *,
    metadata_store: ProviderMetadataStore,
    persistence: Any | None,
    entry_id: str | None,
    selection: SelectionContext,
    profile: dict[str, Any],
) -> PreparedRequest:
    """Build the pipeline and the target body both handlers send.

    The steps here are the same whatever the response mode: build the
    pipeline, convert the request to the target dialect, report a conversion
    failure as the source's ``400``, record the request's fidelity, flatten
    the system message when the model asks for it, and run the request
    stages that pick the transport config.

    The body is already wire-ready when the conversion returns: every
    ``*_to_provider`` strips the internal ``_llm_rosetta_provider_metadata``
    carrier at the converter boundary (#941).  The gateway used to repeat
    that walk here; on a production-sized body it was a second full pass
    over the whole history that always found nothing (#1410).

    Args:
        route: The resolved route the request will take.
        provider_info: The provider's transport config, before the stages.
        body: The source-format request body.
        metadata_store: The provider metadata store the conversion writes to.
        persistence: Request-log persistence, for the failure dump.
        entry_id: The request-log entry this request is recorded under.
        selection: The routing context the request stages read and fill.
        profile: The per-request profile dict, updated in place.

    Returns:
        The pipeline, the converted body, the provider info the stages chose,
        and the model the client named.

    Raises:
        RequestRefused: The conversion failed; ``response`` is the answer and
            ``profile`` is the dict to return with it.
    """
    model = body.get("model", "")

    pipeline = ConversionPipeline(
        route.source_provider,
        route.target_provider,
        target_shim=route.shim_name,
        upstream_model=model,
        model_capabilities=route.model_capabilities,
        reasoning_config_override=route.reasoning_override,
        supports_custom_tools=route.supports_custom_tools,
        max_tool_description_length=route.max_tool_description_length,
        hoist_system_messages=route.hoist_system_messages,
        image_fetcher=fetch_image,
    )

    # Phase 1+2: Source → IR → Target
    try:
        target_body = pipeline.convert_request(
            body, on_ir_ready=metadata_store.inject_into_request
        )
    except ConversionError as exc:
        await dump_error(
            persistence,
            request_body=body,
            response_text=str(exc),
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=400,
            error_phase="conversion",
            upstream_url=str(provider_info.base_url),
            request_log_id=entry_id,
        )
        raise RequestRefused(
            error_response_for_source(route.source_provider, 400, str(exc)), profile
        ) from exc

    profile.update(pipeline.profile)
    # Fields the target required and the request did not name.  The converter
    # records them on its context; the request log carries them on the row, so
    # "the gateway added this" is visible after the fact (#1328).
    injected = pipeline.context.metadata.get("injected")
    if injected:
        profile["injected"] = dict(injected)

    await _fidelity_check_and_dump(
        route, body, target_body, "request", profile, model, persistence, entry_id
    )

    log_original_request(pipeline.ir_request)

    # Per-model system message flattening (gateway config override)
    if route.flatten_system:
        from llm_rosetta.api import flatten_system_content

        target_body = flatten_system_content()(target_body)

    log_converted_request(target_body)

    selection.with_provider_info(provider_info)
    selection.with_ir_request(pipeline.ir_request)
    provider_info = _apply_request_stages(selection)

    return PreparedRequest(
        pipeline=pipeline,
        target_body=target_body,
        provider_info=provider_info,
        model=model,
    )
