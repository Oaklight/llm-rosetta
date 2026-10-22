"""proxy package."""

from __future__ import annotations

from .errors import (  # noqa: F401
    _COLLISION_HINT_STATUSES,
    _upstream_error_passthrough,
    _connection_error_response,
    _handle_401,
    QuotaRefusal,
    _quota_refusal,
    _check_soft_errors,
    _soft_error_response,
    _detect_stream_chunk_error,
    DumpContext,
    error_response_for_source,
)

from .metadata import (  # noqa: F401
    _CacheEntry,
    ProviderMetadataStore,
    _default_metadata_store,
    _strip_internal_metadata,
)

from .fidelity import (  # noqa: F401
    _check_fidelity,
    _fidelity_check_and_dump,
)

from .preflight import (  # noqa: F401
    _PREFLIGHT_TIMEOUT,
    _build_preflight_body,
    _extract_preflight_input_tokens,
    _run_preflight,
)

from .stream_merge import (  # noqa: F401
    _merge_tool_call_delta,
    _merge_choice_delta,
    merge_chat_completion_events,
)

from .engine import (  # noqa: F401
    _inject_stream_flags,
    detect_stream_request,
    extract_model,
    _REQUEST_STAGES,
    _apply_request_stages,
    _log_response_warnings,
    close_resources,
    _non_streaming_from_stream,
    handle_non_streaming,
)

from .streaming import (  # noqa: F401
    _terminal_error_sse,
    _write_back_stream_usage,
    _flush_and_finalize,
    _stream_event_generator,
    handle_streaming,
)


__all__ = [
    "_COLLISION_HINT_STATUSES",
    "_upstream_error_passthrough",
    "_connection_error_response",
    "_handle_401",
    "QuotaRefusal",
    "_quota_refusal",
    "_check_soft_errors",
    "_soft_error_response",
    "_detect_stream_chunk_error",
    "DumpContext",
    "error_response_for_source",
    "_CacheEntry",
    "ProviderMetadataStore",
    "_default_metadata_store",
    "_strip_internal_metadata",
    "_check_fidelity",
    "_fidelity_check_and_dump",
    "_PREFLIGHT_TIMEOUT",
    "_build_preflight_body",
    "_extract_preflight_input_tokens",
    "_run_preflight",
    "_merge_tool_call_delta",
    "_merge_choice_delta",
    "merge_chat_completion_events",
    "_inject_stream_flags",
    "detect_stream_request",
    "extract_model",
    "_REQUEST_STAGES",
    "_apply_request_stages",
    "_log_response_warnings",
    "close_resources",
    "_non_streaming_from_stream",
    "handle_non_streaming",
    "_terminal_error_sse",
    "_write_back_stream_usage",
    "_flush_and_finalize",
    "_stream_event_generator",
    "handle_streaming",
]
