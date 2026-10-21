"""One declarative record per ``(model_type, format)`` pair."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Everything the repository knows about one wire format.

    The record is data only: a *name* for each collaborator (auth builder, SSE
    formatter, usage shape, detector), never the object itself.  The package
    that owns the collaborator resolves the name, so this module stays a leaf
    the shim loader can import before the gateway exists.

    Attributes:
        model_type: ``llm``, ``embedding``, ``rerank`` or ``decision``.
        format: Wire dialect within the model type.
        wire_family: ``openai``, ``anthropic`` or ``google`` — the error
            envelope shape.
        detect: Detector key, for formats ``detect_provider`` can recognise.
        detect_priority: Lower runs first; the first match wins.
        picker_order: Position in the admin provider picker; lower first.
        auth: Auth-builder key (``openai``, ``anthropic``, ``google``).
        url_template: Upstream request URL template, when the format has one.
        stream_url_template: Streaming URL when it differs from ``url_template``.
        base_url_hint: Admin-picker placeholder only — never a runtime default.
        api_key_env_hint: Admin-picker placeholder only.
        supports_custom_tools: Whether the standard accepts ``type: custom``.
        sse: SSE formatter key.
        usage_shape: Usage-normalisation shape key.
        route_path: Client-facing gateway route for a chat format, as a
            router path (``<path:model_path>`` converters included).
        default_path: Default upstream endpoint path, for the path-configured
            model types (embedding, rerank, decision).
        models_path: Path used by the connectivity probe to list models.
    """

    model_type: str
    format: str
    wire_family: str = "openai"
    detect: str | None = None
    detect_priority: int = 100
    picker_order: int = 100
    auth: str | None = None
    url_template: str | None = None
    stream_url_template: str | None = None
    base_url_hint: str | None = None
    api_key_env_hint: str | None = None
    supports_custom_tools: bool = False
    sse: str | None = None
    usage_shape: str | None = None
    route_path: str | None = None
    default_path: str | None = None
    models_path: str | None = None
