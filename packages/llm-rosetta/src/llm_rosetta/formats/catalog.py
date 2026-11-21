"""The format catalog — the one home for per-format facts.

Adding a format means adding one :class:`FormatSpec` here.  Everything that
used to keep its own table derives from this list: the model-type vocabulary,
the converter binding, the gateway's auth/URL tables, the SSE formatter and
error-envelope maps, the usage-shape map and the detection order.

The module is a leaf: it imports nothing from ``converters``, ``gateway`` or
``shims``, so the shim loader can read it while the package is still
initialising.
"""

from __future__ import annotations

from llm_rosetta.formats.spec import FormatSpec

#: Capability categories, in display order.  A category, its shim directory
#: and its route share one spelling, taken from the endpoint the type serves
#: (``/v1/embeddings``, ``/v1/decisions``); ``llm`` keeps its name because chat
#: has no single endpoint.
MODEL_TYPES: tuple[str, ...] = ("llm", "embeddings", "rerank", "decisions")

#: Every format, in display order within its model type.  A pair is unique;
#: format names are namespaced by the model type (``google_generate`` names
#: both a chat and an embedding format).
FORMAT_SPECS: tuple[FormatSpec, ...] = (
    # -- llm ------------------------------------------------------------
    FormatSpec(
        model_type="llm",
        format="chat_completions",
        picker_order=10,
        wire_family="openai",
        detect="chat_completions",
        detect_priority=60,
        auth="openai",
        url_template="{base_url}/chat/completions",
        base_url_hint="https://api.openai.com/v1",
        api_key_env_hint="OPENAI_API_KEY",
        supports_custom_tools=True,
        sse="chat_completions",
        usage_shape="openai_details",
        route_path="/v1/chat/completions",
        models_path="/models",
        # Echo fields a Chat-compatible stream repeats on every chunk.  A
        # non-streaming body has to put them back, and no IR event carries
        # them.
        stream_echo_fields=("service_tier", "system_fingerprint"),
    ),
    FormatSpec(
        model_type="llm",
        format="messages",
        picker_order=30,
        wire_family="anthropic",
        detect="messages",
        detect_priority=50,
        auth="anthropic",
        url_template="{base_url}/v1/messages",
        base_url_hint="https://api.anthropic.com",
        api_key_env_hint="ANTHROPIC_API_KEY",
        sse="messages",
        usage_shape="messages_fields",
        route_path="/v1/messages",
        models_path="/v1/models",
    ),
    FormatSpec(
        model_type="llm",
        format="open_responses",
        picker_order=20,
        wire_family="openai",
        detect="open_responses",
        detect_priority=30,
        auth="openai",
        url_template="{base_url}/responses",
        # A placeholder only.  The hint names a core vendor's host: the
        # aggregator it used to name moved to the contrib source (#1660).
        base_url_hint="https://api.openai.com/v1",
        api_key_env_hint="OPENAI_API_KEY",
        supports_custom_tools=True,
        sse="open_responses",
        usage_shape="openai_details",
        route_path="/v1/responses",
        models_path="/models",
    ),
    FormatSpec(
        model_type="llm",
        format="google_generate",
        picker_order=40,
        wire_family="google",
        detect="google_generate",
        detect_priority=20,
        auth="google",
        url_template="{base_url}/v1beta/models/{model}:generateContent",
        stream_url_template=(
            "{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        ),
        base_url_hint="https://generativelanguage.googleapis.com",
        api_key_env_hint="GOOGLE_API_KEY",
        route_path="/v1beta/models/<path:model_path>",
        sse="google",
        usage_shape="google_metadata",
        models_path="/v1beta/models",
    ),
    FormatSpec(
        model_type="llm",
        format="google_interactions",
        picker_order=50,
        wire_family="google",
        detect="google_interactions",
        detect_priority=40,
        auth="google",
        url_template="{base_url}/v1beta/interactions",
        stream_url_template="{base_url}/v1beta/interactions?alt=sse",
        base_url_hint="https://generativelanguage.googleapis.com",
        api_key_env_hint="GOOGLE_API_KEY",
        route_path="/v1beta/interactions",
        sse="google_interactions",
        models_path="/v1beta/models",
    ),
    FormatSpec(
        model_type="llm",
        format="acp",
        picker_order=95,
        wire_family="openai",
        url_template="{base_url}/acp-turn",
        base_url_hint="http://agy.local",
        # A target and never a source: no client speaks ACP to the gateway,
        # so there is no route to register and no SSE formatter to describe.
        # The flag says that once, rather than each consumer re-deriving it
        # from two absent fields.
        upstream_only=True,
    ),
    # -- embeddings -----------------------------------------------------
    FormatSpec(
        model_type="embeddings",
        format="openai",
        wire_family="openai",
        auth="openai",
        default_path="/v1/embeddings",
        models_path="/models",
    ),
    FormatSpec(model_type="embeddings", format="cohere", default_path="/v2/embed"),
    FormatSpec(model_type="embeddings", format="jina", default_path="/v1/embeddings"),
    FormatSpec(model_type="embeddings", format="voyage", default_path="/v1/embeddings"),
    FormatSpec(
        model_type="embeddings",
        format="google_generate",
        wire_family="google",
        auth="google",
        default_path="/v1beta/models/{model}:embedContent",
        models_path="/v1beta/models",
    ),
    FormatSpec(
        model_type="embeddings", format="minimax", default_path="/v1/embeddings"
    ),
    # -- rerank ---------------------------------------------------------
    FormatSpec(model_type="rerank", format="jina", default_path="/v1/rerank"),
    FormatSpec(model_type="rerank", format="cohere", default_path="/v2/rerank"),
    FormatSpec(model_type="rerank", format="voyage", default_path="/v1/rerank"),
    # -- decisions ------------------------------------------------------
    FormatSpec(
        model_type="decisions", format="systemone", default_path="/v1/systemone"
    ),
    FormatSpec(
        model_type="decisions",
        format="openai_decisions",
        default_path="/v1/decisions",
    ),
)

#: ``(model_type, format) → spec``.
BY_PAIR: dict[tuple[str, str], FormatSpec] = {
    (spec.model_type, spec.format): spec for spec in FORMAT_SPECS
}


def spec_for(model_type: str, format: str) -> FormatSpec | None:
    """The catalog record of one ``(model_type, format)`` pair, if there is one.

    A host reads per-format facts (models path, stream echo fields, ...) here
    rather than through :data:`BY_PAIR`, whose key layout stays internal.  The
    model type is canonicalised first, so a legacy spelling (``embedding``)
    finds its record.

    Args:
        model_type: A model type.
        format: A format name.

    Returns:
        The :class:`FormatSpec`, or ``None`` when the pair is not in the catalog.
    """
    return BY_PAIR.get((model_type, format))


def default_format(model_type: str) -> str:
    """The default (first listed) format of *model_type*."""
    for spec in FORMAT_SPECS:
        if spec.model_type == model_type:
            return spec.format
    raise KeyError(model_type)


def default_path(model_type: str) -> str | None:
    """The upstream endpoint path of *model_type*'s default format."""
    return BY_PAIR[(model_type, default_format(model_type))].default_path
