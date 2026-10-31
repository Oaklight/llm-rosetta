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
#: (``/v1/embeddings``); ``llm`` keeps its name because chat has no single
#: endpoint.
MODEL_TYPES: tuple[str, ...] = ("llm", "embeddings", "rerank", "decision")

#: Spellings an earlier release used for a category.  They are accepted at
#: every boundary that takes a model type from outside the process — a shim's
#: ``model_type``, a config key or ``type:`` value, a request parameter — and
#: rewritten to the canonical name, so the alias never reaches stored state.
LEGACY_MODEL_TYPES: dict[str, str] = {"embedding": "embeddings"}

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
        base_url_hint="https://openrouter.ai/api/v1",
        api_key_env_hint="OPENROUTER_API_KEY",
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
    # -- rerank ---------------------------------------------------------
    FormatSpec(model_type="rerank", format="jina", default_path="/v1/rerank"),
    FormatSpec(model_type="rerank", format="cohere", default_path="/v2/rerank"),
    FormatSpec(model_type="rerank", format="voyage", default_path="/v1/rerank"),
    # -- decision -------------------------------------------------------
    FormatSpec(model_type="decision", format="systemone", default_path="/v1/systemone"),
    FormatSpec(
        model_type="decision",
        format="openai_decisions",
        default_path="/v1/decisions",
    ),
)

#: ``(model_type, format) → spec``.
BY_PAIR: dict[tuple[str, str], FormatSpec] = {
    (spec.model_type, spec.format): spec for spec in FORMAT_SPECS
}


def canonical_model_type(name: str) -> str:
    """The canonical spelling of the model type *name*.

    The one place a legacy spelling is translated.  Callers pass the result
    to anything that stores or compares a model type; the legacy name itself
    is never persisted.

    Args:
        name: A model type as read from a shim, a config or a request.

    Returns:
        The canonical name, unchanged when *name* is already canonical (or is
        not a known type at all).
    """
    return LEGACY_MODEL_TYPES.get(name, name)


def legacy_model_type(name: str) -> str | None:
    """The spelling an earlier release used for canonical *name*, if any.

    The reverse of :func:`canonical_model_type`, for a reader that must also
    accept the old spelling of a key it derives (``embeddings_format`` →
    ``embedding_format``).

    Args:
        name: A canonical model type.

    Returns:
        The legacy spelling, or ``None`` when the name was never renamed.
    """
    for old, new in LEGACY_MODEL_TYPES.items():
        if new == name:
            return old
    return None


def default_format(model_type: str) -> str:
    """The default (first listed) format of *model_type*."""
    model_type = canonical_model_type(model_type)
    for spec in FORMAT_SPECS:
        if spec.model_type == model_type:
            return spec.format
    raise KeyError(model_type)


def default_path(model_type: str) -> str | None:
    """The upstream endpoint path of *model_type*'s default format."""
    return BY_PAIR[
        (canonical_model_type(model_type), default_format(model_type))
    ].default_path
