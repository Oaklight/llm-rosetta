"""
LLM Provider Auto-Detection

自动检测 LLM provider 请求体格式的工具函数
Utility functions for auto-detecting LLM provider request body formats
"""

from typing import Any, Literal

ProviderType = Literal[
    "openai_chat", "openai_responses", "open_responses", "anthropic", "google"
]


_RESPONSES_ITEM_TYPES = frozenset(
    {
        "message",
        "function_call",
        "function_call_output",
        "mcp_call",
        "mcp_call_output",
        "reasoning",
        "system_event",
        "input_text",
        "output_text",
    }
)

_ANTHROPIC_CONTENT_TYPES = frozenset(
    {"image", "tool_use", "tool_result", "thinking", "document"}
)


def _is_google_format(body: dict[str, Any]) -> bool:
    """Check if body matches Google GenAI format (contents with parts)."""
    contents = body.get("contents")
    if not isinstance(contents, list) or len(contents) == 0:
        return False
    first = contents[0]
    return isinstance(first, dict) and "parts" in first


def _is_responses_format(body: dict[str, Any]) -> bool:
    """Check if body matches OpenAI Responses API format (input/output with typed items)."""
    items = body.get("input") or body.get("output")
    if not isinstance(items, list) or len(items) == 0:
        return False
    first = items[0]
    return isinstance(first, dict) and first.get("type") in _RESPONSES_ITEM_TYPES


def _has_anthropic_content_blocks(content: list[Any]) -> bool:
    """Check if any content block in a message uses Anthropic-specific types."""
    for part in content:
        if isinstance(part, dict) and part.get("type") in _ANTHROPIC_CONTENT_TYPES:
            return True
    return False


def _is_anthropic_messages(body: dict[str, Any]) -> bool:
    """Check if a messages-based body is Anthropic rather than OpenAI Chat.

    Both Anthropic and OpenAI Chat use ``messages``, so this inspects
    top-level fields and content-block types to disambiguate.
    """
    # Anthropic-specific top-level fields
    if "system" in body and isinstance(body["system"], (str, list)):
        return True
    if "anthropic_version" in body or "max_tokens_to_sample" in body:
        return True

    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return False

    first_message = messages[0]
    if not isinstance(first_message, dict):
        return False

    content = first_message.get("content")
    if not isinstance(content, list) or len(content) == 0:
        return False

    return _has_anthropic_content_blocks(content)


def detect_provider(body: dict[str, Any]) -> ProviderType | None:
    """Auto-detect provider type from request body structure.

    Args:
        body: Provider request body dict.

    Returns:
        Detected provider type, or ``None`` if unrecognised.

    Examples:
        >>> detect_provider({"messages": [{"role": "user", "content": "Hello"}]})
        'openai_chat'
        >>> detect_provider({"input": [{"type": "message", "role": "user"}]})
        'openai_responses'
        >>> detect_provider({"messages": [{"role": "user", "content": [{"type": "text"}]}]})
        'anthropic'
        >>> detect_provider({"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]})
        'google'
    """
    if not isinstance(body, dict):
        return None

    if _is_google_format(body):
        return "google"

    if ("input" in body or "output" in body) and _is_responses_format(body):
        return "openai_responses"

    if "messages" not in body:
        return None

    if _is_anthropic_messages(body):
        return "anthropic"

    # Check for OpenAI-specific tool_calls in message history
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and "tool_calls" in msg:
                return "openai_chat"

    # Default: OpenAI Chat is the most common messages-based format
    return "openai_chat"


_converter_cache: dict[str, Any] = {}


def get_converter_for_provider(provider: str):
    """Get the corresponding converter for a provider type or shim name.

    Converter instances are cached — the same object is returned for the
    same resolved base provider.  This is safe because converters are
    stateless (all per-request state lives in ``ConversionContext``).

    Args:
        provider: Provider type string or registered shim name.

    Returns:
        Corresponding converter instance (cached).

    Raises:
        ValueError: If the provider is not a known type or shim name.
    """
    if provider in _converter_cache:
        return _converter_cache[provider]

    from .converters.anthropic import AnthropicConverter
    from .converters.google_genai import GoogleConverter
    from .converters.openai_chat import OpenAIChatConverter
    from .converters.openai_responses import OpenAIResponsesConverter
    from .shims import resolve_base

    converter_map = {
        "openai_chat": OpenAIChatConverter,
        "openai_responses": OpenAIResponsesConverter,
        "open_responses": OpenAIResponsesConverter,
        "anthropic": AnthropicConverter,
        "google": GoogleConverter,
    }

    # Direct match against base converter types
    if provider in converter_map:
        instance = converter_map[provider]()
        _converter_cache[provider] = instance
        return instance

    # Resolve through shim registry
    base = resolve_base(provider)
    if base in converter_map:
        instance = converter_map[base]()
        _converter_cache[provider] = instance
        _converter_cache[base] = instance
        return instance

    raise ValueError(f"Unsupported provider: {provider}")


def _detect_source(
    source_body: dict[str, Any],
    source_provider: ProviderType | str | None,
) -> str:
    """Detect or validate the source provider, raising on failure."""
    if source_provider is not None:
        return str(source_provider)
    detected = detect_provider(source_body)
    if detected is None:
        raise ValueError(
            "Unable to detect source provider. "
            "Please specify source_provider explicitly."
        )
    return detected


def convert(
    source_body: dict[str, Any],
    target_provider: ProviderType | str,
    source_provider: ProviderType | str | None = None,
    *,
    model: str | None = None,
    force_conversion: bool = False,
) -> dict[str, Any]:
    """Auto-detect source provider and convert to target provider format.

    Delegates to :class:`~llm_rosetta.pipeline.ConversionPipeline`
    internally, resolving source and target shims from the provider
    names.  For response or streaming conversion, use
    :func:`convert_response` or :class:`ConversionPipeline` directly.

    Args:
        source_body: Source provider request body.
        target_provider: Target provider type or registered shim name.
        source_provider: Optional source provider type or shim name.
            Auto-detected from *source_body* when not provided.
        model: Optional model name passed as ``upstream_model`` to the
            pipeline (used for per-model shim overrides).
        force_conversion: When ``True``, always run the full conversion
            pipeline even when source and target providers are the same.

    Returns:
        Target provider format request body.

    Raises:
        ValueError: If source provider cannot be detected.

    Examples:
        >>> openai_body = {"messages": [{"role": "user", "content": "Hello"}]}
        >>> google_body = convert(openai_body, "google")

        >>> anthropic_body = {"messages": [...]}
        >>> openai_body = convert(anthropic_body, "openai_chat", source_provider="anthropic")

        >>> # With shim transforms
        >>> body = convert(req, "anthropic", source_provider="deepseek", model="deepseek-r1")

        >>> # Force normalisation even for same-provider passthrough
        >>> body = {"messages": [...], "max_tokens": 256}
        >>> normalised = convert(body, "openai_chat", force_conversion=True)
    """
    from .pipeline import ConversionPipeline
    from .shims import get_shim

    src = _detect_source(source_body, source_provider)

    pipeline = ConversionPipeline(
        src,
        str(target_provider),
        source_shim=get_shim(src),
        target_shim=get_shim(str(target_provider)),
        upstream_model=model,
        force_conversion=force_conversion,
        # Library callers expect Google SDK format; gateway uses "rest"
        google_output_format="sdk",
    )
    return pipeline.convert_request(source_body)


def convert_response(
    response_body: dict[str, Any],
    request_body: dict[str, Any],
    source_provider: ProviderType | str,
    target_provider: ProviderType | str,
    *,
    model: str | None = None,
    force_conversion: bool = False,
) -> dict[str, Any]:
    """Convert a response body from target provider format back to source.

    Creates a :class:`~llm_rosetta.pipeline.ConversionPipeline`,
    replays the request conversion to establish context, then converts
    the response.

    Args:
        response_body: Target-format response body from upstream.
        request_body: The original source-format request body (needed
            to establish conversion context, e.g. custom-tool state).
        source_provider: The client/source provider type or shim name.
        target_provider: The upstream/target provider type or shim name.
        model: Optional model name for per-model shim overrides.
        force_conversion: When ``True``, run full conversion even for
            same-provider passthrough.

    Returns:
        Source-format response body.

    Raises:
        ValueError: If conversion fails.
    """
    from .pipeline import ConversionPipeline
    from .shims import get_shim

    src = str(source_provider)
    tgt = str(target_provider)

    pipeline = ConversionPipeline(
        src,
        tgt,
        source_shim=get_shim(src),
        target_shim=get_shim(tgt),
        upstream_model=model,
        force_conversion=force_conversion,
        # Library callers expect Google SDK format; gateway uses "rest"
        google_output_format="sdk",
    )
    # Replay request to populate pipeline context (tool state, metadata)
    pipeline.convert_request(request_body)
    return pipeline.convert_response(response_body)
