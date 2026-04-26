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


def get_converter_for_provider(provider: str):
    """Get the corresponding converter for a provider type or shim name.

    Accepts both base converter types (e.g. ``"openai_chat"``) and
    registered shim names (e.g. ``"deepseek"``).  Shim names are
    resolved to their base converter type via the shim registry.

    Args:
        provider: Provider type string or registered shim name.

    Returns:
        Corresponding converter instance.

    Raises:
        ValueError: If the provider is not a known type or shim name.
    """
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
        return converter_map[provider]()

    # Resolve through shim registry
    base = resolve_base(provider)
    if base in converter_map:
        return converter_map[base]()

    raise ValueError(f"Unsupported provider: {provider}")


def convert(
    source_body: dict[str, Any],
    target_provider: ProviderType | str,
    source_provider: ProviderType | str | None = None,
    *,
    model: str | None = None,
    force_conversion: bool = False,
) -> dict[str, Any]:
    """Auto-detect source provider and convert to target provider format.

    This is a convenience function that auto-detects the source format and
    performs conversion through the IR (Intermediate Representation).

    When *source_provider* or *target_provider* is a registered shim name
    (e.g. ``"deepseek"``), the shim's transforms are applied around the
    base converter.  If *model* is also provided, model-level transforms
    are merged with the provider-level ones via
    :func:`~llm_rosetta.shims.resolve_transforms`.

    Args:
        source_body: Source provider request body.
        target_provider: Target provider type or registered shim name.
        source_provider: Optional source provider type or shim name.
            Auto-detected from *source_body* when not provided.
        model: Optional model name for model-level transform lookup.
        force_conversion: When ``True``, always run the full conversion
            pipeline (source -> IR -> target) even when source and target
            providers are the same.  This normalises parameter names (e.g.
            ``max_tokens`` -> ``max_completion_tokens`` for OpenAI Chat) and
            ensures metadata is round-tripped consistently.

    Returns:
        Target provider format request body.

    Raises:
        ValueError: If source provider cannot be detected or conversion fails.

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
    from .shims import get_shim
    from .shims.transforms import apply_transforms, resolve_transforms

    # Detect source provider
    if source_provider is None:
        source_provider = detect_provider(source_body)
        if source_provider is None:
            raise ValueError(
                "Unable to detect source provider. Please specify source_provider explicitly."
            )

    # Skip conversion when source == target (unless forced)
    if source_provider == target_provider and not force_conversion:
        return source_body

    # --- Resolve shims and transforms ---
    source_shim = get_shim(source_provider)
    target_shim = get_shim(target_provider)

    source_model_shim = (
        source_shim.get_model_shim(model) if source_shim and model else None
    )
    target_model_shim = (
        target_shim.get_model_shim(model) if target_shim and model else None
    )

    # Merge provider + model transforms (provider first, model after)
    if source_shim:
        source_from_t, _ = resolve_transforms(source_shim, source_model_shim)
    else:
        source_from_t = ()

    if target_shim:
        _, target_to_t = resolve_transforms(target_shim, target_model_shim)
    else:
        target_to_t = ()

    # --- Apply source from_transforms ---
    body = apply_transforms(source_from_t, source_body)

    # --- Core conversion: source → IR → target ---
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    ir_request = source_converter.request_from_provider(body)
    target_body, _warnings = target_converter.request_to_provider(ir_request)

    # --- Apply target to_transforms ---
    target_body = apply_transforms(target_to_t, target_body)

    return target_body
