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


def get_converter_for_provider(provider: ProviderType):
    """根据 provider 类型获取对应的转换器
    Get the corresponding converter according to provider type

    Args:
        provider: Provider 类型 Provider type

    Returns:
        对应的转换器实例 Corresponding converter instance

    Raises:
        ValueError: 如果 provider 类型不支持 If provider type is not supported
    """
    from .converters.anthropic import AnthropicConverter
    from .converters.google_genai import GoogleConverter
    from .converters.openai_chat import OpenAIChatConverter
    from .converters.openai_responses import OpenAIResponsesConverter

    converter_map = {
        "openai_chat": OpenAIChatConverter,
        "openai_responses": OpenAIResponsesConverter,
        "open_responses": OpenAIResponsesConverter,
        "anthropic": AnthropicConverter,
        "google": GoogleConverter,
    }

    if provider not in converter_map:
        raise ValueError(f"Unsupported provider: {provider}")

    return converter_map[provider]()


def convert(
    source_body: dict[str, Any],
    target_provider: ProviderType,
    source_provider: ProviderType | None = None,
    *,
    force_conversion: bool = False,
) -> dict[str, Any]:
    """Auto-detect source provider and convert to target provider format.

    This is a convenience function that auto-detects the source format and
    performs conversion through the IR (Intermediate Representation).

    Args:
        source_body: Source provider request body.
        target_provider: Target provider type.
        source_provider: Optional source provider type.  Auto-detected from
            *source_body* when not provided.
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

        >>> # Force normalisation even for same-provider passthrough
        >>> body = {"messages": [...], "max_tokens": 256}
        >>> normalised = convert(body, "openai_chat", force_conversion=True)
    """
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

    # 获取转换器 Get converter
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # 执行转换: source -> IR -> target Perform conversion: source -> IR -> target
    ir_request = source_converter.request_from_provider(source_body)

    # 转换到目标格式 Convert to target format
    target_body, warnings = target_converter.request_to_provider(ir_request)

    # 如果有警告，可以选择记录或返回 If there are warnings, can choose to log or return
    if warnings:
        # 可以添加到结果中或记录日志
        pass

    return target_body
