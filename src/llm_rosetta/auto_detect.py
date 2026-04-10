"""
LLM Provider Auto-Detection

自动检测 LLM provider 请求体格式的工具函数
Utility functions for auto-detecting LLM provider request body formats
"""

from typing import Any, Literal

ProviderType = Literal[
    "openai_chat", "openai_responses", "open_responses", "anthropic", "google"
]


def detect_provider(body: dict[str, Any]) -> ProviderType | None:
    """自动检测 provider 类型
    Auto-detect provider type

    基于请求体的结构特征识别 provider 类型。
    Identify provider type based on request body structure features.

    Args:
        body: Provider 请求体字典 Provider request body dict

    Returns:
        检测到的 provider 类型，如果无法识别则返回 None Detected provider type, returns None if not recognized

    Examples:
        >>> # OpenAI Chat Completions
        >>> detect_provider({"messages": [{"role": "user", "content": "Hello"}]})
        'openai_chat'

        >>> # OpenAI Responses API
        >>> detect_provider({"input": [{"type": "message", "role": "user"}]})
        'openai_responses'

        >>> # Anthropic
        >>> detect_provider({"messages": [{"role": "user", "content": [{"type": "text"}]}]})
        'anthropic'

        >>> # Google GenAI
        >>> detect_provider({"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]})
        'google'
    """
    if not isinstance(body, dict):
        return None

    # 检测 Google GenAI: 必有 contents 字段 Detect Google GenAI: must have contents field
    if "contents" in body:
        contents = body["contents"]
        if isinstance(contents, list) and len(contents) > 0:
            # 验证是否有 parts 结构
            first_content = contents[0]
            if isinstance(first_content, dict) and "parts" in first_content:
                return "google"

    # 检测 OpenAI Responses API: 必有 input 或 output 字段 Detect OpenAI Responses API: must have input or output field
    if "input" in body or "output" in body:
        items = body.get("input") or body.get("output")
        if isinstance(items, list) and len(items) > 0:
            # 验证是否有 type 字段（message, function_call, etc.） Verify if there is a type field (message, function_call, etc.)
            first_item = items[0]
            if isinstance(first_item, dict) and "type" in first_item:
                item_type = first_item["type"]
                # Responses API 特有的类型 Types specific to Responses API
                if item_type in [
                    "message",
                    "function_call",
                    "function_call_output",
                    "mcp_call",
                    "mcp_call_output",
                    "reasoning",
                    "system_event",
                    "input_text",
                    "output_text",
                ]:
                    return "openai_responses"

    # 检测 Anthropic 和 OpenAI Chat: 都有 messages 字段 Detect Anthropic and OpenAI Chat: both have messages field
    if "messages" in body:
        # Anthropic 特有字段检测（优先） Anthropic specific field detection (priority)
        if "system" in body and isinstance(body["system"], (str, list)):
            # Anthropic 使用独立的 system 参数 Anthropic uses a separate system parameter
            return "anthropic"

        if "anthropic_version" in body or "max_tokens_to_sample" in body:
            return "anthropic"

        messages = body["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            first_message = messages[0]
            if isinstance(first_message, dict) and "content" in first_message:
                content = first_message["content"]

                # OpenAI Chat: content 是字符串 OpenAI Chat: content is string
                if isinstance(content, str):
                    return "openai_chat"

                # 如果 content 是列表，需要区分 OpenAI 和 Anthropic If content is a list, need to distinguish OpenAI and Anthropic
                if isinstance(content, list) and len(content) > 0:
                    first_part = content[0]
                    if isinstance(first_part, dict) and "type" in first_part:
                        part_type = first_part["type"]

                        # OpenAI Chat 特有的内容类型 Content types specific to OpenAI Chat
                        if part_type in ["image_url", "input_audio"]:
                            return "openai_chat"

                        # Anthropic 特有的内容类型 Content types specific to Anthropic
                        if part_type in [
                            "image",
                            "tool_use",
                            "tool_result",
                            "thinking",
                            "document",
                        ]:
                            return "anthropic"

                        # 对于 "text" 类型，检查更深层的结构 For "text" type, check deeper structure
                        if part_type == "text":
                            # 检查是否有 Anthropic 特有的嵌套结构
                            # Anthropic 的 image 有 source 字段 Anthropic's image has source field
                            for part in content:
                                if isinstance(part, dict):
                                    if part.get("type") == "image" and "source" in part:
                                        return "anthropic"
                                    if part.get("type") in [
                                        "tool_use",
                                        "thinking",
                                        "document",
                                    ]:
                                        return "anthropic"

                            # 默认 text 类型无法明确区分，返回 openai_chat Default text type cannot be clearly distinguished, return openai_chat
                            # 因为 OpenAI 也支持 [{"type": "text", "text": "..."}] 格式 Because OpenAI also supports [{"type": "text", "text": "..."}] format
                            pass

        # 检查消息中是否有 tool_calls（OpenAI 特有） Check if there are tool_calls in messages (OpenAI specific)
        if messages and isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and "tool_calls" in msg:
                    return "openai_chat"

        # 如果无法明确区分，默认为 OpenAI Chat（更常见） If cannot be clearly distinguished, default to OpenAI Chat (more common)
        return "openai_chat"

    # 无法识别 Cannot recognize
    return None


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
