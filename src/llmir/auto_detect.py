"""
LLM Provider Auto-Detection

自动检测 LLM provider 请求体格式的工具函数
"""

from typing import Any, Dict, Literal, Optional

ProviderType = Literal["openai_chat", "openai_responses", "anthropic", "google"]


def detect_provider(body: Dict[str, Any]) -> Optional[ProviderType]:
    """自动检测 provider 类型

    基于请求体的结构特征识别 provider 类型。

    Args:
        body: Provider 请求体字典

    Returns:
        检测到的 provider 类型，如果无法识别则返回 None

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

    # 检测 Google GenAI: 必有 contents 字段
    if "contents" in body:
        contents = body["contents"]
        if isinstance(contents, list) and len(contents) > 0:
            # 验证是否有 parts 结构
            first_content = contents[0]
            if isinstance(first_content, dict) and "parts" in first_content:
                return "google"

    # 检测 OpenAI Responses API: 必有 input 或 output 字段
    if "input" in body or "output" in body:
        items = body.get("input") or body.get("output")
        if isinstance(items, list) and len(items) > 0:
            # 验证是否有 type 字段（message, function_call, etc.）
            first_item = items[0]
            if isinstance(first_item, dict) and "type" in first_item:
                item_type = first_item["type"]
                # Responses API 特有的类型
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

    # 检测 Anthropic 和 OpenAI Chat: 都有 messages 字段
    if "messages" in body:
        # Anthropic 特有字段检测（优先）
        if "system" in body and isinstance(body["system"], (str, list)):
            # Anthropic 使用独立的 system 参数
            return "anthropic"

        if "anthropic_version" in body or "max_tokens_to_sample" in body:
            return "anthropic"

        messages = body["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            first_message = messages[0]
            if isinstance(first_message, dict) and "content" in first_message:
                content = first_message["content"]

                # OpenAI Chat: content 是字符串
                if isinstance(content, str):
                    return "openai_chat"

                # 如果 content 是列表，需要区分 OpenAI 和 Anthropic
                if isinstance(content, list) and len(content) > 0:
                    first_part = content[0]
                    if isinstance(first_part, dict) and "type" in first_part:
                        part_type = first_part["type"]

                        # OpenAI Chat 特有的内容类型
                        if part_type in ["image_url", "input_audio"]:
                            return "openai_chat"

                        # Anthropic 特有的内容类型
                        if part_type in [
                            "image",
                            "tool_use",
                            "tool_result",
                            "thinking",
                            "document",
                        ]:
                            return "anthropic"

                        # 对于 "text" 类型，检查更深层的结构
                        if part_type == "text":
                            # 检查是否有 Anthropic 特有的嵌套结构
                            # Anthropic 的 image 有 source 字段
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

                            # 默认 text 类型无法明确区分，返回 openai_chat
                            # 因为 OpenAI 也支持 [{"type": "text", "text": "..."}] 格式
                            pass

        # 检查消息中是否有 tool_calls（OpenAI 特有）
        if messages and isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and "tool_calls" in msg:
                    return "openai_chat"

        # 如果无法明确区分，默认为 OpenAI Chat（更常见）
        return "openai_chat"

    # 无法识别
    return None


def get_converter_for_provider(provider: ProviderType):
    """根据 provider 类型获取对应的转换器

    Args:
        provider: Provider 类型

    Returns:
        对应的转换器实例

    Raises:
        ValueError: 如果 provider 类型不支持
    """
    from .converters import (
        AnthropicConverter,
        GoogleConverter,
        OpenAIChatConverter,
        OpenAIResponsesConverter,
    )

    converter_map = {
        "openai_chat": OpenAIChatConverter,
        "openai_responses": OpenAIResponsesConverter,
        "anthropic": AnthropicConverter,
        "google": GoogleConverter,
    }

    if provider not in converter_map:
        raise ValueError(f"Unsupported provider: {provider}")

    return converter_map[provider]()


def convert(
    source_body: Dict[str, Any],
    target_provider: ProviderType,
    source_provider: Optional[ProviderType] = None,
) -> Dict[str, Any]:
    """自动检测源 provider 并转换到目标 provider 格式

    这是一个便捷函数，自动检测源格式并执行转换。

    Args:
        source_body: 源 provider 的请求体
        target_provider: 目标 provider 类型
        source_provider: 可选的源 provider 类型，如果不提供则自动检测

    Returns:
        目标 provider 格式的请求体

    Raises:
        ValueError: 如果无法检测源 provider 或转换失败

    Examples:
        >>> # 自动检测并转换
        >>> openai_body = {"messages": [{"role": "user", "content": "Hello"}]}
        >>> google_body = convert(openai_body, "google")

        >>> # 指定源 provider
        >>> anthropic_body = {"messages": [...]}
        >>> openai_body = convert(anthropic_body, "openai_chat", source_provider="anthropic")
    """
    # 检测源 provider
    if source_provider is None:
        source_provider = detect_provider(source_body)
        if source_provider is None:
            raise ValueError(
                "Unable to detect source provider. Please specify source_provider explicitly."
            )

    # 如果源和目标相同，直接返回
    if source_provider == target_provider:
        return source_body

    # 获取转换器
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # 执行转换: source -> IR -> target
    ir_input = source_converter.from_provider(source_body)

    # 提取工具和工具选择（如果有）
    tools = source_body.get("tools")
    tool_choice = source_body.get("tool_choice")

    # 标准化工具定义格式
    # OpenAI Chat 格式的工具定义是 {"type": "function", "function": {...}}
    # 需要提取为标准格式 {"type": "function", "name": ..., "description": ..., "parameters": ...}
    if tools and isinstance(tools, list) and len(tools) > 0:
        normalized_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # 如果工具有 function 字段，提取它
                if "function" in tool:
                    func = tool["function"]
                    normalized_tool = {
                        "type": tool.get("type", "function"),
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                    normalized_tools.append(normalized_tool)
                else:
                    # 已经是标准格式
                    normalized_tools.append(tool)
        tools = normalized_tools if normalized_tools else tools

    # 转换到目标格式
    target_body, warnings = target_converter.to_provider(
        ir_input, tools=tools, tool_choice=tool_choice
    )

    # 如果有警告，可以选择记录或返回
    if warnings:
        # 可以添加到结果中或记录日志
        pass

    return target_body
