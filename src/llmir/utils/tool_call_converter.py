"""
Tool Call Converter - 工具调用转换工具
Tool Call Converter - Tool call conversion utility

提供IR ToolCallPart与各provider工具调用格式之间的双向转换。
Provides bidirectional conversion between IR ToolCallPart and provider tool call formats.
"""

import json
import uuid
from typing import Any, Dict


class ToolCallConverter:
    """工具调用转换器
    Tool call converter

    处理IR ToolCallPart与各provider工具调用格式之间的转换。
    Handles conversion between IR ToolCallPart and provider tool call formats.
    支持的provider: openai_chat, openai_responses, anthropic, google
    Supported providers: openai_chat, openai_responses, anthropic, google
    """

    # ==================== IR → Provider ====================

    @staticmethod
    def to_openai_chat(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """将IR工具调用转换为OpenAI Chat格式
        Convert IR tool call to OpenAI Chat format

        Args:
            tool_call: IR格式的工具调用 IR format tool call

        Returns:
            OpenAI Chat格式的工具调用 OpenAI Chat format tool call

        Example:
            >>> tool_call = {
                ...     "type": "tool_call",
                ...     "tool_call_id": "call_123",
                ...     "tool_name": "get_weather",
                ...     "tool_input": {"city": "Beijing"},
                ...     "tool_type": "function"
                ... }
            >>> ToolCallConverter.to_openai_chat(tool_call)
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Beijing"}'
                }
            }
        """
        from .field_mapper import FieldMapper

        tool_type = tool_call.get("tool_type", "function")
        tool_call_id = FieldMapper.get_tool_id(tool_call)
        tool_name = FieldMapper.get_tool_name(tool_call)
        tool_input = FieldMapper.get_tool_input(tool_call)

        if tool_type == "function":
            return {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(tool_input)},
            }
        else:
            # 其他工具类型转换为自定义工具 Other tool types converted to custom tool
            return {
                "id": tool_call_id,
                "type": "custom",
                "custom": {
                    "name": f"{tool_type}_{tool_name}",
                    "input": json.dumps(tool_input),
                },
            }

    @staticmethod
    def to_openai_responses(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """将IR工具调用转换为OpenAI Responses API格式
        Convert IR tool call to OpenAI Responses API format

        Args:
            tool_call: IR格式的工具调用 IR format tool call

        Returns:
            OpenAI Responses API格式的工具调用 OpenAI Responses API format tool call

        Example:
            >>> tool_call = {
                ...     "type": "tool_call",
                ...     "tool_call_id": "call_123",
                ...     "tool_name": "mcp://server/tool",
                ...     "tool_input": {"query": "test"},
                ...     "tool_type": "mcp"
                ... }
            >>> ToolCallConverter.to_openai_responses(tool_call)
            {
                "type": "mcp_call",
                "id": "call_123",
                "name": "mcp://server/tool",
                "arguments": '{"query": "test"}',
                "server_label": "default",
                "status": "calling"
            }
        """
        from .field_mapper import FieldMapper

        tool_type = tool_call.get("tool_type", "function")
        tool_call_id = FieldMapper.get_tool_id(tool_call)
        tool_name = FieldMapper.get_tool_name(tool_call)
        tool_input = FieldMapper.get_tool_input(tool_call)

        # 序列化tool_input Serialize tool_input
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else tool_input
        )

        # 检测MCP调用 Detect MCP call
        if tool_name and tool_name.startswith("mcp://"):
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "server_label": tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "mcp":
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "server_label": tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "function":
            return {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
            }
        elif tool_type == "web_search":
            return {
                "type": "function_web_search",
                "call_id": tool_call_id,
                "query": tool_input.get("query", ""),
                "arguments": arguments,
            }
        elif tool_type == "code_interpreter":
            return {
                "type": "code_interpreter_call",
                "call_id": tool_call_id,
                "code": tool_input.get("code", ""),
                "arguments": arguments,
            }
        elif tool_type == "file_search":
            return {
                "type": "file_search_call",
                "call_id": tool_call_id,
                "query": tool_input.get("query", ""),
                "arguments": arguments,
            }
        else:
            # 默认转换为函数调用
            return {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": f"{tool_type}_{tool_name}",
                "arguments": arguments,
            }

    @staticmethod
    def to_anthropic(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """将IR工具调用转换为Anthropic格式
        Convert IR tool call to Anthropic format

        Args:
            tool_call: IR格式的工具调用 IR format tool call

        Returns:
            Anthropic格式的工具调用 Anthropic format tool call

        Example:
            >>> tool_call = {
                ...     "type": "tool_call",
                ...     "tool_call_id": "call_123",
                ...     "tool_name": "get_weather",
                ...     "tool_input": {"city": "Beijing"},
                ...     "tool_type": "function"
                ... }
            >>> ToolCallConverter.to_anthropic(tool_call)
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "get_weather",
                "input": {"city": "Beijing"}
            }
        """
        from .field_mapper import FieldMapper

        tool_type = tool_call.get("tool_type", "function")
        tool_id = FieldMapper.get_tool_id(tool_call)
        tool_name = FieldMapper.get_tool_name(tool_call)
        tool_input = FieldMapper.get_tool_input(tool_call)

        if tool_type == "function":
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            }
        elif tool_type == "web_search":
            # Anthropic有专门的web_search工具 Anthropic has a dedicated web_search tool
            return {
                "type": "server_tool_use",
                "id": tool_id,
                "name": "web_search",
                "input": tool_input,
            }
        else:
            # 其他类型转换为普通工具调用 Other types converted to normal tool call
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": f"{tool_type}_{tool_name}",
                "input": tool_input,
            }

    @staticmethod
    def to_google(
        tool_call: Dict[str, Any], preserve_metadata: bool = True
    ) -> Dict[str, Any]:
        """将IR工具调用转换为Google格式
        Convert IR tool call to Google format

        Args:
            tool_call: IR格式的工具调用 IR format tool call
            preserve_metadata: 是否保留provider_metadata中的thought_signature Whether to preserve thought_signature in provider_metadata

        Returns:
            Google格式的工具调用 Google format tool call

        Example:
            >>> tool_call = {
                ...     "type": "tool_call",
                ...     "tool_name": "get_weather",
                ...     "tool_input": {"city": "Beijing"},
                ...     "provider_metadata": {
                ...         "google": {"thought_signature": "abc123"}
                ...     }
                ... }
            >>> ToolCallConverter.to_google(tool_call)
            {
                "function_call": {
                    "name": "get_weather",
                    "args": {"city": "Beijing"}
                },
                "thoughtSignature": "abc123"
            }
        """
        from .field_mapper import FieldMapper

        tool_name = FieldMapper.get_tool_name(tool_call)
        tool_args = FieldMapper.get_tool_input(tool_call)

        part = {"function_call": {"name": tool_name, "args": tool_args}}

        # 保留thought_signature（对Gemini 3必需，对Gemini 2.5推荐） Preserve thought_signature (required for Gemini 3, recommended for Gemini 2.5)
        if preserve_metadata and "provider_metadata" in tool_call:
            metadata = tool_call["provider_metadata"]
            if "google" in metadata and "thought_signature" in metadata["google"]:
                part["thoughtSignature"] = metadata["google"]["thought_signature"]

        return part

    # ==================== Provider → IR ====================

    @staticmethod
    def from_openai_chat(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """将OpenAI Chat格式的工具调用转换为IR格式
        Convert OpenAI Chat format tool call to IR format

        Args:
            tool_call: OpenAI Chat格式的工具调用 OpenAI Chat format tool call

        Returns:
            IR格式的工具调用 IR format tool call

        Example:
            >>> tool_call = {
                ...     "id": "call_123",
                ...     "type": "function",
                ...     "function": {
                ...         "name": "get_weather",
                ...         "arguments": '{"city": "Beijing"}'
                ...     }
                ... }
            >>> ToolCallConverter.from_openai_chat(tool_call)
            {
                "type": "tool_call",
                "tool_call_id": "call_123",
                "tool_name": "get_weather",
                "tool_input": {"city": "Beijing"},
                "tool_type": "function"
            }
        """
        from ..types.ir import ToolCallPart

        if tool_call["type"] == "function":
            function = tool_call["function"]
            return ToolCallPart(
                type="tool_call",
                tool_call_id=tool_call["id"],
                tool_name=function["name"],
                tool_input=json.loads(function["arguments"]),
                tool_type="function",
            )
        elif tool_call["type"] == "custom":
            custom = tool_call["custom"]
            # 尝试解析工具类型 Try to parse tool type
            name = custom["name"]
            if "_" in name:
                tool_type, tool_name = name.split("_", 1)
            else:
                tool_type = "custom"
                tool_name = name

            return ToolCallPart(
                type="tool_call",
                tool_call_id=tool_call["id"],
                tool_name=tool_name,
                tool_input=json.loads(custom["input"]),
                tool_type=tool_type,
            )
        else:
            raise ValueError(f"Unsupported tool call type: {tool_call['type']}")

    @staticmethod
    def from_anthropic(block: Dict[str, Any]) -> Dict[str, Any]:
        """将Anthropic格式的工具调用转换为IR格式
        Convert Anthropic format tool call to IR format

        Args:
            block: Anthropic格式的工具调用块（tool_use 或 server_tool_use） Anthropic format tool call block (tool_use or server_tool_use)

        Returns:
            IR格式的工具调用 IR format tool call

        Example:
            >>> block = {
                ...     "type": "tool_use",
                ...     "id": "toolu_123",
                ...     "name": "get_weather",
                ...     "input": {"city": "Beijing"}
                ... }
            >>> ToolCallConverter.from_anthropic(block)
            {
                "type": "tool_call",
                "tool_call_id": "toolu_123",
                "tool_name": "get_weather",
                "tool_input": {"city": "Beijing"},
                "tool_type": "function"
            }
        """
        from ..types.ir import ToolCallPart

        block_type = block.get("type")

        if block_type == "tool_use":
            return ToolCallPart(
                type="tool_call",
                tool_call_id=block.get("id", ""),
                tool_name=block.get("name", ""),
                tool_input=block.get("input", {}),
                tool_type="function",
            )
        elif block_type == "server_tool_use":
            # Anthropic的服务器端工具 Anthropic server-side tool
            tool_name = block.get("name", "")
            tool_type = "web_search" if tool_name == "web_search" else "function"
            return ToolCallPart(
                type="tool_call",
                tool_call_id=block.get("id", ""),
                tool_name=tool_name,
                tool_input=block.get("input", {}),
                tool_type=tool_type,
            )
        else:
            raise ValueError(f"Unsupported Anthropic block type: {block_type}")

    @staticmethod
    def from_openai_responses(item: Dict[str, Any]) -> Dict[str, Any]:
        """将OpenAI Responses API格式的工具调用转换为IR格式
        Convert OpenAI Responses API format tool call to IR format

        Args:
            item: OpenAI Responses API格式的工具调用项 OpenAI Responses API format tool call item

        Returns:
            IR格式的工具调用 IR format tool call

        Example:
            >>> item = {
                ...     "type": "function_call",
                ...     "call_id": "call_123",
                ...     "name": "get_weather",
                ...     "arguments": '{"city": "Beijing"}'
                ... }
            >>> ToolCallConverter.from_openai_responses(item)
            {
                "type": "tool_call",
                "tool_call_id": "call_123",
                "tool_name": "get_weather",
                "tool_input": {"city": "Beijing"},
                "tool_type": "function"
            }
        """
        import json

        from ..types.ir import ToolCallPart

        item_type = item.get("type")

        # 解析 arguments Parse arguments
        arguments = item.get("arguments", {})
        if isinstance(arguments, dict):
            tool_input = arguments
        elif isinstance(arguments, str):
            # 尝试解析 JSON 字符串
            try:
                tool_input = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                # 如果不是 JSON，将字符串作为单个参数
                tool_input = {"input": arguments}
        else:
            tool_input = {}

        if item_type == "function_call":
            return ToolCallPart(
                type="tool_call",
                tool_call_id=item.get("call_id", item.get("id", "")),
                tool_name=item["name"],
                tool_input=tool_input,
                tool_type="function",
            )
        elif item_type == "mcp_call":
            # MCP调用可能使用 server/tool 字段或 name 字段 MCP call may use server/tool field or name field
            server = item.get("server", "")
            tool = item.get("tool", item.get("name", ""))
            tool_name = f"mcp://{server}/{tool}" if server and tool else tool

            return ToolCallPart(
                type="tool_call",
                tool_call_id=item["id"],
                tool_name=tool_name,
                tool_input=tool_input,
                tool_type="mcp",
            )
        elif item_type in ["shell_call", "computer_call", "code_interpreter_call"]:
            # 其他工具调用类型 Other tool call types
            tool_type_map = {
                "shell_call": "code_interpreter",
                "computer_call": "computer_use",
                "code_interpreter_call": "code_interpreter",
            }
            return ToolCallPart(
                type="tool_call",
                tool_call_id=item.get("call_id", item.get("id", "")),
                tool_name=item.get("name", item_type),
                tool_input=tool_input,
                tool_type=tool_type_map.get(item_type, "function"),
            )
        else:
            raise ValueError(f"Unsupported OpenAI Responses item type: {item_type}")

    @staticmethod
    def from_google(
        part: Dict[str, Any], preserve_metadata: bool = True
    ) -> Dict[str, Any]:
        """将Google格式的function_call转换为IR格式
        Convert Google format function_call to IR format

        Args:
            part: Google格式的Part（包含function_call） Google format Part (contains function_call)
            preserve_metadata: 是否保留thought_signature到provider_metadata Whether to preserve thought_signature to provider_metadata

        Returns:
            IR格式的工具调用 IR format tool call

        Example:
            >>> part = {
                ...     "function_call": {
                ...         "name": "get_weather",
                ...         "args": {"city": "Beijing"}
                ...     },
                ...     "thoughtSignature": "abc123"
                ... }
            >>> ToolCallConverter.from_google(part)
            {
                "type": "tool_call",
                "tool_call_id": "call_get_weather_...",
                "tool_name": "get_weather",
                "tool_input": {"city": "Beijing"},
                "tool_type": "function",
                "provider_metadata": {
                    "google": {"thought_signature": "abc123"}
                }
            }
        """
        from ..types.ir import ToolCallPart

        # 支持两种命名格式：function_call（SDK）和 functionCall（REST API） Support two naming formats: function_call (SDK) and functionCall (REST API)
        func_call = part.get("function_call") or part.get("functionCall")
        if not func_call:
            raise ValueError("Part does not contain function_call")

        # Google 的 function_call 可能没有 id 字段，我们需要生成一个唯一 ID Google function_call may not have id field, need to generate a unique ID
        tool_call_id = func_call.get("id")
        if not tool_call_id:
            # 生成一个基于函数名的唯一 ID Generate a unique ID based on function name
            tool_call_id = f"call_{func_call['name']}_{uuid.uuid4().hex[:8]}"

        # 构建基本的工具调用部分 Build basic tool call part
        tool_call_kwargs = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "tool_name": func_call["name"],
            "tool_input": func_call.get("args", {}),
            "tool_type": "function",
        }

        # 保存thought_signature到provider_metadata Save thought_signature to provider_metadata
        if preserve_metadata:
            # 支持两种命名：thoughtSignature（REST）和 thought_signature（SDK） Support two naming styles: thoughtSignature (REST) and thought_signature (SDK)
            thought_sig = part.get("thoughtSignature") or part.get("thought_signature")
            if thought_sig:
                tool_call_kwargs["provider_metadata"] = {
                    "google": {"thought_signature": thought_sig}
                }

        return ToolCallPart(**tool_call_kwargs)
