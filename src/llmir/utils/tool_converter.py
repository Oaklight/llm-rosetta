"""
工具定义和选择的转换工具
Tool definition and selection conversion utility

统一处理不同provider之间的工具定义和工具选择配置转换
Unified handling of tool definition and tool choice configuration conversion between different providers
"""

from typing import Any, Dict, List, Optional, Union

from ..types.ir import ToolChoice, ToolDefinition


class ToolConverter:
    """工具定义和选择的转换工具
    Tool definition and selection conversion utility
    """

    @staticmethod
    def convert_tool_definition(
        tool: ToolDefinition, target_format: str
    ) -> Dict[str, Any]:
        """
        转换工具定义到目标provider格式
        Convert tool definition to target provider format

        Args:
            tool: IR格式的工具定义 IR format tool definition
            target_format: 目标格式 ("openai_chat", "openai_responses", "anthropic", "google") Target format

        Returns:
            目标格式的工具定义 Tool definition in target format

        Raises:
            ValueError: 如果目标格式不支持 If target format is not supported
        """
        if target_format == "openai_chat":
            return ToolConverter._to_openai_chat_tool(tool)
        elif target_format == "openai_responses":
            return ToolConverter._to_openai_responses_tool(tool)
        elif target_format == "anthropic":
            return ToolConverter._to_anthropic_tool(tool)
        elif target_format == "google":
            return ToolConverter._to_google_tool(tool)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    @staticmethod
    def _to_openai_chat_tool(tool: ToolDefinition) -> Dict[str, Any]:
        """转换为OpenAI Chat Completions格式
        Convert to OpenAI Chat Completions format
        """
        # 处理嵌套的function结构（测试中使用的格式） Handle nested function structure (format used in tests)
        if "function" in tool and isinstance(tool["function"], dict):
            return tool  # 已经是OpenAI格式，直接返回 Already in OpenAI format, return directly

        if tool["type"] == "function":
            return {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
        else:
            # 其他工具类型转换为自定义工具 Other tool types converted to custom tools
            return {
                "type": "custom",
                "custom": {
                    "name": f"{tool['type']}_{tool['name']}",
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }

    @staticmethod
    def _to_openai_responses_tool(tool: ToolDefinition) -> Dict[str, Any]:
        """转换为OpenAI Responses API格式"""
        if tool["type"] == "function":
            return {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "strict": False,  # 默认使用非strict模式以支持可选参数 Default to non-strict mode to support optional parameters
            }
        else:
            # 其他工具类型 Other tool types
            return {
                "type": "custom",
                "name": f"{tool['type']}_{tool['name']}",
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }

    @staticmethod
    def _to_anthropic_tool(tool: ToolDefinition) -> Dict[str, Any]:
        """转换为Anthropic格式
        Convert to Anthropic format
        """
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
        }

    @staticmethod
    def _to_google_tool(tool: ToolDefinition) -> Dict[str, Any]:
        """转换为Google GenAI格式
        Convert to Google GenAI format
        """
        return {
            "function_declarations": [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
            ]
        }

    @staticmethod
    def convert_tool_choice(
        tool_choice: ToolChoice, target_format: str
    ) -> Union[str, Dict[str, Any], None]:
        """
        转换工具选择配置到目标provider格式
        Convert tool choice configuration to target provider format

        Args:
            tool_choice: IR格式的工具选择配置 IR format tool choice configuration
            target_format: 目标格式 ("openai", "anthropic", "google") Target format

        Returns:
            目标格式的工具选择配置 Tool choice configuration in target format

        Raises:
            ValueError: 如果目标格式不支持 If target format is not supported
        """
        if target_format in ["openai", "openai_chat", "openai_responses"]:
            return ToolConverter._to_openai_tool_choice(tool_choice)
        elif target_format == "anthropic":
            return ToolConverter._to_anthropic_tool_choice(tool_choice)
        elif target_format == "google":
            return ToolConverter._to_google_tool_choice(tool_choice)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    @staticmethod
    def _to_openai_tool_choice(tool_choice: ToolChoice) -> Union[str, Dict[str, Any]]:
        """转换为OpenAI格式
        Convert to OpenAI format
        """
        # 支持测试中使用的"type"字段 Support "type" field used in tests
        mode = tool_choice.get("mode") or tool_choice.get("type")

        if mode == "none":
            return "none"
        elif mode == "auto":
            return "auto"
        elif mode == "any" or mode == "required":
            return "required"
        elif mode == "tool" or mode == "function":
            # 支持多种字段名称 Support multiple field names
            tool_name = tool_choice.get("tool_name")
            if not tool_name and "function" in tool_choice:
                tool_name = tool_choice["function"].get("name")
            if tool_name:
                # 注意：这里是 provider 格式的字典，不是 IR 格式 Note: this is provider format dict, not IR format
                return {"type": "function", "function": {"name": tool_name}}
            else:
                return "required"
        else:
            raise ValueError(f"Unsupported tool choice mode: {mode}")

    @staticmethod
    def _to_anthropic_tool_choice(tool_choice: ToolChoice) -> Dict[str, Any]:
        """转换为Anthropic格式
        Convert to Anthropic format
        """
        mode = tool_choice.get("mode")
        result = {}

        if mode == "none":
            result["type"] = "none"
        elif mode == "auto":
            result["type"] = "auto"
        elif mode == "any":
            result["type"] = "any"
        elif mode == "tool":
            result["type"] = "tool"
            tool_name = tool_choice.get("tool_name")
            if tool_name:
                result["name"] = tool_name
        else:
            raise ValueError(f"Unsupported tool choice mode: {mode}")

        # 处理并行工具使用选项 Handle parallel tool usage option
        if tool_choice.get("disable_parallel"):
            result["disable_parallel_tool_use"] = True

        return result

    @staticmethod
    def _to_google_tool_choice(tool_choice: ToolChoice) -> Optional[Dict[str, Any]]:
        """转换为Google GenAI格式
        Convert to Google GenAI format
        """
        mode = tool_choice.get("mode")

        if mode == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        elif mode == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}
        elif mode == "any":
            return {"function_calling_config": {"mode": "ANY"}}
        elif mode == "tool":
            config = {"function_calling_config": {"mode": "ANY"}}
            tool_name = tool_choice.get("tool_name")
            if tool_name:
                config["function_calling_config"]["allowed_function_names"] = [
                    tool_name
                ]
            return config
        else:
            return None

    @staticmethod
    def batch_convert_tools(
        tools: List[ToolDefinition], target_format: str
    ) -> List[Dict[str, Any]]:
        """
        批量转换工具定义
        Batch convert tool definitions

        Args:
            tools: IR格式的工具定义列表 List of IR format tool definitions
            target_format: 目标格式 Target format

        Returns:
            目标格式的工具定义列表 List of tool definitions in target format
        """
        return [
            ToolConverter.convert_tool_definition(tool, target_format) for tool in tools
        ]
