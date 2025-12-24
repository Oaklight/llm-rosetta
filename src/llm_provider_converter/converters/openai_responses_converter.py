"""
LLM Provider Converter - OpenAI Responses API Converter

实现IR与OpenAI Responses API格式之间的转换
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..types.ir import (
    IRInput,
    ToolChoice,
    ToolDefinition,
    is_extension_item,
    is_message,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
)
from .base import BaseConverter


class OpenAIResponsesConverter(BaseConverter):
    """OpenAI Responses API格式转换器"""

    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        将IR格式转换为OpenAI Responses API格式

        Responses API使用扁平的项目列表，每个项目可以是消息、工具调用或其他类型
        """
        # 验证输入
        validation_errors = self.validate_ir_input(ir_input)
        if validation_errors:
            raise ValueError(f"Invalid IR input: {validation_errors}")

        items = []
        warnings = []

        for item in ir_input:
            if is_message(item):
                message = item  # type: ignore

                if message["role"] in ["system", "user", "developer"]:
                    # 基本消息类型
                    content_list = []
                    tool_calls = []

                    for part in message["content"]:
                        if is_text_part(part):
                            content_list.append(
                                {"type": "input_text", "text": part["text"]}
                            )
                        elif part["type"] == "image":
                            content_list.append(self._convert_image_to_responses(part))
                        elif part["type"] == "file":
                            content_list.append(self._convert_file_to_responses(part))
                        elif is_tool_call_part(part):
                            # Responses API中工具调用是独立的项目
                            tool_calls.append(
                                self._convert_tool_call_to_responses(part)
                            )
                        elif is_tool_result_part(part):
                            # 工具结果转换为函数调用输出
                            # 支持多种字段名称
                            result_content = part.get("result") or part.get("content")
                            items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": part["tool_call_id"],
                                    "output": str(result_content),
                                }
                            )
                        elif part["type"] == "reasoning":
                            # Responses API支持推理
                            items.append(
                                {"type": "reasoning", "reasoning": part["reasoning"]}
                            )

                    # 添加消息（如果有内容）
                    if content_list:
                        items.append(
                            {
                                "type": "message",
                                "role": message["role"],
                                "content": content_list,
                            }
                        )

                    # 添加工具调用
                    items.extend(tool_calls)

                elif message["role"] == "assistant":
                    # Assistant消息处理
                    content_list = []
                    tool_calls = []

                    for part in message["content"]:
                        if is_text_part(part):
                            # 检查是否是推理文本
                            if part.get("reasoning"):
                                items.append(
                                    {"type": "reasoning", "content": part["text"]}
                                )
                            else:
                                content_list.append(
                                    {"type": "input_text", "text": part["text"]}
                                )
                        elif is_tool_call_part(part):
                            tool_calls.append(
                                self._convert_tool_call_to_responses(part)
                            )
                        elif part["type"] == "reasoning":
                            items.append(
                                {"type": "reasoning", "reasoning": part["reasoning"]}
                            )

                    # 添加assistant消息（如果有文本内容）
                    if content_list:
                        items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": content_list,
                            }
                        )

                    # 添加工具调用
                    items.extend(tool_calls)

            elif is_extension_item(item):
                extension = item  # type: ignore
                extension_type = extension.get("type")

                if extension_type == "system_event":
                    # Responses API支持系统事件
                    items.append(
                        {
                            "type": "system_event",
                            "event_type": extension.get("event_type"),
                            "timestamp": extension.get("timestamp"),
                            "message": extension.get("message", ""),
                        }
                    )
                elif extension_type == "tool_chain_node":
                    warnings.append("Tool chain converted to sequential calls")
                    tool_call = extension.get("tool_call")
                    if tool_call:
                        items.append(self._convert_tool_call_to_responses(tool_call))
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果
        result = {"input": items}

        # 添加工具定义
        if tools:
            result["tools"] = [
                self._convert_tool_definition_to_responses(tool) for tool in tools
            ]

        # 添加工具选择
        if tool_choice:
            result["tool_choice"] = self._convert_tool_choice_to_responses(tool_choice)

        return result, warnings

    def from_provider(self, provider_data: Any) -> IRInput:
        """
        将OpenAI Responses API格式转换为IR格式
        """
        if not isinstance(provider_data, dict):
            raise ValueError("OpenAI Responses data must be a dictionary")

        items = provider_data.get("input", [])
        if not isinstance(items, list):
            raise ValueError("OpenAI Responses input must be a list")

        ir_input = []
        current_message = None

        for item in items:
            item_type = item.get("type")

            if item_type == "message":
                # 处理消息
                if current_message:
                    ir_input.append(current_message)

                role = item["role"]
                content = item["content"]
                ir_content = []

                if isinstance(content, str):
                    ir_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for part in content:
                        if part["type"] in ["input_text", "text"]:
                            ir_content.append({"type": "text", "text": part["text"]})
                        elif part["type"] == "input_image":
                            ir_content.append(self._convert_image_from_responses(part))
                        elif part["type"] == "input_file":
                            ir_content.append(self._convert_file_from_responses(part))

                current_message = {"role": role, "content": ir_content}

            elif item_type == "function_call":
                # 函数调用转换为工具调用
                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    tool_input = arguments
                else:
                    tool_input = json.loads(arguments)

                if current_message and current_message["role"] == "assistant":
                    current_message["content"].append(
                        {
                            "type": "tool_call",
                            "tool_call_id": item.get("call_id", item.get("id")),
                            "tool_name": item["name"],
                            "tool_input": tool_input,
                            "tool_type": "function",
                        }
                    )
                else:
                    # 创建新的assistant消息
                    if current_message:
                        ir_input.append(current_message)
                    current_message = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "tool_call_id": item.get("call_id", item.get("id")),
                                "tool_name": item["name"],
                                "tool_input": tool_input,
                                "tool_type": "function",
                            }
                        ],
                    }

            elif item_type == "function_call_output":
                # 函数调用输出转换为工具结果
                if current_message:
                    ir_input.append(current_message)

                current_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": item["call_id"],
                            "result": item["output"],
                        }
                    ],
                }

            elif item_type == "mcp_call":
                # MCP调用转换为工具调用
                server = item.get("server", "")
                tool = item.get("tool", item.get("name", ""))
                tool_name = f"mcp://{server}/{tool}" if server and tool else tool

                if current_message and current_message["role"] == "assistant":
                    current_message["content"].append(
                        {
                            "type": "tool_call",
                            "tool_call_id": item["id"],
                            "tool_name": tool_name,
                            "tool_input": item.get("arguments", {}),
                            "tool_type": "mcp",
                        }
                    )
                else:
                    if current_message:
                        ir_input.append(current_message)
                    current_message = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "tool_call_id": item["id"],
                                "tool_name": tool_name,
                                "tool_input": item.get("arguments", {}),
                                "tool_type": "mcp",
                            }
                        ],
                    }

            elif item_type == "mcp_call_output":
                # MCP调用输出转换为工具结果
                if current_message:
                    ir_input.append(current_message)

                current_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": item["call_id"],
                            "result": item["output"],
                        }
                    ],
                }

            elif item_type == "reasoning":
                # 推理内容转换为带reasoning标志的文本
                reasoning_content = item.get("reasoning") or item.get("content")
                if current_message:
                    current_message["content"].append(
                        {"type": "text", "text": reasoning_content, "reasoning": True}
                    )
                else:
                    # 创建新的assistant消息包含推理
                    current_message = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": reasoning_content,
                                "reasoning": True,
                            }
                        ],
                    }

            elif item_type == "system_event":
                # 系统事件转换为扩展项
                if current_message:
                    ir_input.append(current_message)
                    current_message = None

                ir_input.append(
                    {
                        "type": "system_event",
                        "event_type": item.get("event_type", "unknown"),
                        "timestamp": item.get("timestamp", ""),
                        "message": item.get("message", ""),
                    }
                )

            elif item_type in ["shell_call", "computer_call", "code_interpreter_call"]:
                # 其他工具调用类型
                tool_type_map = {
                    "shell_call": "code_interpreter",
                    "computer_call": "computer_use",
                    "code_interpreter_call": "code_interpreter",
                }

                if current_message and current_message["role"] == "assistant":
                    current_message["content"].append(
                        {
                            "type": "tool_call",
                            "tool_call_id": item.get("call_id", item.get("id")),
                            "tool_name": item.get("name", item_type),
                            "tool_input": item.get("arguments", {}),
                            "tool_type": tool_type_map.get(item_type, "function"),
                        }
                    )

        # 添加最后一个消息
        if current_message:
            ir_input.append(current_message)

        return ir_input

    def _convert_image_to_responses(self, image_part: Dict[str, Any]) -> Dict[str, Any]:
        """将IR图像转换为Responses API格式"""
        result = {"type": "input_image", "detail": image_part.get("detail", "auto")}

        # 支持多种URL字段名称
        url = image_part.get("image_url") or image_part.get("url")
        if url:
            result["image_url"] = url
        elif "image_data" in image_part:
            image_data = image_part["image_data"]
            data_url = f"data:{image_data['media_type']};base64,{image_data['data']}"
            result["image_url"] = data_url
        else:
            raise ValueError("Image part must have either image_url/url or image_data")

        return result

    def _convert_file_to_responses(self, file_part: Dict[str, Any]) -> Dict[str, Any]:
        """将IR文件转换为Responses API格式"""
        result = {
            "type": "input_file",
            "filename": file_part.get("file_name", "unknown"),
        }

        if "file_data" in file_part:
            result["file_data"] = file_part["file_data"]["data"]
        elif "file_url" in file_part:
            result["file_url"] = file_part["file_url"]
        else:
            raise ValueError("File part must have either file_data or file_url")

        return result

    def _convert_tool_call_to_responses(
        self, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将IR工具调用转换为Responses API格式"""
        tool_type = tool_call.get("tool_type", "function")

        # 支持多种字段名称映射
        tool_call_id = tool_call.get("tool_call_id") or tool_call.get("id")
        tool_name = tool_call.get("tool_name") or tool_call.get("name")
        tool_input = tool_call.get("tool_input") or tool_call.get("arguments", {})

        # 检测MCP调用
        if tool_name and tool_name.startswith("mcp://"):
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else tool_input,
                "server_label": tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "mcp":
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else tool_input,
                "server_label": tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "function":
            return {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": tool_name,
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else tool_input,
            }
        elif tool_type == "web_search":
            return {
                "type": "function_web_search",
                "call_id": tool_call_id,
                "query": tool_input.get("query", ""),
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else tool_input,
            }
        elif tool_type == "code_interpreter":
            return {
                "type": "code_interpreter_call",
                "call_id": tool_call_id,
                "code": tool_input.get("code", ""),
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else tool_input,
            }
        elif tool_type == "file_search":
            return {
                "type": "file_search_call",
                "call_id": tool_call_id,
                "query": tool_input.get("query", ""),
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else tool_input,
            }
        else:
            # 默认转换为函数调用
            return {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": f"{tool_type}_{tool_name}",
                "arguments": json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else tool_input,
            }

    def _convert_image_from_responses(
        self, image_part: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将Responses API图像转换为IR格式"""
        result = {"type": "image", "detail": image_part.get("detail", "auto")}

        if "image_url" in image_part:
            url = image_part["image_url"]
            if url.startswith("data:"):
                # Base64数据URL
                import re

                match = re.match(r"data:([^;]+);base64,(.+)", url)
                if match:
                    media_type, data = match.groups()
                    result["image_data"] = {"data": data, "media_type": media_type}
                else:
                    result["image_url"] = url
            else:
                result["image_url"] = url
        elif "file_id" in image_part:
            # 文件ID形式，转换为引用
            result["file_id"] = image_part["file_id"]

        return result

    def _convert_file_from_responses(self, file_part: Dict[str, Any]) -> Dict[str, Any]:
        """将Responses API文件转换为IR格式"""
        result = {"type": "file", "file_name": file_part.get("filename", "unknown")}

        if "file_data" in file_part:
            # 假设是base64编码的数据
            result["file_data"] = {
                "data": file_part["file_data"],
                "media_type": "application/octet-stream",  # 默认类型
            }
        elif "file_url" in file_part:
            result["file_url"] = file_part["file_url"]
        elif "file_id" in file_part:
            result["file_id"] = file_part["file_id"]

        return result

    def _convert_tool_definition_to_responses(
        self, tool: ToolDefinition
    ) -> Dict[str, Any]:
        """将IR工具定义转换为Responses API格式"""
        # 处理测试中传入的OpenAI格式工具定义
        if "function" in tool and isinstance(tool["function"], dict):
            # 这已经是OpenAI格式，直接返回
            return tool

        # 处理IR格式的工具定义
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
            # 其他工具类型转换为自定义工具
            return {
                "type": "custom",
                "custom": {
                    "name": f"{tool['type']}_{tool['name']}",
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }

    def _convert_tool_choice_to_responses(
        self, tool_choice: ToolChoice
    ) -> Union[str, Dict[str, Any]]:
        """将IR工具选择转换为Responses API格式"""
        # 支持测试中使用的"type"字段
        mode = tool_choice.get("mode") or tool_choice.get("type")

        if mode == "none":
            return "none"
        elif mode == "auto":
            return "auto"
        elif mode == "any" or mode == "required":
            return "required"  # Responses API使用"required"表示必须使用工具
        elif mode == "tool" or mode == "function":
            # 支持多种字段名称
            tool_name = tool_choice.get("tool_name")
            if not tool_name and "function" in tool_choice:
                tool_name = tool_choice["function"]["name"]
            return {"type": "function", "function": {"name": tool_name}}
        else:
            raise ValueError(f"Unsupported tool choice mode: {mode}")
