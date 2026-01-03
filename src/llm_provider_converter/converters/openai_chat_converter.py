"""
LLM Provider Converter - OpenAI Chat Completions Converter

实现IR与OpenAI Chat Completions API格式之间的转换
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..types.ir import (
    ContentPart,
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


class OpenAIChatConverter(BaseConverter):
    """OpenAI Chat Completions API格式转换器"""

    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        将IR格式转换为OpenAI Chat Completions格式

        OpenAI使用扁平结构，需要将工具调用提取到消息级别
        """
        # 验证输入
        validation_errors = self.validate_ir_input(ir_input)
        if validation_errors:
            raise ValueError(f"Invalid IR input: {validation_errors}")

        messages = []
        warnings = []

        for item in ir_input:
            if is_message(item):
                message = item  # type: ignore

                if message["role"] == "system":
                    # System消息：直接转换
                    openai_message = {
                        "role": "system",
                        "content": self._extract_text_content(message["content"]),
                    }
                    messages.append(openai_message)

                elif message["role"] == "user":
                    # User消息：处理多模态内容和工具结果
                    content_parts = []
                    tool_results = []

                    for part in message["content"]:
                        if is_text_part(part):
                            content_parts.append({"type": "text", "text": part["text"]})
                        elif part["type"] == "image":
                            content_parts.append(self._convert_image_to_openai(part))
                        elif part["type"] == "file":
                            content_parts.append(self._convert_file_to_openai(part))
                        elif is_tool_result_part(part):
                            # OpenAI使用tool角色处理工具结果
                            tool_results.append(part)
                        elif part["type"] == "reasoning":
                            warnings.append(
                                "Reasoning content not supported in OpenAI Chat Completions, ignored"
                            )

                    # 添加user消息（如果有非工具结果内容）
                    if content_parts:
                        openai_message = {
                            "role": "user",
                            "content": content_parts
                            if len(content_parts) > 1
                            else content_parts[0]["text"]
                            if content_parts
                            else "",
                        }
                        messages.append(openai_message)

                    # 添加tool消息（工具结果）
                    for tool_result in tool_results:
                        # 支持多种字段名称
                        result_content = tool_result.get("result") or tool_result.get(
                            "content"
                        )
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_result["tool_call_id"],
                            "content": str(result_content),
                        }
                        messages.append(tool_message)

                elif message["role"] == "assistant":
                    # Assistant消息：处理工具调用
                    text_parts = []
                    tool_calls = []

                    for part in message["content"]:
                        if is_text_part(part):
                            text_parts.append(part["text"])
                        elif is_tool_call_part(part):
                            tool_calls.append(self._convert_tool_call_to_openai(part))
                        elif part["type"] == "reasoning":
                            warnings.append(
                                "Reasoning content not supported in OpenAI Chat Completions, ignored"
                            )

                    openai_message = {"role": "assistant"}

                    # 添加文本内容
                    if text_parts:
                        openai_message["content"] = " ".join(text_parts)

                    # 添加工具调用
                    if tool_calls:
                        openai_message["tool_calls"] = tool_calls
                        # 当有工具调用时，content应该为None或不设置
                        if not text_parts:
                            openai_message["content"] = None

                    # OpenAI要求assistant消息必须有content或tool_calls
                    if not text_parts and not tool_calls:
                        openai_message["content"] = ""

                    messages.append(openai_message)

            elif is_extension_item(item):
                extension = item  # type: ignore
                extension_type = extension.get("type")

                if extension_type == "system_event":
                    warnings.append(
                        f"System event ignored: {extension.get('event_type', 'unknown')}"
                    )
                elif extension_type == "tool_chain_node":
                    warnings.append("Tool chain converted to sequential calls")
                    # 可以将工具链节点展开为普通工具调用
                    tool_call = extension.get("tool_call")
                    if tool_call:
                        # 创建一个assistant消息包含工具调用
                        openai_message = {
                            "role": "assistant",
                            "tool_calls": [
                                self._convert_tool_call_to_openai(tool_call)
                            ],
                        }
                        messages.append(openai_message)
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果
        result = {"messages": messages}

        # 添加工具定义
        if tools:
            result["tools"] = [
                self._convert_tool_definition_to_openai(tool) for tool in tools
            ]

        # 添加工具选择
        if tool_choice:
            result["tool_choice"] = self._convert_tool_choice_to_openai(tool_choice)

        return result, warnings

    def from_provider(self, provider_data: Any) -> IRInput:
        """
        将OpenAI Chat Completions格式转换为IR格式
        """
        if not isinstance(provider_data, dict):
            raise ValueError("OpenAI data must be a dictionary")

        ir_input = []

        # Handle both a full payload (with a 'messages' key) and a single message dict
        if "messages" in provider_data:
            messages_to_process = provider_data["messages"]
        elif "role" in provider_data:
            messages_to_process = [provider_data]
        else:
            # If it's neither, it might be an empty dict or something else, process nothing.
            messages_to_process = []

        for msg in messages_to_process:
            role = msg["role"]

            if role == "system":
                # System消息
                ir_input.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": msg["content"]}],
                    }
                )

            elif role == "user":
                # User消息
                content = msg["content"]
                ir_content = []

                if isinstance(content, str):
                    ir_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for part in content:
                        if part["type"] == "text":
                            ir_content.append({"type": "text", "text": part["text"]})
                        elif part["type"] == "image_url":
                            ir_content.append(self._convert_image_from_openai(part))
                        elif part["type"] == "input_audio":
                            # 音频暂时转换为文件类型
                            ir_content.append(
                                {
                                    "type": "file",
                                    "file_data": {
                                        "data": part["input_audio"]["data"],
                                        "media_type": f"audio/{part['input_audio']['format']}",
                                    },
                                }
                            )

                ir_input.append({"role": "user", "content": ir_content})

            elif role == "assistant":
                # Assistant消息
                ir_content = []

                # 处理文本内容
                if "content" in msg and msg["content"]:
                    ir_content.append({"type": "text", "text": msg["content"]})

                # 处理工具调用
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        ir_content.append(
                            self._convert_tool_call_from_openai(tool_call)
                        )

                ir_input.append({"role": "assistant", "content": ir_content})

            elif role == "tool":
                # Tool消息转换为user消息中的tool_result
                ir_input.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": msg["tool_call_id"],
                                "result": msg["content"],
                            }
                        ],
                    }
                )

            elif role == "function":
                # 已弃用的function角色，转换为tool_result
                ir_input.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": f"legacy_function_{msg['name']}",
                                "result": msg.get("content", ""),
                            }
                        ],
                    }
                )

        return ir_input

    def _extract_text_content(self, content: List[ContentPart]) -> str:
        """提取文本内容"""
        text_parts = []
        for part in content:
            if is_text_part(part):
                text_parts.append(part["text"])
        return " ".join(text_parts)

    def _convert_image_to_openai(self, image_part: Dict[str, Any]) -> Dict[str, Any]:
        """将IR图像转换为OpenAI格式"""
        # 支持多种URL字段名称
        url = image_part.get("image_url") or image_part.get("url")
        if url:
            return {
                "type": "image_url",
                "image_url": {"url": url, "detail": image_part.get("detail", "auto")},
            }
        elif "image_data" in image_part:
            image_data = image_part["image_data"]
            data_url = f"data:{image_data['media_type']};base64,{image_data['data']}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": image_part.get("detail", "auto"),
                },
            }
        else:
            raise ValueError("Image part must have either image_url/url or image_data")

    def _convert_file_to_openai(self, file_part: Dict[str, Any]) -> Dict[str, Any]:
        """将IR文件转换为OpenAI格式"""
        if "file_data" in file_part:
            return {
                "type": "file",
                "file": {
                    "file_data": file_part["file_data"]["data"],
                    "filename": file_part.get("file_name", "unknown"),
                },
            }
        elif "file_url" in file_part:
            # OpenAI不直接支持file_url，需要先下载
            return {
                "type": "file",
                "file": {
                    "file_url": file_part["file_url"],
                    "filename": file_part.get("file_name", "unknown"),
                },
            }
        else:
            raise ValueError("File part must have either file_data or file_url")

    def _convert_tool_call_to_openai(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """将IR工具调用转换为OpenAI格式"""
        tool_type = tool_call.get("tool_type", "function")

        # 支持多种字段名称映射
        tool_call_id = tool_call.get("tool_call_id") or tool_call.get("id")
        tool_name = tool_call.get("tool_name") or tool_call.get("name")
        tool_input = tool_call.get("tool_input") or tool_call.get("arguments", {})

        if tool_type == "function":
            return {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(tool_input)},
            }
        else:
            # 其他工具类型转换为自定义工具
            return {
                "id": tool_call_id,
                "type": "custom",
                "custom": {
                    "name": f"{tool_type}_{tool_name}",
                    "input": json.dumps(tool_input),
                },
            }

    def _convert_image_from_openai(self, image_part: Dict[str, Any]) -> Dict[str, Any]:
        """将OpenAI图像转换为IR格式"""
        image_url_data = image_part["image_url"]
        url = image_url_data["url"]

        if url.startswith("data:"):
            # Base64数据URL
            import re

            match = re.match(r"data:([^;]+);base64,(.+)", url)
            if match:
                media_type, data = match.groups()
                return {
                    "type": "image",
                    "image_data": {"data": data, "media_type": media_type},
                    "detail": image_url_data.get("detail", "auto"),
                }

        # 普通URL
        return {
            "type": "image",
            "image_url": url,
            "detail": image_url_data.get("detail", "auto"),
        }

    def _convert_tool_call_from_openai(
        self, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将OpenAI工具调用转换为IR格式"""
        if tool_call["type"] == "function":
            function = tool_call["function"]
            return {
                "type": "tool_call",
                "tool_call_id": tool_call["id"],
                "tool_name": function["name"],
                "tool_input": json.loads(function["arguments"]),
                "tool_type": "function",
            }
        elif tool_call["type"] == "custom":
            custom = tool_call["custom"]
            # 尝试解析工具类型
            name = custom["name"]
            if "_" in name:
                tool_type, tool_name = name.split("_", 1)
            else:
                tool_type = "custom"
                tool_name = name

            return {
                "type": "tool_call",
                "tool_call_id": tool_call["id"],
                "tool_name": tool_name,
                "tool_input": json.loads(custom["input"]),
                "tool_type": tool_type,
            }
        else:
            raise ValueError(f"Unsupported tool call type: {tool_call['type']}")

    def _convert_tool_definition_to_openai(
        self, tool: ToolDefinition
    ) -> Dict[str, Any]:
        """将IR工具定义转换为OpenAI格式"""
        # This method now strictly converts from IR format to OpenAI format.
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

    def _convert_tool_choice_to_openai(
        self, tool_choice: ToolChoice
    ) -> Union[str, Dict[str, Any]]:
        """将IR工具选择转换为OpenAI格式"""
        # 增加对字符串输入的健壮性处理
        if isinstance(tool_choice, str):
            if tool_choice in ["none", "auto", "required"]:
                return tool_choice
            else:
                # 如果是函数名，则包装成OpenAI需要的格式
                return {"type": "function", "function": {"name": tool_choice}}

        # 支持测试中使用的"type"字段
        mode = tool_choice.get("mode") or tool_choice.get("type")

        if mode == "none":
            return "none"
        elif mode == "auto":
            return "auto"
        elif mode == "any" or mode == "required":
            return "required"  # OpenAI使用"required"表示必须使用工具
        elif mode == "tool" or mode == "function":
            # 支持多种字段名称
            tool_name = tool_choice.get("tool_name")
            if not tool_name and "function" in tool_choice:
                tool_name = tool_choice["function"]["name"]
            return {"type": "function", "function": {"name": tool_name}}
        else:
            raise ValueError(f"Unsupported tool choice mode: {mode}")
