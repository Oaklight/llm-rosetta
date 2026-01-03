"""
LLM Provider Converter - Google GenAI Converter

实现IR与Google GenAI SDK格式之间的转换
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

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


class GoogleConverter(BaseConverter):
    """Google GenAI格式转换器

    处理IR格式与Google GenAI Content/Part格式之间的转换。

    主要特点：
    - 角色映射: assistant ↔ model, user ↔ user
    - Part架构: 所有内容都通过Part表示
    - 工具调用: 通过function_call和function_response Part
    - system_instruction: 单独处理system消息
    """

    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IR格式转换为Google GenAI格式

        Args:
            ir_input: IR输入列表
            tools: 工具定义列表
            tool_choice: 工具选择配置

        Returns:
            (Google GenAI格式的字典, 警告列表)
        """
        # 对于无效格式，让其在处理过程中自然抛出KeyError或TypeError
        # 不进行预先验证，让错误在访问字段时自然发生

        # 转换消息
        contents = []
        system_instruction = None
        warnings_list = []

        for item in ir_input:
            if is_message(item):
                message = item  # type: ignore
                # 这里会在访问message["role"]时抛出KeyError如果没有role字段
                if message["role"] == "system":
                    # Google使用system_instruction而不是system消息
                    if system_instruction is None:
                        system_instruction = self._convert_message_to_content(
                            message, ir_input
                        )
                    else:
                        # 如果有多个system消息，合并它们
                        existing_parts = system_instruction.get("parts", [])
                        new_parts = self._convert_message_to_content(
                            message, ir_input
                        ).get("parts", [])
                        system_instruction["parts"] = existing_parts + new_parts
                else:
                    content = self._convert_message_to_content(message, ir_input)
                    if content:
                        contents.append(content)
            elif is_extension_item(item):
                # 处理扩展项
                warnings_list.append(
                    f"Google GenAI不支持扩展项类型 '{item.get('type')}'，将被忽略"
                )
            else:
                # 对于既不是消息也不是扩展项的无效项，强制访问role字段来抛出KeyError
                _ = item["role"]  # 这会抛出KeyError

        # 构建结果
        result = {"contents": contents}

        # 添加system_instruction
        if system_instruction:
            result["system_instruction"] = system_instruction

        # 转换工具
        if tools:
            google_tools = self._convert_tools(tools)
            if google_tools:
                result["tools"] = google_tools

        # 转换工具选择配置
        if tool_choice:
            tool_config = self._convert_tool_choice(tool_choice)
            if tool_config:
                result["tool_config"] = tool_config

        return result, warnings_list

    def from_provider(self, provider_data: Any) -> IRInput:
        """将Google GenAI格式转换为IR格式"""
        if isinstance(provider_data, tuple):
            provider_data = provider_data[0]

        if not isinstance(provider_data, dict):
            raise ValueError("Google data must be a dictionary")

        messages = []

        # The actual response is in the 'candidates' field
        candidates = provider_data.get("candidates", [])
        if not candidates:
            # Handle cases with no candidates, e.g., safety block
            prompt_feedback = provider_data.get("prompt_feedback")
            if prompt_feedback and prompt_feedback.get("block_reason"):
                block_reason = prompt_feedback.get("block_reason")
                block_message = f"Request was blocked. Reason: {block_reason}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": block_message}],
                    }
                )
            return messages

        # Process all candidates
        for candidate in candidates:
            content = candidate.get("content")
            if content:
                message = self._convert_content_to_message(content)
                if message:
                    messages.append(message)

        return messages

    def _convert_message_to_content(
        self, message: Dict[str, Any], ir_input: IRInput
    ) -> Dict[str, Any]:
        """将IR消息转换为Google Content"""
        # 角色映射
        role_mapping = {
            "user": "user",
            "assistant": "model",
            "system": "user",  # system在这里不应该被调用，但作为fallback
        }

        google_role = role_mapping.get(message["role"], "user")
        parts = []

        # 转换内容部分
        for content_part in message["content"]:
            part = self._convert_content_part_to_part(content_part, ir_input)
            if part:
                parts.append(part)

        content = {"role": google_role, "parts": parts}
        return content

    def _convert_content_part_to_part(
        self, content_part: Dict[str, Any], ir_input: IRInput
    ) -> Optional[Dict[str, Any]]:
        """将IR内容部分转换为Google Part"""
        if is_text_part(content_part):
            return {"text": content_part["text"]}

        elif content_part["type"] == "image":
            # Google使用inline_data或file_data
            # 支持多种字段格式
            if "image_data" in content_part:
                image_data = content_part["image_data"]
                return {
                    "inline_data": {
                        "mime_type": image_data["media_type"],
                        "data": image_data["data"],
                    }
                }
            elif "data" in content_part and "media_type" in content_part:
                # 直接的data和media_type字段
                return {
                    "inline_data": {
                        "mime_type": content_part["media_type"],
                        "data": content_part["data"],
                    }
                }
            elif "image_url" in content_part:
                # 对于URL，Google需要先上传文件或使用file_data
                warnings.warn(
                    "Google GenAI不直接支持图片URL，需要先上传文件。"
                    "请考虑使用file_data或先转换为inline_data。"
                )
                return None
            elif "url" in content_part:
                # 支持url字段
                warnings.warn(
                    "Google GenAI不直接支持图片URL，需要先上传文件。"
                    "请考虑使用file_data或先转换为inline_data。"
                )
                return None

        elif content_part["type"] == "file":
            if "file_data" in content_part:
                file_data = content_part["file_data"]
                return {
                    "inline_data": {
                        "mime_type": file_data["media_type"],
                        "data": file_data["data"],
                    }
                }
            elif "data" in content_part and "media_type" in content_part:
                # 直接的data和media_type字段
                return {
                    "inline_data": {
                        "mime_type": content_part["media_type"],
                        "data": content_part["data"],
                    }
                }
            elif "file_url" in content_part:
                warnings.warn("Google GenAI不直接支持文件URL，需要先上传文件。")
                return None

        elif content_part["type"] == "audio":
            # 处理音频内容
            if "data" in content_part and "media_type" in content_part:
                return {
                    "inline_data": {
                        "mime_type": content_part["media_type"],
                        "data": content_part["data"],
                    }
                }
            elif "audio_data" in content_part:
                audio_data = content_part["audio_data"]
                return {
                    "inline_data": {
                        "mime_type": audio_data["media_type"],
                        "data": audio_data["data"],
                    }
                }
            else:
                warnings.warn("不支持的音频格式")
                return None

        elif is_tool_call_part(content_part):
            # 支持多种字段名格式
            tool_name = (
                content_part.get("tool_name")
                or content_part.get("name")
                or content_part.get("function", {}).get("name", "")
            )
            tool_args = (
                content_part.get("tool_input")
                or content_part.get("arguments")
                or content_part.get("args", {})
            )

            return {"function_call": {"name": tool_name, "args": tool_args}}

        elif is_tool_result_part(content_part):
            # Google的function_response.name需要是函数名，而不是tool_call_id
            # 我们需要回溯历史记录找到对应的tool_call来获取函数名
            tool_name = None
            for msg in ir_input:
                if not is_message(msg):
                    continue
                for part in msg.get("content", []):
                    if is_tool_call_part(part) and part.get(
                        "tool_call_id"
                    ) == content_part.get("tool_call_id"):
                        tool_name = part.get("tool_name")
                        break
                if tool_name:
                    break

            if not tool_name:
                warnings.warn(
                    f"Could not find corresponding tool call for tool_call_id '{content_part.get('tool_call_id')}'. "
                    "Using tool_call_id as function name, which may cause issues with Google GenAI."
                )
                tool_name = content_part.get("tool_call_id")

            # 支持多种字段名格式
            result_content = (
                content_part.get("result")
                or content_part.get("content")
                or content_part.get("output", "")
            )

            response_data = {"output": result_content}
            if content_part.get("is_error"):
                response_data = {"error": result_content}

            return {
                "function_response": {
                    "name": tool_name,  # 使用找到的函数名
                    "response": response_data,
                }
            }

        else:
            warnings.warn(f"不支持的内容类型: {content_part['type']}")
            return None

    def _convert_content_to_message(
        self, content: Dict[str, Any], force_role: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """将Google Content转换为IR消息"""
        # 角色映射
        role_mapping = {"user": "user", "model": "assistant"}

        google_role = content.get("role", "user")
        ir_role = force_role or role_mapping.get(google_role, "user")

        # 转换parts
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            parts = [parts]

        content_parts = []
        for part in parts:
            converted_parts = self._convert_part_to_content_parts(part)
            if converted_parts:
                content_parts.extend(converted_parts)

        if not content_parts:
            return None

        return {"role": ir_role, "content": content_parts}

    def _convert_part_to_content_parts(
        self, part: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """将Google Part转换为一个或多个IR内容部分"""
        ir_parts = []
        if "text" in part and part["text"] is not None and part["text"] != "":
            ir_parts.append({"type": "text", "text": part["text"]})

        if "function_call" in part and part["function_call"] is not None:
            func_call = part["function_call"]
            ir_parts.append(
                {
                    "type": "tool_call",
                    "tool_call_id": func_call.get("id", func_call.get("name", "")),
                    "tool_name": func_call["name"],
                    "tool_input": func_call.get("args", {}),
                    "tool_type": "function",
                }
            )

        if "inline_data" in part and part["inline_data"] is not None:
            inline_data = part["inline_data"]
            mime_type = inline_data.get("mime_type", "")

            if mime_type.startswith("image/"):
                ir_parts.append(
                    {
                        "type": "image",
                        "data": inline_data["data"],
                        "media_type": mime_type,
                    }
                )
            elif mime_type.startswith("audio/"):
                ir_parts.append(
                    {
                        "type": "audio",
                        "url": None,  # inline data doesn't have URL
                        "media_type": mime_type,
                    }
                )
            else:
                ir_parts.append(
                    {
                        "type": "file",
                        "file_data": {
                            "data": inline_data["data"],
                            "media_type": mime_type,
                        },
                    }
                )

        if "file_data" in part and part["file_data"] is not None:
            file_data = part["file_data"]
            mime_type = file_data.get("mime_type", "")

            if mime_type.startswith("image/"):
                ir_parts.append({"type": "image", "image_url": file_data["file_uri"]})
            elif mime_type.startswith("audio/"):
                ir_parts.append(
                    {
                        "type": "audio",
                        "url": file_data["file_uri"],
                        "media_type": mime_type,
                    }
                )
            else:
                ir_parts.append({"type": "file", "file_url": file_data["file_uri"]})

        if "function_response" in part and part["function_response"] is not None:
            func_response = part["function_response"]
            response_data = func_response.get("response", {})

            # 检查是否是错误响应
            is_error = "error" in response_data
            content = response_data.get("error" if is_error else "output", "")

            ir_parts.append(
                {
                    "type": "tool_result",
                    "tool_call_id": func_response.get(
                        "id", func_response.get("name", "")
                    ),
                    "content": str(content),
                    "is_error": is_error,
                }
            )

        if not ir_parts:
            warnings.warn(f"不支持的Part类型: {list(part.keys())}")

        return ir_parts

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """转换工具定义为Google格式"""
        google_tools = []

        for tool in tools:
            if tool["type"] == "function":
                # 支持嵌套的function结构
                if "function" in tool:
                    func_def = tool["function"]
                    function_declaration = {
                        "name": func_def["name"],
                        "description": func_def.get("description", ""),
                    }
                    # 转换参数schema
                    if "parameters" in func_def:
                        function_declaration["parameters"] = func_def["parameters"]
                else:
                    # 直接的工具定义格式
                    function_declaration = {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                    }
                    # 转换参数schema
                    if "parameters" in tool:
                        function_declaration["parameters"] = tool["parameters"]

                google_tool = {"function_declarations": [function_declaration]}
                google_tools.append(google_tool)

            elif tool["type"] == "mcp":
                # Google原生支持MCP，但需要特殊处理
                warnings.warn(
                    "MCP工具需要通过Google GenAI的MCP适配器处理，"
                    "请参考Google GenAI文档中的MCP集成方式"
                )

            else:
                warnings.warn(f"不支持的工具类型: {tool['type']}")

        return google_tools

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> Optional[Dict[str, Any]]:
        """转换工具选择配置为Google格式"""
        # 支持type和mode字段
        choice_type = tool_choice.get("type") or tool_choice.get("mode")

        if choice_type == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}
        elif choice_type == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        elif choice_type in ["any", "required"]:
            return {"function_calling_config": {"mode": "ANY"}}
        elif choice_type in ["tool", "function"]:
            # 获取函数名
            func_name = None
            if "function" in tool_choice:
                func_name = tool_choice["function"].get("name")
            elif "tool_name" in tool_choice:
                func_name = tool_choice["tool_name"]

            if func_name:
                return {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [func_name],
                    }
                }
            else:
                return {"function_calling_config": {"mode": "ANY"}}
        else:
            warnings.warn(f"不支持的工具选择类型: {choice_type}")
            return None
