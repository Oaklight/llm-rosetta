"""
LLM Provider Converter - OpenAI Responses API Converter

实现IR与OpenAI Responses API格式之间的转换
"""

from typing import Any, Dict, List, Optional, Tuple

from ..types.ir import (
    FilePart,
    ImagePart,
    IRInput,
    Message,
    ReasoningPart,
    TextPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    is_extension_item,
    is_message,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
)
from ..utils import ToolCallConverter, ToolConverter
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
                            # 用户输入使用 input_text
                            # 注意：这里是 provider 格式的字典，不是 IR 格式
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
                                ToolCallConverter.to_openai_responses(part)
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
                            # 注意：这里是 provider 格式的字典，不是 IR 格式
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
                                # 注意：这里是 provider 格式的字典，不是 IR 格式
                                items.append(
                                    {"type": "reasoning", "content": part["text"]}
                                )
                            else:
                                # Assistant 输出使用 output_text
                                # 注意：这里是 provider 格式的字典，不是 IR 格式
                                content_list.append(
                                    {"type": "output_text", "text": part["text"]}
                                )
                        elif is_tool_call_part(part):
                            tool_calls.append(
                                ToolCallConverter.to_openai_responses(part)
                            )
                        elif part["type"] == "reasoning":
                            # 注意：这里是 provider 格式的字典，不是 IR 格式
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
                        items.append(ToolCallConverter.to_openai_responses(tool_call))
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果
        result = {"input": items}

        # 添加工具定义
        if tools:
            result["tools"] = ToolConverter.batch_convert_tools(
                tools, "openai_responses"
            )

        # 添加工具选择
        if tool_choice:
            result["tool_choice"] = ToolConverter.convert_tool_choice(
                tool_choice, "openai_responses"
            )

        return result, warnings

    def from_provider(self, provider_data: Any) -> IRInput:
        """
        将OpenAI Responses API格式转换为IR格式

        注意：Responses API的响应在'output'字段，而请求在'input'字段

        Args:
            provider_data: OpenAI Responses API响应对象或字典
                          自动处理Pydantic模型对象（调用.model_dump()）
        """
        # 自动unwrap Pydantic模型对象
        if hasattr(provider_data, "model_dump"):
            provider_data = provider_data.model_dump()

        if not isinstance(provider_data, dict):
            raise ValueError("OpenAI Responses data must be a dictionary")

        # 响应数据在output字段，请求数据在input字段
        items = provider_data.get("output") or provider_data.get("input", [])
        if not isinstance(items, list):
            raise ValueError("OpenAI Responses output/input must be a list")

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
                    ir_content.append(TextPart(type="text", text=content))
                elif isinstance(content, list):
                    for part in content:
                        part_type = part.get("type", "")
                        # 支持input_text和output_text
                        if part_type in ["input_text", "output_text", "text"]:
                            ir_content.append(TextPart(type="text", text=part["text"]))
                        elif part_type == "input_image":
                            ir_content.append(self._convert_image_from_responses(part))
                        elif part_type == "input_file":
                            ir_content.append(self._convert_file_from_responses(part))

                current_message = Message(role=role, content=ir_content)

            elif item_type == "function_call":
                # 函数调用转换为工具调用
                tool_call = ToolCallConverter.from_openai_responses(item)

                if current_message and current_message["role"] == "assistant":
                    current_message["content"].append(tool_call)
                else:
                    # 创建新的assistant消息
                    if current_message:
                        ir_input.append(current_message)
                    current_message = Message(
                        role="assistant",
                        content=[tool_call],
                    )

            elif item_type == "function_call_output":
                # 函数调用输出转换为工具结果
                if current_message:
                    ir_input.append(current_message)

                current_message = Message(
                    role="user",
                    content=[
                        ToolResultPart(
                            type="tool_result",
                            tool_call_id=item["call_id"],
                            result=item["output"],
                        )
                    ],
                )

            elif item_type == "mcp_call":
                # MCP调用转换为工具调用
                tool_call = ToolCallConverter.from_openai_responses(item)

                if current_message and current_message["role"] == "assistant":
                    current_message["content"].append(tool_call)
                else:
                    if current_message:
                        ir_input.append(current_message)
                    current_message = Message(
                        role="assistant",
                        content=[tool_call],
                    )

            elif item_type == "mcp_call_output":
                # MCP调用输出转换为工具结果
                if current_message:
                    ir_input.append(current_message)

                current_message = Message(
                    role="user",
                    content=[
                        ToolResultPart(
                            type="tool_result",
                            tool_call_id=item["call_id"],
                            result=item["output"],
                        )
                    ],
                )

            elif item_type == "reasoning":
                # 推理内容转换为ReasoningPart
                # o4-mini的reasoning可能有content字段为null的情况
                reasoning_content = item.get("reasoning") or item.get("content")

                # 只有当reasoning_content不为None时才添加
                if reasoning_content:
                    reasoning_part = ReasoningPart(
                        type="reasoning", reasoning=str(reasoning_content)
                    )
                    if current_message:
                        current_message["content"].append(reasoning_part)
                    else:
                        # 创建新的assistant消息包含推理
                        current_message = Message(
                            role="assistant",
                            content=[reasoning_part],
                        )
                # 如果reasoning_content为None，跳过这个reasoning event

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
                tool_call = ToolCallConverter.from_openai_responses(item)

                if current_message and current_message["role"] == "assistant":
                    current_message["content"].append(tool_call)
                else:
                    if current_message:
                        ir_input.append(current_message)
                    current_message = Message(
                        role="assistant",
                        content=[tool_call],
                    )

        # 添加最后一个消息
        if current_message:
            ir_input.append(current_message)

        return ir_input

    def _convert_image_to_responses(self, image_part: Dict[str, Any]) -> Dict[str, Any]:
        """将IR图像转换为Responses API格式

        注意：图像始终使用 input_image，因为图像只能作为输入
        """
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
        """将IR文件转换为Responses API格式

        注意：文件始终使用 input_file，因为文件只能作为输入
        """
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

    def _convert_image_from_responses(
        self, image_part: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将Responses API图像转换为IR格式"""
        detail = image_part.get("detail", "auto")

        if "image_url" in image_part:
            url = image_part["image_url"]
            if url.startswith("data:"):
                # Base64数据URL
                import re

                match = re.match(r"data:([^;]+);base64,(.+)", url)
                if match:
                    media_type, data = match.groups()
                    return ImagePart(
                        type="image",
                        image_data={"data": data, "media_type": media_type},
                        detail=detail,
                    )
                else:
                    return ImagePart(type="image", image_url=url, detail=detail)
            else:
                return ImagePart(type="image", image_url=url, detail=detail)
        elif "file_id" in image_part:
            # 文件ID形式，转换为引用
            # 注意：file_id 不在 ImagePart 定义中，使用字典字面量
            return {"type": "image", "file_id": image_part["file_id"], "detail": detail}  # type: ignore

        return ImagePart(type="image", detail=detail)

    def _convert_file_from_responses(self, file_part: Dict[str, Any]) -> Dict[str, Any]:
        """将Responses API文件转换为IR格式"""
        file_name = file_part.get("filename", "unknown")

        if "file_data" in file_part:
            # 假设是base64编码的数据
            return FilePart(
                type="file",
                file_name=file_name,
                file_data={
                    "data": file_part["file_data"],
                    "media_type": "application/octet-stream",  # 默认类型
                },
            )
        elif "file_url" in file_part:
            return FilePart(
                type="file", file_name=file_name, file_url=file_part["file_url"]
            )
        elif "file_id" in file_part:
            # 注意：file_id 不在 FilePart 定义中，使用字典字面量
            return {
                "type": "file",
                "file_name": file_name,
                "file_id": file_part["file_id"],
            }  # type: ignore

        return FilePart(type="file", file_name=file_name)
