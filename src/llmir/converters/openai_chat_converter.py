"""
LLM Provider Converter - OpenAI Chat Completions Converter

实现IR与OpenAI Chat Completions API格式之间的转换
"""

from typing import Any, Dict, List, Optional, Tuple

from ..types.ir import (
    ContentPart,
    FileData,
    FilePart,
    ImagePart,
    IRInput,
    Message,
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
from ..utils import FieldMapper, ToolCallConverter, ToolConverter
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
                            # 注意：这里是 provider 格式的字典，不是 IR 的 TextPart
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
                        # 如果只有一个文本部分，使用字符串；否则使用列表
                        if (
                            len(content_parts) == 1
                            and content_parts[0].get("type") == "text"
                        ):
                            content = content_parts[0]["text"]
                        else:
                            content = content_parts

                        openai_message = {
                            "role": "user",
                            "content": content,
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
                            tool_calls.append(ToolCallConverter.to_openai_chat(part))
                        elif part["type"] == "reasoning":
                            warnings.append(
                                "Reasoning content not supported in OpenAI Chat Completions, ignored"
                            )

                    # 注意：这里不能使用 Message() 因为这是 provider 格式，不是 IR 格式
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
                            "tool_calls": [ToolCallConverter.to_openai_chat(tool_call)],
                        }
                        messages.append(openai_message)
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果
        result = {"messages": messages}

        # 添加工具定义
        if tools:
            result["tools"] = ToolConverter.batch_convert_tools(tools, "openai_chat")

        # 添加工具选择
        if tool_choice:
            result["tool_choice"] = ToolConverter.convert_tool_choice(
                tool_choice, "openai"
            )

        return result, warnings

    def from_provider(self, provider_data: Any) -> IRInput:
        """
        将OpenAI Chat Completions格式转换为IR格式

        Args:
            provider_data: OpenAI Chat Completions响应对象或字典
                          可以是：
                          1. API响应（包含choices字段）
                          2. 消息列表（包含messages字段）
                          3. 单个消息对象（包含role字段）
                          自动处理Pydantic模型对象（调用.model_dump()）
        """
        # 自动unwrap Pydantic模型对象
        if hasattr(provider_data, "model_dump"):
            provider_data = provider_data.model_dump()

        if not isinstance(provider_data, dict):
            raise ValueError("OpenAI data must be a dictionary")

        ir_input = []

        # Handle different input formats
        if "choices" in provider_data:
            # This is an API response with choices
            messages_to_process = []
            for choice in provider_data["choices"]:
                if "message" in choice:
                    messages_to_process.append(choice["message"])
                elif "delta" in choice:
                    # Streaming response
                    messages_to_process.append(choice["delta"])
        elif "messages" in provider_data:
            # This is a payload with messages
            messages_to_process = provider_data["messages"]
        elif "role" in provider_data:
            # This is a single message
            messages_to_process = [provider_data]
        else:
            # Empty or unknown format
            messages_to_process = []

        for msg in messages_to_process:
            role = msg["role"]

            if role == "system":
                # System消息
                ir_input.append(
                    Message(
                        role="system",
                        content=[TextPart(type="text", text=msg["content"])],
                    )
                )

            elif role == "user":
                # User消息
                content = msg["content"]
                ir_content = []

                if isinstance(content, str):
                    ir_content.append(TextPart(type="text", text=content))
                elif isinstance(content, list):
                    for part in content:
                        if part["type"] == "text":
                            ir_content.append(TextPart(type="text", text=part["text"]))
                        elif part["type"] == "image_url":
                            ir_content.append(self._convert_image_from_openai(part))
                        elif part["type"] == "input_audio":
                            # 音频暂时转换为文件类型
                            ir_content.append(
                                FilePart(
                                    type="file",
                                    file_data=FileData(
                                        data=part["input_audio"]["data"],
                                        media_type=f"audio/{part['input_audio']['format']}",
                                    ),
                                )
                            )

                ir_input.append(Message(role="user", content=ir_content))

            elif role == "assistant":
                # Assistant消息
                ir_content = []

                # 处理文本内容
                if "content" in msg and msg["content"]:
                    ir_content.append(TextPart(type="text", text=msg["content"]))

                # 处理工具调用
                if "tool_calls" in msg and msg["tool_calls"] is not None:
                    for tool_call in msg["tool_calls"]:
                        ir_content.append(ToolCallConverter.from_openai_chat(tool_call))

                ir_input.append(Message(role="assistant", content=ir_content))

            elif role == "tool":
                # Tool消息转换为user消息中的tool_result
                ir_input.append(
                    Message(
                        role="user",
                        content=[
                            ToolResultPart(
                                type="tool_result",
                                tool_call_id=msg["tool_call_id"],
                                result=msg["content"],
                            )
                        ],
                    )
                )

            elif role == "function":
                # 已弃用的function角色，转换为tool_result
                ir_input.append(
                    Message(
                        role="user",
                        content=[
                            ToolResultPart(
                                type="tool_result",
                                tool_call_id=f"legacy_function_{msg['name']}",
                                result=msg.get("content", ""),
                            )
                        ],
                    )
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
        url = FieldMapper.get_image_url(image_part)
        image_data = FieldMapper.get_image_data(image_part)
        detail = image_part.get("detail", "auto")

        if url:
            return {
                "type": "image_url",
                "image_url": {"url": url, "detail": detail},
            }
        elif image_data:
            # 创建data URL
            data_url = f"data:{image_data['media_type']};base64,{image_data['data']}"
            return {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": detail},
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

    def _convert_image_from_openai(self, image_part: Dict[str, Any]) -> Dict[str, Any]:
        """将OpenAI图像转换为IR格式"""
        image_url_data = image_part["image_url"]
        url = image_url_data["url"]
        detail = image_url_data.get("detail", "auto")

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

        # 普通URL
        return ImagePart(
            type="image",
            image_url=url,
            detail=detail,
        )
