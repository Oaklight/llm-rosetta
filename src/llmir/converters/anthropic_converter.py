"""
LLM Provider Converter - Anthropic Converter

实现IR与Anthropic格式之间的转换
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..types.ir import (
    ContentPart,
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
)
from ..utils import FieldMapper, ToolCallConverter, ToolConverter
from .base import BaseConverter


class AnthropicConverter(BaseConverter):
    """Anthropic格式转换器"""

    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        将IR格式转换为Anthropic格式

        Anthropic使用嵌套结构，与IR设计最接近，转换相对简单
        """
        # 验证输入
        validation_errors = self.validate_ir_input(ir_input)
        if validation_errors:
            raise ValueError(f"Invalid IR input: {validation_errors}")

        messages = []
        warnings = []
        system_messages = []

        for item in ir_input:
            if is_message(item):
                message = item  # type: ignore

                if message["role"] == "system":
                    # Anthropic的system消息通过API参数传递
                    system_content = self._convert_content_to_anthropic(
                        message["content"]
                    )
                    # 提取文本内容作为system消息
                    for block in system_content:
                        if block.get("type") == "text":
                            system_messages.append(
                                TextPart(type="text", text=block["text"])
                            )
                else:
                    # 普通消息：直接转换
                    anthropic_message = {
                        "role": message["role"],
                        "content": self._convert_content_to_anthropic(
                            message["content"]
                        ),
                    }
                    messages.append(anthropic_message)

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
                        anthropic_message = {
                            "role": "assistant",
                            "content": [ToolCallConverter.to_anthropic(tool_call)],
                        }
                        messages.append(anthropic_message)
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果
        result = {"messages": messages}

        # 添加system消息（如果有）
        if system_messages:
            result["system"] = system_messages

        # 添加工具定义
        if tools:
            result["tools"] = ToolConverter.batch_convert_tools(tools, "anthropic")

        # 添加工具选择
        if tool_choice:
            result["tool_choice"] = ToolConverter.convert_tool_choice(
                tool_choice, "anthropic"
            )

        return result, warnings

    def from_provider(self, provider_data: Any) -> IRInput:
        """
        将Anthropic格式转换为IR格式

        Args:
            provider_data: Anthropic响应对象或字典
                          自动处理Pydantic模型对象（调用.model_dump()）
        """
        # 自动unwrap Pydantic模型对象
        if hasattr(provider_data, "model_dump"):
            provider_data = provider_data.model_dump()

        if not isinstance(provider_data, dict):
            raise ValueError("Anthropic data must be a dictionary")

        ir_input = []

        # 处理system消息
        system_content = provider_data.get("system")
        if system_content:
            if isinstance(system_content, str):
                # 简单字符串形式
                ir_input.append(
                    Message(
                        role="system",
                        content=[TextPart(type="text", text=system_content)],
                    )
                )
            elif isinstance(system_content, list):
                # 内容块列表形式
                ir_input.append(
                    Message(
                        role="system",
                        content=self._convert_content_from_anthropic(system_content),
                    )
                )

        # 处理普通消息
        # Handle both a full payload (with a 'messages' key) and a single message dict
        if "messages" in provider_data:
            messages_to_process = provider_data["messages"]
        elif "role" in provider_data:
            messages_to_process = [provider_data]
        else:
            messages_to_process = []

        for msg in messages_to_process:
            ir_message = Message(
                role=msg["role"],
                content=self._convert_content_from_anthropic(msg["content"]),
            )
            ir_input.append(ir_message)

        return ir_input

    def _convert_content_to_anthropic(
        self, content: List[ContentPart]
    ) -> List[Dict[str, Any]]:
        """将IR内容部分转换为Anthropic内容块"""
        blocks = []

        for part in content:
            part_type = part.get("type")

            if part_type == "text":
                blocks.append(TextPart(type="text", text=part["text"]))

            elif part_type == "image":
                image_url = FieldMapper.get_image_url(part)
                image_data = FieldMapper.get_image_data(part)

                if image_data:
                    # base64形式
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_data["media_type"],
                                "data": image_data["data"],
                            },
                        }
                    )
                elif image_url:
                    # URL形式
                    # 注意：这里是 provider 格式的字典，不是 IR 格式
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": image_url},
                        }
                    )

            elif part_type == "file":
                # Anthropic支持文档类型
                if "file_data" in part:
                    file_data = part["file_data"]
                    blocks.append(
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": file_data["media_type"],
                                "data": file_data["data"],
                            },
                        }
                    )
                elif "file_url" in part:
                    # 注意：这里是 provider 格式的字典，不是 IR 格式
                    blocks.append(
                        {
                            "type": "document",
                            "source": {"type": "url", "url": part["file_url"]},
                        }
                    )

            elif part_type == "tool_call":
                blocks.append(ToolCallConverter.to_anthropic(part))

            elif part_type == "tool_result":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": part["tool_call_id"],
                        "content": part["result"],
                        "is_error": part.get("is_error", False),
                    }
                )

            elif part_type == "reasoning":
                # Anthropic支持thinking块
                # 注意：这里是 provider 格式的字典，不是 IR 格式
                blocks.append({"type": "thinking", "thinking": part["reasoning"]})

        return blocks

    def _convert_content_from_anthropic(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> List[ContentPart]:
        """将Anthropic内容转换为IR内容部分"""
        if isinstance(content, str):
            return [TextPart(type="text", text=content)]

        ir_content = []

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                ir_content.append(TextPart(type="text", text=block["text"]))

            elif block_type == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    ir_content.append(
                        ImagePart(
                            type="image",
                            image_data={
                                "data": source.get("data", ""),
                                "media_type": source.get("media_type", ""),
                            },
                        )
                    )
                elif source.get("type") == "url":
                    ir_content.append(
                        ImagePart(type="image", image_url=source.get("url", ""))
                    )

            elif block_type == "document":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    ir_content.append(
                        FilePart(
                            type="file",
                            file_data={
                                "data": source["data"],
                                "media_type": source["media_type"],
                            },
                        )
                    )
                elif source.get("type") == "url":
                    ir_content.append(FilePart(type="file", file_url=source["url"]))

            elif block_type in ["tool_use", "server_tool_use"]:
                ir_content.append(ToolCallConverter.from_anthropic(block))

            elif block_type == "tool_result":
                ir_content.append(
                    ToolResultPart(
                        type="tool_result",
                        tool_call_id=block.get("tool_use_id", ""),
                        result=block.get("content", ""),
                        is_error=block.get("is_error", False),
                    )
                )

            elif block_type == "thinking":
                ir_content.append(
                    ReasoningPart(type="reasoning", reasoning=block["thinking"])
                )

        return ir_content
