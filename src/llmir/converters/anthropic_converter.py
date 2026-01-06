"""
LLMIR - Anthropic Converter

实现IR与Anthropic格式之间的转换
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..exceptions import (
    ConversionError,
    ErrorCategory,
    ErrorSeverity,
    WarningInfo,
    create_warning,
)
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
from ..utils import ErrorConverter, FieldMapper, ToolCallConverter, ToolConverter
from .base import BaseConverter


class AnthropicConverter(BaseConverter):
    """Anthropic格式转换器
    Anthropic format converter
    """

    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[Union[str, WarningInfo]]]:
        """
        将IR格式转换为Anthropic格式
        Convert IR format to Anthropic format

        Anthropic使用嵌套结构，与IR设计最接近，转换相对简单
        Anthropic uses nested structure, closest to IR design, conversion is relatively simple
        """
        # 验证输入 Validate input
        self.validate_and_raise(ir_input)

        messages = []
        warnings = []
        system_messages = []

        try:
            for i, item in enumerate(ir_input):
                try:
                    if is_message(item):
                        message = item  # type: ignore

                        if message["role"] == "system":
                            # Anthropic的system消息通过API参数传递 Anthropic system messages are passed through API parameters
                            system_content = self._convert_content_to_anthropic(
                                message["content"], i
                            )
                            # 提取文本内容作为system消息 Extract text content as system message
                            for block in system_content:
                                if block.get("type") == "text":
                                    system_messages.append(
                                        TextPart(type="text", text=block["text"])
                                    )
                        else:
                            # 普通消息：直接转换 Normal messages: direct conversion
                            anthropic_message = {
                                "role": message["role"],
                                "content": self._convert_content_to_anthropic(
                                    message["content"], i
                                ),
                            }
                            messages.append(anthropic_message)

                    elif is_extension_item(item):
                        extension = item  # type: ignore
                        extension_type = extension.get("type")

                        if extension_type == "system_event":
                            warnings.append(
                                create_warning(
                                    f"System event ignored: {extension.get('event_type', 'unknown')}",
                                    category=ErrorCategory.CONVERSION,
                                    severity=ErrorSeverity.LOW,
                                    context={
                                        "item_index": i,
                                        "extension_type": extension_type,
                                        "event_type": extension.get("event_type"),
                                    },
                                    suggestions=[
                                        "Anthropic does not support system events"
                                    ],
                                )
                            )
                        elif extension_type == "tool_chain_node":
                            warnings.append(
                                create_warning(
                                    "Tool chain converted to sequential calls",
                                    category=ErrorCategory.CONVERSION,
                                    severity=ErrorSeverity.MEDIUM,
                                    context={
                                        "item_index": i,
                                        "extension_type": extension_type,
                                    },
                                    suggestions=[
                                        "Consider using sequential tool calls instead of tool chains"
                                    ],
                                )
                            )
                            # 可以将工具链节点展开为普通工具调用 Can expand tool chain nodes into normal tool calls
                            tool_call = extension.get("tool_call")
                            if tool_call:
                                # 创建一个assistant消息包含工具调用 Create an assistant message containing tool call
                                anthropic_message = {
                                    "role": "assistant",
                                    "content": [
                                        ToolCallConverter.to_anthropic(tool_call)
                                    ],
                                }
                                messages.append(anthropic_message)
                        elif extension_type in ["batch_marker", "session_control"]:
                            warnings.append(
                                create_warning(
                                    f"Extension item ignored: {extension_type}",
                                    category=ErrorCategory.CONVERSION,
                                    severity=ErrorSeverity.LOW,
                                    context={
                                        "item_index": i,
                                        "extension_type": extension_type,
                                    },
                                    suggestions=[
                                        "Anthropic does not support this extension type"
                                    ],
                                )
                            )

                except Exception as e:
                    # 转换单个项目时出错，包装为ConversionError
                    conversion_error = self.handle_conversion_error(
                        e, "IR", "Anthropic", i, {"item": item}
                    )
                    raise conversion_error

            # 构建结果 Build result
            result = {"messages": messages}

            # 添加system消息（如果有） Add system message (if any)
            if system_messages:
                result["system"] = system_messages

            # 添加工具定义 Add tool definitions
            if tools:
                try:
                    result["tools"] = ToolConverter.batch_convert_tools(
                        tools, "anthropic"
                    )
                except Exception as e:
                    conversion_error = self.handle_conversion_error(
                        e, "IR tools", "Anthropic tools", context={"tools": tools}
                    )
                    raise conversion_error

            # 添加工具选择 Add tool choice
            if tool_choice:
                try:
                    result["tool_choice"] = ToolConverter.convert_tool_choice(
                        tool_choice, "anthropic"
                    )
                except Exception as e:
                    conversion_error = self.handle_conversion_error(
                        e,
                        "IR tool_choice",
                        "Anthropic tool_choice",
                        context={"tool_choice": tool_choice},
                    )
                    raise conversion_error

            return result, warnings

        except ConversionError:
            # 重新抛出ConversionError
            raise
        except Exception as e:
            # 包装其他异常为ConversionError
            conversion_error = self.handle_conversion_error(
                e, "IR", "Anthropic", context={"ir_input_length": len(ir_input)}
            )
            raise conversion_error

    def from_provider(self, provider_data: Any) -> IRInput:
        """
        将Anthropic格式转换为IR格式
        Convert Anthropic format to IR format

        Args:
            provider_data: Anthropic响应对象或字典 Anthropic response object or dict
                          自动处理Pydantic模型对象（调用.model_dump()） Automatically handles Pydantic model objects (calls .model_dump())
        """
        # 自动unwrap Pydantic模型对象 Auto unwrap Pydantic model objects
        if hasattr(provider_data, "model_dump"):
            provider_data = provider_data.model_dump()

        if not isinstance(provider_data, dict):
            raise ValueError("Anthropic data must be a dictionary")

        ir_input = []

        # 处理system消息 Handle system messages
        system_content = provider_data.get("system")
        if system_content:
            if isinstance(system_content, str):
                # 简单字符串形式 Simple string form
                ir_input.append(
                    Message(
                        role="system",
                        content=[TextPart(type="text", text=system_content)],
                    )
                )
            elif isinstance(system_content, list):
                # 内容块列表形式 Content block list form
                ir_input.append(
                    Message(
                        role="system",
                        content=self._convert_content_from_anthropic(system_content),
                    )
                )

        # 处理普通消息 Handle normal messages
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
        self, content: List[ContentPart], item_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """将IR内容部分转换为Anthropic内容块
        Convert IR content parts to Anthropic content blocks
        """
        blocks = []

        for part_index, part in enumerate(content):
            try:
                part_type = part.get("type")

                if part_type == "text":
                    if "text" not in part:
                        raise ConversionError(
                            "Text part missing 'text' field",
                            source_format="IR",
                            target_format="Anthropic",
                            item_index=item_index,
                            context={"part_index": part_index, "part": part},
                        )
                    blocks.append(TextPart(type="text", text=part["text"]))

                elif part_type == "image":
                    image_url = FieldMapper.get_image_url(part)
                    image_data = FieldMapper.get_image_data(part)

                    if image_data:
                        # base64形式 base64 form
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
                        # URL形式 URL form
                        # 注意：这里是 provider 格式的字典，不是 IR 格式
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": image_url},
                            }
                        )
                    else:
                        raise ConversionError(
                            "Image part must have either image_url or image_data",
                            source_format="IR",
                            target_format="Anthropic",
                            item_index=item_index,
                            context={"part_index": part_index, "part": part},
                        )

                elif part_type == "file":
                    # Anthropic支持文档类型 Anthropic supports document type
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
                    else:
                        raise ConversionError(
                            "File part must have either file_data or file_url",
                            source_format="IR",
                            target_format="Anthropic",
                            item_index=item_index,
                            context={"part_index": part_index, "part": part},
                        )

                elif part_type == "tool_call":
                    try:
                        blocks.append(ToolCallConverter.to_anthropic(part))
                    except Exception as e:
                        raise ConversionError(
                            f"Failed to convert tool call: {str(e)}",
                            source_format="IR",
                            target_format="Anthropic",
                            item_index=item_index,
                            context={"part_index": part_index, "part": part},
                            original_error=e,
                        )

                elif part_type == "tool_result":
                    # 使用ErrorConverter处理工具结果
                    tool_result_part = ErrorConverter.convert_ir_error_to_provider(
                        part, "anthropic"
                    )
                    blocks.append(tool_result_part)

                elif part_type == "reasoning":
                    # Anthropic支持thinking块 Anthropic supports thinking blocks
                    if "reasoning" not in part:
                        raise ConversionError(
                            "Reasoning part missing 'reasoning' field",
                            source_format="IR",
                            target_format="Anthropic",
                            item_index=item_index,
                            context={"part_index": part_index, "part": part},
                        )
                    # 注意：这里是 provider 格式的字典，不是 IR 格式
                    blocks.append({"type": "thinking", "thinking": part["reasoning"]})

                else:
                    raise ConversionError(
                        f"Unsupported content part type: {part_type}",
                        source_format="IR",
                        target_format="Anthropic",
                        item_index=item_index,
                        context={"part_index": part_index, "part": part},
                        suggestions=[
                            "Use supported content types: text, image, file, tool_call, tool_result, reasoning",
                            "Check the IR specification for valid content part types",
                        ],
                    )

            except ConversionError:
                # 重新抛出ConversionError
                raise
            except Exception as e:
                # 包装其他异常
                raise ConversionError(
                    f"Unexpected error converting content part: {str(e)}",
                    source_format="IR",
                    target_format="Anthropic",
                    item_index=item_index,
                    context={"part_index": part_index, "part": part},
                    original_error=e,
                )

        return blocks

    def _convert_content_from_anthropic(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> List[ContentPart]:
        """将Anthropic内容转换为IR内容部分
        Convert Anthropic content to IR content parts
        """
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
