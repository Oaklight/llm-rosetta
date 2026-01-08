"""
LLMIR - Anthropic Converter

实现IR与Anthropic格式之间的转换
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ...types.ir import (
    FilePart,
    ImagePart,
    IRInput,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    is_extension_item,
    is_message,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
)
from ...types.ir_request import IRRequest
from ...types.ir_response import IRResponse
from ...utils import FieldMapper, ToolCallConverter, ToolConverter
from ..base import BaseConverter


class AnthropicConverter(BaseConverter):
    """Anthropic格式转换器
    Anthropic format converter
    """

    def to_provider(
        self,
        ir_input: Union[IRInput, IRRequest],
        tools: Optional[Iterable[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        将IR格式转换为Anthropic格式
        Convert IR format to Anthropic format

        Anthropic使用嵌套结构，与IR设计最接近，转换相对简单
        Anthropic uses nested structure, closest to IR design, conversion is relatively simple
        """
        if isinstance(ir_input, dict) and "messages" in ir_input:
            # Handle IRRequest
            return self._ir_request_to_p(ir_input)

        # 验证输入 Validate input
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
                    # Anthropic的system消息通过API参数传递 Anthropic system messages are passed through API parameters
                    system_content, msg_warnings = self._ir_message_to_p(
                        message, ir_input
                    )
                    warnings.extend(msg_warnings)
                    # 提取文本内容作为system消息 Extract text content as system message
                    if isinstance(system_content, list):
                        for block in system_content:
                            if block.get("type") == "text":
                                system_messages.append(
                                    TextPart(type="text", text=block["text"])
                                )
                    elif (
                        isinstance(system_content, dict) and "content" in system_content
                    ):
                        for block in system_content["content"]:
                            if block.get("type") == "text":
                                system_messages.append(
                                    TextPart(type="text", text=block["text"])
                                )
                else:
                    # 普通消息：直接转换 Normal messages: direct conversion
                    anthropic_message, msg_warnings = self._ir_message_to_p(
                        message, ir_input
                    )
                    warnings.extend(msg_warnings)
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
                    # 可以将工具链节点展开为普通工具调用 Can expand tool chain nodes into normal tool calls
                    tool_call = extension.get("tool_call")
                    if tool_call:
                        # 创建一个assistant消息包含工具调用 Create an assistant message containing tool call
                        anthropic_message = {
                            "role": "assistant",
                            "content": [self._ir_tool_call_to_p(tool_call)],
                        }
                        messages.append(anthropic_message)
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果 Build result
        result = {"messages": messages}

        # 添加system消息（如果有） Add system message (if any)
        if system_messages:
            result["system"] = system_messages

        # 添加工具定义 Add tool definitions
        if tools:
            result["tools"] = [self._ir_tool_to_p(tool) for tool in tools]

        # 添加工具选择 Add tool choice
        if tool_choice:
            result["tool_choice"] = self._ir_tool_choice_to_p(tool_choice)

        return result, warnings

    def from_provider(self, provider_data: Any) -> Union[IRInput, IRResponse]:
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

        # If it's a full API response, convert to IRResponse
        if (
            "id" in provider_data
            and "type" in provider_data
            and provider_data["type"] == "message"
        ):
            return self._p_response_to_ir(provider_data)

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
                ir_content = []
                for part in system_content:
                    converted_parts = self._p_content_part_to_ir(part)
                    ir_content.extend(converted_parts)
                ir_input.append(
                    Message(
                        role="system",
                        content=ir_content,
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
            ir_message = self._p_message_to_ir(msg)
            ir_input.append(ir_message)

        return ir_input

    # ==================== 分层方法 Layer methods ====================

    def _ir_request_to_p(
        self, ir_request: IRRequest
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IRRequest转换为Anthropic请求参数
        Convert IRRequest to Anthropic request parameters
        """
        warnings = []
        result = {
            "model": ir_request["model"],
        }

        # 1. 处理消息 / Handle messages
        messages = []

        # 处理 system_instruction / Handle system_instruction
        system_instruction = ir_request.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                result["system"] = system_instruction
            elif isinstance(system_instruction, list):
                result["system"] = self._ir_text_to_p_batch(system_instruction)

        # 处理 messages / Handle messages
        ir_input = ir_request["messages"]
        for item in ir_input:
            if is_message(item):
                if item["role"] == "system":
                    # Anthropic system messages are handled above
                    continue
                converted, msg_warnings = self._ir_message_to_p(item, ir_input)
                warnings.extend(msg_warnings)
                if converted:
                    messages.append(converted)
            elif is_extension_item(item):
                extension_type = item.get("type")
                if extension_type == "tool_chain_node":
                    tool_call = item.get("tool_call")
                    if tool_call:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [self._ir_tool_call_to_p(tool_call)],
                            }
                        )
                else:
                    warnings.append(
                        f"Extension item ignored in request: {extension_type}"
                    )

        result["messages"] = messages

        # 2. 处理工具 / Handle tools
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = [self._ir_tool_to_p(tool) for tool in tools]

        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            result["tool_choice"] = self._ir_tool_choice_to_p(tool_choice)

        tool_config = ir_request.get("tool_config")
        if tool_config:
            if "disable_parallel" in tool_config:
                # Anthropic handles this in tool_choice or as a separate param depending on SDK version
                # Here we follow the mapping doc which suggests it's part of tool_choice
                if "tool_choice" not in result:
                    result["tool_choice"] = {"type": "auto"}
                result["tool_choice"]["disable_parallel_tool_use"] = tool_config[
                    "disable_parallel"
                ]
            if "max_calls" in tool_config:
                warnings.append("Anthropic does not support max_tool_calls, ignored")

        # 3. 处理生成配置 / Handle generation config
        gen_config = ir_request.get("generation", {})

        # Anthropic requires max_tokens
        result["max_tokens"] = gen_config.get("max_tokens", 4096)

        if gen_config:
            if "temperature" in gen_config:
                # Anthropic temperature is 0.0-1.0
                result["temperature"] = min(gen_config["temperature"], 1.0)
            if "top_p" in gen_config:
                result["top_p"] = gen_config["top_p"]
            if "top_k" in gen_config:
                result["top_k"] = gen_config["top_k"]
            if "stop_sequences" in gen_config:
                result["stop_sequences"] = list(gen_config["stop_sequences"])

            # Unmapped fields
            for field in [
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "seed",
                "logprobs",
                "n",
            ]:
                if field in gen_config:
                    warnings.append(f"Anthropic does not support {field}, ignored")

        # 4. 处理响应格式 / Handle response format
        if "response_format" in ir_request:
            warnings.append(
                "Anthropic does not support response_format, use system instructions or tools instead"
            )

        # 5. 处理推理配置 / Handle reasoning config
        reasoning = ir_request.get("reasoning")
        if reasoning:
            thinking = {}
            if reasoning.get("type") == "enabled":
                thinking["type"] = "enabled"
                if "budget_tokens" in reasoning:
                    thinking["budget_tokens"] = reasoning["budget_tokens"]
                result["thinking"] = thinking
            elif reasoning.get("type") == "disabled":
                result["thinking"] = {"type": "disabled"}

        # 6. 处理流式配置 / Handle stream config
        stream = ir_request.get("stream")
        if stream:
            if "enabled" in stream:
                result["stream"] = stream["enabled"]

        # 7. 处理缓存配置 / Handle cache config
        # Anthropic cache is block-level, not request-level.
        # For POC, we might ignore it or apply to system/tools if needed.
        if "cache" in ir_request:
            warnings.append(
                "Anthropic cache control is block-level, request-level cache config ignored"
            )

        # 8. 处理扩展参数 / Handle provider extensions
        extensions = ir_request.get("provider_extensions")
        if extensions:
            result.update(extensions)

        return result, warnings

    def _ir_text_to_p_batch(self, content: List[TextPart]) -> str:
        """批量转换文本内容为字符串 / Batch convert text content to string"""
        text_parts = []
        for part in content:
            if is_text_part(part):
                text_parts.append(part["text"])
        return " ".join(text_parts)

    def _p_response_to_ir(self, provider_response: Dict[str, Any]) -> IRResponse:
        """将Anthropic响应转换为IRResponse
        Convert Anthropic response to IRResponse
        """
        import time

        # Anthropic response is a single message, we wrap it in choices[0]
        ir_message = self._p_message_to_ir(provider_response)

        stop_reason_val = provider_response.get("stop_reason")
        reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
            "refusal": "refusal",
        }

        choice_info = {
            "index": 0,
            "message": ir_message,
            "finish_reason": {"reason": reason_map.get(stop_reason_val, "stop")},
        }

        if "stop_sequence" in provider_response:
            choice_info["finish_reason"]["stop_sequence"] = provider_response[
                "stop_sequence"
            ]

        ir_response = {
            "id": provider_response.get("id", ""),
            "object": "response",
            "created": int(time.time()),  # Anthropic doesn't provide timestamp
            "model": provider_response.get("model", ""),
            "choices": [choice_info],
        }

        # 处理使用统计 / Handle usage
        p_usage = provider_response.get("usage")
        if p_usage:
            input_tokens = p_usage.get("input_tokens", 0)
            output_tokens = p_usage.get("output_tokens", 0)
            usage_info = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

            if "cache_read_input_tokens" in p_usage:
                usage_info["cache_read_tokens"] = p_usage["cache_read_input_tokens"]

            # Detailed completion tokens (reasoning)
            # Anthropic thinking tokens are part of output_tokens
            # If we have thinking blocks, we could potentially count them,
            # but usually it's provided in usage if supported.

            ir_response["usage"] = usage_info

        return ir_response

    def _ir_message_to_p(
        self, message: Dict[str, Any], ir_input: IRInput
    ) -> Tuple[Any, List[str]]:
        """IR Message → Provider Message / IR消息转换为Anthropic消息

        Returns:
            Tuple[转换后的消息, 警告列表] / Tuple[converted message, warning list]
        """
        role = message["role"]
        content = message["content"]
        warnings = []

        # 转换内容部分
        anthropic_content = []
        for part in content:
            converted, part_warnings = self._ir_content_part_to_p(part, ir_input)
            warnings.extend(part_warnings)
            if converted:
                anthropic_content.append(converted)

        if role == "system":
            # System消息返回内容列表，由调用者处理
            return anthropic_content, warnings
        else:
            # 普通消息返回完整消息结构
            return {
                "role": role,
                "content": anthropic_content,
            }, warnings

    def _ir_content_part_to_p(
        self, content_part: Dict[str, Any], ir_input: IRInput
    ) -> Tuple[Any, List[str]]:
        """IR ContentPart → Provider Content/Part / IR内容部分转换为Anthropic内容"""
        part_type = content_part.get("type")
        warnings = []

        if is_text_part(content_part):
            return self._ir_text_to_p(content_part), warnings

        elif part_type == "image":
            return self._ir_image_to_p(content_part), warnings

        elif part_type == "file":
            return self._ir_file_to_p(content_part), warnings

        elif is_tool_call_part(content_part):
            return self._ir_tool_call_to_p(content_part), warnings

        elif is_tool_result_part(content_part):
            return self._ir_tool_result_to_p(content_part), warnings

        elif part_type == "reasoning":
            return self._ir_reasoning_to_p(content_part), warnings

        return None, warnings

    def _p_message_to_ir(self, provider_message: Any) -> Dict[str, Any]:
        """Provider Message → IR Message / Anthropic消息转换为IR消息"""
        if not isinstance(provider_message, dict):
            return None

        role = provider_message.get("role")
        content = provider_message.get("content")

        ir_content = []
        if isinstance(content, str):
            ir_content.append(TextPart(type="text", text=content))
        elif isinstance(content, list):
            for part in content:
                converted_parts = self._p_content_part_to_ir(part)
                ir_content.extend(converted_parts)

        return Message(role=role, content=ir_content)

    def _p_content_part_to_ir(self, provider_part: Any) -> List[Dict[str, Any]]:
        """Provider Content/Part → IR ContentPart(s) / Anthropic内容部分转换为IR内容"""
        if isinstance(provider_part, str):
            return [TextPart(type="text", text=provider_part)]

        if not isinstance(provider_part, dict):
            return []

        part_type = provider_part.get("type")

        if part_type == "text":
            return [self._p_text_to_ir(provider_part)]

        elif part_type == "image":
            return [self._p_image_to_ir(provider_part)]

        elif part_type == "document":
            return [self._p_file_to_ir(provider_part)]

        elif part_type in ["tool_use", "server_tool_use"]:
            return [self._p_tool_call_to_ir(provider_part)]

        elif part_type == "tool_result":
            return [self._p_tool_result_to_ir(provider_part)]

        elif part_type == "thinking":
            return [self._p_reasoning_to_ir(provider_part)]

        return []

    # ==================== 内容类型转换方法 Content type conversion methods ====================

    def _ir_text_to_p(self, text_part: TextPart) -> Any:
        """IR TextPart → Provider Text Content / IR文本部分转换为Anthropic文本内容"""
        return TextPart(type="text", text=text_part["text"])

    def _p_text_to_ir(self, provider_text: Any) -> TextPart:
        """Provider Text Content → IR TextPart / Anthropic文本内容转换为IR文本部分"""
        if isinstance(provider_text, str):
            return TextPart(type="text", text=provider_text)
        if isinstance(provider_text, dict) and provider_text.get("type") == "text":
            return TextPart(type="text", text=provider_text["text"])
        return None

    def _ir_image_to_p(self, image_part: ImagePart) -> Any:
        """IR ImagePart → Provider Image Content / IR图像部分转换为Anthropic图像内容"""
        image_url = FieldMapper.get_image_url(image_part)
        image_data = FieldMapper.get_image_data(image_part)

        if image_data:
            # base64形式 base64 form
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_data["media_type"],
                    "data": image_data["data"],
                },
            }
        elif image_url:
            # URL形式 URL form
            return {
                "type": "image",
                "source": {"type": "url", "url": image_url},
            }
        else:
            raise ValueError("Image part must have either image_url or image_data")

    def _p_image_to_ir(self, provider_image: Any) -> ImagePart:
        """Provider Image Content → IR ImagePart / Anthropic图像内容转换为IR图像部分"""
        source = provider_image.get("source", {})
        if source.get("type") == "base64":
            return ImagePart(
                type="image",
                image_data={
                    "data": source.get("data", ""),
                    "media_type": source.get("media_type", ""),
                },
            )
        elif source.get("type") == "url":
            return ImagePart(type="image", image_url=source.get("url", ""))
        return ImagePart(type="image")

    def _ir_file_to_p(self, file_part: FilePart) -> Any:
        """IR FilePart → Provider File Content / IR文件部分转换为Anthropic文件内容"""
        if "file_data" in file_part:
            file_data = file_part["file_data"]
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": file_data["media_type"],
                    "data": file_data["data"],
                },
            }
        elif "file_url" in file_part:
            return {
                "type": "document",
                "source": {"type": "url", "url": file_part["file_url"]},
            }
        else:
            raise ValueError("File part must have either file_data or file_url")

    def _p_file_to_ir(self, provider_file: Any) -> FilePart:
        """Provider File Content → IR FilePart / Anthropic文件内容转换为IR文件部分"""
        source = provider_file.get("source", {})
        if source.get("type") == "base64":
            return FilePart(
                type="file",
                file_data={
                    "data": source["data"],
                    "media_type": source["media_type"],
                },
            )
        elif source.get("type") == "url":
            return FilePart(type="file", file_url=source["url"])
        return FilePart(type="file")

    def _ir_tool_call_to_p(self, tool_call_part: ToolCallPart) -> Any:
        """IR ToolCallPart → Provider Tool Call / IR工具调用部分转换为Anthropic工具调用"""
        return ToolCallConverter.to_anthropic(tool_call_part)

    def _p_tool_call_to_ir(self, provider_tool_call: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart / Anthropic工具调用转换为IR工具调用部分"""
        return ToolCallConverter.from_anthropic(provider_tool_call)

    def _ir_tool_result_to_p(self, tool_result_part: ToolResultPart) -> Any:
        """IR ToolResultPart → Provider Tool Result / IR工具结果部分转换为Anthropic工具结果"""
        return {
            "type": "tool_result",
            "tool_use_id": tool_result_part["tool_call_id"],
            "content": tool_result_part["result"],
            "is_error": tool_result_part.get("is_error", False),
        }

    def _p_tool_result_to_ir(self, provider_tool_result: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart / Anthropic工具结果转换为IR工具结果部分"""
        return ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("tool_use_id", ""),
            result=provider_tool_result.get("content", ""),
            is_error=provider_tool_result.get("is_error", False),
        )

    def _ir_reasoning_to_p(self, reasoning_part: Dict[str, Any]) -> Any:
        """IR ReasoningPart → Provider Reasoning Content / IR推理部分转换为Anthropic推理内容"""
        return {"type": "thinking", "thinking": reasoning_part["reasoning"]}

    def _p_reasoning_to_ir(self, provider_reasoning: Any) -> Dict[str, Any]:
        """Provider Reasoning Content → IR ReasoningPart / Anthropic推理内容转换为IR推理部分"""
        return ReasoningPart(type="reasoning", reasoning=provider_reasoning["thinking"])

    def _ir_tool_to_p(self, tool: ToolDefinition) -> Any:
        """IR ToolDefinition → Provider Tool Definition / IR工具定义转换为Anthropic工具定义"""
        return ToolConverter.convert_tool_definition(tool, "anthropic")

    def _p_tool_to_ir(self, provider_tool: Any) -> ToolDefinition:
        """Provider Tool Definition → IR ToolDefinition / Anthropic工具定义转换为IR工具定义"""
        # Anthropic工具定义格式 / Anthropic tool definition format
        return {
            "type": "function",
            "name": provider_tool.get("name", ""),
            "description": provider_tool.get("description", ""),
            "parameters": provider_tool.get("input_schema", {}),
        }

    def _ir_tool_choice_to_p(self, tool_choice: ToolChoice) -> Any:
        """IR ToolChoice → Provider Tool Choice Config / IR工具选择转换为Anthropic工具选择配置"""
        return ToolConverter.convert_tool_choice(tool_choice, "anthropic")

    def _p_tool_choice_to_ir(self, provider_tool_choice: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice / Anthropic工具选择配置转换为IR工具选择"""
        # Anthropic工具选择格式 / Anthropic tool choice format
        if isinstance(provider_tool_choice, dict):
            choice_type = provider_tool_choice.get("type", "auto")
            if choice_type == "auto":
                return {"mode": "auto"}
            elif choice_type == "any":
                return {"mode": "required"}
            elif choice_type == "tool":
                tool_name = provider_tool_choice.get("name", "")
                return {"mode": "tool", "tool_name": tool_name}
        return {"mode": "auto"}
