"""
LLMIR - OpenAI Responses API Converter

实现IR与OpenAI Responses API格式之间的转换
Implement conversion between IR and OpenAI Responses API format
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ...types.ir import (
    FileData,
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


class OpenAIResponsesConverter(BaseConverter):
    """OpenAI Responses API格式转换器
    OpenAI Responses API format converter

    Responses API使用扁平的项目列表，每个项目可以是消息、工具调用或其他类型
    Responses API uses a flat list of items, each item can be message, tool call or other types
    """

    def to_provider(
        self,
        ir_input: Union[IRInput, IRRequest],
        tools: Optional[Iterable[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IR格式转换为OpenAI Responses API格式
        Convert IR format to OpenAI Responses API format

        Returns:
            Tuple[转换后的数据, 警告信息列表] / Tuple[converted data, warning list]
        """
        if isinstance(ir_input, dict) and "messages" in ir_input:
            # Handle IRRequest
            return self._ir_request_to_p(ir_input)

        # 验证输入 / Validate input
        validation_errors = self.validate_ir_input(ir_input)
        if validation_errors:
            raise ValueError(f"Invalid IR input: {validation_errors}")

        items = []
        warnings = []

        for item in ir_input:
            if is_message(item):
                message = item  # type: ignore
                converted, msg_warnings = self._ir_message_to_p(message, ir_input)
                warnings.extend(msg_warnings)
                if isinstance(converted, list):
                    items.extend(converted)
                elif converted:
                    items.append(converted)
            elif is_extension_item(item):
                extension = item  # type: ignore
                extension_type = extension.get("type")

                if extension_type == "system_event":
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
                        items.append(self._ir_tool_call_to_p(tool_call))
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果 / Build result
        result = {"input": items}

        # 添加工具定义 / Add tool definitions
        if tools:
            result["tools"] = ToolConverter.batch_convert_tools(
                tools, "openai_responses"
            )

        # 添加工具选择 / Add tool choice
        if tool_choice:
            result["tool_choice"] = ToolConverter.convert_tool_choice(
                tool_choice, "openai_responses"
            )

        return result, warnings

    def from_provider(
        self,
        provider_data: Any,
        **kwargs: Any,
    ) -> Union[IRInput, IRResponse]:
        """将OpenAI Responses API格式转换为IR格式
        Convert OpenAI Responses API format to IR format

        注意：Responses API的响应在'output'字段，而请求在'input'字段
        Note: Responses API response is in 'output' field, while request is in 'input' field

        Args:
            provider_data: OpenAI Responses API响应对象或字典
                          OpenAI Responses API response object or dict
                          自动处理Pydantic模型对象（调用.model_dump()）
                          Automatically handles Pydantic model objects (calls .model_dump())
        """
        # 自动unwrap Pydantic模型对象 / Auto unwrap Pydantic model objects
        if hasattr(provider_data, "model_dump"):
            provider_data = provider_data.model_dump()

        if not isinstance(provider_data, dict):
            raise ValueError("OpenAI Responses data must be a dictionary")

        # If it's a full API response, convert to IRResponse
        if "id" in provider_data and "output" in provider_data:
            return self._p_response_to_ir(provider_data)

        # 响应数据在output字段，请求数据在input字段
        # Response data in output field, request data in input field
        items = provider_data.get("output") or provider_data.get("input", [])
        if not isinstance(items, list):
            raise ValueError("OpenAI Responses output/input must be a list")

        ir_input = []
        current_message = None

        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else None

            if item_type == "message":
                # 消息类型：创建新消息
                new_message = self._p_message_to_ir(item)
                if new_message:
                    # 保存当前消息
                    if current_message:
                        ir_input.append(current_message)
                    current_message = new_message

            elif item_type in [
                "function_call",
                "mcp_call",
                "shell_call",
                "computer_call",
                "code_interpreter_call",
            ]:
                # 工具调用：转换为ToolCallPart
                tool_call = self._p_tool_call_to_ir(item)
                if current_message and current_message.get("role") == "assistant":
                    # 如果有assistant消息，直接附加
                    current_message["content"].append(tool_call)
                else:
                    # 否则创建新的assistant消息
                    if current_message:
                        ir_input.append(current_message)
                    current_message = Message(role="assistant", content=[tool_call])

            elif item_type in ["function_call_output", "mcp_call_output"]:
                # 工具结果：转换为ToolResultPart
                tool_result = self._p_tool_result_to_ir(item)
                if current_message and current_message.get("role") == "user":
                    current_message["content"].append(tool_result)
                else:
                    # 创建新的user消息
                    if current_message:
                        ir_input.append(current_message)
                    current_message = Message(role="user", content=[tool_result])

            elif item_type == "reasoning":
                # 推理内容：返回ReasoningPart
                reasoning = self._p_reasoning_to_ir(item)
                if reasoning:
                    if current_message and current_message.get("role") == "assistant":
                        current_message["content"].append(reasoning)
                    else:
                        # 创建新的assistant消息
                        if current_message:
                            ir_input.append(current_message)
                        current_message = Message(role="assistant", content=[reasoning])

            elif item_type == "system_event":
                # 系统事件：直接添加到输出
                # System event: directly added to output
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

        # 处理最后一个消息 / Handle the last message
        if current_message:
            if current_message.get("content"):
                ir_input.append(current_message)

        return ir_input

    # ==================== 分层方法 Layer methods ====================

    def _ir_request_to_p(
        self, ir_request: IRRequest
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IRRequest转换为OpenAI Responses API请求参数
        Convert IRRequest to OpenAI Responses API request parameters
        """
        warnings = []
        result = {
            "model": ir_request["model"],
        }

        # 1. 处理消息 / Handle messages
        items = []

        # 处理 system_instruction / Handle system_instruction
        system_instruction = ir_request.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                result["instructions"] = system_instruction
            elif isinstance(system_instruction, list):
                result["instructions"] = self._ir_text_to_p_batch(system_instruction)

        # 处理 messages / Handle messages
        ir_input = ir_request["messages"]
        for item in ir_input:
            if is_message(item):
                converted, msg_warnings = self._ir_message_to_p(item, ir_input)
                warnings.extend(msg_warnings)
                if isinstance(converted, list):
                    items.extend(converted)
                elif converted:
                    items.append(converted)
            elif is_extension_item(item):
                # 扩展项处理逻辑与 to_provider 一致
                extension_type = item.get("type")
                if extension_type == "system_event":
                    items.append(
                        {
                            "type": "system_event",
                            "event_type": item.get("event_type"),
                            "timestamp": item.get("timestamp"),
                            "message": item.get("message", ""),
                        }
                    )
                elif extension_type == "tool_chain_node":
                    tool_call = item.get("tool_call")
                    if tool_call:
                        items.append(self._ir_tool_call_to_p(tool_call))
                else:
                    warnings.append(
                        f"Extension item ignored in request: {extension_type}"
                    )

        result["input"] = items

        # 2. 处理工具 / Handle tools
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = ToolConverter.batch_convert_tools(
                tools, "openai_responses"
            )

        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            result["tool_choice"] = ToolConverter.convert_tool_choice(
                tool_choice, "openai_responses"
            )

        tool_config = ir_request.get("tool_config")
        if tool_config:
            if "disable_parallel" in tool_config:
                result["parallel_tool_calls"] = not tool_config["disable_parallel"]
            if "max_calls" in tool_config:
                result["max_tool_calls"] = tool_config["max_calls"]

        # 3. 处理生成配置 / Handle generation config
        gen_config = ir_request.get("generation", {})
        if gen_config:
            for ir_field, p_field in [
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_tokens", "max_output_tokens"),
                ("top_logprobs", "top_logprobs"),
            ]:
                if ir_field in gen_config:
                    result[p_field] = gen_config[ir_field]

            if "truncation" in gen_config:
                result["truncation"] = gen_config["truncation"]

            # Unmapped fields
            for field in [
                "top_k",
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "seed",
                "n",
                "stop_sequences",
            ]:
                if field in gen_config:
                    warnings.append(
                        f"OpenAI Responses API does not support {field}, ignored"
                    )

        # 4. 处理响应格式 / Handle response format
        resp_format = ir_request.get("response_format")
        if resp_format:
            fmt_type = resp_format.get("type")
            if fmt_type == "text":
                result["text"] = {"type": "text"}
            elif fmt_type == "json_object":
                result["text"] = {"type": "json_object"}
            elif fmt_type == "json_schema":
                result["text"] = {
                    "type": "json_schema",
                    "json_schema": resp_format.get("json_schema", {}),
                }

        # 5. 处理推理配置 / Handle reasoning config
        reasoning = ir_request.get("reasoning")
        if reasoning:
            reasoning_p = {}
            if "type" in reasoning:
                reasoning_p["type"] = reasoning["type"]
            if "effort" in reasoning:
                reasoning_p["effort"] = reasoning["effort"]
            if reasoning_p:
                result["reasoning"] = reasoning_p

        # 6. 处理流式配置 / Handle stream config
        stream = ir_request.get("stream")
        if stream:
            if "enabled" in stream:
                result["stream"] = stream["enabled"]
            if stream.get("include_usage") and result.get("stream"):
                result["stream_options"] = {"include_obfuscation": True}

        # 7. 处理缓存配置 / Handle cache config
        cache = ir_request.get("cache")
        if cache:
            if "key" in cache:
                result["prompt_cache_key"] = cache["key"]
            if "retention" in cache:
                result["prompt_cache_retention"] = cache["retention"]

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
        """将OpenAI Responses API响应转换为IRResponse
        Convert OpenAI Responses API response to IRResponse
        """
        # OpenAI Responses API uses 'output' list instead of 'choices'
        # We convert each item in 'output' to a choice if it's a message
        choices = []
        output_items = provider_response.get("output", [])

        # Determine finish reason from status
        status = provider_response.get("status")
        finish_reason_val = "stop"
        if status == "completed":
            finish_reason_val = "stop"
        elif status == "incomplete":
            incomplete_details = provider_response.get("incomplete_details", {})
            reason = incomplete_details.get("reason")
            if reason == "max_output_tokens":
                finish_reason_val = "length"
            elif reason == "content_filter":
                finish_reason_val = "content_filter"
            else:
                finish_reason_val = "stop"
        elif status == "failed":
            finish_reason_val = "error"
        elif status == "cancelled":
            finish_reason_val = "cancelled"

        # Group output items into messages
        # In Responses API, output can contain multiple items (message, tool_call, reasoning)
        # We'll try to reconstruct a single assistant message for the first choice
        ir_items = []
        for item in output_items:
            converted = self._p_item_to_ir(item)
            if isinstance(converted, list):
                ir_items.extend(converted)
            elif converted:
                ir_items.append(converted)

        # Filter for content parts that belong to a message
        message_content = [
            item
            for item in ir_items
            if isinstance(item, dict)
            and "type" in item
            and item["type"] not in ["system_event"]
        ]

        if message_content:
            choices.append(
                {
                    "index": 0,
                    "message": Message(role="assistant", content=message_content),
                    "finish_reason": {"reason": finish_reason_val},
                }
            )

        ir_response = {
            "id": provider_response.get("id", ""),
            "object": "response",
            "created": int(provider_response.get("created_at", 0)),
            "model": provider_response.get("model", ""),
            "choices": choices,
        }

        # 处理使用统计 / Handle usage
        p_usage = provider_response.get("usage")
        if p_usage:
            usage_info = {
                "prompt_tokens": p_usage.get("input_tokens", 0),
                "completion_tokens": p_usage.get("output_tokens", 0),
                "total_tokens": p_usage.get("total_tokens", 0),
            }

            # 处理详细统计 / Handle detailed statistics
            p_input_details = p_usage.get("input_tokens_details")
            if p_input_details:
                if "cached_tokens" in p_input_details:
                    usage_info["cache_read_tokens"] = p_input_details["cached_tokens"]

            p_output_details = p_usage.get("output_tokens_details")
            if p_output_details:
                if "reasoning_tokens" in p_output_details:
                    usage_info["reasoning_tokens"] = p_output_details[
                        "reasoning_tokens"
                    ]

            ir_response["usage"] = usage_info

        if "service_tier" in provider_response:
            ir_response["service_tier"] = provider_response["service_tier"]

        return ir_response

    def _ir_message_to_p(
        self, message: Dict[str, Any], ir_input: IRInput, **kwargs: Any
    ) -> Tuple[Any, List[str]]:
        """IR Message → Provider Message/Item(s) / IR消息转换为Responses API项目

        Returns:
            Tuple[转换后的项目列表, 警告列表] / Tuple[converted items, warning list]
        """
        role = message["role"]
        content = message["content"]
        warnings = []
        items = []

        if role in ["system", "user", "developer"]:
            # 基本消息类型 / Basic message types
            content_parts = []
            tool_items = []

            for part in content:
                converted, part_warnings = self._ir_content_part_to_p(part, ir_input)
                warnings.extend(part_warnings)
                if converted:
                    if isinstance(converted, dict):
                        # 检查是否是工具调用（需要在消息级别处理为独立项目）
                        # Check if it's tool call (handled at message level as independent item)
                        if converted.get("type") == "tool_call_item":
                            tool_items.append(
                                self._ir_tool_call_to_p(converted["tool_call"])
                            )
                        # 检查是否是工具结果（需要在消息级别处理）
                        # Check if it's tool result (handled at message level)
                        elif converted.get("type") == "function_call_output":
                            items.append(converted)
                        elif converted.get("type") == "reasoning_item":
                            items.append(
                                {
                                    "type": "reasoning",
                                    "content": converted.get("reasoning"),
                                }
                            )
                        else:
                            content_parts.append(converted)
                    elif isinstance(converted, dict) and "tool_call_id" in converted:
                        # 工具调用是独立的项目 / Tool calls are independent items
                        tool_items.append(self._ir_tool_call_to_p(converted))
                    else:
                        content_parts.append(converted)

            # 添加消息（如果有内容）/ Add message (if there is content)
            if content_parts:
                # system/user/developer 消息使用 input_text
                # assistant 消息使用 output_text
                content_type = (
                    "input_text"
                    if role in ["system", "user", "developer"]
                    else "output_text"
                )
                openai_content = []
                for cp in content_parts:
                    if cp.get("type") == "text":
                        openai_content.append(
                            {"type": content_type, "text": cp["text"]}
                        )
                    else:
                        openai_content.append(cp)

                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": openai_content,
                    }
                )

            # 添加工具调用 / Add tool calls
            items.extend(tool_items)

        elif role == "assistant":
            # Assistant消息处理 / Assistant message handling
            content_parts = []
            tool_items = []
            reasoning_items = []

            for part in content:
                if is_text_part(part):
                    # 检查是否是推理文本 / Check if it's reasoning text
                    if part.get("reasoning"):
                        reasoning_items.append(
                            {"type": "reasoning", "content": part["text"]}
                        )
                    else:
                        content_parts.append(
                            {"type": "output_text", "text": part["text"]}
                        )
                elif is_tool_call_part(part):
                    tool_items.append(self._ir_tool_call_to_p(part))
                elif part.get("type") == "reasoning":
                    reasoning_items.append(
                        {"type": "reasoning", "content": part.get("reasoning")}
                    )

            # 添加推理内容 / Add reasoning content
            items.extend(reasoning_items)

            # 添加assistant消息（如果有文本内容）/ Add assistant message (if there is text content)
            if content_parts:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content_parts,
                    }
                )

            # 添加工具调用 / Add tool calls
            items.extend(tool_items)

        return items, warnings

    def _ir_content_part_to_p(
        self, content_part: Dict[str, Any], ir_input: IRInput, **kwargs: Any
    ) -> Tuple[Any, List[str]]:
        """IR ContentPart → Provider Content/Item / IR内容部分转换为Responses API内容

        Returns:
            Tuple[转换后的内容部分, 警告列表] / Tuple[converted content part, warning list]
        """
        part_type = content_part.get("type")
        warnings = []

        if is_text_part(content_part):
            return {"type": "text", "text": content_part["text"]}, warnings

        elif part_type == "image":
            return self._ir_image_to_p(content_part), warnings

        elif part_type == "file":
            return self._ir_file_to_p(content_part), warnings

        elif is_tool_call_part(content_part):
            # 工具调用返回特殊标记，由消息层处理为独立项目
            # Tool call returns special marker, handled at message level as independent item
            return {
                "type": "tool_call_item",
                "tool_call": content_part,
            }, warnings

        elif is_tool_result_part(content_part):
            # 工具结果转换为函数调用输出
            # Tool results converted to function call output
            result_content = content_part.get("result") or content_part.get("content")
            return {
                "type": "function_call_output",
                "call_id": content_part["tool_call_id"],
                "output": str(result_content),
            }, warnings

        elif part_type == "reasoning":
            # 推理内容返回特殊标记，由消息层处理
            # Reasoning content returns special marker, handled at message level
            return {
                "type": "reasoning_item",
                "reasoning": content_part.get("reasoning"),
            }, warnings

        return None, warnings

    def _p_item_to_ir(self, provider_item: Any, current_message: Any = None) -> Any:
        """Provider Item → IR Message/Extension / Responses API项目转换为IR消息或扩展项

        Args:
            provider_item: Responses API项目
            current_message: 当前消息（未使用，保留为了API兼容性）

        Returns:
            转换后的内容部分列表或扩展项 / Converted content parts or extension
        """
        if not isinstance(provider_item, dict):
            return None

        item_type = provider_item.get("type")

        if item_type == "message":
            msg = self._p_message_to_ir(provider_item)
            return msg.get("content", []) if msg else []

        elif item_type in [
            "function_call",
            "mcp_call",
            "shell_call",
            "computer_call",
            "code_interpreter_call",
        ]:
            # 工具调用转换为ToolCallPart
            # Tool call converted to ToolCallPart
            tool_call = self._p_tool_call_to_ir(provider_item)
            return [tool_call]

        elif item_type in ["function_call_output", "mcp_call_output"]:
            # 工具结果转换为ToolResultPart
            # Tool result converted to ToolResultPart
            tool_result = self._p_tool_result_to_ir(provider_item)
            return [tool_result]

        elif item_type == "reasoning":
            # 推理内容：返回ReasoningPart
            # Reasoning content: returns ReasoningPart
            reasoning = self._p_reasoning_to_ir(provider_item)
            if reasoning:
                return reasoning
            return None

        elif item_type == "system_event":
            # 系统事件转换为扩展项 / System event converted to extension item
            return {
                "type": "system_event",
                "event_type": provider_item.get("event_type", "unknown"),
                "timestamp": provider_item.get("timestamp", ""),
                "message": provider_item.get("message", ""),
            }

        return None

    def _p_message_to_ir(self, provider_message: Any, **kwargs: Any) -> Dict[str, Any]:
        """Provider Message → IR Message / Responses API消息转换为IR消息"""
        if not isinstance(provider_message, dict):
            return None

        role = provider_message.get("role")
        content = provider_message.get("content")

        ir_content = []

        if isinstance(content, str):
            ir_content.append(TextPart(type="text", text=content))
        elif isinstance(content, list):
            for part in content:
                converted = self._p_content_part_to_ir(part)
                if converted:
                    ir_content.extend(converted)

        # 空消息（content为[]）也创建消息，因为可能有后续工具调用需要附加
        # Empty messages (content=[]) are also created because subsequent tool calls may need to be appended
        return Message(role=role, content=ir_content)

    def _p_content_part_to_ir(
        self, provider_part: Any, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Provider Content/Part → IR ContentPart(s) / Responses API内容部分转换为IR内容"""
        if isinstance(provider_part, str):
            return [TextPart(type="text", text=provider_part)]

        if not isinstance(provider_part, dict):
            return []

        part_type = provider_part.get("type")

        # 支持input_text和output_text / Support input_text and output_text
        if part_type in ["input_text", "output_text", "text"]:
            return [TextPart(type="text", text=provider_part["text"])]

        elif part_type == "input_image":
            return [self._p_image_to_ir(provider_part)]

        elif part_type == "input_file":
            return [self._p_file_to_ir(provider_part)]

        return []

    # ==================== 内容类型转换方法 Content type conversion methods ====================

    def _ir_text_to_p(self, text_part: TextPart, **kwargs: Any) -> Any:
        """IR TextPart → Provider Text Content / IR文本部分转换为Responses API文本内容"""
        return {"type": "text", "text": text_part["text"]}

    def _p_text_to_ir(self, provider_text: Any, **kwargs: Any) -> TextPart:
        """Provider Text Content → IR TextPart / Responses API文本内容转换为IR文本部分"""
        if isinstance(provider_text, str):
            return TextPart(type="text", text=provider_text)
        if isinstance(provider_text, dict) and provider_text.get("type") in [
            "input_text",
            "output_text",
            "text",
        ]:
            return TextPart(type="text", text=provider_text["text"])
        return None

    def _ir_image_to_p(self, image_part: ImagePart, **kwargs: Any) -> Any:
        """IR ImagePart → Provider Image Content / IR图像部分转换为Responses API图像内容

        注意：图像始终使用 input_image，因为图像只能作为输入
        Note: images always use input_image, because images can only be input
        """
        result: Dict[str, Any] = {
            "type": "input_image",
            "detail": image_part.get("detail", "auto"),
        }

        # 支持多种URL字段名称 / Support multiple URL field names
        url = FieldMapper.get_image_url(image_part)
        image_data = FieldMapper.get_image_data(image_part)

        if url:
            result["image_url"] = url
        elif image_data:
            data_url = f"data:{image_data['media_type']};base64,{image_data['data']}"
            result["image_url"] = data_url
        else:
            raise ValueError("Image part must have either image_url/url or image_data")

        return result

    def _p_image_to_ir(self, provider_image: Any, **kwargs: Any) -> ImagePart:
        """Provider Image Content → IR ImagePart / Responses API图像内容转换为IR图像部分"""
        detail = provider_image.get("detail", "auto")

        if "image_url" in provider_image:
            url = provider_image["image_url"]
            if url.startswith("data:"):
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
        elif "file_id" in provider_image:
            # 文件ID形式，转换为引用 / File ID form, converted to reference
            return {
                "type": "image",
                "file_id": provider_image["file_id"],
                "detail": detail,
            }  # type: ignore

        return ImagePart(type="image", detail=detail)

    def _ir_file_to_p(self, file_part: FilePart, **kwargs: Any) -> Any:
        """IR FilePart → Provider File Content / IR文件部分转换为Responses API文件内容

        注意：文件始终使用 input_file，因为文件只能作为输入
        Note: files always use input_file, because files can only be input
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

    def _p_file_to_ir(self, provider_file: Any, **kwargs: Any) -> FilePart:
        """Provider File Content → IR FilePart / Responses API文件内容转换为IR文件部分"""
        file_name = provider_file.get("filename", "unknown")

        if "file_data" in provider_file:
            # 假设是base64编码的数据 / Assume it's base64 encoded data
            return FilePart(
                type="file",
                file_name=file_name,
                file_data=FileData(
                    data=provider_file["file_data"],
                    media_type="application/octet-stream",  # 默认类型 / Default type
                ),
            )
        elif "file_url" in provider_file:
            return FilePart(
                type="file", file_name=file_name, file_url=provider_file["file_url"]
            )
        elif "file_id" in provider_file:
            return {
                "type": "file",
                "file_name": file_name,
                "file_id": provider_file["file_id"],
            }  # type: ignore

        return FilePart(type="file", file_name=file_name)

    def _ir_tool_call_to_p(self, tool_call_part: ToolCallPart, **kwargs: Any) -> Any:
        """IR ToolCallPart → Provider Tool Call / IR工具调用部分转换为Responses API工具调用"""
        return ToolCallConverter.to_openai_responses(tool_call_part)

    def _p_tool_call_to_ir(
        self, provider_tool_call: Any, **kwargs: Any
    ) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart / Responses API工具调用转换为IR工具调用部分

        Returns:
            ToolCallPart
        """
        tool_call = ToolCallConverter.from_openai_responses(provider_tool_call)
        return tool_call

    def _ir_tool_result_to_p(
        self, tool_result_part: ToolResultPart, **kwargs: Any
    ) -> Any:
        """IR ToolResultPart → Provider Tool Result / IR工具结果部分转换为Responses API工具结果"""
        return {
            "type": "function_call_output",
            "call_id": tool_result_part["tool_call_id"],
            "output": str(
                tool_result_part.get("result") or tool_result_part.get("content", "")
            ),
        }

    def _p_tool_result_to_ir(
        self, provider_tool_result: Any, **kwargs: Any
    ) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart / Responses API工具结果转换为IR工具结果部分

        Returns:
            ToolResultPart
        """
        tool_result = ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("call_id", ""),
            result=provider_tool_result.get("output", ""),
            is_error=provider_tool_result.get("is_error", False),
        )
        return tool_result

    def _p_reasoning_to_ir(self, provider_reasoning: Any) -> ReasoningPart:
        """Provider Reasoning → IR ReasoningPart / Responses API推理内容转换为IR推理部分

        Returns:
            ReasoningPart or None
        """
        # o4-mini的reasoning可能有content字段为null的情况
        # o4-mini reasoning may have null content field
        reasoning_content = provider_reasoning.get(
            "reasoning"
        ) or provider_reasoning.get("content")

        if reasoning_content:
            reasoning_part = ReasoningPart(
                type="reasoning", reasoning=str(reasoning_content)
            )
            return reasoning_part

        return None

    def _ir_tool_to_p(self, tool: ToolDefinition, **kwargs: Any) -> Any:
        """IR ToolDefinition → Provider Tool Definition / IR工具定义转换为Responses API工具定义"""
        return ToolConverter.convert_tool_definition(tool, "openai_responses")

    def _p_tool_to_ir(self, provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """Provider Tool Definition → IR ToolDefinition / Responses API工具定义转换为IR工具定义"""
        # Responses API 工具定义格式 / Responses API tool definition format
        if "function" in provider_tool:
            func = provider_tool["function"]
            return {
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
        return provider_tool

    def _ir_tool_choice_to_p(self, tool_choice: ToolChoice, **kwargs: Any) -> Any:
        """IR ToolChoice → Provider Tool Choice Config / IR工具选择转换为Responses API工具选择配置"""
        return ToolConverter.convert_tool_choice(tool_choice, "openai_responses")

    def _p_tool_choice_to_ir(
        self, provider_tool_choice: Any, **kwargs: Any
    ) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice / Responses API工具选择配置转换为IR工具选择"""
        # Responses API tool choice format / Responses API tool choice format
        if isinstance(provider_tool_choice, str):
            mode = provider_tool_choice
            if mode == "none":
                return {"mode": "none"}
            elif mode == "auto":
                return {"mode": "auto"}
            elif mode == "required":
                return {"mode": "required"}
            return {"mode": "auto"}
        elif isinstance(provider_tool_choice, dict):
            if provider_tool_choice.get("type") == "function":
                func = provider_tool_choice.get("function", {})
                return {"mode": "tool", "tool_name": func.get("name")}
        return {"mode": "auto"}

    # ==================== 兼容性别名 Compatibility aliases ====================

    def _convert_image_to_responses(self, image_part: Dict[str, Any]) -> Any:
        """将IR图像转换为Responses API格式（兼容性别名）
        Convert IR image to Responses API format (compatibility alias)
        """
        return self._ir_image_to_p(image_part)

    def _convert_file_to_responses(self, file_part: Dict[str, Any]) -> Any:
        """将IR文件转换为Responses API格式（兼容性别名）
        Convert IR file to Responses API format (compatibility alias)
        """
        return self._ir_file_to_p(file_part)

    def _convert_image_from_responses(self, image_part: Dict[str, Any]) -> Any:
        """将Responses API图像转换为IR格式（兼容性别名）
        Convert Responses API image to IR format (compatibility alias)
        """
        return self._p_image_to_ir(image_part)

    def _convert_file_from_responses(self, file_part: Dict[str, Any]) -> Any:
        """将Responses API文件转换为IR格式（兼容性别名）
        Convert Responses API file to IR format (compatibility alias)
        """
        return self._p_file_to_ir(file_part)
