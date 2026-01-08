"""
LLMIR - OpenAI Chat Completions Converter

实现IR与OpenAI Chat Completions API格式之间的转换
Implement conversion between IR and OpenAI Chat Completions API format
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ...types.ir import (
    FileData,
    FilePart,
    ImagePart,
    IRInput,
    Message,
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
from ...utils import FieldMapper, ToolCallConverter, ToolConverter
from ...types.ir_response import IRResponse
from ..base import BaseConverter


class OpenAIChatConverter(BaseConverter):
    """OpenAI Chat Completions API格式转换器
    OpenAI Chat Completions API format converter
    """

    def to_provider(
        self,
        ir_input: Union[IRInput, IRRequest],
        tools: Optional[Iterable[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IR格式转换为OpenAI Chat Completions格式
        Convert IR format to OpenAI Chat Completions format

        OpenAI使用扁平结构，需要将工具调用提取到消息级别
        OpenAI uses flat structure, need to extract tool calls to message level
        """
        if isinstance(ir_input, dict) and "messages" in ir_input:
            # Handle IRRequest
            return self._ir_request_to_p(ir_input)

        # 验证输入 / Validate input
        validation_errors = self.validate_ir_input(ir_input)
        if validation_errors:
            raise ValueError(f"Invalid IR input: {validation_errors}")

        messages = []
        warnings = []

        for item in ir_input:
            if is_message(item):
                message = item  # type: ignore
                converted, msg_warnings = self._ir_message_to_p(message, ir_input)
                warnings.extend(msg_warnings)
                if isinstance(converted, list):
                    messages.extend(converted)
                elif converted:
                    messages.append(converted)
            elif is_extension_item(item):
                extension = item  # type: ignore
                extension_type = extension.get("type")

                if extension_type == "system_event":
                    warnings.append(
                        f"System event ignored: {extension.get('event_type', 'unknown')}"
                    )
                elif extension_type == "tool_chain_node":
                    warnings.append("Tool chain converted to sequential calls")
                    tool_call = extension.get("tool_call")
                    if tool_call:
                        openai_message = {
                            "role": "assistant",
                            "tool_calls": [self._ir_tool_call_to_p(tool_call)],
                        }
                        messages.append(openai_message)
                elif extension_type in ["batch_marker", "session_control"]:
                    warnings.append(f"Extension item ignored: {extension_type}")

        # 构建结果 / Build result
        result = {"messages": messages}

        # 添加工具定义 / Add tool definitions
        if tools:
            result["tools"] = ToolConverter.batch_convert_tools(tools, "openai_chat")

        # 添加工具选择 / Add tool choice
        if tool_choice:
            result["tool_choice"] = ToolConverter.convert_tool_choice(
                tool_choice, "openai"
            )

        return result, warnings

    def from_provider(self, provider_data: Any) -> Union[IRInput, IRResponse]:
        """将OpenAI Chat Completions格式转换为IR格式
        Convert OpenAI Chat Completions format to IR format

        Args:
            provider_data: OpenAI Chat Completions响应对象或字典
                          OpenAI Chat Completions response object or dict
                          可以是： / Can be:
                          1. API响应（包含choices字段） / API response (contains choices field)
                          2. 消息列表（包含messages字段） / Message list (contains messages field)
                          3. 单个消息对象（包含role字段） / Single message object (contains role field)
                          自动处理Pydantic模型对象（调用.model_dump()）
                          Automatically handles Pydantic model objects (calls .model_dump())
        """
        # 自动unwrap Pydantic模型对象 / Auto unwrap Pydantic model objects
        if hasattr(provider_data, "model_dump"):
            provider_data = provider_data.model_dump()

        if not isinstance(provider_data, dict):
            raise ValueError("OpenAI data must be a dictionary")

        # If it's a full API response, convert to IRResponse
        if "choices" in provider_data and "id" in provider_data:
            return self._p_response_to_ir(provider_data)

        ir_input = []

        # Handle different input formats / 处理不同的输入格式
        if "choices" in provider_data:
            # This is an API response with choices / 这是一个包含choices的API响应
            messages_to_process = []
            for choice in provider_data["choices"]:
                if "message" in choice:
                    messages_to_process.append(choice["message"])
                elif "delta" in choice:
                    # Streaming response / 流式响应
                    messages_to_process.append(choice["delta"])
        elif "messages" in provider_data:
            # This is a payload with messages / 这是一个包含messages的payload
            messages_to_process = provider_data["messages"]
        elif "role" in provider_data:
            # This is a single message / 这是一个单条消息
            messages_to_process = [provider_data]
        else:
            # Empty or unknown format / 空或未知格式
            messages_to_process = []

        for msg in messages_to_process:
            converted = self._p_message_to_ir(msg)
            if converted:
                ir_input.append(converted)

        return ir_input

    # ==================== 分层方法 Layer methods ====================

    def _ir_request_to_p(
        self, ir_request: IRRequest
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IRRequest转换为OpenAI Chat Completions请求参数
        Convert IRRequest to OpenAI Chat Completions request parameters
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
                messages.append({"role": "system", "content": system_instruction})
            elif isinstance(system_instruction, list):
                messages.append(
                    {
                        "role": "system",
                        "content": self._ir_text_to_p_batch(system_instruction),
                    }
                )

        # 处理 messages / Handle messages
        ir_input = ir_request["messages"]
        for item in ir_input:
            if is_message(item):
                converted, msg_warnings = self._ir_message_to_p(item, ir_input)
                warnings.extend(msg_warnings)
                if isinstance(converted, list):
                    messages.extend(converted)
                elif converted:
                    messages.append(converted)
            elif is_extension_item(item):
                # 扩展项处理逻辑与 to_provider 一致
                extension_type = item.get("type")
                if extension_type == "tool_chain_node":
                    tool_call = item.get("tool_call")
                    if tool_call:
                        messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [self._ir_tool_call_to_p(tool_call)],
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
            result["tools"] = ToolConverter.batch_convert_tools(tools, "openai_chat")

        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            result["tool_choice"] = ToolConverter.convert_tool_choice(
                tool_choice, "openai"
            )

        tool_config = ir_request.get("tool_config")
        if tool_config:
            if "disable_parallel" in tool_config:
                result["parallel_tool_calls"] = not tool_config["disable_parallel"]
            if "max_calls" in tool_config:
                warnings.append("OpenAI Chat does not support max_tool_calls, ignored")

        # 3. 处理生成配置 / Handle generation config
        gen_config = ir_request.get("generation", {})
        if gen_config:
            # 直接映射的字段 / Directly mapped fields
            for ir_field, p_field in [
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("frequency_penalty", "frequency_penalty"),
                ("presence_penalty", "presence_penalty"),
                ("logit_bias", "logit_bias"),
                ("seed", "seed"),
                ("logprobs", "logprobs"),
                ("top_logprobs", "top_logprobs"),
                ("n", "n"),
            ]:
                if ir_field in gen_config:
                    result[p_field] = gen_config[ir_field]

            # 特殊处理的字段 / Specially handled fields
            if "max_tokens" in gen_config:
                result["max_completion_tokens"] = gen_config["max_tokens"]

            if "stop_sequences" in gen_config:
                stop = list(gen_config["stop_sequences"])
                if len(stop) == 1:
                    result["stop"] = stop[0]
                elif len(stop) > 1:
                    result["stop"] = stop

            if "top_k" in gen_config:
                warnings.append("OpenAI Chat does not support top_k, ignored")

        # 4. 处理响应格式 / Handle response format
        resp_format = ir_request.get("response_format")
        if resp_format:
            fmt_type = resp_format.get("type")
            if fmt_type == "text":
                result["response_format"] = {"type": "text"}
            elif fmt_type == "json_object":
                result["response_format"] = {"type": "json_object"}
            elif fmt_type == "json_schema":
                result["response_format"] = {
                    "type": "json_schema",
                    "json_schema": resp_format.get("json_schema", {}),
                }

        # 5. 处理推理配置 / Handle reasoning config
        reasoning = ir_request.get("reasoning")
        if reasoning:
            if "effort" in reasoning:
                result["reasoning_effort"] = reasoning["effort"]
            if "budget_tokens" in reasoning:
                warnings.append(
                    "OpenAI Chat does not support reasoning budget_tokens, ignored"
                )

        # 6. 处理流式配置 / Handle stream config
        stream = ir_request.get("stream")
        if stream:
            if "enabled" in stream:
                result["stream"] = stream["enabled"]
            if stream.get("include_usage") and result.get("stream"):
                result["stream_options"] = {"include_usage": True}

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

    def _ir_message_to_p(
        self, message: Dict[str, Any], ir_input: IRInput
    ) -> Tuple[Any, List[str]]:
        """IR Message → Provider Message / IR消息转换为OpenAI消息

        Returns:
            Tuple[转换后的消息, 警告列表] / Tuple[converted message, warning list]
        """
        role = message["role"]
        content = message["content"]
        warnings = []

        if role == "system":
            # System消息：直接转换 / System message: direct conversion
            return {
                "role": "system",
                "content": self._ir_text_to_p_batch(content),
            }, warnings

        elif role == "user":
            # User消息：处理多模态内容和工具结果
            # User message: handle multimodal content and tool results
            content_parts = []
            tool_messages = []

            for part in content:
                converted, part_warnings = self._ir_content_part_to_p(part, ir_input)
                warnings.extend(part_warnings)
                if converted:
                    content_parts.append(converted)

            # 分离content和tool role消息 / Separate content and tool role messages
            user_content = []
            for cp in content_parts:
                if isinstance(cp, dict) and cp.get("type") == "tool_result":
                    # Tool result becomes separate tool role message
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": cp["tool_call_id"],
                            "content": str(cp.get("result") or cp.get("content", "")),
                        }
                    )
                else:
                    user_content.append(cp)

            messages = []
            # 只有当有用户内容或者没有工具消息时才创建用户消息
            # Only create user message when there's user content or no tool messages
            if user_content or not tool_messages:
                if user_content:
                    # 如果只有一个文本部分，使用字符串；否则使用列表
                    # If only one text part, use string; otherwise use list
                    if len(user_content) == 1 and user_content[0].get("type") == "text":
                        content_val = user_content[0]["text"]
                    else:
                        content_val = user_content
                else:
                    # 如果没有有效内容（比如文件被忽略），创建空字符串内容
                    # If no valid content (e.g., files ignored), create empty string content
                    content_val = ""

                messages.append({"role": "user", "content": content_val})

            messages.extend(tool_messages)
            return messages, warnings

        elif role == "assistant":
            # Assistant消息：处理工具调用
            # Assistant message: handle tool calls
            text_parts = []
            tool_calls = []

            for part in content:
                if is_text_part(part):
                    text_parts.append(part["text"])
                elif is_tool_call_part(part):
                    tool_calls.append(self._ir_tool_call_to_p(part))
                elif part.get("type") == "reasoning":
                    # 不支持 / Not supported
                    warnings.append(
                        "Reasoning content not supported in OpenAI Chat Completions, ignored"
                    )

            openai_message = {"role": "assistant"}

            # 添加文本内容 / Add text content
            if text_parts:
                openai_message["content"] = " ".join(text_parts)

            # 添加工具调用 / Add tool calls
            if tool_calls:
                openai_message["tool_calls"] = tool_calls
                if not text_parts:
                    openai_message["content"] = None

            # OpenAI要求assistant消息必须有content或tool_calls
            # OpenAI requires assistant messages to have content or tool_calls
            if not text_parts and not tool_calls:
                openai_message["content"] = ""

            return openai_message, warnings

        return None, warnings

    def _ir_content_part_to_p(
        self, content_part: Dict[str, Any], ir_input: IRInput
    ) -> Tuple[Any, List[str]]:
        """IR ContentPart → Provider Content/Part / IR内容部分转换为OpenAI内容

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
            warnings.append(
                "File content not supported in OpenAI Chat Completions, ignored. "
                "Use OpenAI Responses API converter for file support."
            )
            return None, warnings

        elif is_tool_result_part(content_part):
            # Tool result will be handled at message level
            return content_part, warnings

        elif part_type == "reasoning":
            # 不支持 / Not supported
            warnings.append(
                "Reasoning content not supported in OpenAI Chat Completions, ignored"
            )
            return None, warnings

        return None, warnings

    def _p_response_to_ir(self, provider_response: Dict[str, Any]) -> IRResponse:
        """将OpenAI Chat Completions响应转换为IRResponse
        Convert OpenAI Chat Completions response to IRResponse
        """
        choices = []
        for p_choice in provider_response.get("choices", []):
            message = self._p_message_to_ir(
                p_choice.get("message", p_choice.get("delta", {}))
            )

            finish_reason_val = p_choice.get("finish_reason")
            # 映射停止原因 / Map finish reason
            reason_map = {
                "stop": "stop",
                "length": "length",
                "tool_calls": "tool_calls",
                "content_filter": "content_filter",
                "function_call": "tool_calls",
            }

            choice_info = {
                "index": p_choice.get("index", 0),
                "message": message,
                "finish_reason": {"reason": reason_map.get(finish_reason_val, "stop")},
            }

            if "logprobs" in p_choice:
                choice_info["logprobs"] = p_choice["logprobs"]

            choices.append(choice_info)

        ir_response = {
            "id": provider_response.get("id", ""),
            "object": "response",
            "created": provider_response.get("created", 0),
            "model": provider_response.get("model", ""),
            "choices": choices,
        }

        # 处理使用统计 / Handle usage
        p_usage = provider_response.get("usage")
        if p_usage:
            usage_info = {
                "prompt_tokens": p_usage.get("prompt_tokens", 0),
                "completion_tokens": p_usage.get("completion_tokens", 0),
                "total_tokens": p_usage.get("total_tokens", 0),
            }

            # 处理详细统计 / Handle detailed statistics
            p_prompt_details = p_usage.get("prompt_tokens_details")
            if p_prompt_details:
                usage_info["prompt_tokens_details"] = p_prompt_details
                if "cached_tokens" in p_prompt_details:
                    usage_info["cache_read_tokens"] = p_prompt_details["cached_tokens"]

            p_completion_details = p_usage.get("completion_tokens_details")
            if p_completion_details:
                usage_info["completion_tokens_details"] = p_completion_details
                if "reasoning_tokens" in p_completion_details:
                    usage_info["reasoning_tokens"] = p_completion_details[
                        "reasoning_tokens"
                    ]

            ir_response["usage"] = usage_info

        if "service_tier" in provider_response:
            ir_response["service_tier"] = provider_response["service_tier"]

        if "system_fingerprint" in provider_response:
            ir_response["system_fingerprint"] = provider_response["system_fingerprint"]

        return ir_response

    def _p_message_to_ir(self, provider_message: Any) -> Dict[str, Any]:
        """Provider Message → IR Message / OpenAI消息转换为IR消息"""
        if not isinstance(provider_message, dict):
            return None

        role = provider_message.get("role")
        content = provider_message.get("content")

        if role == "system":
            return Message(
                role="system",
                content=[TextPart(type="text", text=content)],
            )

        elif role == "user":
            return self._p_user_message_to_ir(content)

        elif role == "assistant":
            return self._p_assistant_message_to_ir(provider_message)

        elif role == "tool":
            # Tool消息转换为user消息中的tool_result
            # Tool message converted to tool_result in user message
            return Message(
                role="user",
                content=[
                    ToolResultPart(
                        type="tool_result",
                        tool_call_id=provider_message["tool_call_id"],
                        result=provider_message["content"],
                    )
                ],
            )

        elif role == "function":
            # 已弃用的function角色，转换为tool_result
            # Deprecated function role, converted to tool_result
            return Message(
                role="user",
                content=[
                    ToolResultPart(
                        type="tool_result",
                        tool_call_id=f"legacy_function_{provider_message['name']}",
                        result=provider_message.get("content", ""),
                    )
                ],
            )

        return None

    def _p_content_part_to_ir(self, provider_part: Any) -> List[Dict[str, Any]]:
        """Provider Content/Part → IR ContentPart(s) / OpenAI内容部分转换为IR内容"""
        if isinstance(provider_part, str):
            return [TextPart(type="text", text=provider_part)]

        if not isinstance(provider_part, dict):
            return []

        part_type = provider_part.get("type")

        if part_type == "text":
            return [TextPart(type="text", text=provider_part["text"])]

        elif part_type == "image_url":
            return [self._p_image_to_ir(provider_part)]

        elif part_type == "input_audio":
            # 音频暂时转换为文件类型 / Audio temporarily converted to file type
            return [
                FilePart(
                    type="file",
                    file_data=FileData(
                        data=provider_part["input_audio"]["data"],
                        media_type=f"audio/{provider_part['input_audio']['format']}",
                    ),
                )
            ]

        return []

    # ==================== 内容类型转换方法 Content type conversion methods ====================

    def _ir_text_to_p(self, text_part: TextPart) -> Any:
        """IR TextPart → Provider Text Content / IR文本部分转换为OpenAI文本内容"""
        return {"type": "text", "text": text_part["text"]}

    def _ir_text_to_p_batch(self, content: List[TextPart]) -> str:
        """批量转换文本内容为字符串 / Batch convert text content to string"""
        text_parts = []
        for part in content:
            if is_text_part(part):
                text_parts.append(part["text"])
        return " ".join(text_parts)

    def _p_text_to_ir(self, provider_text: Any) -> TextPart:
        """Provider Text Content → IR TextPart / OpenAI文本内容转换为IR文本部分"""
        if isinstance(provider_text, str):
            return TextPart(type="text", text=provider_text)
        if isinstance(provider_text, dict) and provider_text.get("type") == "text":
            return TextPart(type="text", text=provider_text["text"])
        return None

    def _ir_image_to_p(self, image_part: ImagePart) -> Any:
        """IR ImagePart → Provider Image Content / IR图像部分转换为OpenAI图像内容"""
        url = FieldMapper.get_image_url(image_part)
        image_data = FieldMapper.get_image_data(image_part)
        detail = image_part.get("detail", "auto")

        if url:
            return {"type": "image_url", "image_url": {"url": url, "detail": detail}}
        elif image_data:
            data_url = f"data:{image_data['media_type']};base64,{image_data['data']}"
            return {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": detail},
            }
        else:
            raise ValueError("Image part must have either image_url/url or image_data")

    def _p_image_to_ir(self, provider_image: Any) -> ImagePart:
        """Provider Image Content → IR ImagePart / OpenAI图像内容转换为IR图像部分"""
        image_url_data = provider_image.get("image_url", {})
        url = image_url_data.get("url")
        detail = image_url_data.get("detail", "auto")

        if url and url.startswith("data:"):
            match = re.match(r"data:([^;]+);base64,(.+)", url)
            if match:
                media_type, data = match.groups()
                return ImagePart(
                    type="image",
                    image_data={"data": data, "media_type": media_type},
                    detail=detail,
                )

        return ImagePart(type="image", image_url=url, detail=detail)

    def _ir_file_to_p(self, file_part: FilePart) -> Any:
        """IR FilePart → Provider File Content / IR文件部分转换为OpenAI文件内容

        Note: OpenAI Chat Completion endpoint does not support file input
        """
        raise NotImplementedError(
            "OpenAI Chat Completion endpoint does not support file input. "
            "Use OpenAI Responses API converter for file support."
        )

    def _p_file_to_ir(self, provider_file: Any) -> FilePart:
        """Provider File Content → IR FilePart / OpenAI文件内容转换为IR文件部分

        Note: OpenAI Chat Completion endpoint does not support file input
        """
        raise NotImplementedError(
            "OpenAI Chat Completion endpoint does not support file input. "
            "Use OpenAI Responses API converter for file support."
        )

    def _ir_tool_call_to_p(self, tool_call_part: ToolCallPart) -> Any:
        """IR ToolCallPart → Provider Tool Call / IR工具调用部分转换为OpenAI工具调用"""
        return ToolCallConverter.to_openai_chat(tool_call_part)

    def _p_tool_call_to_ir(self, provider_tool_call: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart / OpenAI工具调用转换为IR工具调用部分"""
        return ToolCallConverter.from_openai_chat(provider_tool_call)

    def _ir_tool_result_to_p(self, tool_result_part: ToolResultPart) -> Any:
        """IR ToolResultPart → Provider Tool Result / IR工具结果部分转换为OpenAI工具结果"""
        return {
            "role": "tool",
            "tool_call_id": tool_result_part["tool_call_id"],
            "content": str(
                tool_result_part.get("result") or tool_result_part.get("content", "")
            ),
        }

    def _p_tool_result_to_ir(self, provider_tool_result: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart / OpenAI工具结果转换为IR工具结果部分"""
        return ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("tool_call_id", ""),
            result=provider_tool_result.get("content", ""),
            is_error=provider_tool_result.get("is_error", False),
        )

    def _ir_tool_to_p(self, tool: ToolDefinition) -> Any:
        """IR ToolDefinition → Provider Tool Definition / IR工具定义转换为OpenAI工具定义"""
        return ToolConverter.convert_tool_definition(tool, "openai_chat")

    def _p_tool_to_ir(self, provider_tool: Any) -> ToolDefinition:
        """Provider Tool Definition → IR ToolDefinition / OpenAI工具定义转换为IR工具定义"""
        # OpenAI Chat 工具定义格式 / OpenAI Chat tool definition format
        if "function" in provider_tool:
            func = provider_tool["function"]
            return {
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
        return provider_tool

    def _ir_tool_choice_to_p(self, tool_choice: ToolChoice) -> Any:
        """IR ToolChoice → Provider Tool Choice Config / IR工具选择转换为OpenAI工具选择配置"""
        return ToolConverter.convert_tool_choice(tool_choice, "openai")

    def _p_tool_choice_to_ir(self, provider_tool_choice: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice / OpenAI工具选择配置转换为IR工具选择"""
        # OpenAI tool choice format / OpenAI工具选择格式
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

    # ==================== 辅助方法 Helper methods ====================

    def _p_user_message_to_ir(self, content: Any) -> Message:
        """处理OpenAI user消息转换为IR / Process OpenAI user message to IR"""
        ir_content = []

        if isinstance(content, str):
            ir_content.append(TextPart(type="text", text=content))
        elif isinstance(content, list):
            for part in content:
                converted = self._p_content_part_to_ir(part)
                if converted:
                    ir_content.extend(converted)

        return Message(role="user", content=ir_content)

    def _p_assistant_message_to_ir(self, provider_message: Any) -> Message:
        """处理OpenAI assistant消息转换为IR / Process OpenAI assistant message to IR"""
        ir_content = []

        # 处理文本内容 / Handle text content
        content = provider_message.get("content")
        if content:
            if isinstance(content, str):
                ir_content.append(TextPart(type="text", text=content))
            elif isinstance(content, list):
                for part in content:
                    converted = self._p_content_part_to_ir(part)
                    if converted:
                        ir_content.extend(converted)

        # 处理工具调用 / Handle tool calls
        tool_calls = provider_message.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                converted = self._p_tool_call_to_ir(tool_call)
                if converted:
                    ir_content.append(converted)

        return Message(role="assistant", content=ir_content)

    # ==================== 兼容性别名 Compatibility aliases ====================

    # 保留旧方法名以保持测试兼容性 / Keep old method names for test compatibility
    def _convert_image_to_openai(self, image_part: Dict[str, Any]) -> Any:
        """将IR图像转换为OpenAI格式（兼容性别名）
        Convert IR image to OpenAI format (compatibility alias)
        """
        return self._ir_image_to_p(image_part)

    def _convert_file_to_openai(self, file_part: Dict[str, Any]) -> Any:
        """将IR文件转换为OpenAI格式（兼容性别名）
        Convert IR file to OpenAI format (compatibility alias)

        Note: OpenAI Chat Completion endpoint does not support file input
        """
        raise NotImplementedError(
            "OpenAI Chat Completion endpoint does not support file input. "
            "Use OpenAI Responses API converter for file support."
        )
