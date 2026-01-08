"""
LLMIR - Google GenAI Converter

实现IR与Google GenAI SDK格式之间的转换
"""

import warnings
import time
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


class GoogleConverter(BaseConverter):
    """Google GenAI格式转换器
    Google GenAI format converter

    处理IR格式与Google GenAI Content/Part格式之间的转换。
    Handles conversion between IR format and Google GenAI Content/Part format.

    主要特点：
    - 角色映射: assistant ↔ model, user ↔ user
    - Part架构: 所有内容都通过Part表示
    - 工具调用: 通过function_call和function_response Part
    - system_instruction: 单独处理system消息
    Key features:
    - Role mapping: assistant ↔ model, user ↔ user
    - Part architecture: all content represented through Parts
    - Tool calls: through function_call and function_response Parts
    - system_instruction: separate handling of system messages
    """

    def build_config(
        self,
        tools: Optional[Iterable[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Optional[Dict[str, Any]]:
        """构建Google GenAI的config参数
        Build Google GenAI config parameters

        这个方法用于从tools和tool_choice构建Google API调用所需的config字典。
        在多次调用同一个模型时，可以预先构建config以提高效率。
        This method builds the config dict required for Google API calls from tools and tool_choice.
        When calling the same model multiple times, config can be pre-built for efficiency.

        Args:
            tools: 工具定义列表 Tool definition list
            tool_choice: 工具选择配置 Tool choice configuration

        Returns:
            Google GenAI的config字典，如果没有工具配置则返回None Google GenAI config dict, returns None if no tool configuration

        Example:
            >>> converter = GoogleConverter()
            >>> config = converter.build_config(tools=tools_spec)
            >>> # 在多次调用中重用config Reuse config in multiple calls
            >>> response1 = client.models.generate_content(
            ...     model=model_name,
            ...     contents=contents1,
            ...     config=config
            ... )
            >>> response2 = client.models.generate_content(
            ...     model=model_name,
            ...     contents=contents2,
            ...     config=config
            ... )
        """
        config = {}

        # 转换工具 Convert tools
        if tools:
            config["tools"] = ToolConverter.batch_convert_tools(tools, "google")

        # 转换工具选择配置 Convert tool choice configuration
        if tool_choice:
            tool_config = ToolConverter.convert_tool_choice(tool_choice, "google")
            if tool_config:
                config["tool_config"] = tool_config

        return config if config else None

    def to_provider(
        self,
        ir_input: Union[IRInput, IRRequest],
        tools: Optional[Iterable[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IR格式转换为Google GenAI格式
        Convert IR format to Google GenAI format

        Args:
            ir_input: IR输入列表或请求对象 IR input list or request object
            tools: 工具定义列表 Tool definition list
            tool_choice: 工具选择配置 Tool choice configuration

        Returns:
            (Google GenAI格式的字典, 警告列表) (Google GenAI format dict, warning list)
        """
        if isinstance(ir_input, dict) and "messages" in ir_input:
            # Handle IRRequest
            return self._ir_request_to_p(ir_input)

        # 对于无效格式，让其在处理过程中自然抛出KeyError或TypeError For invalid formats, let KeyError or TypeError be thrown naturally during processing
        # 不进行预先验证，让错误在访问字段时自然发生 No pre-validation, let errors occur naturally when accessing fields

        # 转换消息 Convert messages
        contents = []
        system_instruction = None
        warnings_list = []

        for item in ir_input:
            if is_message(item):
                message = item  # type: ignore
                # 这里会在访问message["role"]时抛出KeyError如果没有role字段 Will throw KeyError when accessing message["role"] if no role field
                if message["role"] == "system":
                    # Google使用system_instruction而不是system消息 Google uses system_instruction instead of system messages
                    if system_instruction is None:
                        system_instruction = self._ir_message_to_p(message, ir_input)
                    else:
                        # 如果有多个system消息，合并它们 If there are multiple system messages, merge them
                        existing_parts = system_instruction.get("parts", [])
                        new_parts = self._ir_message_to_p(message, ir_input).get(
                            "parts", []
                        )
                        system_instruction["parts"] = existing_parts + new_parts
                else:
                    content = self._ir_message_to_p(message, ir_input)
                    if content:
                        contents.append(content)
            elif is_extension_item(item):
                # 处理扩展项 Handle extension items
                warnings_list.append(
                    f"Google GenAI不支持扩展项类型 '{item.get('type')}'，将被忽略 Google GenAI does not support extension item type '{item.get('type')}', will be ignored"
                )
            else:
                # 对于既不是消息也不是扩展项的无效项，强制访问role字段来抛出KeyError For invalid items that are neither messages nor extension items, force access to role field to throw KeyError
                _ = item["role"]  # 这会抛出KeyError This will throw KeyError

        # 构建结果 Build result
        result = {"contents": contents}

        # 添加system_instruction Add system_instruction
        if system_instruction:
            result["system_instruction"] = system_instruction

        # 转换工具 Convert tools
        if tools:
            result["tools"] = ToolConverter.batch_convert_tools(tools, "google")

        # 转换工具选择配置 Convert tool choice configuration
        if tool_choice:
            tool_config = ToolConverter.convert_tool_choice(tool_choice, "google")
            if tool_config:
                result["tool_config"] = tool_config

        return result, warnings_list

    def from_provider(self, provider_data: Any) -> Union[IRInput, IRResponse]:
        """将Google GenAI格式转换为IR格式

        Args:
            provider_data: Google GenAI响应对象或字典
                          自动处理Pydantic model对象（调用.model_dump()）
        """
        # 自动unwrap Pydantic模型对象
        if isinstance(provider_data, tuple):
            provider_data = provider_data[0]

        if hasattr(provider_data, "model_dump"):
            provider_data = provider_data.model_dump()

        if not isinstance(provider_data, dict):
            raise ValueError("Google data must be a dictionary")

        # If it's a full API response, convert to IRResponse
        if "candidates" in provider_data and "response_id" in provider_data:
            return self._p_response_to_ir(provider_data)

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
                    Message(
                        role="assistant",
                        content=[TextPart(type="text", text=block_message)],
                    )
                )
            return messages

        # Process all candidates
        for candidate in candidates:
            content = candidate.get("content")
            if content:
                message = self._p_message_to_ir(content)
                if message:
                    messages.append(message)

        return messages

    # ==================== 分层转换方法实现 Layered conversion method implementations ====================

    def _ir_request_to_p(
        self, ir_request: IRRequest
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IRRequest转换为Google GenAI请求参数
        Convert IRRequest to Google GenAI request parameters
        """
        warnings_list = []
        result = {
            "model": ir_request["model"],
        }

        # 1. 处理消息 / Handle messages
        contents = []
        system_instruction = None

        # 处理 system_instruction / Handle system_instruction
        ir_system = ir_request.get("system_instruction")
        if ir_system:
            if isinstance(ir_system, str):
                system_instruction = {"role": "user", "parts": [{"text": ir_system}]}
            elif isinstance(ir_system, list):
                parts = []
                for part in ir_system:
                    if is_text_part(part):
                        parts.append({"text": part["text"]})
                system_instruction = {"role": "user", "parts": parts}

        # 处理 messages / Handle messages
        ir_input = ir_request["messages"]
        for item in ir_input:
            if is_message(item):
                if item["role"] == "system":
                    # Merge into system_instruction
                    msg_p = self._ir_message_to_p(item, ir_input)
                    if system_instruction is None:
                        system_instruction = msg_p
                    else:
                        system_instruction["parts"].extend(msg_p.get("parts", []))
                else:
                    content = self._ir_message_to_p(item, ir_input)
                    if content:
                        contents.append(content)
            elif is_extension_item(item):
                warnings_list.append(
                    f"Extension item ignored in request: {item.get('type')}"
                )

        result["contents"] = contents
        if system_instruction:
            result["system_instruction"] = system_instruction

        # 2. 处理工具 / Handle tools
        tools = ir_request.get("tools")
        tool_choice = ir_request.get("tool_choice")
        tool_config = ir_request.get("tool_config")

        config = {}
        if tools:
            config["tools"] = ToolConverter.batch_convert_tools(tools, "google")

        if tool_choice:
            tc_p = ToolConverter.convert_tool_choice(tool_choice, "google")
            if tc_p:
                config["tool_config"] = tc_p

        if tool_config:
            if "disable_parallel" in tool_config:
                # Google handles this in function_calling_config
                if "tool_config" not in config:
                    config["tool_config"] = {"function_calling_config": {}}
                # Note: Google SDK might not have a direct "disable_parallel" field in tool_config
                # but some versions support it. We follow the mapping or common patterns.
                # For now, we just pass it through if it's a known field in some versions.
                pass

        # 3. 处理生成配置 / Handle generation config
        gen_config = ir_request.get("generation", {})
        if gen_config:
            for ir_field, p_field in [
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("top_k", "top_k"),
                ("max_tokens", "max_output_tokens"),
                ("stop_sequences", "stop_sequences"),
                ("frequency_penalty", "frequency_penalty"),
                ("presence_penalty", "presence_penalty"),
                ("seed", "seed"),
                ("candidate_count", "candidate_count"),
            ]:
                if ir_field in gen_config:
                    config[p_field] = gen_config[ir_field]

        # 4. 响应格式 / Response format
        resp_format = ir_request.get("response_format")
        if resp_format:
            fmt_type = resp_format.get("type")
            if fmt_type == "json_object":
                config["response_mime_type"] = "application/json"
            elif fmt_type == "json_schema":
                config["response_mime_type"] = "application/json"
                config["response_schema"] = resp_format.get("json_schema")

        result["config"] = config

        # 5. 推理配置 / Reasoning config
        # Google reasoning is usually automatic or model-specific (e.g. Gemini 2.0 Thinking)
        # We don't have a direct config field for it in GenerateContentConfig yet,
        # but we can pass it through provider_extensions if needed.

        # 6. 流式配置 / Stream config
        # Handled by the client method (generate_content vs generate_content_stream)

        # 7. 扩展参数 / Provider extensions
        extensions = ir_request.get("provider_extensions")
        if extensions:
            # Merge into config or top level depending on the field
            # For Google, most go into config
            config.update(extensions)

        return result, warnings_list

    def _p_response_to_ir(self, provider_response: Dict[str, Any]) -> IRResponse:
        """将Google GenAI响应转换为IRResponse
        Convert Google GenAI response to IRResponse
        """
        choices = []
        candidates = provider_response.get("candidates", [])

        for p_candidate in candidates:
            content = p_candidate.get("content")
            message = (
                self._p_message_to_ir(content)
                if content
                else Message(role="assistant", content=[])
            )

            finish_reason_val = p_candidate.get("finish_reason")
            # 映射停止原因 / Map finish reason
            # Google values: STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
            reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "error",
            }

            choice_info = {
                "index": p_candidate.get("index", 0),
                "message": message,
                "finish_reason": {"reason": reason_map.get(finish_reason_val, "stop")},
            }
            choices.append(choice_info)

        ir_response = {
            "id": provider_response.get("response_id", ""),
            "object": "response",
            "created": int(time.time()),  # Google doesn't provide timestamp
            "model": provider_response.get("model_version", ""),
            "choices": choices,
        }

        # 处理使用统计 / Handle usage
        p_usage = provider_response.get("usage_metadata")
        if p_usage:
            usage_info = {
                "prompt_tokens": p_usage.get("prompt_token_count", 0),
                "completion_tokens": p_usage.get("candidates_token_count", 0),
                "total_tokens": p_usage.get("total_token_count", 0),
            }

            # 推理 Token / Reasoning tokens
            if "thoughts_token_count" in p_usage:
                usage_info["reasoning_tokens"] = p_usage["thoughts_token_count"]

            ir_response["usage"] = usage_info

        return ir_response

    def _ir_message_to_p(self, message: Dict[str, Any], ir_input: IRInput) -> Any:
        """IR Message → Provider Message / IR消息转换为Provider消息"""
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
            part = self._ir_content_part_to_p(content_part, ir_input)
            if part:
                parts.append(part)

        content = {"role": google_role, "parts": parts}
        return content

    def _p_message_to_ir(self, provider_message: Any) -> Dict[str, Any]:
        """Provider Message → IR Message / Provider消息转换为IR消息"""
        # 角色映射
        role_mapping = {"user": "user", "model": "assistant"}

        google_role = provider_message.get("role", "user")
        ir_role = role_mapping.get(google_role, "user")

        # 转换parts
        parts = provider_message.get("parts", [])
        if not isinstance(parts, list):
            parts = [parts]

        content_parts = []
        for part in parts:
            # Handle reasoning (thoughts)
            if part.get("thought") is True:
                content_parts.append(
                    ReasoningPart(type="reasoning", reasoning=part.get("text", ""))
                )
                continue

            converted_parts = self._p_content_part_to_ir(part)
            if converted_parts:
                content_parts.extend(converted_parts)

        if not content_parts:
            return None

        return {"role": ir_role, "content": content_parts}

    def _ir_content_part_to_p(
        self, content_part: Dict[str, Any], ir_input: IRInput
    ) -> Any:
        """IR ContentPart → Provider Content/Part / IR内容部分转换为Provider内容/Part"""
        if is_text_part(content_part):
            return self._ir_text_to_p(content_part)
        elif content_part["type"] == "image":
            return self._ir_image_to_p(content_part)
        elif content_part["type"] == "file":
            return self._ir_file_to_p(content_part)
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
            return self._ir_tool_call_to_p(content_part)
        elif is_tool_result_part(content_part):
            return self._ir_tool_result_to_p_with_context(content_part, ir_input)
        else:
            warnings.warn(f"不支持的内容类型: {content_part['type']}")
            return None

    def _p_content_part_to_ir(self, provider_part: Any) -> List[Dict[str, Any]]:
        """Provider Content/Part → IR ContentPart(s) / Provider内容/Part转换为IR内容部分"""
        ir_parts = []
        if (
            "text" in provider_part
            and provider_part["text"] is not None
            and provider_part["text"] != ""
        ):
            ir_parts.append(self._p_text_to_ir(provider_part))

        # 支持两种命名格式：function_call（SDK）和 functionCall（REST API）
        func_call = provider_part.get("function_call") or provider_part.get(
            "functionCall"
        )
        if func_call is not None:
            ir_parts.append(self._p_tool_call_to_ir(provider_part))

        if "inline_data" in provider_part and provider_part["inline_data"] is not None:
            inline_data = provider_part["inline_data"]
            mime_type = inline_data.get("mime_type", "")

            if mime_type.startswith("image/"):
                ir_parts.append(self._p_image_to_ir(provider_part))
            elif mime_type.startswith("audio/"):
                # 保持音频类型为 "audio"，这是测试期望的
                ir_parts.append(
                    {
                        "type": "audio",
                        "url": None,  # inline data doesn't have URL
                        "media_type": mime_type,
                    }
                )
            else:
                ir_parts.append(self._p_file_to_ir(provider_part))

        if "file_data" in provider_part and provider_part["file_data"] is not None:
            file_data = provider_part["file_data"]
            mime_type = file_data.get("mime_type", "")

            if mime_type.startswith("image/"):
                ir_parts.append(
                    ImagePart(type="image", image_url=file_data["file_uri"])
                )
            elif mime_type.startswith("audio/"):
                # 保持音频类型为 "audio"，这是测试期望的
                ir_parts.append(
                    {
                        "type": "audio",
                        "url": file_data["file_uri"],
                        "media_type": mime_type,
                    }
                )
            else:
                ir_parts.append(FilePart(type="file", file_url=file_data["file_uri"]))

        # 支持两种命名格式：function_response（SDK）和 functionResponse（REST API）
        func_response = provider_part.get("function_response") or provider_part.get(
            "functionResponse"
        )
        if func_response is not None:
            ir_parts.append(self._p_tool_result_to_ir(provider_part))

        # 处理独立的thoughtSignature（在text或其他part中）
        # 这种情况下，signature会附加到最后一个part上
        thought_sig = provider_part.get("thoughtSignature") or provider_part.get(
            "thought_signature"
        )
        if thought_sig and ir_parts:
            # 将signature添加到最后一个part的metadata中
            last_part = ir_parts[-1]
            if "provider_metadata" not in last_part:
                last_part["provider_metadata"] = {}
            if "google" not in last_part["provider_metadata"]:
                last_part["provider_metadata"]["google"] = {}
            last_part["provider_metadata"]["google"]["thought_signature"] = thought_sig

        if not ir_parts:
            # 过滤掉已知的可忽略字段
            ignorable_keys = {"thoughtSignature", "thought_signature"}
            unknown_keys = set(provider_part.keys()) - ignorable_keys
            if unknown_keys:
                warnings.warn(f"不支持的Part类型: {list(unknown_keys)}")

        return ir_parts

    # ==================== 类型特定转换方法实现 Type-specific conversion method implementations ====================

    def _ir_text_to_p(self, text_part: TextPart) -> Any:
        """IR TextPart → Provider Text Content / IR文本部分转换为Provider文本内容"""
        part = {"text": text_part["text"]}
        # 检查是否有thought_signature需要保留
        if "provider_metadata" in text_part:
            metadata = text_part["provider_metadata"]
            if "google" in metadata and "thought_signature" in metadata["google"]:
                part["thoughtSignature"] = metadata["google"]["thought_signature"]
        return part

    def _p_text_to_ir(self, provider_text: Any) -> TextPart:
        """Provider Text Content → IR TextPart / Provider文本内容转换为IR文本部分"""
        return TextPart(type="text", text=provider_text["text"])

    def _ir_image_to_p(self, image_part: ImagePart) -> Any:
        """IR ImagePart → Provider Image Content / IR图像部分转换为Provider图像内容"""
        # Google使用inline_data或file_data
        # 支持多种字段格式
        if "image_data" in image_part:
            image_data = image_part["image_data"]
            return {
                "inline_data": {
                    "mime_type": image_data["media_type"],
                    "data": image_data["data"],
                }
            }
        elif "data" in image_part and "media_type" in image_part:
            # 直接的data和media_type字段
            return {
                "inline_data": {
                    "mime_type": image_part["media_type"],
                    "data": image_part["data"],
                }
            }
        elif "image_url" in image_part:
            # 对于URL，Google需要先上传文件或使用file_data
            warnings.warn(
                "Google GenAI不直接支持图片URL，需要先上传文件。"
                "请考虑使用file_data或先转换为inline_data。"
            )
            return None
        elif "url" in image_part:
            # 支持url字段
            warnings.warn(
                "Google GenAI不直接支持图片URL，需要先上传文件。"
                "请考虑使用file_data或先转换为inline_data。"
            )
            return None

    def _p_image_to_ir(self, provider_image: Any) -> ImagePart:
        """Provider Image Content → IR ImagePart / Provider图像内容转换为IR图像部分"""
        inline_data = provider_image["inline_data"]
        return {
            "type": "image",
            "data": inline_data["data"],
            "media_type": inline_data["mime_type"],
        }

    def _ir_file_to_p(self, file_part: FilePart) -> Any:
        """IR FilePart → Provider File Content / IR文件部分转换为Provider文件内容"""
        if "file_data" in file_part:
            file_data = file_part["file_data"]
            return {
                "inline_data": {
                    "mime_type": file_data["media_type"],
                    "data": file_data["data"],
                }
            }
        elif "data" in file_part and "media_type" in file_part:
            # 直接的data和media_type字段
            return {
                "inline_data": {
                    "mime_type": file_part["media_type"],
                    "data": file_part["data"],
                }
            }
        elif "file_url" in file_part:
            warnings.warn("Google GenAI不直接支持文件URL，需要先上传文件。")
            return None

    def _p_file_to_ir(self, provider_file: Any) -> FilePart:
        """Provider File Content → IR FilePart / Provider文件内容转换为IR文件部分"""
        inline_data = provider_file["inline_data"]
        return {
            "type": "file",
            "file_data": {
                "data": inline_data["data"],
                "media_type": inline_data["mime_type"],
            },
        }

    def _ir_tool_call_to_p(self, tool_call_part: ToolCallPart) -> Any:
        """IR ToolCallPart → Provider Tool Call / IR工具调用部分转换为Provider工具调用"""
        return ToolCallConverter.to_google(tool_call_part, preserve_metadata=True)

    def _p_tool_call_to_ir(self, provider_tool_call: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart / Provider工具调用转换为IR工具调用部分"""
        return ToolCallConverter.from_google(provider_tool_call, preserve_metadata=True)

    def _ir_tool_result_to_p(self, tool_result_part: ToolResultPart) -> Any:
        """IR ToolResultPart → Provider Tool Result / IR工具结果部分转换为Provider工具结果"""
        # 注意：Google转换器需要额外的ir_input参数来查找对应的tool_call
        # 这里我们使用tool_call_id作为函数名的fallback
        tool_name = tool_result_part.get("tool_call_id")

        # 使用FieldMapper统一处理字段名
        result_content = FieldMapper.get_result_content(tool_result_part)

        response_data = {"output": result_content}
        if tool_result_part.get("is_error"):
            response_data = {"error": result_content}

        return {
            "function_response": {
                "name": tool_name,  # 使用tool_call_id作为函数名
                "response": response_data,
            }
        }

    def _ir_tool_result_to_p_with_context(
        self, tool_result_part: ToolResultPart, ir_input: IRInput
    ) -> Any:
        """IR ToolResultPart → Provider Tool Result with context / 带上下文的IR工具结果部分转换为Provider工具结果"""
        # Google的function_response.name需要是函数名，而不是tool_call_id
        # 我们需要回溯历史记录找到对应的tool_call来获取函数名
        tool_name = None
        for msg in ir_input:
            if not is_message(msg):
                continue
            for part in msg.get("content", []):
                if is_tool_call_part(part) and part.get(
                    "tool_call_id"
                ) == tool_result_part.get("tool_call_id"):
                    tool_name = part.get("tool_name")
                    break
            if tool_name:
                break

        if not tool_name:
            warnings.warn(
                f"Could not find corresponding tool call for tool_call_id '{tool_result_part.get('tool_call_id')}'. "
                "Using tool_call_id as function name, which may cause issues with Google GenAI."
            )
            tool_name = tool_result_part.get("tool_call_id")

        # 使用FieldMapper统一处理字段名
        result_content = FieldMapper.get_result_content(tool_result_part)

        response_data = {"output": result_content}
        if tool_result_part.get("is_error"):
            response_data = {"error": result_content}

        return {
            "function_response": {
                "name": tool_name,  # 使用找到的函数名
                "response": response_data,
            }
        }

    def _p_tool_result_to_ir(self, provider_tool_result: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart / Provider工具结果转换为IR工具结果部分"""
        func_response = provider_tool_result.get(
            "function_response"
        ) or provider_tool_result.get("functionResponse")
        response_data = func_response.get("response", {})

        # 检查是否是错误响应
        is_error = "error" in response_data
        content = response_data.get("error" if is_error else "output", "")

        return ToolResultPart(
            type="tool_result",
            tool_call_id=func_response.get("id", func_response.get("name", "")),
            result=str(content),
            is_error=is_error,
        )

    def _ir_tool_to_p(self, tool: ToolDefinition) -> Any:
        """IR ToolDefinition → Provider Tool Definition / IR工具定义转换为Provider工具定义"""
        return ToolConverter.convert_tool(tool, "google")

    def _p_tool_to_ir(self, provider_tool: Any) -> ToolDefinition:
        """Provider Tool Definition → IR ToolDefinition / Provider工具定义转换为IR工具定义"""
        # 这个方法在当前实现中不需要，因为from_provider不处理工具定义
        # 但为了完整性，提供一个基本实现
        raise NotImplementedError(
            "Google converter does not support converting tools from provider format"
        )

    def _ir_tool_choice_to_p(self, tool_choice: ToolChoice) -> Any:
        """IR ToolChoice → Provider Tool Choice Config / IR工具选择转换为Provider工具选择配置"""
        return ToolConverter.convert_tool_choice(tool_choice, "google")

    def _p_tool_choice_to_ir(self, provider_tool_choice: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice / Provider工具选择配置转换为IR工具选择"""
        # 这个方法在当前实现中不需要，因为from_provider不处理工具选择
        # 但为了完整性，提供一个基本实现
        raise NotImplementedError(
            "Google converter does not support converting tool choice from provider format"
        )
