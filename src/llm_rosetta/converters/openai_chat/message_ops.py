"""
LLM-Rosetta - OpenAI Chat Message Operations

OpenAI Chat Completions API message conversion operations.
Handles bidirectional conversion of system, user, assistant, and tool messages.

This layer calls content_ops and tool_ops for part-level conversions.
"""

from typing import Any, Dict, Iterable, List, Tuple, Union

from ...types.ir import (
    ExtensionItem,
    Message,
    TextPart,
    ToolResultPart,
    is_extension_item,
    is_message,
    is_part_type,
)
from ..base import BaseMessageOps
from .content_ops import OpenAIChatContentOps
from .tool_ops import OpenAIChatToolOps


class OpenAIChatMessageOps(BaseMessageOps):
    """OpenAI Chat Completions message conversion operations.

    Stateful: holds references to content_ops and tool_ops instances.
    Handles system/user/assistant/tool message bidirectional conversion.
    """

    def __init__(
        self,
        content_ops: OpenAIChatContentOps,
        tool_ops: OpenAIChatToolOps,
    ):
        self.content_ops = content_ops
        self.tool_ops = tool_ops

    # ==================== IR → Provider ====================

    def ir_messages_to_p(
        self,
        ir_messages: Iterable[Union[Message, ExtensionItem]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[str]]:
        """IR Messages → OpenAI Chat messages.

        Processes each IR message by role and converts to OpenAI format.
        User messages containing ToolResultParts are split into separate
        tool role messages.

        Args:
            ir_messages: IR message list (may contain ExtensionItems).

        Returns:
            Tuple of (converted messages list, warnings list).
        """
        messages: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for item in ir_messages:
            if is_message(item):
                converted, msg_warnings = self._ir_message_to_p(item)
                warnings.extend(msg_warnings)
                if isinstance(converted, list):
                    messages.extend(converted)
                elif converted is not None:
                    messages.append(converted)
            elif is_extension_item(item):
                ext_warnings = self._handle_extension_item(item, messages)
                warnings.extend(ext_warnings)

        return messages, warnings

    def _ir_message_to_p(self, message: Dict[str, Any]) -> Tuple[Any, List[str]]:
        """Convert a single IR message to OpenAI format.

        Args:
            message: IR message dict.

        Returns:
            Tuple of (converted message or list of messages, warnings).
        """
        role = message.get("role")
        content = message.get("content", [])
        warnings: List[str] = []

        if role == "system":
            return self._ir_system_to_p(content), warnings
        elif role == "user":
            return self._ir_user_to_p(content, warnings)
        elif role == "assistant":
            return self._ir_assistant_to_p(content, warnings)
        elif role == "tool":
            return self._ir_tool_messages_to_p(content, warnings)

        return None, warnings

    def _ir_system_to_p(self, content: Iterable) -> Dict[str, Any]:
        """Convert IR system message content to OpenAI system message.

        Concatenates all text parts into a single string.
        """
        text_parts = []
        for part in content:
            if is_part_type(part, TextPart):
                text_parts.append(part["text"])
        return {"role": "system", "content": " ".join(text_parts)}

    def _ir_user_to_p(
        self, content: Iterable, warnings: List[str]
    ) -> Tuple[Any, List[str]]:
        """Convert IR user message content to OpenAI user message(s).

        ToolResultParts in user messages are split into separate tool role messages.
        """
        user_content_parts: List[Dict[str, Any]] = []
        tool_messages: List[Dict[str, Any]] = []

        for part in content:
            part_type = part.get("type")

            if part_type == "text":
                user_content_parts.append(self.content_ops.ir_text_to_p(part))
            elif part_type == "image":
                user_content_parts.append(self.content_ops.ir_image_to_p(part))
            elif part_type == "tool_result":
                # ToolResultPart in user message → separate tool role message
                tool_messages.append(self.tool_ops.ir_tool_result_to_p(part))
            elif part_type == "file":
                warnings.append(
                    "File content not supported in OpenAI Chat Completions, ignored. "
                    "Use OpenAI Responses API converter for file support."
                )
            elif part_type == "reasoning":
                warnings.append(
                    "Reasoning content not supported in OpenAI Chat Completions, ignored"
                )
            else:
                warnings.append(
                    f"Unsupported content part type in user message: {part_type}"
                )

        result_messages: List[Dict[str, Any]] = []

        # Build user message if there's user content or no tool messages
        if user_content_parts or not tool_messages:
            if user_content_parts:
                # Single text part → use string; otherwise use list
                if (
                    len(user_content_parts) == 1
                    and user_content_parts[0].get("type") == "text"
                ):
                    content_val: Any = user_content_parts[0]["text"]
                else:
                    content_val = user_content_parts
            else:
                content_val = ""

            result_messages.append({"role": "user", "content": content_val})

        result_messages.extend(tool_messages)
        return result_messages, warnings

    def _ir_assistant_to_p(
        self, content: Iterable, warnings: List[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Convert IR assistant message content to OpenAI assistant message.

        Text parts are concatenated. Tool calls are collected into tool_calls list.
        """
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        refusal_text = None

        for part in content:
            part_type = part.get("type")

            if part_type == "text":
                text_parts.append(part["text"])
            elif part_type == "tool_call":
                tool_calls.append(self.tool_ops.ir_tool_call_to_p(part))
            elif part_type == "reasoning":
                warnings.append(
                    "Reasoning content not supported in OpenAI Chat Completions, ignored"
                )
            elif part_type == "refusal":
                refusal_text = part.get("refusal", "")
            elif part_type == "citation":
                # Citations are annotations, handled at response level
                pass
            else:
                warnings.append(
                    f"Unsupported content part type in assistant message: {part_type}"
                )

        openai_message: Dict[str, Any] = {"role": "assistant"}

        # Set text content
        if text_parts:
            openai_message["content"] = " ".join(text_parts)

        # Set tool calls
        if tool_calls:
            openai_message["tool_calls"] = tool_calls
            if not text_parts:
                openai_message["content"] = None

        # OpenAI requires assistant messages to have content or tool_calls
        if not text_parts and not tool_calls:
            openai_message["content"] = ""

        # Set refusal if present
        if refusal_text is not None:
            openai_message["refusal"] = refusal_text

        return openai_message, warnings

    def _ir_tool_messages_to_p(
        self, content: Iterable, warnings: List[str]
    ) -> Tuple[Any, List[str]]:
        """Convert IR tool message content to OpenAI tool role message(s).

        Each ToolResultPart becomes a separate tool role message.
        """
        tool_messages: List[Dict[str, Any]] = []

        for part in content:
            if is_part_type(part, ToolResultPart):
                tool_messages.append(self.tool_ops.ir_tool_result_to_p(part))

        if len(tool_messages) == 1:
            return tool_messages[0], warnings
        return tool_messages, warnings

    def _handle_extension_item(
        self, item: Dict[str, Any], messages: List[Dict[str, Any]]
    ) -> List[str]:
        """Handle extension items during IR → Provider conversion.

        Returns list of warnings.
        """
        warnings: List[str] = []
        extension_type = item.get("type")

        if extension_type == "system_event":
            warnings.append(
                f"System event ignored: {item.get('event_type', 'unknown')}"
            )
        elif extension_type == "tool_chain_node":
            warnings.append("Tool chain converted to sequential calls")
            tool_call = item.get("tool_call")
            if tool_call:
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [self.tool_ops.ir_tool_call_to_p(tool_call)],
                    }
                )
        elif extension_type in ("batch_marker", "session_control"):
            warnings.append(f"Extension item ignored: {extension_type}")

        return warnings

    # ==================== Provider → IR ====================

    def p_messages_to_ir(
        self,
        provider_messages: List[Any],
        **kwargs: Any,
    ) -> List[Union[Message, ExtensionItem]]:
        """OpenAI Chat messages → IR Messages.

        Converts each OpenAI message to the appropriate IR message type.

        Args:
            provider_messages: List of OpenAI Chat message dicts.

        Returns:
            List of IR messages.
        """
        ir_messages: List[Union[Message, ExtensionItem]] = []

        for msg in provider_messages:
            converted = self._p_message_to_ir(msg)
            if converted is not None:
                ir_messages.append(converted)

        return ir_messages

    def _p_message_to_ir(self, provider_message: Any) -> Any:
        """Convert a single OpenAI message to IR format.

        Args:
            provider_message: OpenAI message dict.

        Returns:
            IR message dict, or None.
        """
        if not isinstance(provider_message, dict):
            return None

        role = provider_message.get("role")

        if role == "system":
            return self._p_system_to_ir(provider_message)
        elif role == "user":
            return self._p_user_to_ir(provider_message)
        elif role == "assistant":
            return self._p_assistant_to_ir(provider_message)
        elif role == "tool":
            return self._p_tool_to_ir(provider_message)
        elif role == "function":
            return self._p_function_to_ir(provider_message)

        return None

    def _p_system_to_ir(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI system message → IR SystemMessage."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return {
                "role": "system",
                "content": [TextPart(type="text", text=content)],
            }
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(TextPart(type="text", text=part["text"]))
            return {"role": "system", "content": parts}
        return {"role": "system", "content": [TextPart(type="text", text=str(content))]}

    def _p_user_to_ir(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI user message → IR UserMessage."""
        content = msg.get("content", "")
        ir_content: List[Dict[str, Any]] = []

        if isinstance(content, str):
            ir_content.append(TextPart(type="text", text=content))
        elif isinstance(content, list):
            for part in content:
                converted = self._p_content_part_to_ir(part)
                ir_content.extend(converted)

        return {"role": "user", "content": ir_content}

    def _p_assistant_to_ir(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI assistant message → IR AssistantMessage."""
        ir_content: List[Dict[str, Any]] = []

        # Handle text content
        content = msg.get("content")
        if content:
            if isinstance(content, str):
                ir_content.append(TextPart(type="text", text=content))
            elif isinstance(content, list):
                for part in content:
                    converted = self._p_content_part_to_ir(part)
                    ir_content.extend(converted)

        # Handle refusal
        refusal = msg.get("refusal")
        if refusal:
            ir_content.append({"type": "refusal", "refusal": refusal})

        # Handle tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                ir_content.append(self.tool_ops.p_tool_call_to_ir(tc))

        # Handle annotations (citations)
        annotations = msg.get("annotations")
        if annotations:
            for ann in annotations:
                ir_content.append(self.content_ops.p_citation_to_ir(ann))

        return {"role": "assistant", "content": ir_content}

    def _p_tool_to_ir(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI tool role message → IR ToolMessage."""
        return {
            "role": "tool",
            "content": [
                ToolResultPart(
                    type="tool_result",
                    tool_call_id=msg.get("tool_call_id", ""),
                    result=msg.get("content", ""),
                )
            ],
        }

    def _p_function_to_ir(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI deprecated function role message → IR ToolMessage.

        Generates a legacy tool_call_id from the function name.
        """
        return {
            "role": "tool",
            "content": [
                ToolResultPart(
                    type="tool_result",
                    tool_call_id=f"legacy_function_{msg.get('name', 'unknown')}",
                    result=msg.get("content", ""),
                )
            ],
        }

    def _p_content_part_to_ir(self, provider_part: Any) -> List[Dict[str, Any]]:
        """Convert a single OpenAI content part to IR content part(s).

        Args:
            provider_part: OpenAI content part (string or dict).

        Returns:
            List of IR content parts.
        """
        if isinstance(provider_part, str):
            return [self.content_ops.p_text_to_ir(provider_part)]

        if not isinstance(provider_part, dict):
            return []

        part_type = provider_part.get("type")

        if part_type == "text":
            return [self.content_ops.p_text_to_ir(provider_part)]
        elif part_type == "image_url":
            return [self.content_ops.p_image_to_ir(provider_part)]
        elif part_type == "input_audio":
            # Audio input → FilePart as fallback
            audio_data = provider_part.get("input_audio", {})
            return [
                {
                    "type": "file",
                    "file_data": {
                        "data": audio_data.get("data", ""),
                        "media_type": f"audio/{audio_data.get('format', 'wav')}",
                    },
                }
            ]

        return []
