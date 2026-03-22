"""
LLM-Rosetta - OpenAI Responses Message Operations

OpenAI Responses API message conversion operations.
Handles bidirectional conversion of input items (user/system/developer messages)
and output items (assistant messages, function calls, reasoning, etc.).

Note: Responses API uses a flat list of items instead of nested messages.
Input items include messages, function_call_output, etc.
Output items include messages, function_call, reasoning, etc.

This layer calls content_ops and tool_ops for part-level conversions.
"""

from typing import Any, cast
from collections.abc import Iterable

from ...types.ir import (
    ContentPart,
    ExtensionItem,
    Message,
    TextPart,
    is_extension_item,
    is_file_part,
    is_image_part,
    is_message,
    is_reasoning_part,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
)
from ..base import BaseMessageOps
from .content_ops import OpenAIResponsesContentOps
from .tool_ops import OpenAIResponsesToolOps


class OpenAIResponsesMessageOps(BaseMessageOps):
    """OpenAI Responses API message conversion operations.

    Stateful: holds references to content_ops and tool_ops instances.
    Handles conversion between IR messages and Responses API flat items.
    """

    def __init__(
        self,
        content_ops: OpenAIResponsesContentOps,
        tool_ops: OpenAIResponsesToolOps,
    ):
        self.content_ops = content_ops
        self.tool_ops = tool_ops

    # ==================== IR → Provider ====================

    def ir_messages_to_p(
        self,
        ir_messages: Iterable[Message | ExtensionItem],
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """IR Messages → OpenAI Responses input items.

        Converts IR messages to a flat list of Responses API items.
        Each IR message may produce multiple items (e.g., an assistant
        message with tool calls produces a message item + function_call items).

        Args:
            ir_messages: IR message list (may contain ExtensionItems).

        Returns:
            Tuple of (converted items list, warnings list).
        """
        items: list[dict[str, Any]] = []
        warnings: list[str] = []

        for item in ir_messages:
            if is_message(item):
                converted, msg_warnings = self._ir_message_to_p(cast(Message, item))
                warnings.extend(msg_warnings)
                if isinstance(converted, list):
                    items.extend(converted)
                elif converted is not None:
                    items.append(converted)
            elif is_extension_item(item):
                ext_warnings = self._handle_extension_item(
                    cast(dict[str, Any], item), items
                )
                warnings.extend(ext_warnings)

        return items, warnings

    def _ir_message_to_p(self, message: Message) -> tuple[Any, list[str]]:
        """Convert a single IR message to Responses API items.

        Args:
            message: IR message dict.

        Returns:
            Tuple of (converted items list, warnings).
        """
        role = message.get("role")
        content = message.get("content", [])
        warnings: list[str] = []

        if role in ("system", "user", "developer"):
            return self._ir_input_message_to_p(role, content, warnings)
        elif role == "assistant":
            return self._ir_assistant_to_p(content, warnings)
        elif role == "tool":
            return self._ir_tool_messages_to_p(content, warnings)

        return [], warnings

    def _ir_input_message_to_p(
        self, role: str, content: Iterable, warnings: list[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert IR system/user/developer message to Responses API items.

        Content parts are converted to input_text/input_image/input_file.
        Tool calls and tool results are extracted as separate items.
        """
        content_parts: list[dict[str, Any]] = []
        extra_items: list[dict[str, Any]] = []

        for part in content:
            if is_text_part(part):
                content_parts.append(
                    self.content_ops.ir_text_to_p(part, context="input")
                )
            elif is_image_part(part):
                content_parts.append(self.content_ops.ir_image_to_p(part))
            elif is_file_part(part):
                content_parts.append(self.content_ops.ir_file_to_p(part))
            elif is_tool_call_part(part):
                # Tool calls become separate function_call items
                extra_items.append(self.tool_ops.ir_tool_call_to_p(part))
            elif is_tool_result_part(part):
                # Tool results become separate function_call_output items
                extra_items.append(self.tool_ops.ir_tool_result_to_p(part))
            elif is_reasoning_part(part):
                # Reasoning becomes a separate reasoning item
                extra_items.append(self.content_ops.ir_reasoning_to_p(part))
            else:
                warnings.append(
                    f"Unsupported content part type in {role} message: {part.get('type')}"
                )

        result_items: list[dict[str, Any]] = []

        # Add message item if there are content parts
        if content_parts:
            result_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": content_parts,
                }
            )

        # Add extra items (tool calls, tool results, reasoning)
        result_items.extend(extra_items)

        return result_items, warnings

    def _ir_assistant_to_p(
        self, content: Iterable, warnings: list[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert IR assistant message to Responses API items.

        Text parts become output_text in a message item.
        Tool calls become separate function_call items.
        Reasoning parts become separate reasoning items.
        """
        content_parts: list[dict[str, Any]] = []
        tool_items: list[dict[str, Any]] = []
        reasoning_items: list[dict[str, Any]] = []

        for part in content:
            if is_text_part(part):
                # Check if it's reasoning text (legacy format)
                if part.get("reasoning"):
                    reasoning_items.append(
                        {"type": "reasoning", "content": part["text"]}
                    )
                else:
                    content_parts.append({"type": "output_text", "text": part["text"]})
            elif is_tool_call_part(part):
                tool_items.append(self.tool_ops.ir_tool_call_to_p(part))
            elif is_reasoning_part(part):
                reasoning_items.append(self.content_ops.ir_reasoning_to_p(part))
            else:
                warnings.append(
                    f"Unsupported content part type in assistant message: "
                    f"{part.get('type')}"
                )

        result_items: list[dict[str, Any]] = []

        # Add reasoning items first (they come before the message)
        result_items.extend(reasoning_items)

        # Add assistant message if there are text content parts
        if content_parts:
            result_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": content_parts,
                }
            )

        # Add tool call items
        result_items.extend(tool_items)

        return result_items, warnings

    def _ir_tool_messages_to_p(
        self, content: Iterable, warnings: list[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert IR tool message content to Responses API function_call_output items.

        Each ToolResultPart becomes a separate function_call_output item.
        """
        tool_result_items: list[dict[str, Any]] = []

        for part in content:
            if is_tool_result_part(part):
                tool_result_items.append(self.tool_ops.ir_tool_result_to_p(part))

        return tool_result_items, warnings

    def _handle_extension_item(
        self, item: dict[str, Any], items: list[dict[str, Any]]
    ) -> list[str]:
        """Handle extension items during IR → Provider conversion.

        Returns list of warnings.
        """
        warnings: list[str] = []
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
            warnings.append("Tool chain converted to sequential calls")
            tool_call = item.get("tool_call")
            if tool_call:
                items.append(self.tool_ops.ir_tool_call_to_p(tool_call))
        elif extension_type in ("batch_marker", "session_control"):
            warnings.append(f"Extension item ignored: {extension_type}")

        return warnings

    # ==================== Provider → IR ====================

    def p_messages_to_ir(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """OpenAI Responses items → IR Messages.

        Converts a flat list of Responses API items to IR messages.
        Groups consecutive items of the same role together.

        Args:
            provider_messages: List of Responses API item dicts.

        Returns:
            List of IR messages.
        """
        ir_input: list[Any] = []
        current_message: dict[str, Any] | None = None

        for item in provider_messages:
            # Normalize shorthand items: {"role": "user", "content": "..."}
            # → {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}
            if isinstance(item, dict) and "type" not in item and "role" in item:
                content = item.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "input_text", "text": content}]
                elif isinstance(content, list):
                    normalized = []
                    for part in content:
                        if isinstance(part, str):
                            normalized.append({"type": "input_text", "text": part})
                        elif isinstance(part, dict) and "type" not in part:
                            normalized.append({"type": "input_text", **part})
                        else:
                            normalized.append(part)
                    content = normalized
                item = {"type": "message", "role": item["role"], "content": content}

            item_type = item.get("type") if isinstance(item, dict) else None

            if item_type == "message":
                # Message type: create new message
                new_message = self._p_message_to_ir(item)
                if new_message:
                    if current_message:
                        ir_input.append(current_message)
                    current_message = new_message

            elif item_type in (
                "function_call",
                "mcp_call",
                "shell_call",
                "computer_call",
                "code_interpreter_call",
            ):
                # Tool call: convert to ToolCallPart
                tool_call = self.tool_ops.p_tool_call_to_ir(item)
                if current_message and current_message.get("role") == "assistant":
                    cast(list, current_message["content"]).append(tool_call)
                else:
                    if current_message:
                        ir_input.append(current_message)
                    current_message = {"role": "assistant", "content": [tool_call]}

            elif item_type in ("function_call_output", "mcp_call_output"):
                # Tool result: convert to ToolResultPart
                # Use role="tool" (not "user") so fix_orphaned_tool_calls_ir
                # can correctly detect answered tool calls across formats.
                tool_result = self.tool_ops.p_tool_result_to_ir(item)
                if current_message and current_message.get("role") == "tool":
                    cast(list, current_message["content"]).append(tool_result)
                else:
                    if current_message:
                        ir_input.append(current_message)
                    current_message = {"role": "tool", "content": [tool_result]}

            elif item_type == "reasoning":
                # Reasoning content
                reasoning = self.content_ops.p_reasoning_to_ir(item)
                if reasoning:
                    if current_message and current_message.get("role") == "assistant":
                        cast(list, current_message["content"]).append(reasoning)
                    else:
                        if current_message:
                            ir_input.append(current_message)
                        current_message = {"role": "assistant", "content": [reasoning]}

            elif item_type == "system_event":
                # System event: add as extension item
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

        # Handle the last message
        if current_message:
            if current_message.get("content"):
                ir_input.append(current_message)

        return ir_input

    def _p_message_to_ir(self, provider_message: Any) -> Any:
        """Convert a single Responses API message item to IR format.

        Args:
            provider_message: Responses API message item dict.

        Returns:
            IR message dict, or None.
        """
        if not isinstance(provider_message, dict):
            return None

        role = provider_message.get("role")
        content = provider_message.get("content")

        ir_content: list[ContentPart] = []

        if isinstance(content, str):
            ir_content.append(TextPart(type="text", text=content))
        elif isinstance(content, list):
            for part in content:
                converted = self._p_content_part_to_ir(part)
                if converted:
                    ir_content.extend(converted)

        # Empty messages are also created because subsequent tool calls
        # may need to be appended
        return {"role": role, "content": ir_content}

    def _p_content_part_to_ir(self, provider_part: Any) -> list[ContentPart]:
        """Convert a single Responses API content part to IR content part(s).

        Args:
            provider_part: Responses API content part (string or dict).

        Returns:
            List of IR content parts.
        """
        if isinstance(provider_part, str):
            return [self.content_ops.p_text_to_ir(provider_part)]

        if not isinstance(provider_part, dict):
            return []

        part_type = provider_part.get("type")

        # Support input_text, output_text, and text
        if part_type in ("input_text", "output_text", "text"):
            return [self.content_ops.p_text_to_ir(provider_part)]
        elif part_type == "input_image":
            return [self.content_ops.p_image_to_ir(provider_part)]
        elif part_type == "input_file":
            return [self.content_ops.p_file_to_ir(provider_part)]

        return []
