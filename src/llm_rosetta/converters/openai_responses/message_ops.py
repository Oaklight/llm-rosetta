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

from collections.abc import Sequence
from typing import Any, cast

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
from ...types.ir.messages import MessageMetadata
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
        ir_messages: Sequence[Message | ExtensionItem],
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

        metadata = message.get("metadata")
        if role in ("system", "user", "developer"):
            return self._ir_input_message_to_p(role, content, warnings)
        elif role == "assistant":
            return self._ir_assistant_to_p(content, warnings, metadata=metadata)
        elif role == "tool":
            return self._ir_tool_messages_to_p(content, warnings)

        return [], warnings

    def _ir_input_message_to_p(
        self, role: str, content: list, warnings: list[str]
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
        self,
        content: list,
        warnings: list[str],
        *,
        metadata: MessageMetadata | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert IR assistant message to Responses API items.

        Text parts become output_text in a message item.
        Tool calls become separate function_call items.
        Reasoning parts become separate reasoning items.
        Slug-prefixed passthrough items are emitted verbatim.
        """
        # Emit slug-prefixed passthrough items verbatim
        passthrough_items: list[dict[str, Any]] = []
        if metadata:
            custom = metadata.get("custom", {})
            passthrough_items = custom.get("_passthrough_items", [])

        content_parts: list[dict[str, Any]] = []
        tool_items: list[dict[str, Any]] = []
        reasoning_items: list[dict[str, Any]] = []

        for part in content:
            if is_text_part(part):
                # Check for passthrough content part (slug-prefixed)
                pt_meta = (part.get("provider_metadata") or {}).get(
                    "_passthrough_content_part"
                )
                if pt_meta:
                    content_parts.append(pt_meta)
                    continue
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

        # Add slug-prefixed passthrough items verbatim
        result_items.extend(passthrough_items)

        return result_items, warnings

    def _ir_tool_messages_to_p(
        self, content: list, warnings: list[str]
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

    @staticmethod
    def _normalize_shorthand_item(item: dict[str, Any]) -> dict[str, Any]:
        """Normalize shorthand ``{"role": ..., "content": ...}`` to typed item.

        Converts plain ``{"role": "user", "content": "hi"}`` into
        ``{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}``.
        """
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
        return {"type": "message", "role": item["role"], "content": content}

    @staticmethod
    def _append_or_start(
        ir_input: list[Any],
        current_message: dict[str, Any] | None,
        part: Any,
        target_role: str,
    ) -> dict[str, Any]:
        """Append *part* to *current_message* if roles match, else flush and start a new message."""
        if current_message and current_message.get("role") == target_role:
            cast(list, current_message["content"]).append(part)
            return current_message
        if current_message:
            ir_input.append(current_message)
        return {"role": target_role, "content": [part]}

    _TOOL_CALL_TYPES = frozenset(
        {
            "function_call",
            "mcp_call",
            "shell_call",
            "computer_call",
            "code_interpreter_call",
        }
    )

    _TOOL_RESULT_TYPES = frozenset({"function_call_output", "mcp_call_output"})

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
            if isinstance(item, dict) and "type" not in item and "role" in item:
                item = self._normalize_shorthand_item(item)

            item_type = item.get("type") if isinstance(item, dict) else None

            if item_type == "message":
                new_message = self._p_message_to_ir(item)
                if new_message:
                    if current_message:
                        ir_input.append(current_message)
                    current_message = new_message

            elif item_type in self._TOOL_CALL_TYPES:
                tool_call = self.tool_ops.p_tool_call_to_ir(item)
                current_message = self._append_or_start(
                    ir_input, current_message, tool_call, "assistant"
                )

            elif item_type in self._TOOL_RESULT_TYPES:
                tool_result = self.tool_ops.p_tool_result_to_ir(item)
                current_message = self._append_or_start(
                    ir_input, current_message, tool_result, "tool"
                )

            elif item_type == "reasoning":
                reasoning = self.content_ops.p_reasoning_to_ir(item)
                if reasoning:
                    current_message = self._append_or_start(
                        ir_input, current_message, reasoning, "assistant"
                    )

            elif item_type == "system_event":
                if current_message:
                    ir_input.append(current_message)
                    current_message = None
                ir_input.append(self._make_system_event(item))

            elif isinstance(item_type, str) and ":" in item_type:
                current_message = self._handle_p_extension_item(
                    item, current_message, ir_input
                )

        if current_message and current_message.get("content"):
            ir_input.append(current_message)

        return ir_input

    @staticmethod
    def _make_system_event(item: dict[str, Any]) -> dict[str, Any]:
        """Build an IR system_event from a Responses API system_event item."""
        return {
            "type": "system_event",
            "event_type": item.get("event_type", "unknown"),
            "timestamp": item.get("timestamp", ""),
            "message": item.get("message", ""),
        }

    @staticmethod
    def _handle_p_extension_item(
        item: dict[str, Any],
        current_message: dict[str, Any] | None,
        ir_input: list[Any],
    ) -> dict[str, Any]:
        """Handle a slug-prefixed extension item, preserving it opaquely."""
        if current_message and current_message.get("role") == "assistant":
            metadata = current_message.setdefault("metadata", {})
            custom = metadata.setdefault("custom", {})
            custom.setdefault("_passthrough_items", []).append(dict(item))
            return current_message
        if current_message:
            ir_input.append(current_message)
        return {
            "role": "assistant",
            "content": [],
            "metadata": {"custom": {"_passthrough_items": [dict(item)]}},
        }

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

        # Map Responses API "developer" role to IR "system"
        ir_role = "system" if role == "developer" else role

        # Empty messages are also created because subsequent tool calls
        # may need to be appended
        return {"role": ir_role, "content": ir_content}

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

        # Slug-prefixed content part (e.g. "openai:summary_text") — preserve opaquely
        if isinstance(part_type, str) and ":" in part_type:
            return [
                TextPart(
                    type="text",
                    text="",
                    provider_metadata={
                        "_passthrough_content_part": dict(provider_part)
                    },
                )
            ]

        return []
