"""
LLM-Rosetta - Anthropic Message Operations

Anthropic Messages API message conversion operations.
Handles bidirectional conversion of system, user, assistant, and tool messages.

This layer calls content_ops and tool_ops for part-level conversions.

Key Anthropic differences:
- System messages are passed via top-level ``system`` parameter, not in messages
- Tool results are content blocks within user messages
- All content is block-based (no string shorthand in structured mode)
"""

from typing import Any, cast
from collections.abc import Iterable

from ...types.ir import (
    ContentPart,
    ExtensionItem,
    Message,
    TextPart,
    is_audio_part,
    is_citation_part,
    is_extension_item,
    is_file_part,
    is_image_part,
    is_message,
    is_reasoning_part,
    is_refusal_part,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
)
from ..base import BaseMessageOps
from .content_ops import AnthropicContentOps
from .tool_ops import AnthropicToolOps


class AnthropicMessageOps(BaseMessageOps):
    """Anthropic Messages API message conversion operations.

    Stateful: holds references to content_ops and tool_ops instances.
    Handles system/user/assistant message bidirectional conversion.
    """

    def __init__(
        self,
        content_ops: AnthropicContentOps,
        tool_ops: AnthropicToolOps,
    ):
        self.content_ops = content_ops
        self.tool_ops = tool_ops

    # ==================== IR → Provider ====================

    def ir_messages_to_p(
        self,
        ir_messages: Iterable[Message | ExtensionItem],
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """IR Messages → Anthropic messages.

        Processes each IR message by role and converts to Anthropic format.
        System messages are skipped here (handled at converter level).

        Args:
            ir_messages: IR message list (may contain ExtensionItems).

        Returns:
            Tuple of (converted messages list, warnings list).
        """
        messages: list[dict[str, Any]] = []
        warnings: list[str] = []

        for item in ir_messages:
            if is_message(item):
                msg = cast(Message, item)
                role = msg.get("role")
                if role == "system":
                    # System messages handled at converter level
                    continue
                converted, msg_warnings = self._ir_message_to_p(msg)
                warnings.extend(msg_warnings)
                if isinstance(converted, list):
                    messages.extend(converted)
                elif converted is not None:
                    messages.append(converted)
            elif is_extension_item(item):
                ext_warnings = self._handle_extension_item(
                    cast(dict[str, Any], item), messages
                )
                warnings.extend(ext_warnings)

        return messages, warnings

    def _ir_message_to_p(self, message: Message) -> tuple[Any, list[str]]:
        """Convert a single IR message to Anthropic format.

        Args:
            message: IR message dict.

        Returns:
            Tuple of (converted message or list of messages, warnings).
        """
        role = message.get("role")
        content = message.get("content", [])
        warnings: list[str] = []

        if role == "system":
            return self._ir_system_to_p(content), warnings
        elif role == "user":
            return self._ir_user_to_p(content, warnings)
        elif role == "assistant":
            return self._ir_assistant_to_p(content, warnings)
        elif role == "tool":
            return self._ir_tool_to_p(content, warnings)

        return None, warnings

    def _ir_system_to_p(self, content: Iterable) -> list[dict[str, Any]]:
        """Convert IR system message content to Anthropic system content blocks.

        Returns list of text blocks for the top-level ``system`` parameter.
        """
        blocks: list[dict[str, Any]] = []
        for part in content:
            if is_text_part(part):
                blocks.append(self.content_ops.ir_text_to_p(part))
        return blocks

    def _ir_user_to_p(
        self, content: Iterable, warnings: list[str]
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IR user message content to Anthropic user message.

        All content parts become content blocks in the user message.
        """
        anthropic_content: list[dict[str, Any]] = []

        for part in content:
            if is_text_part(part):
                anthropic_content.append(self.content_ops.ir_text_to_p(part))
            elif is_image_part(part):
                anthropic_content.append(self.content_ops.ir_image_to_p(part))
            elif is_file_part(part):
                anthropic_content.append(self.content_ops.ir_file_to_p(part))
            elif is_tool_result_part(part):
                anthropic_content.append(self.tool_ops.ir_tool_result_to_p(part))
            elif is_audio_part(part):
                warnings.append(
                    "Audio content not supported in Anthropic Messages API, ignored"
                )
            elif is_reasoning_part(part):
                warnings.append(
                    "Reasoning content in user message not supported, ignored"
                )
            else:
                warnings.append(
                    f"Unsupported content part type in user message: {part.get('type')}"
                )

        return {"role": "user", "content": anthropic_content}, warnings

    def _ir_assistant_to_p(
        self, content: Iterable, warnings: list[str]
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IR assistant message content to Anthropic assistant message.

        All content parts become content blocks in the assistant message.
        Anthropic supports mixed content blocks (text, tool_use, thinking).
        """
        anthropic_content: list[dict[str, Any]] = []

        for part in content:
            if is_text_part(part):
                anthropic_content.append(self.content_ops.ir_text_to_p(part))
            elif is_tool_call_part(part):
                anthropic_content.append(self.tool_ops.ir_tool_call_to_p(part))
            elif is_reasoning_part(part):
                anthropic_content.append(self.content_ops.ir_reasoning_to_p(part))
            elif is_refusal_part(part):
                # Convert refusal to text since Anthropic has no refusal type
                refusal_block = self.content_ops.ir_refusal_to_p(part)
                if refusal_block:
                    anthropic_content.append(refusal_block)
            elif is_citation_part(part):
                # Citations are part of TextBlock in Anthropic, skip
                warnings.append(
                    "Citation content is part of TextBlock in Anthropic, ignored"
                )
            else:
                warnings.append(
                    f"Unsupported content part type in assistant message: {part.get('type')}"
                )

        return {"role": "assistant", "content": anthropic_content}, warnings

    def _ir_tool_to_p(
        self, content: Iterable, warnings: list[str]
    ) -> tuple[Any, list[str]]:
        """Convert IR tool message content to Anthropic user message with tool_result.

        In Anthropic, tool results are sent as content blocks in a user message.
        """
        anthropic_content: list[dict[str, Any]] = []

        for part in content:
            if is_tool_result_part(part):
                anthropic_content.append(self.tool_ops.ir_tool_result_to_p(part))

        return {"role": "user", "content": anthropic_content}, warnings

    def _handle_extension_item(
        self, item: dict[str, Any], messages: list[dict[str, Any]]
    ) -> list[str]:
        """Handle extension items during IR → Provider conversion.

        Returns list of warnings.
        """
        warnings: list[str] = []
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
                        "content": [self.tool_ops.ir_tool_call_to_p(tool_call)],
                    }
                )
        elif extension_type in ("batch_marker", "session_control"):
            warnings.append(f"Extension item ignored: {extension_type}")

        return warnings

    # ==================== Provider → IR ====================

    def p_messages_to_ir(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """Anthropic messages → IR Messages.

        Converts each Anthropic message to the appropriate IR message type.

        Args:
            provider_messages: List of Anthropic message dicts.

        Returns:
            List of IR messages.
        """
        ir_messages: list[Message | ExtensionItem] = []

        for msg in provider_messages:
            converted = self._p_message_to_ir(msg)
            if converted is not None:
                ir_messages.append(converted)

        return ir_messages

    def _p_message_to_ir(self, provider_message: Any) -> Any:
        """Convert a single Anthropic message to IR format.

        Args:
            provider_message: Anthropic message dict.

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
                converted_parts = self._p_content_part_to_ir(part)
                ir_content.extend(converted_parts)

        return {"role": role, "content": ir_content}

    def _p_content_part_to_ir(self, provider_part: Any) -> list[ContentPart]:
        """Convert a single Anthropic content block to IR content part(s).

        Args:
            provider_part: Anthropic content block (string or dict).

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
        elif part_type == "image":
            return [self.content_ops.p_image_to_ir(provider_part)]
        elif part_type == "document":
            return [self.content_ops.p_file_to_ir(provider_part)]
        elif part_type in ("tool_use", "server_tool_use"):
            return [self.tool_ops.p_tool_call_to_ir(provider_part)]
        elif part_type == "tool_result":
            return [self.tool_ops.p_tool_result_to_ir(provider_part)]
        elif part_type == "thinking":
            return [self.content_ops.p_reasoning_to_ir(provider_part)]

        return []
