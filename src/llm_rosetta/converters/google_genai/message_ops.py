"""
LLM-Rosetta - Google GenAI Message Operations

Google GenAI API message conversion operations.
Handles bidirectional conversion of user, model (assistant), and system messages.

This layer calls content_ops and tool_ops for part-level conversions.

Google-specific:
- Messages are Content objects with role + parts list
- System messages are NOT in contents; they go to system_instruction
- Role mapping: user ↔ user, assistant ↔ model
- All content is represented as Part objects in a flat list
"""

import warnings
from typing import Any, cast
from collections.abc import Iterable

from ...types.ir import (
    ContentPart,
    ExtensionItem,
    Message,
    ReasoningPart,
    is_audio_part,
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
from .content_ops import GoogleGenAIContentOps
from .tool_ops import GoogleGenAIToolOps


# Role mapping constants
_IR_TO_GOOGLE_ROLE = {
    "user": "user",
    "assistant": "model",
    "tool": "user",  # Tool results go in user-role Content with functionResponse parts
    "system": "user",  # Fallback; system should be handled separately
}

_GOOGLE_TO_IR_ROLE = {
    "user": "user",
    "model": "assistant",
}


class GoogleGenAIMessageOps(BaseMessageOps):
    """Google GenAI message conversion operations.

    Holds references to content_ops and tool_ops instances.
    Handles user/model/system message bidirectional conversion.

    Note: System messages are extracted to system_instruction at the
    converter level, not handled here as regular messages.
    """

    def __init__(
        self,
        content_ops: GoogleGenAIContentOps,
        tool_ops: GoogleGenAIToolOps,
    ):
        self.content_ops = content_ops
        self.tool_ops = tool_ops

    # ==================== IR → Provider ====================

    def ir_messages_to_p(
        self,
        ir_messages: Iterable[Message | ExtensionItem],
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """IR Messages → Google GenAI Content list + system_instruction.

        Processes each IR message by role and converts to Google format.
        System messages are collected separately for system_instruction.

        The returned tuple contains:
        - A dict with 'contents' and optionally 'system_instruction'
        - A list of warning strings

        However, to match the BaseMessageOps interface (which returns
        List[messages], List[warnings]), we return the contents list
        and warnings. System instruction extraction is handled at the
        converter level.

        Args:
            ir_messages: IR message list (may contain ExtensionItems).
            **kwargs: May contain 'ir_input' for tool result context lookup.

        Returns:
            Tuple of (converted Content list, warnings list).
        """
        contents: list[dict[str, Any]] = []
        warnings_list: list[str] = []

        # Convert ir_messages to list for context lookup
        ir_input_list = (
            list(ir_messages) if not isinstance(ir_messages, list) else ir_messages
        )

        for item in ir_input_list:
            if is_message(item):
                msg = cast(Message, item)
                role = msg.get("role")
                if role == "system":
                    # System messages are handled at converter level
                    # Skip them here
                    continue
                content = self._ir_message_to_p(msg, ir_input_list)
                if content:
                    contents.append(content)
            elif is_extension_item(item):
                warnings_list.append(
                    f"Google GenAI不支持扩展项类型 '{item.get('type')}'，将被忽略 "
                    f"Google GenAI does not support extension item type "
                    f"'{item.get('type')}', will be ignored"
                )
            else:
                # Unknown item type, skip
                warnings.warn("Unknown item type in ir_messages, will be ignored")

        return contents, warnings_list

    def _ir_message_to_p(
        self, message: Message, ir_input: Any = None
    ) -> dict[str, Any]:
        """Convert a single IR message to Google Content format.

        Args:
            message: IR message dict.
            ir_input: Full IR input for tool result context lookup.

        Returns:
            Google Content dict with role and parts.
        """
        google_role = _IR_TO_GOOGLE_ROLE.get(message["role"], "user")
        parts: list[dict[str, Any]] = []

        for content_part in message.get("content", []):
            part = self._ir_content_part_to_p(content_part, ir_input)
            if part is not None:
                parts.append(part)

        return {"role": google_role, "parts": parts}

    def _ir_content_part_to_p(
        self, content_part: ContentPart, ir_input: Any = None
    ) -> Any:
        """Convert a single IR content part to Google Part format.

        Dispatches to the appropriate content_ops or tool_ops method.

        Args:
            content_part: IR content part dict.
            ir_input: Full IR input for tool result context lookup.

        Returns:
            Google Part dict, or None if unsupported.
        """
        if is_text_part(content_part):
            return self.content_ops.ir_text_to_p(content_part)
        elif is_image_part(content_part):
            return self.content_ops.ir_image_to_p(content_part)
        elif is_file_part(content_part):
            return self.content_ops.ir_file_to_p(content_part)
        elif is_audio_part(content_part):
            return self.content_ops.ir_audio_to_p(content_part)
        elif is_reasoning_part(content_part):
            return self.content_ops.ir_reasoning_to_p(content_part)
        elif is_tool_call_part(content_part):
            return self.tool_ops.ir_tool_call_to_p(content_part)
        elif is_tool_result_part(content_part):
            if ir_input is not None:
                return self.tool_ops.ir_tool_result_to_p_with_context(
                    content_part, ir_input
                )
            return self.tool_ops.ir_tool_result_to_p(content_part)
        else:
            warnings.warn(f"不支持的内容类型: {content_part.get('type')}")
            return None

    # ==================== Provider → IR ====================

    def p_messages_to_ir(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """Google GenAI Content list → IR Messages.

        Converts each Google Content to the appropriate IR message type.
        After conversion, reconciles tool_call_ids between tool_call parts
        and tool_result parts by matching on function name, since Google's
        ``functionCall`` has no ID field and the converter generates UUIDs
        that won't match the IDs assigned by the client SDK (e.g. Gemini
        CLI) in ``functionResponse``.

        Args:
            provider_messages: List of Google Content dicts.

        Returns:
            List of IR messages.
        """
        ir_messages: list[Message | ExtensionItem] = []

        for msg in provider_messages:
            converted = self._p_message_to_ir(msg)
            if converted is None:
                continue
            if isinstance(converted, list):
                ir_messages.extend(converted)
            else:
                ir_messages.append(converted)

        # Reconcile tool_call_ids: map tool_result IDs to the matching
        # tool_call IDs by function name.
        self._reconcile_tool_call_ids(ir_messages)

        return ir_messages

    @staticmethod
    def _reconcile_tool_call_ids(
        ir_messages: list[Message | ExtensionItem],
    ) -> None:
        """Match tool_result tool_call_ids to tool_call tool_call_ids by name.

        Google ``functionCall`` parts do not carry an ID.  During P→IR
        conversion, unique IDs are generated for tool_call parts.  When the
        client sends ``functionResponse`` parts back, it uses its *own* IDs
        (or the function name).  This method patches tool_result
        ``tool_call_id`` values in-place so that they reference the IDs
        generated for the corresponding tool_call parts.

        Matching strategy: for each tool_result, find the *first*
        unmatched tool_call with the same ``tool_name`` and adopt its
        ``tool_call_id``.  This handles parallel calls to the same
        function correctly (FIFO pairing).

        Args:
            ir_messages: IR messages list (modified in-place).
        """
        # Build ordered list of (tool_call_id, tool_name) from tool_call parts.
        call_queue: dict[str, list[str]] = {}  # tool_name → [tool_call_ids]
        for msg in ir_messages:
            if not is_message(msg) or msg.get("role") != "assistant":
                continue
            for part in msg.get("content", []):
                if is_tool_call_part(part):
                    name = part.get("tool_name", "")
                    call_queue.setdefault(name, []).append(part.get("tool_call_id", ""))

        if not call_queue:
            return

        # Track which tool_call_ids have already been consumed.
        consumed: dict[str, int] = {}  # tool_name → next index in call_queue

        for msg in ir_messages:
            if not is_message(msg) or msg.get("role") != "tool":
                continue
            for part in msg.get("content", []):
                if not is_tool_result_part(part):
                    continue
                # Determine the function name for this result.
                # The tool_call_id in the result may actually be the
                # function name (Google convention) or a client-generated
                # ID.  We need to find the matching tool_call by name.
                result_id = part.get("tool_call_id", "")
                # Try to identify the function name: check if result_id
                # matches any known tool_name directly.
                tool_name = result_id  # default guess
                for name in call_queue:
                    # Match if the result_id starts with the tool name
                    # (Gemini CLI format: "<name>_<timestamp>_<index>")
                    # or equals the tool name exactly.
                    if result_id == name or result_id.startswith(name + "_"):
                        tool_name = name
                        break

                ids = call_queue.get(tool_name)
                if not ids:
                    continue
                idx = consumed.get(tool_name, 0)
                if idx < len(ids):
                    part["tool_call_id"] = ids[idx]
                    consumed[tool_name] = idx + 1

    def _p_message_to_ir(self, provider_message: Any) -> Any | list[Any]:
        """Convert a single Google Content to IR format.

        A Google Content dict with ``role: "user"`` may contain a mix of
        regular content parts and ``functionResponse`` parts.  In the IR,
        tool results must live in ``role: "tool"`` messages whereas regular
        user content must remain in ``role: "user"`` messages.  When both
        kinds coexist in one Content, this method returns **a list** of IR
        messages (one ``"user"`` + one ``"tool"``).

        Args:
            provider_message: Google Content dict with role and parts.

        Returns:
            A single IR message dict, a list of IR messages, or None.
        """
        if not isinstance(provider_message, dict):
            return None

        google_role = provider_message.get("role", "user")
        ir_role = _GOOGLE_TO_IR_ROLE.get(google_role, "user")

        # Convert parts
        parts = provider_message.get("parts", [])
        if not isinstance(parts, list):
            parts = [parts]

        content_parts: list[ContentPart] = []
        tool_result_parts: list[ContentPart] = []

        for part in parts:
            # Handle reasoning (thoughts)
            if part.get("thought") is True:
                content_parts.append(
                    ReasoningPart(type="reasoning", reasoning=part.get("text", ""))
                )
                continue

            # Handle function_call and function_response via tool_ops
            func_call = part.get("function_call") or part.get("functionCall")
            if func_call is not None:
                content_parts.append(self.tool_ops.p_tool_call_to_ir(part))
                continue

            func_response = part.get("function_response") or part.get(
                "functionResponse"
            )
            if func_response is not None:
                # Tool results must go into role:"tool" messages so that
                # fix_orphaned_tool_calls_ir can detect and fix ID
                # mismatches correctly.
                tool_result_parts.append(self.tool_ops.p_tool_result_to_ir(part))
                continue

            # Handle content parts (text, image, file, audio)
            converted_parts = self.content_ops.p_part_to_ir(part)
            if converted_parts:
                content_parts.extend(cast(list, converted_parts))
            else:
                # Check for unknown part types
                ignorable_keys = {"thoughtSignature", "thought_signature"}
                unknown_keys = set(part.keys()) - ignorable_keys
                if unknown_keys:
                    warnings.warn(f"不支持的Part类型: {list(unknown_keys)}")

        if not content_parts and not tool_result_parts:
            return None

        # If there are only tool results and no other content, return a
        # single role:"tool" message.
        if tool_result_parts and not content_parts:
            return {"role": "tool", "content": tool_result_parts}

        # If there are only regular content parts, return a single message.
        if content_parts and not tool_result_parts:
            return {"role": ir_role, "content": content_parts}

        # Mixed: tool results first (to keep them adjacent to the preceding
        # assistant tool_calls), then regular content.
        return [
            {"role": "tool", "content": tool_result_parts},
            {"role": ir_role, "content": content_parts},
        ]

    # ==================== System Instruction Helpers ====================

    @staticmethod
    def extract_system_instruction(
        ir_messages: Iterable[Message | ExtensionItem],
    ) -> tuple[Any, list[Message | ExtensionItem]]:
        """Extract system messages from IR message list.

        Returns the system_instruction Content dict and the remaining
        non-system messages.

        Args:
            ir_messages: IR message list.

        Returns:
            Tuple of (system_instruction Content dict or None, remaining messages).
        """
        system_instruction: dict[str, Any] | None = None
        remaining: list[Message | ExtensionItem] = []

        for item in ir_messages:
            if is_message(item) and item.get("role") == "system":
                parts: list[dict[str, str]] = []
                for part in cast(Iterable[ContentPart], item.get("content", [])):
                    if is_text_part(part):
                        parts.append({"text": part["text"]})
                if system_instruction is None:
                    system_instruction = {"role": "user", "parts": parts}
                else:
                    existing_parts = system_instruction["parts"]
                    if isinstance(existing_parts, list):
                        existing_parts.extend(parts)
            else:
                remaining.append(item)

        return system_instruction, remaining
