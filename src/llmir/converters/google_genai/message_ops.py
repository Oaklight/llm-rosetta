"""
LLMIR - Google GenAI Message Operations

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
from typing import Any, Dict, Iterable, List, Tuple, Union

from ...types.ir import (
    ExtensionItem,
    Message,
    ReasoningPart,
    is_extension_item,
    is_message,
    is_text_part,
)
from ..base import BaseMessageOps
from .content_ops import GoogleGenAIContentOps
from .tool_ops import GoogleGenAIToolOps


# Role mapping constants
_IR_TO_GOOGLE_ROLE = {
    "user": "user",
    "assistant": "model",
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
        ir_messages: Iterable[Union[Message, ExtensionItem]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[str]]:
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
        contents: List[Dict[str, Any]] = []
        warnings_list: List[str] = []

        # Convert ir_messages to list for context lookup
        ir_input_list = (
            list(ir_messages) if not isinstance(ir_messages, list) else ir_messages
        )

        for item in ir_input_list:
            if is_message(item):
                role = item.get("role")
                if role == "system":
                    # System messages are handled at converter level
                    # Skip them here
                    continue
                content = self._ir_message_to_p(item, ir_input_list)
                if content:
                    contents.append(content)
            elif is_extension_item(item):
                warnings_list.append(
                    f"Google GenAI不支持扩展项类型 '{item.get('type')}'，将被忽略 "
                    f"Google GenAI does not support extension item type "
                    f"'{item.get('type')}', will be ignored"
                )
            else:
                # Force access to role field to throw KeyError for invalid items
                _ = item["role"]

        return contents, warnings_list

    def _ir_message_to_p(
        self, message: Dict[str, Any], ir_input: Any = None
    ) -> Dict[str, Any]:
        """Convert a single IR message to Google Content format.

        Args:
            message: IR message dict.
            ir_input: Full IR input for tool result context lookup.

        Returns:
            Google Content dict with role and parts.
        """
        google_role = _IR_TO_GOOGLE_ROLE.get(message["role"], "user")
        parts: List[Dict[str, Any]] = []

        for content_part in message.get("content", []):
            part = self._ir_content_part_to_p(content_part, ir_input)
            if part is not None:
                parts.append(part)

        return {"role": google_role, "parts": parts}

    def _ir_content_part_to_p(
        self, content_part: Dict[str, Any], ir_input: Any = None
    ) -> Any:
        """Convert a single IR content part to Google Part format.

        Dispatches to the appropriate content_ops or tool_ops method.

        Args:
            content_part: IR content part dict.
            ir_input: Full IR input for tool result context lookup.

        Returns:
            Google Part dict, or None if unsupported.
        """
        part_type = content_part.get("type")

        if part_type == "text":
            return self.content_ops.ir_text_to_p(content_part)
        elif part_type == "image":
            return self.content_ops.ir_image_to_p(content_part)
        elif part_type == "file":
            return self.content_ops.ir_file_to_p(content_part)
        elif part_type == "audio":
            return self.content_ops.ir_audio_to_p(content_part)
        elif part_type == "reasoning":
            return self.content_ops.ir_reasoning_to_p(content_part)
        elif part_type == "tool_call":
            return self.tool_ops.ir_tool_call_to_p(content_part)
        elif part_type == "tool_result":
            if ir_input is not None:
                return self.tool_ops.ir_tool_result_to_p_with_context(
                    content_part, ir_input
                )
            return self.tool_ops.ir_tool_result_to_p(content_part)
        else:
            warnings.warn(f"不支持的内容类型: {part_type}")
            return None

    # ==================== Provider → IR ====================

    def p_messages_to_ir(
        self,
        provider_messages: List[Any],
        **kwargs: Any,
    ) -> List[Union[Message, ExtensionItem]]:
        """Google GenAI Content list → IR Messages.

        Converts each Google Content to the appropriate IR message type.

        Args:
            provider_messages: List of Google Content dicts.

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
        """Convert a single Google Content to IR format.

        Args:
            provider_message: Google Content dict with role and parts.

        Returns:
            IR message dict, or None.
        """
        if not isinstance(provider_message, dict):
            return None

        google_role = provider_message.get("role", "user")
        ir_role = _GOOGLE_TO_IR_ROLE.get(google_role, "user")

        # Convert parts
        parts = provider_message.get("parts", [])
        if not isinstance(parts, list):
            parts = [parts]

        content_parts: List[Dict[str, Any]] = []
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
                content_parts.append(self.tool_ops.p_tool_result_to_ir(part))
                continue

            # Handle content parts (text, image, file, audio)
            converted_parts = self.content_ops.p_part_to_ir(part)
            if converted_parts:
                content_parts.extend(converted_parts)
            else:
                # Check for unknown part types
                ignorable_keys = {"thoughtSignature", "thought_signature"}
                unknown_keys = set(part.keys()) - ignorable_keys
                if unknown_keys:
                    warnings.warn(f"不支持的Part类型: {list(unknown_keys)}")

        if not content_parts:
            return None

        return {"role": ir_role, "content": content_parts}

    # ==================== System Instruction Helpers ====================

    @staticmethod
    def extract_system_instruction(
        ir_messages: Iterable[Union[Message, ExtensionItem]],
    ) -> Tuple[Any, List[Union[Message, ExtensionItem]]]:
        """Extract system messages from IR message list.

        Returns the system_instruction Content dict and the remaining
        non-system messages.

        Args:
            ir_messages: IR message list.

        Returns:
            Tuple of (system_instruction Content dict or None, remaining messages).
        """
        system_instruction = None
        remaining: List[Union[Message, ExtensionItem]] = []

        for item in ir_messages:
            if is_message(item) and item.get("role") == "system":
                parts = []
                for part in item.get("content", []):
                    if is_text_part(part):
                        parts.append({"text": part["text"]})
                if system_instruction is None:
                    system_instruction = {"role": "user", "parts": parts}
                else:
                    system_instruction["parts"].extend(parts)
            else:
                remaining.append(item)

        return system_instruction, remaining
