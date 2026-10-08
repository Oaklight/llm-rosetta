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

import uuid
import warnings
from collections.abc import Sequence
from typing import Any, cast

from ...types.ir import (
    ContentPart,
    IRInputItem,
    Message,
    is_audio_part,
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
from ..base.tools.batch import assign_tool_batch_ids, merge_tool_messages
from ..base.tools.intrinsic import (
    get_intrinsic_kind,
    make_intrinsic_tool_call,
    make_intrinsic_tool_result,
)
from .content_ops import GoogleGenerateContentOps
from .tool_ops import GoogleGenerateToolOps


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


def _match_tool_name(result_id: str, known_names: dict[str, list[str]]) -> str:
    """Find the tool_name that best matches a result ID.

    Checks for exact match first, then Gemini CLI prefix format
    (``<name>_<timestamp>_<index>``).  Falls back to *result_id* itself.
    """
    for name in known_names:
        if result_id == name or result_id.startswith(name + "_"):
            return name
    return result_id


def _ir_intrinsic_tool_call_to_google(content_part: Any) -> dict[str, Any] | None:
    if get_intrinsic_kind(content_part) != "code_execution":
        return None
    return {
        "executableCode": {
            "code": content_part.get("tool_input", {}).get("code", ""),
            "language": content_part.get("tool_input", {}).get("language", "PYTHON"),
        }
    }


def _ir_intrinsic_tool_result_to_google(content_part: Any) -> dict[str, Any] | None:
    if get_intrinsic_kind(content_part) != "code_execution":
        return None
    outcome = "OUTCOME_FAILED" if content_part.get("is_error") else "OUTCOME_OK"
    result = content_part.get("result", "")
    return {
        "codeExecutionResult": {
            "output": result if isinstance(result, str) else str(result),
            "outcome": outcome,
        }
    }


class GoogleGenerateMessageOps(BaseMessageOps):
    """Google GenAI message conversion operations.

    Holds references to content_ops and tool_ops instances.
    Handles user/model/system message bidirectional conversion.

    Note: System messages are extracted to system_instruction at the
    converter level, not handled here as regular messages.
    """

    def __init__(
        self,
        content_ops: GoogleGenerateContentOps,
        tool_ops: GoogleGenerateToolOps,
    ):
        self.content_ops = content_ops
        self.tool_ops = tool_ops

    # ==================== IR → Provider ====================

    def ir_messages_to_p(
        self,
        ir_messages: Sequence[IRInputItem],
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
        # tool_call_id → tool_name, built incrementally as we iterate
        tool_call_index: dict[str, str] = {}

        # Merge tool messages from the same batch before conversion
        ir_input_list = merge_tool_messages(ir_messages)

        for item in ir_input_list:
            passthrough_warnings = self._restore_provider_passthrough_item(
                item,
                contents,
                target_provider=kwargs.get("target_provider", ""),
            )
            if passthrough_warnings is not None:
                warnings_list.extend(passthrough_warnings)
                continue
            if is_message(item):
                msg = cast(Message, item)
                role = msg.get("role")
                if role == "system":
                    # System messages are handled at converter level
                    # Skip them here
                    continue
                # Accumulate tool_call_id → tool_name before converting,
                # so tool_results in this or later messages can look up
                # the name in O(1) instead of scanning the full list.
                for part in msg.get("content", []):
                    if is_tool_call_part(part):
                        tc_id = part.get("tool_call_id")
                        tc_name = part.get("tool_name")
                        if tc_id and tc_name:
                            tool_call_index[tc_id] = tc_name
                converted = self._ir_message_to_p(msg, tool_call_index)
                if converted:
                    contents.append(converted)
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
        self,
        message: Message,
        tool_call_index: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Convert a single IR message to Google Content format.

        Args:
            message: IR message dict.
            tool_call_index: Accumulated ``{tool_call_id: tool_name}``
                mapping for O(1) tool result name resolution.

        Returns:
            Google Content dict with role and parts.
        """
        google_role = _IR_TO_GOOGLE_ROLE.get(message["role"], "user")
        parts: list[dict[str, Any]] = []

        for content_part in message.get("content", []):
            part = self._ir_content_part_to_p(content_part, tool_call_index)
            if part is not None:
                parts.append(part)

        return {"role": google_role, "parts": parts}

    def _ir_content_part_to_p(
        self,
        content_part: ContentPart,
        tool_call_index: dict[str, str] | None = None,
    ) -> Any:
        """Convert a single IR content part to Google Part format.

        Args:
            content_part: IR content part dict.
            tool_call_index: Accumulated ``{tool_call_id: tool_name}``
                for O(1) tool result name resolution.

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
        elif is_refusal_part(content_part):
            return self.content_ops.ir_refusal_to_p(content_part)
        elif is_tool_call_part(content_part):
            if content_part.get("tool_type") == "intrinsic":
                return _ir_intrinsic_tool_call_to_google(content_part)
            return self.tool_ops.ir_tool_call_to_p(content_part)
        elif is_tool_result_part(content_part):
            if content_part.get("tool_type") == "intrinsic":
                return _ir_intrinsic_tool_result_to_google(content_part)
            tool_call_id = content_part.get("tool_call_id")
            tool_name = (
                tool_call_index.get(tool_call_id)
                if tool_call_index and tool_call_id
                else None
            )
            if tool_name is not None:
                return self.tool_ops.ir_tool_result_to_p_named(content_part, tool_name)
            return self.tool_ops.ir_tool_result_to_p(content_part)
        else:
            warnings.warn(f"不支持的内容类型: {content_part.get('type')}")
            return None

    # ==================== Provider → IR ====================

    def p_messages_to_ir(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[IRInputItem]:
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
        ir_messages: list[IRInputItem] = []

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

        assign_tool_batch_ids(ir_messages)
        return ir_messages

    @staticmethod
    def _reconcile_tool_call_ids(
        ir_messages: Sequence[IRInputItem],
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
                tool_name = _match_tool_name(result_id, call_queue)

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
                content_parts.append(self.content_ops.p_reasoning_to_ir(part))
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

            # Handle executableCode / codeExecutionResult as intrinsic tools
            exec_code = part.get("executableCode") or part.get("executable_code")
            if exec_code is not None:
                call_id = f"google_exec_{uuid.uuid4().hex[:12]}"
                content_parts.append(
                    cast(
                        ContentPart,
                        make_intrinsic_tool_call(
                            tool_call_id=call_id,
                            tool_name="code_execution",
                            tool_input={
                                "code": exec_code.get("code", ""),
                                "language": exec_code.get("language", "PYTHON"),
                            },
                            intrinsic_kind="code_execution",
                        ),
                    )
                )
                continue

            code_result = part.get("codeExecutionResult") or part.get(
                "code_execution_result"
            )
            if code_result is not None:
                # Google's codeExecutionResult has no ID field to pair with
                # executableCode; _reconcile_tool_call_ids handles matching.
                tool_result_parts.append(
                    cast(
                        ContentPart,
                        make_intrinsic_tool_result(
                            tool_call_id="",
                            result=code_result.get("output", ""),
                            intrinsic_kind="code_execution",
                            is_error=code_result.get("outcome", "OUTCOME_OK")
                            != "OUTCOME_OK",
                        ),
                    )
                )
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
        ir_messages: Sequence[IRInputItem],
    ) -> tuple[Any, list[IRInputItem]]:
        """Extract system messages from IR message list.

        Returns the system_instruction Content dict and the remaining
        non-system messages.

        Args:
            ir_messages: IR message list.

        Returns:
            Tuple of (system_instruction Content dict or None, remaining messages).
        """
        system_instruction: dict[str, Any] | None = None
        remaining: list[IRInputItem] = []

        for item in ir_messages:
            if is_message(item) and item.get("role") == "system":
                parts: list[dict[str, str]] = []
                for part in cast(list[ContentPart], item.get("content", [])):
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
