"""
LLM-Rosetta - OpenAI Chat Message Operations

OpenAI Chat Completions API message conversion operations.
Handles bidirectional conversion of system, user, assistant, and tool messages.

This layer calls content_ops and tool_ops for part-level conversions.
"""

from typing import Any, cast
from collections.abc import Iterable

from ...types.ir import (
    ContentPart,
    ExtensionItem,
    FileData,
    FilePart,
    Message,
    RefusalPart,
    TextPart,
    ToolResultPart,
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
        ir_messages: Iterable[Message | ExtensionItem],
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """IR Messages → OpenAI Chat messages.

        Processes each IR message by role and converts to OpenAI format.
        User messages containing ToolResultParts are split into separate
        tool role messages.

        Args:
            ir_messages: IR message list (may contain ExtensionItems).

        Returns:
            Tuple of (converted messages list, warnings list).
        """
        messages: list[dict[str, Any]] = []
        warnings: list[str] = []

        for item in ir_messages:
            if is_message(item):
                converted, msg_warnings = self._ir_message_to_p(cast(Message, item))
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

        messages = self._reorder_tool_messages(messages, warnings)
        return messages, warnings

    @staticmethod
    def _reorder_tool_messages(
        messages: list[dict[str, Any]], warnings: list[str]
    ) -> list[dict[str, Any]]:
        """Reorder tool messages so each sits right after the assistant that called it.

        The Chat Completions API requires ``role: "tool"`` messages to appear
        immediately after the ``role: "assistant"`` message whose ``tool_calls``
        they answer.  Codex CLI interleaves ``function_call_output`` items
        with other items during multi-turn conversations, which after
        Responses→IR→Chat conversion produces tool messages that are
        separated from their assistant message.  Upstream providers
        (e.g. DeepSeek) reject this with a 400.

        Workaround for upstream bug: https://github.com/openai/codex/pull/7038
        Remove when Codex CLI fixes its item ordering.
        """
        # Separate tool messages from everything else
        tool_msgs: list[dict[str, Any]] = []
        non_tool: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "tool":
                tool_msgs.append(m)
            else:
                non_tool.append(m)

        if not tool_msgs:
            return messages

        # Preserve all tool messages per tool_call_id instead of overwriting
        # duplicates. Some clients may emit multiple tool results with the
        # same tool_call_id, and we should keep them all rather than silently
        # dropping earlier entries.
        tool_by_id: dict[str, list[dict[str, Any]]] = {}
        for tm in tool_msgs:
            tcid = tm.get("tool_call_id")
            if tcid:
                tool_by_id.setdefault(tcid, []).append(tm)

        # Rebuild: after each assistant message with tool_calls, insert
        # matching tool messages in tool_calls order.
        result: list[dict[str, Any]] = []
        matched_msg_ids: set[int] = set()
        for m in non_tool:
            result.append(m)
            if m.get("role") == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    tcid = tc.get("id")
                    if tcid and tcid in tool_by_id:
                        for tool_msg in tool_by_id[tcid]:
                            result.append(tool_msg)
                            matched_msg_ids.add(id(tool_msg))

        # Warn about unmatched tool messages (don't silently drop them)
        for tm in tool_msgs:
            tcid = tm.get("tool_call_id")
            if id(tm) not in matched_msg_ids:
                if tcid:
                    warnings.append(
                        f"Tool message with tool_call_id='{tcid}' has no matching "
                        "assistant tool_calls entry"
                    )
                else:
                    warnings.append(
                        "Tool message with no tool_call_id cannot be reordered"
                    )
                # Append at end so the request isn't silently truncated
                result.append(tm)

        # Only warn if the order actually changed
        if result != messages:
            warnings.append(
                "Reordered tool messages to follow assistant tool_calls "
                "(workaround for Codex CLI item ordering, see openai/codex#7038)"
            )

        return result

    def _ir_message_to_p(self, message: Message) -> tuple[Any, list[str]]:
        """Convert a single IR message to OpenAI format.

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
            return self._ir_tool_messages_to_p(content, warnings)

        return None, warnings

    def _ir_system_to_p(self, content: Iterable) -> dict[str, Any]:
        """Convert IR system message content to OpenAI system message.

        Concatenates all text parts into a single string.
        """
        text_parts = []
        for part in content:
            if is_text_part(part):
                text_parts.append(part["text"])
        return {"role": "system", "content": " ".join(text_parts)}

    def _ir_user_to_p(
        self, content: Iterable, warnings: list[str]
    ) -> tuple[Any, list[str]]:
        """Convert IR user message content to OpenAI user message(s).

        ToolResultParts in user messages are split into separate tool role messages.
        """
        user_content_parts: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []

        for part in content:
            if is_text_part(part):
                user_content_parts.append(self.content_ops.ir_text_to_p(part))
            elif is_image_part(part):
                user_content_parts.append(self.content_ops.ir_image_to_p(part))
            elif is_tool_result_part(part):
                # ToolResultPart in user message → separate tool role message
                tool_messages.append(self.tool_ops.ir_tool_result_to_p(part))
            elif is_file_part(part):
                warnings.append(
                    "File content not supported in OpenAI Chat Completions, ignored. "
                    "Use OpenAI Responses API converter for file support."
                )
            elif is_reasoning_part(part):
                warnings.append(
                    "Reasoning content not supported in OpenAI Chat Completions, ignored"
                )
            else:
                warnings.append(
                    f"Unsupported content part type in user message: {part.get('type')}"
                )

        result_messages: list[dict[str, Any]] = []

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
        self, content: Iterable, warnings: list[str]
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IR assistant message content to OpenAI assistant message.

        Text parts are concatenated. Tool calls are collected into tool_calls list.
        """
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        refusal_text = None

        for part in content:
            if is_text_part(part):
                text_parts.append(part["text"])
            elif is_tool_call_part(part):
                tool_calls.append(self.tool_ops.ir_tool_call_to_p(part))
            elif is_reasoning_part(part):
                warnings.append(
                    "Reasoning content not supported in OpenAI Chat Completions, ignored"
                )
            elif is_refusal_part(part):
                refusal_text = part.get("refusal", "")
            elif is_citation_part(part):
                # Citations are annotations, handled at response level
                pass
            else:
                warnings.append(
                    f"Unsupported content part type in assistant message: {part.get('type')}"
                )

        openai_message: dict[str, Any] = {"role": "assistant"}

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
        self, content: Iterable, warnings: list[str]
    ) -> tuple[Any, list[str]]:
        """Convert IR tool message content to OpenAI tool role message(s).

        Each ToolResultPart becomes a separate tool role message.
        """
        tool_messages: list[dict[str, Any]] = []

        for part in content:
            if is_tool_result_part(part):
                tool_messages.append(self.tool_ops.ir_tool_result_to_p(part))

        if len(tool_messages) == 1:
            return tool_messages[0], warnings
        return tool_messages, warnings

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
                        "tool_calls": [self.tool_ops.ir_tool_call_to_p(tool_call)],
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
        """OpenAI Chat messages → IR Messages.

        Converts each OpenAI message to the appropriate IR message type.

        Args:
            provider_messages: List of OpenAI Chat message dicts.

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

    def _p_system_to_ir(self, msg: dict[str, Any]) -> dict[str, Any]:
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

    def _p_user_to_ir(self, msg: dict[str, Any]) -> dict[str, Any]:
        """OpenAI user message → IR UserMessage."""
        content = msg.get("content", "")
        ir_content: list[ContentPart] = []

        if isinstance(content, str):
            ir_content.append(TextPart(type="text", text=content))
        elif isinstance(content, list):
            for part in content:
                converted = self._p_content_part_to_ir(part)
                ir_content.extend(converted)

        return {"role": "user", "content": ir_content}

    def _p_assistant_to_ir(self, msg: dict[str, Any]) -> dict[str, Any]:
        """OpenAI assistant message → IR AssistantMessage."""
        ir_content: list[ContentPart] = []

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
            ir_content.append(RefusalPart(type="refusal", refusal=refusal))

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

    def _p_tool_to_ir(self, msg: dict[str, Any]) -> dict[str, Any]:
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

    def _p_function_to_ir(self, msg: dict[str, Any]) -> dict[str, Any]:
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

    def _p_content_part_to_ir(self, provider_part: Any) -> list[ContentPart]:
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
                FilePart(
                    type="file",
                    file_data=FileData(
                        data=audio_data.get("data", ""),
                        media_type=f"audio/{audio_data.get('format', 'wav')}",
                    ),
                )
            ]

        return []
