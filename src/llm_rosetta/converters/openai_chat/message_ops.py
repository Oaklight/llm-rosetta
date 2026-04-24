"""
LLM-Rosetta - OpenAI Chat Message Operations

OpenAI Chat Completions API message conversion operations.
Handles bidirectional conversion of system, user, assistant, and tool messages.

This layer calls content_ops and tool_ops for part-level conversions.
Also handles multimodal tool result packing/unpacking (Phase 2 dual encoding).
"""

import logging
from collections.abc import Sequence
from typing import Any, cast

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
from ._constants import TOOL_CONTENT_CLOSE_TAG, TOOL_CONTENT_OPEN_TAG_RE
from .content_ops import OpenAIChatContentOps
from .tool_ops import OpenAIChatToolOps

logger = logging.getLogger(__name__)


def _has_multimodal_content(result: Any) -> bool:
    """Check if a tool result contains multimodal content blocks.

    Returns True if ``result`` is a list containing at least one
    non-text content block (image, file, etc.).
    """
    if not isinstance(result, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") not in ("text", None)
        for block in result
    )


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
        ir_messages: Sequence[Message | ExtensionItem],
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
        multimodal_packs: dict[str, list[dict[str, Any]]] = {}

        for item in ir_messages:
            if is_message(item):
                converted, msg_warnings = self._ir_message_to_p(
                    cast(Message, item), multimodal_packs
                )
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
        if multimodal_packs:
            messages = self._inject_packed_tool_content(messages, multimodal_packs)
        return messages, warnings

    @staticmethod
    def _reorder_tool_messages(
        messages: list[dict[str, Any]], warnings: list[str]
    ) -> list[dict[str, Any]]:
        """Reorder tool messages so each sits right after the assistant that called it.

        The Chat Completions API requires ``role: "tool"`` messages to appear
        immediately after the ``role: "assistant"`` message whose ``tool_calls``
        they answer.  Codex CLI interleaves ``function_call_output`` items
        with other items in Responses API format (see
        https://github.com/openai/codex/pull/7038); after conversion the tool
        messages can end up separated from their assistant message, causing
        upstream 400 errors.

        Args:
            messages: Flat list of converted OpenAI Chat messages.
            warnings: Warning list to append reorder notices to.

        Returns:
            Reordered message list.
        """
        tool_msgs: list[dict[str, Any]] = []
        non_tool: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "tool":
                tool_msgs.append(m)
            else:
                non_tool.append(m)

        if not tool_msgs:
            return messages

        # Group tool messages by tool_call_id
        tool_by_id: dict[str, list[dict[str, Any]]] = {}
        for tm in tool_msgs:
            tcid = tm.get("tool_call_id")
            if tcid:
                tool_by_id.setdefault(tcid, []).append(tm)

        # Rebuild: after each assistant message with tool_calls, insert
        # matching tool messages in tool_calls order.
        result: list[dict[str, Any]] = []
        matched_ids: set[int] = set()
        for m in non_tool:
            result.append(m)
            if m.get("role") == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    tcid = tc.get("id")
                    if tcid and tcid in tool_by_id:
                        for tool_msg in tool_by_id[tcid]:
                            result.append(tool_msg)
                            matched_ids.add(id(tool_msg))

        # Append unmatched tool messages at end (don't silently drop them)
        for tm in tool_msgs:
            if id(tm) not in matched_ids:
                tcid = tm.get("tool_call_id")
                warnings.append(
                    f"Tool message with tool_call_id='{tcid}' has no matching "
                    "assistant tool_calls entry"
                )
                result.append(tm)

        if result != messages:
            warnings.append(
                "Reordered tool messages to follow assistant tool_calls "
                "(workaround for Codex CLI item ordering)"
            )

        return result

    def _ir_message_to_p(
        self,
        message: Message,
        multimodal_packs: dict[str, list[dict[str, Any]]],
    ) -> tuple[Any, list[str]]:
        """Convert a single IR message to OpenAI format.

        Args:
            message: IR message dict.
            multimodal_packs: Accumulator for multimodal tool result content.

        Returns:
            Tuple of (converted message or list of messages, warnings).
        """
        role = message.get("role")
        content = message.get("content", [])
        warnings: list[str] = []

        if role == "system":
            return self._ir_system_to_p(content), warnings
        elif role == "user":
            return self._ir_user_to_p(content, warnings, multimodal_packs)
        elif role == "assistant":
            return self._ir_assistant_to_p(content, warnings)
        elif role == "tool":
            return self._ir_tool_messages_to_p(content, warnings, multimodal_packs)

        return None, warnings

    def _ir_system_to_p(self, content: list) -> dict[str, Any]:
        """Convert IR system message content to OpenAI system message.

        Concatenates all text parts into a single string.
        """
        text_parts = []
        for part in content:
            if is_text_part(part):
                text_parts.append(part["text"])
        return {"role": "system", "content": " ".join(text_parts)}

    def _ir_user_to_p(
        self,
        content: list,
        warnings: list[str],
        multimodal_packs: dict[str, list[dict[str, Any]]],
    ) -> tuple[Any, list[str]]:
        """Convert IR user message content to OpenAI user message(s).

        ToolResultParts in user messages are split into separate tool role messages.
        Multimodal tool results are packed for dual encoding.
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
                tool_messages.append(
                    self._convert_tool_result_with_packing(
                        part, multimodal_packs, warnings
                    )
                )
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
        self, content: list, warnings: list[str]
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IR assistant message content to OpenAI assistant message.

        Text parts are concatenated. Tool calls are collected into tool_calls list.
        """
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        refusal_text = None
        reasoning_text: str | None = None

        for part in content:
            if is_text_part(part):
                text_parts.append(part["text"])
            elif is_tool_call_part(part):
                tool_calls.append(self.tool_ops.ir_tool_call_to_p(part))
            elif is_reasoning_part(part):
                reasoning_text = part.get("reasoning", "")
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

        # Set reasoning content (DeepSeek / extended OpenAI Chat providers)
        if reasoning_text is not None:
            openai_message["reasoning_content"] = reasoning_text

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
        self,
        content: list,
        warnings: list[str],
        multimodal_packs: dict[str, list[dict[str, Any]]],
    ) -> tuple[Any, list[str]]:
        """Convert IR tool message content to OpenAI tool role message(s).

        Each ToolResultPart becomes a separate tool role message.
        Multimodal tool results are packed for dual encoding.
        """
        tool_messages: list[dict[str, Any]] = []

        for part in content:
            if is_tool_result_part(part):
                tool_messages.append(
                    self._convert_tool_result_with_packing(
                        part, multimodal_packs, warnings
                    )
                )

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

    # ==================== Multimodal Packing ====================

    def _convert_tool_result_with_packing(
        self,
        part: ToolResultPart,
        multimodal_packs: dict[str, list[dict[str, Any]]],
        warnings: list[str],
    ) -> dict[str, Any]:
        """Convert an IR ToolResultPart, packing multimodal content for dual encoding.

        If the result contains multimodal content (images, files, etc.), the
        visual content blocks are extracted and stored in ``multimodal_packs``
        for later injection as a synthetic user message. The tool message itself
        keeps ``json.dumps(result)`` as content (Phase 1 fallback).

        Text-only results delegate directly to ``tool_ops.ir_tool_result_to_p()``.

        Args:
            part: IR tool result part.
            multimodal_packs: Accumulator mapping call_id → provider content blocks.
            warnings: Warning list.

        Returns:
            OpenAI tool role message dict.
        """
        result = part.get("result", "")
        if not _has_multimodal_content(result):
            return self.tool_ops.ir_tool_result_to_p(part)

        # Multimodal: extract visual content for synthetic user message
        call_id = part["tool_call_id"]
        packed_parts: list[dict[str, Any]] = []

        for block in result:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                packed_parts.append(self.content_ops.ir_text_to_p(block))
            elif block_type == "image":
                try:
                    packed_parts.append(self.content_ops.ir_image_to_p(block))
                except ValueError as e:
                    warnings.append(f"Skipped image in tool result packing: {e}")
            elif block_type == "file":
                warnings.append(
                    "File content not supported in OpenAI Chat tool result packing, "
                    "skipped"
                )
            else:
                warnings.append(
                    f"Unsupported block type in tool result packing: {block_type}"
                )

        if packed_parts:
            multimodal_packs[call_id] = packed_parts

        # Tool message keeps json.dumps fallback (existing Phase 1 behavior)
        return self.tool_ops.ir_tool_result_to_p(part)

    @staticmethod
    def _inject_packed_tool_content(
        messages: list[dict[str, Any]],
        multimodal_packs: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Inject synthetic user message with packed multimodal tool content.

        Walks the reordered message list and, after each group of consecutive
        tool messages following an assistant, inserts a synthetic user message
        containing ``<tool-content call-id="...">`` tagged content blocks for
        any tool results that have multimodal content.

        Args:
            messages: Reordered OpenAI Chat messages.
            multimodal_packs: Mapping of call_id → provider content blocks.

        Returns:
            Messages list with synthetic user messages injected.
        """
        if not multimodal_packs:
            return messages

        result: list[dict[str, Any]] = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            result.append(msg)
            i += 1

            # After tool messages group, check for packed content
            if msg.get("role") != "tool":
                continue

            # Collect consecutive tool messages (already appended first one)
            tool_call_ids = [msg.get("tool_call_id")]
            while i < len(messages) and messages[i].get("role") == "tool":
                result.append(messages[i])
                tool_call_ids.append(messages[i].get("tool_call_id"))
                i += 1

            # Build synthetic user message for packed call_ids
            synthetic_parts: list[dict[str, Any]] = []
            for tcid in tool_call_ids:
                if tcid and tcid in multimodal_packs:
                    synthetic_parts.append(
                        {"type": "text", "text": f'<tool-content call-id="{tcid}">'}
                    )
                    synthetic_parts.extend(multimodal_packs[tcid])
                    synthetic_parts.append(
                        {"type": "text", "text": TOOL_CONTENT_CLOSE_TAG}
                    )

            if synthetic_parts:
                result.append({"role": "user", "content": synthetic_parts})

        return result

    # ==================== Provider → IR ====================

    def p_messages_to_ir(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """OpenAI Chat messages → IR Messages.

        Pre-processes synthetic user messages (from dual encoding packing)
        before converting each OpenAI message to the appropriate IR type.

        Args:
            provider_messages: List of OpenAI Chat message dicts.

        Returns:
            List of IR messages.
        """
        unpacked_content, clean_messages = self._unpack_tool_content(provider_messages)

        ir_messages: list[Message | ExtensionItem] = []
        for msg in clean_messages:
            converted = self._p_message_to_ir(msg, unpacked_content)
            if converted is not None:
                ir_messages.append(converted)

        return ir_messages

    def _p_message_to_ir(
        self,
        provider_message: Any,
        unpacked_content: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Any:
        """Convert a single OpenAI message to IR format.

        Args:
            provider_message: OpenAI message dict.
            unpacked_content: Mapping of call_id → provider content blocks
                extracted from synthetic user messages.

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
            return self._p_tool_to_ir(provider_message, unpacked_content)
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

        # Handle reasoning_content (DeepSeek / extended OpenAI Chat providers)
        # Prepend before text content to match convention (reasoning first)
        reasoning_content = msg.get("reasoning_content")
        if reasoning_content:
            ir_content.append(self.content_ops.p_reasoning_to_ir(reasoning_content))

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

    def _p_tool_to_ir(
        self,
        msg: dict[str, Any],
        unpacked_content: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """OpenAI tool role message → IR ToolMessage.

        If unpacked multimodal content exists for this tool_call_id (from a
        synthetic user message), the visual content blocks are converted to IR
        and used as the result. Otherwise, the tool message content string is
        used as-is.

        Args:
            msg: OpenAI tool role message dict.
            unpacked_content: Mapping of call_id → provider content blocks.
        """
        call_id = msg.get("tool_call_id", "")

        if call_id and unpacked_content and call_id in unpacked_content:
            # Restore multimodal content from synthetic user message
            ir_parts: list[ContentPart] = []
            for block in unpacked_content[call_id]:
                converted = self._p_content_part_to_ir(block)
                ir_parts.extend(converted)
            return {
                "role": "tool",
                "content": [
                    ToolResultPart(
                        type="tool_result",
                        tool_call_id=call_id,
                        result=ir_parts,
                    )
                ],
            }

        return {
            "role": "tool",
            "content": [
                ToolResultPart(
                    type="tool_result",
                    tool_call_id=call_id,
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

    # ==================== Multimodal Unpacking ====================

    @staticmethod
    def _is_synthetic_tool_content_msg(msg: dict[str, Any]) -> bool:
        """Check if a user message is a synthetic tool content message.

        A synthetic message has ``role: "user"`` and its content list starts
        with a text part matching the ``<tool-content call-id="...">`` tag.

        Args:
            msg: OpenAI message dict.

        Returns:
            True if the message is a synthetic tool content message.
        """
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            return False
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            return bool(TOOL_CONTENT_OPEN_TAG_RE.match(first.get("text", "")))
        return False

    @staticmethod
    def _unpack_tool_content(
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        """Extract multimodal content from synthetic user messages.

        Scans for synthetic user messages containing ``<tool-content>`` tags,
        parses the tags to extract ``call_id → content blocks`` mapping, and
        removes the synthetic messages from the message list.

        Args:
            messages: OpenAI Chat messages (may contain synthetic user messages).

        Returns:
            Tuple of (unpacked_content mapping, clean message list).
        """
        unpacked: dict[str, list[dict[str, Any]]] = {}
        clean: list[dict[str, Any]] = []

        for msg in messages:
            if not OpenAIChatMessageOps._is_synthetic_tool_content_msg(msg):
                clean.append(msg)
                continue

            # Parse <tool-content call-id="..."> sections
            content = msg.get("content", [])
            current_call_id: str | None = None
            current_blocks: list[dict[str, Any]] = []

            for part in content:
                if not isinstance(part, dict):
                    continue

                if part.get("type") == "text":
                    text = part.get("text", "")

                    # Check for open tag
                    open_match = TOOL_CONTENT_OPEN_TAG_RE.match(text)
                    if open_match:
                        # Save previous section if any
                        if current_call_id and current_blocks:
                            unpacked[current_call_id] = current_blocks
                        current_call_id = open_match.group(1)
                        current_blocks = []
                        continue

                    # Check for close tag
                    if text == TOOL_CONTENT_CLOSE_TAG:
                        if current_call_id and current_blocks:
                            unpacked[current_call_id] = current_blocks
                        current_call_id = None
                        current_blocks = []
                        continue

                # Content block within a section
                if current_call_id is not None:
                    current_blocks.append(part)

            # Handle unclosed section
            if current_call_id and current_blocks:
                unpacked[current_call_id] = current_blocks

        return unpacked, clean

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
