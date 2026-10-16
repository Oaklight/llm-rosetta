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

from collections.abc import Sequence
from typing import Any, cast

from ...shims.provider_shim import ReasoningCapability
from ...types.ir import (
    ContentPart,
    IRInputItem,
    Message,
    ReasoningPart,
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
from ..base.helpers.citations import report_dropped_citation
from ..base.tools import intrinsic_kind_of, is_intrinsic_part
from ..base.tools.batch import assign_tool_batch_ids, merge_tool_messages
from .content_ops import MessagesContentOps
from .tool_ops import MessagesToolOps


def _server_call_owners(messages: list[Any]) -> dict[str, int]:
    """Map each renderable server call id to the assistant message holding it."""
    owner: dict[str, int] = {}
    for i, message in enumerate(messages):
        if not (isinstance(message, dict) and message.get("role") == "assistant"):
            continue
        for part in message.get("content") or []:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_call"
                and is_intrinsic_part(part)
                and MessagesToolOps.intrinsic_renderable(
                    intrinsic_kind_of(part), {}, leg="history"
                )
            ):
                owner[part.get("tool_call_id", "")] = i
    return owner


def _is_movable_server_result(part: Any, owner: dict[str, int], at: int) -> int | None:
    """Index of the assistant message *part* belongs to, if it can move back."""
    if not (
        isinstance(part, dict)
        and part.get("type") == "tool_result"
        and is_intrinsic_part(part)
    ):
        return None
    idx = owner.get(part.get("tool_call_id", ""))
    return idx if idx is not None and idx < at else None


def _inline_server_results(ir_messages: Any) -> list[Any]:
    """Move server results in tool messages back into the assistant turn.

    Anthropic requires a ``server_tool_use`` and its result block in the same
    assistant message; other dialects keep the result in a ``role: "tool"``
    message, which would otherwise be written as a later user turn (400:
    tool use found without a corresponding result).
    """
    messages = [dict(m) if isinstance(m, dict) else m for m in ir_messages]
    owner = _server_call_owners(messages)
    if not owner:
        return messages
    out: list[Any] = []
    for i, message in enumerate(messages):
        if not (isinstance(message, dict) and message.get("role") == "tool"):
            out.append(message)
            continue
        keep: list[Any] = []
        for part in message.get("content") or []:
            target = _is_movable_server_result(part, owner, i)
            if target is None:
                keep.append(part)
            else:
                messages[target]["content"] = [*messages[target]["content"], part]
        if keep:
            out.append({**message, "content": keep})
    return out


class MessagesMessageOps(BaseMessageOps):
    """Anthropic Messages API message conversion operations.

    Stateful: holds references to content_ops and tool_ops instances.
    Handles system/user/assistant message bidirectional conversion.
    """

    def __init__(
        self,
        content_ops: MessagesContentOps,
        tool_ops: MessagesToolOps,
    ):
        self.content_ops = content_ops
        self.tool_ops = tool_ops

    # ==================== IR → Provider ====================

    def ir_messages_to_p(
        self,
        ir_messages: Sequence[IRInputItem],
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
        reasoning_cap = kwargs.get("reasoning_cap")
        if not isinstance(reasoning_cap, ReasoningCapability):
            reasoning_cap = None

        ir_messages = _inline_server_results(merge_tool_messages(ir_messages))

        for item in ir_messages:
            passthrough_warnings = self._restore_provider_passthrough_item(
                item,
                messages,
                target_provider=kwargs.get("target_provider", ""),
            )
            if passthrough_warnings is not None:
                warnings.extend(passthrough_warnings)
                continue
            if is_message(item):
                msg = cast(Message, item)
                role = msg.get("role")
                if role == "system":
                    # System messages handled at converter level
                    continue
                converted, msg_warnings = self._ir_message_to_p(
                    msg, reasoning_cap=reasoning_cap
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

        return messages, warnings

    def _ir_message_to_p(
        self,
        message: Message,
        *,
        reasoning_cap: ReasoningCapability | None = None,
    ) -> tuple[Any, list[str]]:
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
            return self._ir_assistant_to_p(
                content, warnings, reasoning_cap=reasoning_cap
            )
        elif role == "tool":
            return self._ir_tool_to_p(content, warnings)

        return None, warnings

    def _ir_system_to_p(self, content: list) -> list[dict[str, Any]]:
        """Convert IR system message content to Anthropic system content blocks.

        Returns list of text blocks for the top-level ``system`` parameter.
        """
        blocks: list[dict[str, Any]] = []
        for part in content:
            if is_text_part(part):
                blocks.append(self.content_ops.ir_text_to_p(part))
        return blocks

    def _ir_user_to_p(
        self, content: list, warnings: list[str]
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
                anthropic_content.append(
                    self.tool_ops.ir_tool_result_to_p(part, leg="history")
                )
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
        self,
        content: list,
        warnings: list[str],
        *,
        reasoning_cap: ReasoningCapability | None = None,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Convert IR assistant message content to Anthropic assistant message.

        All content parts become content blocks in the assistant message.
        Anthropic supports mixed content blocks (text, tool_use, thinking).
        """
        anthropic_content: list[dict[str, Any]] = []

        for part in content:
            if is_text_part(part):
                anthropic_content.append(self.content_ops.ir_text_to_p(part))
            elif is_tool_call_part(part):
                # A server call Anthropic has no server tool for is dropped
                # (never handed to the client as a call it must run), so the
                # writer may answer with nothing; say so rather than append an
                # empty block.
                block = self.tool_ops.ir_tool_call_to_p(
                    part, leg="history", warnings=warnings
                )
                if block:
                    anthropic_content.append(block)
            elif is_reasoning_part(part):
                unsigned_policy = (
                    reasoning_cap.unsigned_blocks
                    if reasoning_cap is not None
                    else "as_is"
                )
                if unsigned_policy == "preserve" and not part.get("signature"):
                    self._preserve_unsigned_reasoning(part)
                    warnings.append(
                        "Unsigned reasoning content in assistant message not supported by target provider, preserved in metadata"
                    )
                    continue
                reasoning_block = self.content_ops.ir_reasoning_to_p(part)
                if reasoning_block is not None:
                    anthropic_content.append(reasoning_block)
            elif is_refusal_part(part):
                # Convert refusal to text since Anthropic has no refusal type
                refusal_block = self.content_ops.ir_refusal_to_p(part)
                if refusal_block:
                    anthropic_content.append(refusal_block)
            elif is_citation_part(part):
                self._attach_citation(part, anthropic_content, warnings)
            elif is_tool_result_part(part):
                self._append_assistant_server_result(part, anthropic_content, warnings)
            else:
                warnings.append(
                    f"Unsupported content part type in assistant message: {part.get('type')}"
                )

        if not anthropic_content:
            warnings.append(
                "Assistant message has no Anthropic-compatible content, ignored"
            )
            return None, warnings

        return {"role": "assistant", "content": anthropic_content}, warnings

    def _attach_citation(
        self,
        part: Any,
        anthropic_content: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        """Hang a citation on the text block it annotates.

        Anthropic has no standalone citation block, so a citation belongs to the
        text before it; one that arrives with no text block to annotate is
        reported rather than dropped in silence.
        """
        citation = self.content_ops.ir_citation_to_p(part)
        if citation is None:
            report_dropped_citation(part, warnings=warnings)
        elif anthropic_content and anthropic_content[-1].get("type") == "text":
            anthropic_content[-1].setdefault("citations", []).append(citation)
        else:
            report_dropped_citation(
                part, warnings=warnings, why="no text block to annotate"
            )

    @staticmethod
    def _append_assistant_server_result(
        part: Any, content: list[dict[str, Any]], warnings: list[str]
    ) -> None:
        """Render a server result inline in the assistant turn, or report it.

        A server result the assistant turn produced (web search, code
        execution) rides inline next to its call.  A result Anthropic cannot
        render here — a client result, or a server result for a kind Anthropic
        has no server tool for — is dropped with a warning rather than raising.

        Args:
            part: IR tool result part.
            content: The Anthropic assistant content list, extended in place.
            warnings: Conversion warnings, appended to in place.
        """
        result_block = MessagesToolOps.ir_server_tool_result_to_p(part)
        if result_block is not None:
            content.append(result_block)
        elif is_intrinsic_part(part):
            warnings.append(
                "Server tool result of kind "
                f"{part.get('intrinsic_kind')!r} has no Anthropic server tool; "
                "dropped from the assistant message"
            )
        else:
            warnings.append(
                "Client tool result in assistant message not supported by "
                "Anthropic, ignored"
            )

    @staticmethod
    def _preserve_unsigned_reasoning(part: ReasoningPart) -> None:
        provider_metadata = part.setdefault("provider_metadata", {})
        anthropic_meta = provider_metadata.setdefault("messages", {})
        preserved = anthropic_meta.setdefault("unsigned_reasoning_blocks", [])
        preserved.append(
            {
                "reasoning": part.get("reasoning", ""),
                "signature": part.get("signature"),
                "status": part.get("status"),
            }
        )

    def _ir_tool_to_p(
        self, content: list, warnings: list[str]
    ) -> tuple[Any, list[str]]:
        """Convert IR tool message content to Anthropic user message with tool_result.

        In Anthropic, tool results are sent as content blocks in a user message.
        """
        anthropic_content: list[dict[str, Any]] = []

        for part in content:
            if is_tool_result_part(part):
                # A server result the target cannot render is dropped (falsy).
                block = self.tool_ops.ir_tool_result_to_p(part)
                if block:
                    anthropic_content.append(block)

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
    ) -> list[IRInputItem]:
        """Anthropic messages → IR Messages.

        Converts each Anthropic message to the appropriate IR message type.

        Args:
            provider_messages: List of Anthropic message dicts.

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

        assign_tool_batch_ids(ir_messages)
        return ir_messages

    def _p_message_to_ir(self, provider_message: Any) -> Message | list[Message] | None:
        """Convert a single Anthropic message to IR format.

        Anthropic inlines a *client's* ``tool_result`` blocks, but IR
        normalizes them to ``role: "tool"`` (like OpenAI).  The split keys on
        whether the result is a client result, not on the role:

        - A ``user`` message with **only** ``tool_result`` parts becomes a
          ``"tool"`` message; mixed with other parts it is **split** into a
          ``"tool"`` message followed by a ``"user"`` message.
        - An ``assistant`` message is split the same way for a client result:
          ``[assistant(other parts), tool(results)]``.
        - A **server** result (``tool_type == "intrinsic"``) is the exception.
          The provider produced it inside the assistant turn that ran the tool
          (web search, code execution), so it stays inline and the turn keeps
          every part it owns, in order.

        Args:
            provider_message: Anthropic message dict.

        Returns:
            A single IR message, a list of IR messages (when splitting is
            needed), or ``None``.
        """
        if not isinstance(provider_message, dict):
            return None

        role = provider_message.get("role")
        ir_content = self._p_message_content_to_ir(provider_message.get("content"))

        split = self._split_client_tool_results(role, ir_content)
        if split is not None:
            return split

        return cast(Message, {"role": role, "content": ir_content})

    def _p_message_content_to_ir(self, content: Any) -> list[ContentPart]:
        """Convert an Anthropic message's content to IR content parts."""
        ir_content: list[ContentPart] = []
        if isinstance(content, str):
            ir_content.append(TextPart(type="text", text=content))
        elif isinstance(content, list):
            for part in content:
                ir_content.extend(self._p_content_part_to_ir(part))
        return ir_content

    @staticmethod
    def _split_client_tool_results(
        role: Any, ir_content: list[ContentPart]
    ) -> Message | list[Message] | None:
        """Split a *client's* tool_result parts out of ``ir_content``.

        IR requires a client result in a ``role: "tool"`` message; Anthropic
        inlines it.  A *server* result (``tool_type == "intrinsic"``) is the
        exception and stays inline in the assistant turn, so it is not
        displaced.  Returns ``None`` when nothing needs splitting.
        """
        if role not in ("user", "assistant") or not ir_content:
            return None

        if role == "assistant":
            displaced = [
                p
                for p in ir_content
                if is_tool_result_part(p) and not is_intrinsic_part(p)
            ]
            if not displaced:
                return None
            other_parts = [
                p
                for p in ir_content
                if not is_tool_result_part(p) or is_intrinsic_part(p)
            ]
            if not other_parts:
                return cast(Message, {"role": "tool", "content": ir_content})
            # Client results into a tool message, the rest stays in assistant.
            return [
                cast(Message, {"role": "assistant", "content": other_parts}),
                cast(Message, {"role": "tool", "content": displaced}),
            ]

        tool_result_parts = [p for p in ir_content if is_tool_result_part(p)]
        if not tool_result_parts:
            return None
        other_parts = [p for p in ir_content if not is_tool_result_part(p)]
        if not other_parts:
            return cast(Message, {"role": "tool", "content": ir_content})
        # Mixed: results into a tool message, the rest stays in user.
        return [
            cast(Message, {"role": "tool", "content": tool_result_parts}),
            cast(Message, {"role": "user", "content": other_parts}),
        ]

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
            parts: list[ContentPart] = [self.content_ops.p_text_to_ir(provider_part)]
            # Anthropic embeds citations in the text block; the IR keeps them as
            # their own parts, right after the text they annotate.
            citations = provider_part.get("citations")
            if isinstance(citations, list):
                for citation in citations:
                    if isinstance(citation, dict):
                        parts.append(self.content_ops.p_citation_to_ir(citation))
            return parts
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
        elif part_type == "redacted_thinking":
            return [self.content_ops.p_redacted_thinking_to_ir(provider_part)]
        elif isinstance(part_type, str) and part_type.endswith("_tool_result"):
            return [self.tool_ops.p_server_tool_result_to_ir(provider_part)]

        return []


# Backward-compatible alias (deprecated) — the base type was renamed
# ``anthropic`` → ``messages``.
AnthropicMessageOps = MessagesMessageOps
