"""
LLM-Rosetta - Messages API stream writer (IR events -> provider SSE events).

The per-event handlers :class:`MessagesConverter` dispatches to through
``_IR_TO_P_DISPATCH``, plus the block-tracking helpers they share; kept here
so the converter module stays focused on the request/response interfaces.

``_handle_ir_citation_delta_to_p`` and ``_handle_ir_refusal_delta_to_p`` are
deliberate overrides of the same-named defaults on
:class:`~llm_rosetta.converters.base.converter.BaseConverter`, not copies of
them: this mixin precedes ``BaseConverter`` in the MRO, so these bodies win,
and an edit to the base versions does not reach Messages.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_rosetta.ir.model.stream import (
    CitationDeltaEvent,
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    RefusalDeltaEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    UsageEvent,
)
from ...base.errors import require_representable_finish_reason
from llm_rosetta.conversion_warnings import DOWNGRADED, report_loss
from llm_rosetta.ir.context import (
    StreamContext,
)
from ...base.tools import (
    intrinsic_kind_of,
    prefixed_tool_call_id,
    sanitize_tool_call_id,
)
from ...base.helpers.citations import (
    report_dropped_citation,
)
from ...base.helpers.opaque import (
    get_opaque_origin,
    is_own_origin,
)
from ._constants import (
    ANTHROPIC_REASON_TO_PROVIDER,
    AnthropicEventType,
)
from .tool_ops import MessagesToolOps


@dataclass
class _AnthropicBlockState:
    """Track outbound block identity separately from the currently active block."""

    next_index: int = 0
    tool_indices: dict[str, int] = field(default_factory=dict)
    tool_index_to_block: dict[int, int] = field(default_factory=dict)
    open_indices: set[int] = field(default_factory=set)
    block_types: dict[int, str] = field(default_factory=dict)
    pending_tool_index: int | None = None

    def open(
        self,
        context: StreamContext,
        index: int | None = None,
        *,
        block_type: str | None = None,
    ) -> int:
        if index is None:
            index = max(self.next_index, context.current_block_index + 1)
        self.next_index = max(self.next_index, index + 1)
        self.open_indices.add(index)
        if block_type is not None:
            self.block_types[index] = block_type
        context.current_block_index = index
        return index

    def close(self, index: int) -> None:
        """Close a block and discard tool identities registered to its index."""
        self.open_indices.discard(index)
        stale_ids = [
            tool_id
            for tool_id, tool_index in self.tool_indices.items()
            if tool_index == index
        ]
        for tool_id in stale_ids:
            del self.tool_indices[tool_id]


def _stream_blocks(context: StreamContext) -> _AnthropicBlockState:
    """Keep provider-private state on the stream, including generic contexts."""
    state = context.metadata.get("_anthropic_blocks")
    if not isinstance(state, _AnthropicBlockState):
        state = _AnthropicBlockState()
        context.metadata["_anthropic_blocks"] = state
    return state


def _close_content_blocks(context: StreamContext) -> list[dict[str, Any]]:
    """Close every open block once without resetting the allocation watermark."""
    state = _stream_blocks(context)
    if context.current_block_index >= 0:
        state.open(context, context.current_block_index)
    closing_indices = sorted(state.open_indices)
    events = [
        {"type": AnthropicEventType.CONTENT_BLOCK_STOP, "index": index}
        for index in closing_indices
    ]
    for index in closing_indices:
        state.close(index)
    state.pending_tool_index = None
    context.current_block_index = -1
    context.current_block_type = None
    return events


def _anthropic_stop_reason(
    reason: str, response_id: str, metadata: Mapping[str, Any] | None
) -> str:
    """Map an IR reason to a ``stop_reason``, refusing the failures.

    Anthropic's ``stop_reason`` vocabulary is a model of a *finished* turn —
    ``end_turn``, ``max_tokens``, ``tool_use``, ``stop_sequence``, ``refusal``.
    It has no value for an upstream that failed or a generation that was
    cancelled, and defaulting those to ``end_turn`` tells the client the model
    replied normally when nothing was generated.

    Args:
        reason: The IR finish reason.
        response_id: The response the reason belongs to.
        metadata: Upstream metadata recorded with the reason.

    Returns:
        The Anthropic ``stop_reason``.

    Raises:
        UnrepresentableFinishReason: For ``error`` and ``cancelled``.
    """
    require_representable_finish_reason(
        "The Messages API",
        reason,
        response_id=response_id,
        provider_metadata=metadata,
    )
    return ANTHROPIC_REASON_TO_PROVIDER.get(reason, "end_turn")


class MessagesStreamWriter:
    """Mixin holding the IR stream event -> Messages SSE event handlers."""

    if TYPE_CHECKING:
        # Provided by ``MessagesConverter`` / ``BaseConverter`` at runtime.
        _CONVERTER_TAG: str
        content_ops: Any
        tool_ops: Any

        @staticmethod
        def _build_ir_usage_to_p(ir_usage: Mapping[str, Any]) -> dict[str, Any]: ...

        @staticmethod
        def _build_message_delta_usage(
            ir_usage: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]: ...

        def _get_response_id_prefix(self, context: Any = None) -> str: ...

        @staticmethod
        def add_response_id_prefix(stem: str, prefix: str) -> str: ...

    def _handle_ir_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamStartEvent → message_start."""
        ir_usage: Mapping[str, Any] = {}
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.mark_started()
            if context.pending_usage is not None:
                ir_usage = context.pending_usage
            elif context.preflight_usage is not None:
                ir_usage = context.preflight_usage

        p_usage = self._build_ir_usage_to_p(ir_usage)

        return {
            "type": AnthropicEventType.MESSAGE_START,
            "message": {
                "id": self.add_response_id_prefix(
                    event["response_id"], self._get_response_id_prefix(context)
                ),
                "type": "message",
                "role": "assistant",
                "model": event["model"],
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "stop_details": None,
                "container": None,
                "usage": p_usage,
            },
        }

    def _handle_ir_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle StreamEndEvent → message_stop (with optional pending finish flush)."""
        results: list[dict[str, Any]] = []
        if context is not None:
            results.extend(_close_content_blocks(context))
            # Flush any buffered finish that never got a UsageEvent
            finish = context.pop_pending_finish()
            if finish is not None:
                usage = context.pop_pending_usage()
                results.append(
                    {
                        "type": AnthropicEventType.MESSAGE_DELTA,
                        "delta": finish,
                        "usage": self._build_message_delta_usage(usage),
                    }
                )
            context.mark_ended()
        results.append({"type": AnthropicEventType.MESSAGE_STOP})
        return results if len(results) > 1 else results[0]

    def _handle_ir_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockStartEvent → content_block_start."""
        block_index = event["block_index"]
        block_type = event["block_type"]

        if context is not None:
            # Anchor context to the explicit block_index from the IR event
            # instead of auto-incrementing, so subsequent deltas that read
            # context.current_block_index stay in sync.  (#246)
            state = _stream_blocks(context)
            state.open(context, block_index, block_type=block_type)
            state.pending_tool_index = (
                block_index if block_type in ("tool_use", "server_tool_use") else None
            )
            context.current_block_type = block_type

        if block_type == "text":
            return {
                "type": AnthropicEventType.CONTENT_BLOCK_START,
                "index": block_index,
                "content_block": {"type": "text", "text": ""},
            }
        elif block_type == "thinking":
            return {
                "type": AnthropicEventType.CONTENT_BLOCK_START,
                "index": block_index,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        else:
            return {}

    def _handle_ir_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockEndEvent → content_block_stop."""
        if context is not None:
            index = event["block_index"]
            state = _stream_blocks(context)
            state.close(index)
            state.next_index = max(state.next_index, index + 1)
            if state.pending_tool_index == index:
                state.pending_tool_index = None
            if context.current_block_index == index:
                context.current_block_index = -1
                context.current_block_type = None
        return {
            "type": AnthropicEventType.CONTENT_BLOCK_STOP,
            "index": event["block_index"],
        }

    def _handle_ir_text_delta_to_p(
        self, event: TextDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle TextDeltaEvent → content_block_delta (with synthetic start if needed)."""
        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
            "delta": {
                "type": "text_delta",
                "text": event["text"],
            },
        }
        # Prefer explicit block_index from the IR event (#246);
        # fall back to context for providers that don't emit block indexes.
        explicit_idx: int | None = event.get("block_index")
        if explicit_idx is not None:
            result["index"] = explicit_idx
            if context is not None:
                _stream_blocks(context).open(context, explicit_idx, block_type="text")
                context.current_block_type = "text"
        elif context is not None:
            needs_new_block = context.current_block_index < 0 or (
                context.current_block_type is not None
                and context.current_block_type != "text"
            )
            if needs_new_block:
                preamble = _close_content_blocks(context)
                _stream_blocks(context).open(context, block_type="text")
                context.current_block_type = "text"
                result["index"] = context.current_block_index
                preamble.append(
                    {
                        "type": AnthropicEventType.CONTENT_BLOCK_START,
                        "index": context.current_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                preamble.append(result)
                return preamble
            result["index"] = context.current_block_index
        return result

    def _handle_ir_citation_delta_to_p(
        self, event: CitationDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle CitationDeltaEvent → content_block_delta (citations_delta).

        Anthropic's own citations hang off a text block, so the delta is
        addressed to the block the citation annotates: the one the event names,
        or the block currently open.
        """
        citation = self.content_ops.ir_citation_to_p(event["citation"])
        if citation is None:
            report_dropped_citation(
                event["citation"], warnings=getattr(context, "warnings", None)
            )
            return {}
        index = event.get("block_index")
        if index is None and context is not None:
            index = context.current_block_index
        if index is None:
            return {}
        return {
            "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
            "index": index,
            "delta": {"type": "citations_delta", "citation": citation},
        }

    def _handle_ir_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle ReasoningDeltaEvent → content_block_delta (thinking and/or signature)."""
        deltas = self._ir_reasoning_deltas(event)
        if not deltas:
            return {}
        rd_results: list[dict[str, Any]] = [
            {"type": AnthropicEventType.CONTENT_BLOCK_DELTA, "delta": delta}
            for delta in deltas
        ]
        # Prefer explicit block_index from the IR event (#246)
        explicit_idx: int | None = event.get("block_index")
        if explicit_idx is not None:
            for result in rd_results:
                result["index"] = explicit_idx
            if context is not None:
                _stream_blocks(context).open(
                    context, explicit_idx, block_type="thinking"
                )
                context.current_block_type = "thinking"
        elif context is not None:
            needs_new_block = context.current_block_index < 0 or (
                context.current_block_type is not None
                and context.current_block_type != "thinking"
            )
            if needs_new_block:
                preamble = _close_content_blocks(context)
                _stream_blocks(context).open(context, block_type="thinking")
                context.current_block_type = "thinking"
                index = context.current_block_index
                for result in rd_results:
                    result["index"] = index
                return [
                    *preamble,
                    {
                        "type": AnthropicEventType.CONTENT_BLOCK_START,
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                    *rd_results,
                ]
            for result in rd_results:
                result["index"] = context.current_block_index
        return rd_results[0] if len(rd_results) == 1 else rd_results

    def _ir_reasoning_deltas(self, event: ReasoningDeltaEvent) -> list[dict[str, Any]]:
        """Split an IR reasoning delta event into Anthropic ``delta`` payloads.

        Anthropic carries the visible thinking text and the verification
        signature as separate deltas, so an event holding both yields two
        payloads: the text first, then the signature.  A foreign signature is
        dropped without taking the text along with it.

        Both fields are tested for emptiness rather than presence: an empty
        signature is not a signature, and emitting an empty ``signature_delta``
        would be a delta with no content.  The Interactions writer drops empty
        signatures the same way.

        Dropping an all-empty event keeps the client from ever seeing an empty
        thinking block.  That matters because an Anthropic client replays what
        it receives, and a thinking block cannot be sent back without a
        signature — the API answers ``400
        messages.N.content.0.thinking.signature: Field required`` — so the
        response side must not hand one out.  The request side drops the same
        shape in ``MessagesContentOps.ir_reasoning_to_p``.

        Two things this does not cover.  A non-empty reasoning part with no
        signature still leaves the request path unsigned and is rejected the
        same way; suppressing that is the shim's ``unsigned_blocks: preserve``
        policy, which ``anthropic--messages`` does not declare yet (#954).  And
        the Chat Completions writer keeps the opposite rule on purpose —
        DeepSeek's thinking mode needs ``reasoning_content`` present even when
        it is empty (#948) — so the two are not a divergence to unify.

        Args:
            event: IR reasoning delta event.

        Returns:
            ``content_block_delta`` payloads to emit, in order.
        """
        signature = event.get("signature")
        if signature and not is_own_origin(
            get_opaque_origin(event, "signature"), self._CONVERTER_TAG
        ):
            # Foreign verification signature — not valid for Anthropic.
            signature = None
        deltas: list[dict[str, Any]] = []
        if event["reasoning"]:
            deltas.append({"type": "thinking_delta", "thinking": event["reasoning"]})
        if signature:
            deltas.append({"type": "signature_delta", "signature": signature})
        return deltas

    def _handle_ir_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle ToolCallStartEvent → content_block_start for a tool call.

        Mirrors ``MessagesToolOps.ir_tool_call_to_p``: a call the server
        executed (``tool_type == "intrinsic"``) is re-emitted as
        ``server_tool_use`` with its canonical kind as the block name, and
        everything else as ``tool_use``.  The kind comes from the event's
        first-class ``intrinsic_kind`` field (legacy metadata still accepted),
        never from the tool name.

        A server call Anthropic has no server tool for is dropped — never
        downgraded to a ``tool_use`` the client would be asked to run — and
        the call id is remembered so the fragments that follow it are dropped
        with it rather than attached to whatever block is open.
        """
        block_type = "tool_use"
        tool_name = event["tool_name"]
        if event.get("tool_type", "function") == "intrinsic":
            kind = intrinsic_kind_of(event)
            if not self.tool_ops.intrinsic_renderable(kind, {}, leg="response"):
                report_loss(
                    f"Server tool call of kind {kind!r} has no Anthropic "
                    "server tool; dropped from the streamed response",
                    subject=kind,
                    warnings=getattr(context, "warnings", None),
                )
                if context is not None:
                    dropped = context.metadata.setdefault(
                        "_dropped_intrinsic_calls", set()
                    )
                    dropped.add(event["tool_call_id"])
                return {}
            block_type = "server_tool_use"
            tool_name = kind
        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_START,
            "content_block": {
                "type": block_type,
                "id": (
                    prefixed_tool_call_id(event["tool_call_id"], "srvtoolu_")
                    if block_type == "server_tool_use"
                    else sanitize_tool_call_id(event["tool_call_id"])
                ),
                "name": tool_name,
                "input": {},
            },
        }
        # Mirrors ``ir_tool_call_to_p``'s tail: the cache breakpoint always, and
        # a non-default ``caller`` only on a client ``tool_use`` — the
        # non-streaming writer makes the same restriction.
        block = result["content_block"]
        cache_hint = event.get("cache_hint")
        if cache_hint is not None:
            block["cache_control"] = cache_hint
        caller = (event.get("provider_metadata") or {}).get("anthropic_caller")
        if caller is not None and block_type == "tool_use":
            block["caller"] = caller
        if context is not None:
            state = _stream_blocks(context)
            tool_id = event["tool_call_id"]
            if state.tool_indices.get(tool_id) in state.open_indices:
                return {}
            preamble: list[dict[str, Any]] = []
            if context.current_block_type not in (None, "tool_use", "server_tool_use"):
                preamble = _close_content_blocks(context)
            index = state.pending_tool_index
            state.pending_tool_index = None
            if index is None and context.current_block_type is None:
                # Honor a caller-preallocated generic context index.
                if context.current_block_index >= state.next_index:
                    index = context.current_block_index
            index = state.open(context, index, block_type=block_type)
            state.tool_indices[tool_id] = index
            tool_call_index = event.get("tool_call_index")
            if tool_call_index is not None:
                state.tool_index_to_block[tool_call_index] = index
            context.current_block_type = block_type
            result["index"] = index
            if preamble:
                preamble.append(result)
                return preamble
        return result

    def _handle_ir_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallDeltaEvent → content_block_delta with input_json_delta.

        A fragment whose call was dropped at its start belongs to no block:
        writing it would open one for a server call the client was never told
        about, so it is dropped too.
        """
        tool_call_id = event.get("tool_call_id", "")
        explicit_idx: int | None = event.get("block_index")

        if context is None:
            # Stateless conversion: an explicit block index is all we have.
            if explicit_idx is None:
                report_loss(
                    "Dropping Messages tool-call delta without a context or "
                    "an explicit block index",
                    subject="tool_call_delta",
                )
                return {}
            return {
                "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": event["arguments_delta"],
                },
                "index": explicit_idx,
            }

        dropped = context.metadata.get("_dropped_intrinsic_calls")
        if dropped and tool_call_id in dropped:
            return {}

        state = _stream_blocks(context)
        registered_index = (
            state.tool_indices.get(tool_call_id) if tool_call_id else None
        )

        # The tool ID is the authority: the block it was registered to is the
        # one this fragment belongs to.  An explicit block_index is a
        # source-provider index (#246) and only breaks ties when no identity
        # is known.
        index: int | None = None
        if registered_index is not None:
            index = registered_index
            if explicit_idx is not None and explicit_idx != registered_index:
                report_loss(
                    f"Messages tool-call delta block_index={explicit_idx} "
                    f"disagrees with the registered index {registered_index} "
                    f"for {tool_call_id}; using the registered index",
                    code=DOWNGRADED,
                    subject=tool_call_id,
                    warnings=context.warnings,
                )
        elif explicit_idx is not None:
            index = explicit_idx
        else:
            tool_call_index = event.get("tool_call_index")
            if tool_call_index is not None:
                index = state.tool_index_to_block.get(tool_call_index)
            if index is None:
                index = state.pending_tool_index
            if (
                index is None
                and context.current_block_index >= 0
                and context.current_block_type in (None, "tool_use", "server_tool_use")
            ):
                if len(state.open_indices) > 1:
                    report_loss(
                        "Dropping ambiguous tool-call argument delta for "
                        f"{event.get('tool_call_id') or '<unknown>'}",
                        subject=str(event.get("tool_call_id") or ""),
                        warnings=context.warnings,
                    )
                    return {}
                index = context.current_block_index

        block_type = state.block_types.get(index) if index is not None else None
        if (
            index is None
            or index not in state.open_indices
            or block_type not in ("tool_use", "server_tool_use")
        ):
            report_loss(
                "Dropping Messages tool-call delta for an unregistered, closed, "
                f"or non-tool block (tool_call_id={tool_call_id}, "
                f"block_index={explicit_idx}, index={index})",
                subject=tool_call_id,
                warnings=context.warnings,
            )
            return {}

        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
            "delta": {
                "type": "input_json_delta",
                "partial_json": event["arguments_delta"],
            },
            "index": index,
        }
        context.current_block_index = index
        context.current_block_type = block_type
        return result

    def _handle_ir_tool_result_to_p(
        self, event: ToolResultEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolResultEvent → content_block_start for a server-tool result.

        A server-tool result arrives whole, so the writer emits the matching
        ``*_tool_result`` content block in a single ``content_block_start``.
        When the block was read from this same dialect, the stored ``_native``
        payload is replayed verbatim, preserving fields the IR does not model
        (the original ``content`` shape and ``caller``).  Otherwise the block is
        rebuilt from the IR fields.

        A server result for a kind Anthropic has no server tool for is dropped
        with a warning, the same way its call is: the client can neither run
        the tool nor recognize its result.
        """
        if event.get("tool_type") == "intrinsic" and not (
            self.tool_ops.intrinsic_renderable(
                intrinsic_kind_of(event), None, leg="response"
            )
        ):
            report_loss(
                f"Server tool result of kind {intrinsic_kind_of(event)!r} has no "
                "Anthropic server tool; dropped from the streamed response",
                subject=intrinsic_kind_of(event),
                warnings=getattr(context, "warnings", None),
            )
            return {}
        content_block = self._ir_tool_result_block(event)

        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_START,
            "content_block": content_block,
        }
        index = event.get("block_index")
        if context is not None:
            state = _stream_blocks(context)
            index = state.open(context, index, block_type=content_block.get("type"))
            context.current_block_type = content_block.get("type")
            result["index"] = index
        elif index is not None:
            result["index"] = index
        return result

    def _ir_tool_result_block(self, event: ToolResultEvent) -> dict[str, Any]:
        """Build the Anthropic ``*_tool_result`` block for a ToolResultEvent.

        Shares the writer with the non-streaming assistant turn
        (``MessagesToolOps.ir_server_tool_result_to_p``), so a same-dialect read
        restores the stored ``_native`` block and a foreign one is rebuilt from
        the IR fields.  A ``ToolResultEvent`` is always a server result, so the
        writer never returns ``None`` here.
        """
        block = MessagesToolOps.ir_server_tool_result_to_p(event)
        if block is None:  # pragma: no cover - defensive; events are intrinsic
            # Generic ``tool_result`` shape: the reader only emits this event for
            # ``*_tool_result`` blocks, but if one ever arrives without an
            # intrinsic kind keep the full block — both fields are valid on the
            # generic block, unlike on a ``*_tool_result`` one (#973).
            block = {
                "type": "tool_result",
                "tool_use_id": sanitize_tool_call_id(event["tool_call_id"]),
                "content": event.get("result", []),
            }
            if event.get("is_error") is not None:
                block["is_error"] = event["is_error"]
            cache_hint = event.get("cache_hint")
            if cache_hint is not None:
                block["cache_control"] = cache_hint
        return block

    def _handle_ir_refusal_delta_to_p(
        self, event: RefusalDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle RefusalDeltaEvent → nothing yet; the words ride the finish.

        Anthropic has no refusal *delta*: the refusal is a content block, and
        the structured ``stop_details`` is answered with the stop reason.  So
        the words are held until the finish delta, where the body leg puts
        them too.
        """
        if context is not None and event["refusal"]:
            previous = context.metadata.get("_refusal_text", "")
            context.metadata["_refusal_text"] = previous + event["refusal"]
        return {}

    def _handle_ir_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle FinishEvent → buffered message_delta + optional content_block_stop."""
        reason = event["finish_reason"]["reason"]
        pm = event.get("provider_metadata") or {}
        stop_reason = _anthropic_stop_reason(
            reason,
            context.response_id if context is not None else "",
            pm,
        )

        # The sequence the model actually stopped on, when it stopped on one.
        # Anthropic puts it on the same delta as ``stop_reason``; dropping it
        # left the stream saying ``null`` while the body carried the word.
        stop_sequence = event["finish_reason"].get("stop_sequence")

        # Structured refusal details, when the turn actually carries them.
        # A reason of ``refusal`` alone says the model declined, not why; the
        # body writer sets ``stop_details`` only for a refusal *part*, so
        # inventing ``{'type': 'refusal', 'explanation': ''}`` here would let
        # the two legs disagree about a turn nobody wrote words for.
        stop_details = None
        if reason == "refusal":
            category = pm.get("anthropic_refusal_category")
            explanation = pm.get("refusal_text", "") or (
                context.metadata.get("_refusal_text", "") if context is not None else ""
            )
            if category or explanation:
                stop_details = {
                    "type": "refusal",
                    "category": category,
                    "explanation": explanation,
                }

        if context is not None:
            results = _close_content_blocks(context)
            usage = context.pop_pending_usage()
            if usage is not None:
                delta_payload: dict[str, Any] = {"stop_reason": stop_reason}
                if stop_details:
                    delta_payload["stop_details"] = stop_details
                delta_payload.setdefault("stop_sequence", stop_sequence)
                delta_payload.setdefault("stop_details", None)
                delta_payload.setdefault("container", None)
                results.append(
                    {
                        "type": AnthropicEventType.MESSAGE_DELTA,
                        "delta": delta_payload,
                        "usage": self._build_message_delta_usage(usage),
                    }
                )
            else:
                # Buffer finish for later UsageEvent or StreamEnd flush.
                finish_delta: dict[str, Any] = {"stop_reason": stop_reason}
                if stop_details:
                    finish_delta["stop_details"] = stop_details
                finish_delta.setdefault("stop_sequence", stop_sequence)
                finish_delta.setdefault("stop_details", None)
                finish_delta.setdefault("container", None)
                context.buffer_finish(finish_delta)
            return results if results else {}
        return {
            "type": AnthropicEventType.MESSAGE_DELTA,
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence,
                "stop_details": stop_details,
                "container": None,
            },
            "usage": self._build_message_delta_usage(),
        }

    def _handle_ir_usage_to_p(
        self, event: UsageEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle UsageEvent → message_delta (merged with pending finish).

        When a pending_finish exists, merges usage into it and emits a
        message_delta immediately.  Otherwise buffers the usage for a
        later FinishEvent or StreamEndEvent to consume, preventing the
        extra empty message_delta that caused round-trip inflation.
        """
        usage = event["usage"]
        if context is not None:
            delta = context.pop_pending_finish()
            if delta is not None:
                return {
                    "type": AnthropicEventType.MESSAGE_DELTA,
                    "delta": delta,
                    "usage": self._build_message_delta_usage(usage),
                }
            # No pending finish — buffer for later merge.
            context.buffer_usage(usage)
            return {}
        return {
            "type": AnthropicEventType.MESSAGE_DELTA,
            "delta": {},
            "usage": self._build_message_delta_usage(usage),
        }
