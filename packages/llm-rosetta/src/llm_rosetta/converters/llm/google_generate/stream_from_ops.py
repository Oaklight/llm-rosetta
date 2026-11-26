"""
LLM-Rosetta - Google GenAI stream reader

Provider-chunk to IR-stream-event handlers for the converter.
"""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast


from llm_rosetta.ir.model.response import UsageInfo
from llm_rosetta.ir.model.stream import (
    CitationDeltaEvent,
    FinishEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    UsageEvent,
)
from ...base.context import (
    ConversionContext,
    StreamContext,
    carry_registered_call_identity,
)
from llm_rosetta.ir.tools.intrinsic import (
    intrinsic_kind_of,
)
from ...base.helpers.opaque import (
    set_google_thought_signature,
    set_opaque_origin,
)
from llm_rosetta.ir.primitives.native import record_native
from ._constants import (
    GOOGLE_REASON_FROM_PROVIDER,
    generate_tool_call_id,
)

from .response_ops import (
    _search_metadata,
    _citation_is_new,
    _stash_search_metadata,
    _take_search_metadata,
    _grounding_citations,
    _native_word_lost,
)

if TYPE_CHECKING:
    from .tool_ops import GoogleGenerateToolOps


class GoogleGenerateStreamFromOps:
    """Mixin: Google stream chunks to IR stream events."""

    if TYPE_CHECKING:
        _CONVERTER_TAG: str
        tool_ops: GoogleGenerateToolOps

        @staticmethod
        def _normalize(data: Any) -> dict: ...

        @staticmethod
        def strip_response_id_prefix(raw_id: str, prefix: str) -> str: ...

        def _get_response_id_prefix(
            self, context: ConversionContext | StreamContext | None = None
        ) -> str: ...

        def finalize_stream_finish(
            self,
            events: list[IRStreamEvent],
            context: StreamContext | None,
        ) -> list[IRStreamEvent]: ...

        def _hosted_call_kind(
            self,
            tool_name: str,
            context: ConversionContext | StreamContext | None,
        ) -> str: ...

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any] | None,
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert a Google GenAI stream chunk to IR stream events.

        Google GenAI stream chunks are complete ``GenerateContentResponse``
        objects. Each chunk contains incremental content in
        ``candidates[].content.parts[]``.

        When a ``context`` is provided, lifecycle events (``StreamStartEvent``,
        ``StreamEndEvent``) are emitted and cross-chunk state is tracked.

        Pass ``None`` as the chunk to signal end-of-stream; nothing of this
        dialect is held back across chunks, so it flushes nothing.

        Args:
            chunk: Google GenAI stream chunk dict (or SDK object), or
                ``None`` to signal end-of-stream.
            context: Optional stream context for stateful conversions.

        Returns:
            List of IR stream events extracted from the chunk.
        """
        if chunk is None:
            # The EOF sentinel: this dialect holds nothing back across chunks
            # except an end that waited for a ``usageMetadata`` chunk which
            # never came, so it flushes that.  Callers pass it —
            # ``BaseConverter.iter_stream_from_provider`` and the gateway's
            # per-chunk ``StreamProcessor`` both do — so it must not reach the
            # normalizing below.
            if context is not None and context._stream_end_deferred:
                context._stream_end_deferred = False
                context.mark_ended()
                return [StreamEndEvent(type="stream_end")]
            return []

        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []

        if context is not None and not context.is_started:
            self._handle_p_stream_start_to_ir(chunk, context, events)

        has_finish_reason = False
        deferred_finish: FinishEvent | None = None

        for candidate in chunk.get("candidates", []):
            finish = self._process_stream_candidate(candidate, context, events)
            if finish is not None:
                has_finish_reason = True
                deferred_finish = finish

        # Handle prompt-level blocks in streaming (no candidates)
        if not has_finish_reason and not chunk.get("candidates"):
            prompt_feedback = (
                chunk.get("prompt_feedback") or chunk.get("promptFeedback") or {}
            )
            block_reason = prompt_feedback.get("block_reason") or prompt_feedback.get(
                "blockReason"
            )
            if block_reason:
                has_finish_reason = True
                block_ir_reason = GOOGLE_REASON_FROM_PROVIDER.get(
                    block_reason, "content_filter"
                )
                deferred_finish = FinishEvent(
                    type="finish",
                    finish_reason={"reason": block_ir_reason},  # ty: ignore[invalid-argument-type]
                )
                record_native(
                    deferred_finish.setdefault("provider_metadata", {}),
                    origin=self._CONVERTER_TAG,
                    prompt_feedback=prompt_feedback,
                )

        self._handle_p_usage_to_ir(chunk, events)

        if deferred_finish is not None:
            events.append(deferred_finish)

        if (
            context is not None
            and context._stream_end_deferred
            and any(e["type"] == "usage" for e in events)
        ):
            # The usage chunk the deferral waited for: it has been emitted
            # above, so the end may follow it.
            context._stream_end_deferred = False
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

        if context is not None and has_finish_reason:
            # Google sends ``usageMetadata`` in a chunk *after* the one that
            # carries ``finishReason``.  Ending the stream here put the
            # ``usage`` event behind ``stream_end``, and a target dialect that
            # stops reading at the end dropped it (#1314).  Wait for the
            # usage chunk; the EOF sentinel flushes an end that never came.
            if chunk.get("usageMetadata"):
                context.mark_ended()
                events.append(StreamEndEvent(type="stream_end"))
            else:
                context._stream_end_deferred = True

        return self.finalize_stream_finish(events, context)

    def _process_stream_candidate(
        self,
        candidate: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> FinishEvent | None:
        """Process a single candidate from a stream chunk.

        Handles part dispatch, text deferral for compound chunks, and
        finish reason detection.  Returns a ``FinishEvent`` if the
        candidate signals completion, otherwise ``None``.
        """
        choice_index = candidate.get("index", 0)
        cand_content = candidate.get("content", {})
        finish_reason = candidate.get("finish_reason") or candidate.get("finishReason")
        grounding = candidate.get("grounding_metadata") or candidate.get(
            "groundingMetadata"
        )

        pre_parts_len = len(events)
        for part in cand_content.get("parts", []):
            self._handle_p_part_to_ir(part, choice_index, context, events)

        if finish_reason and context is not None:
            new_events: list[IRStreamEvent] = events[pre_parts_len:]
            deferred_texts: list[str] = []
            kept_new: list[IRStreamEvent] = []
            for ev in new_events:
                if ev["type"] == "text_delta":
                    deferred_texts.append(ev["text"])
                else:
                    kept_new.append(ev)
            if deferred_texts:
                context.pending_text = "".join(deferred_texts)
                events[pre_parts_len:] = kept_new

        if isinstance(grounding, dict):
            # Citations ride their own events (one per source, see the IR's
            # ``citation_delta``), emitted where Google reports them rather than
            # deferred to the finish: a reader that waits for a finish loses
            # them when the candidate produced no text.  The search-level
            # metadata is not a citation and has no event, so it is held for the
            # finish event.
            for citation in _grounding_citations(grounding):
                if context is None or _citation_is_new(context, citation):
                    events.append(
                        CitationDeltaEvent(
                            type="citation_delta",
                            citation=citation,
                            choice_index=choice_index,
                        )
                    )
            if context is not None:
                _stash_search_metadata(context, _search_metadata(grounding))

        if finish_reason:
            # Google finishes in a chunk with no content, so the call that
            # makes this a tool-call turn arrived in an earlier one: ask the
            # context for every call the response has opened, not this chunk.
            finish_reason_ir = GOOGLE_REASON_FROM_PROVIDER.get(finish_reason, "stop")  # ty: ignore[invalid-argument-type]
            finish_event = FinishEvent(
                type="finish",
                finish_reason={"reason": finish_reason_ir},  # ty: ignore[invalid-argument-type]
                choice_index=choice_index,
            )
            # Google's search-level metadata (the queries and the rendered
            # entry point) is not a citation and has no event of its own; it
            # rides the finish, which every reader receives.  The citations
            # themselves were emitted as ``citation_delta`` where they arrived.
            finish_metadata: dict[str, Any] = {}
            search = _take_search_metadata(context)
            if search:
                finish_metadata["google"] = search
            # Same record the JSON reader leaves: the word Google used for a
            # stop the IR has no room for, so a Google target can write it back
            # and only a *foreign* failure raises.
            if _native_word_lost(finish_reason_ir, finish_reason):
                record_native(
                    finish_metadata,
                    origin=self._CONVERTER_TAG,
                    finish_reason=finish_reason,
                )
            if finish_metadata:
                finish_event["provider_metadata"] = finish_metadata
            return finish_event
        return None

    @staticmethod
    def _apply_candidate_grounding(
        message: dict[str, Any], grounding: dict[str, Any]
    ) -> None:
        """Put a candidate's grounding on the IR message.

        The sources become citation parts after the text; the search-level
        metadata (the queries and the rendered entry point) is not a citation
        and describes the turn, so it rides ``provider_metadata.google`` once
        instead of being repeated on every source.
        """
        citations = _grounding_citations(grounding)
        content = message.get("content")
        if isinstance(content, list):
            content.extend(citations)
        else:
            message["content"] = citations
        search = _search_metadata(grounding)
        if search:
            existing = cast(dict[str, Any], message.get("provider_metadata") or {})
            google_metadata = dict(existing.get("google") or {})
            google_metadata.update(search)
            message["provider_metadata"] = {**existing, "google": google_metadata}

    def _handle_p_stream_start_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit StreamStartEvent on the first chunk."""
        response_id = chunk.get("response_id") or chunk.get("responseId") or ""
        model = chunk.get("model_version") or chunk.get("modelVersion") or ""
        prefix = self._get_response_id_prefix(context)
        context.response_id = self.strip_response_id_prefix(response_id, prefix)
        context.model = model
        context.mark_started()
        events.append(
            StreamStartEvent(
                type="stream_start",
                response_id=response_id,
                model=model,
            )
        )

    def _handle_p_part_to_ir(
        self,
        part: dict[str, Any],
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Process a single part from a candidate's content."""
        is_thought = part.get("thought", False)
        text = part.get("text")
        func_call = part.get("function_call") or part.get("functionCall")
        thought_sig = part.get("thoughtSignature") or part.get("thought_signature")
        # Google puts a turn's thought signature on its own part, where the
        # empty ``text`` is a placeholder for the signature rather than content.
        signature_only = bool(thought_sig) and text == ""

        if text is not None and not signature_only:
            if is_thought:
                events.append(
                    ReasoningDeltaEvent(
                        type="reasoning_delta",
                        reasoning=text,
                        choice_index=choice_index,
                    )
                )
            else:
                events.append(
                    TextDeltaEvent(
                        type="text_delta",
                        text=text,
                        choice_index=choice_index,
                    )
                )

        # The signature cannot ride on a text delta -- TextDeltaEvent has no
        # opaque slot -- so it travels on a signature-only reasoning delta, the
        # same shape the Anthropic and Interactions readers use.  A functionCall
        # part is excluded: _handle_p_function_call_to_ir already carries its
        # signature on the tool_call_start, and emitting it here too would put
        # one signature on two IR carriers.
        if thought_sig and not func_call:
            sig_event = ReasoningDeltaEvent(
                type="reasoning_delta",
                reasoning="",
                signature=thought_sig,
                choice_index=choice_index,
            )
            set_opaque_origin(sig_event, signature=self._CONVERTER_TAG)
            set_google_thought_signature(sig_event, thought_sig)
            events.append(sig_event)

        if func_call:
            self._handle_p_function_call_to_ir(
                func_call, part, choice_index, context, events
            )
            return

        # Server-tool parts.  Google delivers ``executableCode`` and
        # ``codeExecutionResult`` whole, in their own parts.  A server call
        # must never surface as a client call, so both are read with the same
        # intrinsic reader the non-streaming path uses — the kind comes from
        # the reader, never from a tool name.  (#986)
        intrinsic_call = self.tool_ops.p_intrinsic_call_to_ir(part)
        if intrinsic_call is not None:
            self._handle_p_intrinsic_call_to_ir(
                cast("dict[str, Any]", intrinsic_call),
                part,
                choice_index,
                context,
                events,
            )
            return

        intrinsic_result = self.tool_ops.p_intrinsic_result_to_ir(part)
        if intrinsic_result is not None:
            self._handle_p_intrinsic_result_to_ir(
                cast("dict[str, Any]", intrinsic_result),
                part,
                choice_index,
                context,
                events,
            )

    def _handle_p_intrinsic_call_to_ir(
        self,
        ir_call: dict[str, Any],
        native_part: dict[str, Any],
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit ``ToolCallStartEvent`` (+ input delta) for a server call.

        Google sends the call whole, so the start event carries the native
        part in ``provider_metadata`` for a same-dialect restore.  A
        ``ToolCallDeltaEvent`` accompanies it so every other format receives
        the input too.
        """
        tool_call_id = ir_call["tool_call_id"]
        tool_name = ir_call["tool_name"]
        native_meta = {
            "_native": {"base": self._CONVERTER_TAG, "part": dict(native_part)}
        }
        if context is not None:
            context.register_tool_call(
                tool_call_id,
                tool_name,
                tool_type="intrinsic",
                provider_metadata=native_meta,
                intrinsic_kind=intrinsic_kind_of(ir_call),
            )
        tc_index = context.tool_call_count - 1 if context is not None else 0
        start_event: dict[str, Any] = {
            "type": "tool_call_start",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_type": "intrinsic",
            "choice_index": choice_index,
            "tool_call_index": tc_index,
            "provider_metadata": dict(native_meta),
        }
        start_event["intrinsic_kind"] = intrinsic_kind_of(ir_call)
        events.append(cast(ToolCallStartEvent, start_event))

        args_json = json.dumps(ir_call.get("tool_input", {}))
        delta_evt = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id=tool_call_id,
            arguments_delta=args_json,
            choice_index=choice_index,
        )
        delta_evt["tool_call_index"] = tc_index
        if context is not None:
            # The fragment repeats the call's canonical kind so a reader that
            # sees only the delta is not left guessing from the tool name.
            # The native part (and its thoughtSignature) stays on the start
            # event: Google sends the call whole, one signature, one carrier.
            carry_registered_call_identity(
                delta_evt, context, tool_call_id, include_metadata=False
            )
        events.append(delta_evt)
        if context is not None:
            context.append_tool_call_args(tool_call_id, args_json)

    def _handle_p_intrinsic_result_to_ir(
        self,
        ir_result: dict[str, Any],
        native_part: dict[str, Any],
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit a one-shot ``ToolResultEvent`` for a server result.

        The result arrives whole with no deltas to follow, so everything
        rides on the event.  A result event that carries an id — a
        cross-format hop, where the source dialect knew which call it answered
        — pairs by that id.  Google's own ``codeExecutionResult`` carries no
        id, so for Google wire the pairing falls back to position: the most
        recent intrinsic call the same stream registered.  See
        :meth:`_last_intrinsic_id`.
        """
        tool_call_id = ir_result.get("tool_call_id") or self._last_intrinsic_id(context)
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id=tool_call_id,
            result=ir_result.get("result", ""),
            is_error=ir_result.get("is_error", False),
            tool_type="intrinsic",
        )
        event["intrinsic_kind"] = intrinsic_kind_of(ir_result)
        event.setdefault("provider_metadata", {})["_native"] = {
            "base": self._CONVERTER_TAG,
            "part": dict(native_part),
        }
        events.append(event)

    @staticmethod
    def _last_intrinsic_id(context: StreamContext | None) -> str:
        """Return the id of the most recent intrinsic call, or ``""``."""
        if context is None:
            return ""
        for call_id in reversed(context.tool_call_ids):
            if context.get_tool_type(call_id) == "intrinsic":
                return call_id
        return ""

    def _handle_p_function_call_to_ir(
        self,
        func_call: dict[str, Any],
        part: dict[str, Any],
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Process a function_call part into ToolCallStart + ToolCallDelta events.

        A ``functionCall`` may name a hosted tool the request enabled, which
        Google delivers through the same part as a client function.  Such a
        call is registered and emitted as intrinsic, exactly like the dialect's
        own server parts, so the finish event reads the turn as complete and no
        client is asked to run a search it cannot run.
        """
        tool_call_id = func_call.get("id") or generate_tool_call_id()
        tool_name = func_call.get("name", "")
        args = func_call.get("args", {})
        intrinsic_kind = self._hosted_call_kind(tool_name, context)
        thought_sig = part.get("thoughtSignature") or part.get("thought_signature")

        # The native part travels with an intrinsic call so a Google target can
        # write the same ``functionCall`` back on either leg (the counterpart of
        # the ``executableCode`` native part below).
        provider_metadata: dict[str, Any] = {}
        if intrinsic_kind:
            provider_metadata["_native"] = {
                "base": self._CONVERTER_TAG,
                "part": dict(part),
            }
        if thought_sig:
            provider_metadata.setdefault("google", {})["thought_signature"] = (
                thought_sig
            )

        if context is not None:
            context.register_tool_call(
                tool_call_id,
                tool_name,
                tool_type="intrinsic" if intrinsic_kind else "function",
                provider_metadata=provider_metadata or None,
                intrinsic_kind=intrinsic_kind,
            )

        tc_index = context.tool_call_count - 1 if context is not None else 0
        start_event: dict[str, Any] = {
            "type": "tool_call_start",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "choice_index": choice_index,
            "tool_call_index": tc_index,
        }
        if intrinsic_kind:
            start_event["tool_type"] = "intrinsic"
            start_event["intrinsic_kind"] = intrinsic_kind

        if provider_metadata:
            start_event["provider_metadata"] = {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in provider_metadata.items()
            }

        events.append(cast(ToolCallStartEvent, start_event))

        args_json = json.dumps(args) if isinstance(args, dict) else str(args)
        delta_evt = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id=tool_call_id,
            arguments_delta=args_json,
            choice_index=choice_index,
        )
        delta_evt["tool_call_index"] = tc_index
        if intrinsic_kind:
            # The fragment repeats the call's canonical kind, as the dialect's
            # own server call does, so a reader that sees only the delta does
            # not fall back to the tool name.
            delta_evt["intrinsic_kind"] = intrinsic_kind
        events.append(delta_evt)

        if context is not None:
            context.append_tool_call_args(tool_call_id, args_json)

    def _handle_p_usage_to_ir(
        self,
        chunk: dict[str, Any],
        events: list[IRStreamEvent],
    ) -> None:
        """Emit UsageEvent from chunk usage metadata."""
        usage = chunk.get("usage_metadata") or chunk.get("usageMetadata")
        if not usage:
            return

        usage_info: dict[str, Any] = {
            "prompt_tokens": usage.get(
                "prompt_token_count", usage.get("promptTokenCount", 0)
            ),
            "completion_tokens": usage.get(
                "candidates_token_count",
                usage.get("candidatesTokenCount", 0),
            ),
            "total_tokens": usage.get(
                "total_token_count", usage.get("totalTokenCount", 0)
            ),
        }

        thoughts = usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount")
        if thoughts is not None:
            usage_info["reasoning_tokens"] = thoughts

        cached = usage.get("cached_content_token_count") or usage.get(
            "cachedContentTokenCount"
        )
        if cached is not None:
            usage_info["cache_read_tokens"] = cached

        events.append(
            UsageEvent(
                type="usage",
                usage=cast(UsageInfo, usage_info),
            )
        )

    # --- to_provider ---

    @staticmethod
    def _inject_stream_metadata(
        chunk: dict[str, Any], context: StreamContext | None
    ) -> dict[str, Any]:
        """Inject ``responseId`` and ``modelVersion`` into a non-empty chunk.

        Google's streaming API includes these fields on every chunk.
        """
        if not chunk or context is None:
            return chunk
        if context.response_id:
            chunk["responseId"] = context.response_id
        if context.model:
            chunk["modelVersion"] = context.model
        return chunk

    @staticmethod
    def _build_stream_usage_metadata(usage: Mapping[str, Any]) -> dict[str, Any]:
        """Build a Google ``usageMetadata`` dict from IR usage for streaming.

        Lighter than ``_build_ir_usage_to_p`` — omits the details arrays
        which are not typically present in streaming usage events.
        """
        usage_metadata: dict[str, Any] = {
            "promptTokenCount": usage.get("prompt_tokens") or 0,
            "candidatesTokenCount": usage.get("completion_tokens") or 0,
            "totalTokenCount": usage.get("total_tokens") or 0,
        }
        if "reasoning_tokens" in usage:
            usage_metadata["thoughtsTokenCount"] = usage["reasoning_tokens"]
        if "cache_read_tokens" in usage:
            usage_metadata["cachedContentTokenCount"] = usage["cache_read_tokens"]
        return usage_metadata
