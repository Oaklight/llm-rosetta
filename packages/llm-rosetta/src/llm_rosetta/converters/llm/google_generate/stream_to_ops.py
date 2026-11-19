"""
LLM-Rosetta - Google GenAI stream writer

IR-stream-event to provider-chunk handlers for the converter.
"""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast


from ....types.ir.stream import (
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
from llm_rosetta.conversion_warnings import report_loss
from ...base.context import (
    StreamContext,
)
from ...base.tools import (
    sanitize_tool_call_id,
)
from ...base.tools.intrinsic import (
    intrinsic_kind_of,
)
from ...base.helpers.citations import (
    report_dropped_citation,
)
from ...base.helpers.opaque import (
    get_opaque_origin,
    google_thought_signature,
    inject_google_function_call_signature,
    is_own_origin,
)

from .response_ops import (
    _grounding_metadata_from_ir,
    _google_finish_reason,
    _native_prompt_feedback,
)

if TYPE_CHECKING:
    from ....types.ir import ToolResultPart
    from .tool_ops import GoogleGenerateToolOps


class GoogleGenerateStreamToOps:
    """Mixin: IR stream events to Google stream chunks."""

    if TYPE_CHECKING:
        _CONVERTER_TAG: str
        tool_ops: GoogleGenerateToolOps

        @staticmethod
        def _inject_stream_metadata(
            chunk: dict[str, Any], context: StreamContext | None
        ) -> dict[str, Any]: ...

        @staticmethod
        def _build_stream_usage_metadata(
            usage: Mapping[str, Any],
        ) -> dict[str, Any]: ...

    def _handle_ir_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamStartEvent → store metadata, no output."""
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.mark_started()
        return {}

    def _handle_ir_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamEndEvent → flush any buffered usage, mark ended.

        When usage arrives after the finish chunk (e.g. OpenAI streaming
        sends usage in a separate final chunk), pending_usage won't have
        been merged into the finish chunk. Emit it here so it's not lost.
        """
        if context is not None:
            usage = context.pop_pending_usage()
            context.mark_ended()
            if usage is not None:
                return self._inject_stream_metadata(
                    {"usageMetadata": self._build_stream_usage_metadata(usage)},
                    context,
                )
        return {}

    def _handle_ir_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockStartEvent → no-op for Google GenAI."""
        return {}

    def _handle_ir_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockEndEvent → no-op for Google GenAI."""
        return {}

    def _handle_ir_text_delta_to_p(
        self, event: TextDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle TextDeltaEvent → text part chunk.

        Returns empty for empty-text deltas (e.g. padding in Google
        finish chunks) to avoid inflating the output event count.
        """
        if not event["text"]:
            return {}
        choice_index = event.get("choice_index", 0)
        return self._inject_stream_metadata(
            {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [{"text": event["text"]}],
                        },
                    }
                ]
            },
            context,
        )

    def _handle_ir_refusal_delta_to_p(
        self, event: RefusalDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle RefusalDeltaEvent → a text part chunk.

        Google has no refusal part and its stream carries no field for one:
        a real ``streamGenerateContent`` probe (2026-10-30, gemini-3.7-flash)
        showed part keys of ``text`` and ``thoughtSignature`` only, on a
        normal turn and on one the model declined.  The body writer already
        folds a refusal part into a text part, so the stream does the same and
        the words survive both legs.
        """
        if not event["refusal"]:
            return {}
        choice_index = event.get("choice_index", 0)
        return self._inject_stream_metadata(
            {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [{"text": event["refusal"]}],
                        },
                    }
                ]
            },
            context,
        )

    def _handle_ir_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ReasoningDeltaEvent → thought text part chunk."""
        choice_index = event.get("choice_index", 0)
        part: dict[str, Any] = {"thought": True, "text": event["reasoning"]}
        # Prefer the Google-namespace carrier (survives a foreign hop); a flat
        # signature is used only when it is Google-origin or unrecorded.
        signature = google_thought_signature(event)
        if signature is None and is_own_origin(
            get_opaque_origin(event, "signature"), self._CONVERTER_TAG
        ):
            signature = event.get("signature")
        if signature:
            part["thoughtSignature"] = signature
        return self._inject_stream_metadata(
            {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {"role": "model", "parts": [part]},
                    }
                ]
            },
            context,
        )

    def _handle_ir_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallStartEvent → register, and write a server call.

        A client call stays pending until the finish flush, which writes it as
        ``functionCall``.  A server call (``tool_type == "intrinsic"``) is
        re-emitted as Google's own part type, never as a client
        ``functionCall``: ``executableCode`` is written here, at the position
        Google sent it, when the event carries the native part from a
        same-dialect read.  Without a native part (a cross-format hop) it is
        deferred to the finish flush, which rebuilds it from the accumulated
        input.  An intrinsic kind Google cannot render is dropped at the flush
        with a warning, but never as a client call.  (#986)
        """
        tool_type = event.get("tool_type", "function")
        if context is not None:
            context.register_tool_call(
                event["tool_call_id"],
                event["tool_name"],
                tool_type=tool_type,
                provider_metadata=event.get("provider_metadata"),
                intrinsic_kind=intrinsic_kind_of(event),
            )
        if tool_type != "intrinsic":
            return {}
        native = self._native_executable_code(event.get("provider_metadata"))
        if native is None:
            return {}
        return self._executable_code_chunk(
            native, event.get("choice_index", 0), context
        )

    def _handle_ir_tool_result_to_p(
        self, event: ToolResultEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolResultEvent → a Google server-result part.

        Only an intrinsic result Google can render
        (``intrinsic_renderable``) becomes a ``codeExecutionResult`` part; a
        native part from a same-dialect read is replayed verbatim so its
        ``outcome`` survives.  Anything else is dropped with a warning — a
        server result is never downgraded to a client ``functionResponse``.
        (#986)
        """
        if event.get("tool_type") != "intrinsic":
            return {}
        kind = intrinsic_kind_of(event)
        if not self.tool_ops.intrinsic_renderable(kind, None, leg="response"):
            report_loss(
                f"Unsupported intrinsic kind {kind!r} for Google Generate "
                "stream result — dropping",
                subject=kind,
                leg="stream",
                warnings=getattr(context, "warnings", None),
            )
            return {}
        native = self._native_code_execution_result(event.get("provider_metadata"))
        result_part: dict[str, Any]
        if native is not None:
            result_part = {"codeExecutionResult": dict(native)}
        else:
            ir_part: dict[str, Any] = {
                "type": "tool_result",
                "tool_call_id": event["tool_call_id"],
                "result": event.get("result", ""),
                "tool_type": "intrinsic",
            }
            if event.get("is_error"):
                ir_part["is_error"] = True
            ir_part["intrinsic_kind"] = kind
            converted = self.tool_ops.ir_intrinsic_result_to_p(
                cast("ToolResultPart", ir_part)
            )
            if converted is None:
                return {}
            result_part = converted
        return self._inject_stream_metadata(
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [result_part]},
                    }
                ]
            },
            context,
        )

    def _native_executable_code(
        self, provider_metadata: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        """Return the native ``executableCode`` for a Google-origin call."""
        native = (provider_metadata or {}).get("_native")
        if isinstance(native, dict) and native.get("base") == self._CONVERTER_TAG:
            part = native.get("part")
            if isinstance(part, dict) and isinstance(part.get("executableCode"), dict):
                return part["executableCode"]
        return None

    def _native_code_execution_result(
        self, provider_metadata: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        """Return the native ``codeExecutionResult``, if Google-origin."""
        native = (provider_metadata or {}).get("_native")
        if isinstance(native, dict) and native.get("base") == self._CONVERTER_TAG:
            part = native.get("part")
            if isinstance(part, dict) and isinstance(
                part.get("codeExecutionResult"), dict
            ):
                return part["codeExecutionResult"]
        return None

    def _executable_code_chunk(
        self,
        executable: Mapping[str, Any],
        choice_index: int,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        """Build a stream chunk carrying a single ``executableCode`` part."""
        part = {
            "executableCode": {
                "code": executable.get("code", ""),
                "language": executable.get("language", "PYTHON"),
            }
        }
        return self._inject_stream_metadata(
            {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {"role": "model", "parts": [part]},
                    }
                ]
            },
            context,
        )

    def _handle_ir_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallDeltaEvent → accumulate args, no output."""
        if context is not None:
            context.append_tool_call_args(
                event["tool_call_id"], event["arguments_delta"]
            )
        return {}

    def _flushed_intrinsic_part(
        self,
        kind: str,
        provider_metadata: Mapping[str, Any],
        args_str: str,
        *,
        tool_name: str = "",
        tool_call_id: str = "",
    ) -> dict[str, Any] | None:
        """Build the finish part for a pending intrinsic call, or ``None``.

        A Google-origin hosted call is flushed as the ``functionCall`` it
        arrived as — name, accumulated args, id and thought signature — which
        is the shape a client call takes and the shape master replayed.  It
        stays intrinsic in the IR; only the wire form is restored.

        ``None`` means nothing is written here: the call was already written on
        its start event (it carried a native ``executableCode``), or Google
        cannot render the kind.  Either way a server call is never downgraded
        to a client ``functionCall``.  (#986)
        """
        native_call = self.tool_ops.native_function_call(
            {"provider_metadata": provider_metadata}
        )
        if native_call is not None:
            try:
                args = json.loads(args_str) if args_str.strip() else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            func_call: dict[str, Any] = {
                "name": tool_name or native_call.get("name", ""),
                "args": args,
            }
            if tool_call_id:
                func_call["id"] = sanitize_tool_call_id(tool_call_id)
            function_part: dict[str, Any] = {"functionCall": func_call}
            thought_sig = google_thought_signature(
                {"provider_metadata": provider_metadata}
            )
            if thought_sig:
                function_part["thoughtSignature"] = thought_sig
            return function_part
        if self._native_executable_code(provider_metadata) is not None:
            return None
        if not self.tool_ops.intrinsic_renderable(kind, None, leg="response"):
            if kind:
                report_loss(
                    f"Unsupported intrinsic kind {kind!r} for Google Generate "
                    "stream call — dropping",
                    subject=kind,
                    leg="stream",
                )
            return None
        if not args_str.strip():
            # No input ever arrived (a cross-format source that emitted the
            # result without an argument delta): there is nothing to render.
            return None
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            # A truncated fragment (a stream cut mid-call) parses to nothing,
            # which the guard below then drops.
            args = {}
        if not isinstance(args, dict) or not args.get("code"):
            # ``executableCode`` with an empty ``code`` is a part no API
            # accepts, so drop it the way an unrenderable kind is dropped.  A
            # fragment that parses to a list or a string has no ``code`` either.
            return None
        return {
            "executableCode": {
                "code": args["code"],
                "language": args.get("language", "PYTHON"),
            }
        }

    def _handle_ir_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> list[dict[str, Any]]:
        """Handle FinishEvent → flush tool calls + finish chunk."""
        choice_index = event.get("choice_index", 0)
        reason = event["finish_reason"]["reason"]

        # A blocked prompt streams as the field Google sent, with no candidate:
        # there is no model turn to write.
        blocked = _native_prompt_feedback(
            event.get("provider_metadata") or {}, self._CONVERTER_TAG
        )
        if blocked is not None:
            chunk: dict[str, Any] = {"promptFeedback": blocked}
            if context is not None:
                usage = context.pop_pending_usage()
                if usage is not None:
                    chunk["usageMetadata"] = self._build_stream_usage_metadata(usage)
            return [self._inject_stream_metadata(chunk, context)]

        chunks: list[dict[str, Any]] = []

        # Merge deferred text and tool calls into the finish chunk's
        # parts array, matching Google's native format where a single
        # candidate carries content parts alongside finishReason.
        parts: list[dict[str, Any]] = []
        if context is not None and context.pending_text is not None:
            parts.append({"text": context.pending_text})
            context.pending_text = None

        # A refusal whose words arrived only on the finish event has no part
        # yet.  Google carries it as text, so write it as one: a reader then
        # sees the words rather than a terminal state with nothing behind it
        # (#1314).
        refusal_text = (event.get("provider_metadata") or {}).get("refusal_text")
        if reason == "refusal" and refusal_text:
            parts.append({"text": refusal_text})

        if context is not None:
            parts.extend(self._flushed_tool_parts(context))

        # Gemini 3 requires a thought signature on replayed function-call
        # history; inject the documented dummy when the real one is absent.
        inject_google_function_call_signature(parts)

        event_metadata = event.get("provider_metadata") or {}
        candidate: dict[str, Any] = {
            "index": choice_index,
            "content": {"role": "model", "parts": parts},
            "finishReason": _google_finish_reason(
                reason,
                native=event_metadata,
                base=self._CONVERTER_TAG,
                response_id=context.response_id if context is not None else "",
                metadata=event_metadata,
            ),
        }
        if context is not None:
            self._attach_stream_grounding(candidate, context)
        finish_chunk: dict[str, Any] = {"candidates": [candidate]}

        # Merge buffered usage into the finish chunk so that
        # finishReason and usageMetadata stay in a single chunk,
        # matching the original Google format.
        if context is not None:
            usage = context.pop_pending_usage()
        else:
            usage = None
        if usage is not None:
            finish_chunk["usageMetadata"] = self._build_stream_usage_metadata(usage)

        chunks.append(self._inject_stream_metadata(finish_chunk, context))

        return chunks

    def _handle_ir_citation_delta_to_p(
        self, event: CitationDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Hold a streamed citation until the finish chunk can carry it.

        Gemini has no citation delta: the sources belong to the candidate's
        ``groundingMetadata``, which is why the reader emits them where Google
        reports them and the writer collects them back into the final chunk.
        """
        if context is None:
            report_dropped_citation(
                event.get("citation") or {}, why="no stream context to hold it"
            )
            return {}
        held = context.metadata.setdefault("_google_grounding_citations_out", [])
        held.append(event["citation"])
        return {}

    def _flushed_tool_parts(self, context: StreamContext) -> list[dict[str, Any]]:
        """The tool calls held during the stream, as finish-chunk parts."""
        parts: list[dict[str, Any]] = []
        for call_id, tool_name, args_str in context.get_pending_tool_calls():
            if context.get_tool_type(call_id) == "intrinsic":
                provider_metadata = (
                    context.get_tool_call_provider_metadata(call_id) or {}
                )
                intrinsic_part = self._flushed_intrinsic_part(
                    context.get_tool_call_intrinsic_kind(call_id),
                    provider_metadata,
                    args_str,
                    tool_name=tool_name,
                    tool_call_id=call_id,
                )
                if intrinsic_part is not None:
                    parts.append(intrinsic_part)
                continue
            try:
                args = json.loads(args_str) if args_str else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            fc: dict[str, Any] = {"name": tool_name, "args": args}
            if call_id:
                fc["id"] = sanitize_tool_call_id(call_id)
            flushed_part: dict[str, Any] = {"functionCall": fc}
            thought_sig = google_thought_signature(
                {
                    "provider_metadata": context.get_tool_call_provider_metadata(
                        call_id
                    )
                    or {}
                }
            )
            if thought_sig:
                flushed_part["thoughtSignature"] = thought_sig
            parts.append(flushed_part)
        return parts

    @staticmethod
    def _attach_stream_grounding(
        candidate: dict[str, Any], context: StreamContext
    ) -> None:
        """Hang the citations held during the stream on the finish candidate.

        Gemini has no citation delta, so the writer collects them and rebuilds
        the candidate's ``groundingMetadata``, the inverse of the reader.
        """
        held = context.metadata.pop("_google_grounding_citations_out", [])
        if not held:
            return
        grounding = _grounding_metadata_from_ir(held, {}, warnings=None)
        if grounding:
            candidate["groundingMetadata"] = grounding

    def _handle_ir_usage_to_p(
        self, event: UsageEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle UsageEvent → buffer for FinishEvent merge.

        When context is provided, buffers usage in pending_usage so
        FinishEvent can emit a single combined chunk with both
        finishReason and usageMetadata, matching the original Google
        format and preventing round-trip inflation.
        """
        usage = event["usage"]
        if context is not None:
            context.buffer_usage(usage)
            return {}
        return {"usageMetadata": self._build_stream_usage_metadata(usage)}
