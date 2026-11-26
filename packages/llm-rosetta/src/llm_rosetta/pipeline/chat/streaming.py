"""Stream processors: stateful chunk-by-chunk response conversion."""

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol, runtime_checkable

from llm_rosetta.ir.passes import (
    ToolNameMap,
    restore_client_tool_names,
    CustomToolRestorer,
)
from llm_rosetta.conversion_warnings import recording_to
from llm_rosetta.contracts.usage.semantics import GOOGLE_METADATA, USAGE_SHAPES
from llm_rosetta.ir.primitives.body import Transform, apply_transforms

logger = logging.getLogger("llm_rosetta.pipeline.chat")


@runtime_checkable
class StreamProcessorProtocol(Protocol):
    """Shared interface for StreamProcessor and BaselineStreamProcessor."""

    @property
    def source_context(self) -> Any: ...

    def process_chunk(self, chunk: dict[str, Any] | None) -> list[dict[str, Any]]: ...

    def iter_events(
        self, chunks: Iterable[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]: ...


class BaselineStreamProcessor:
    """No-op stream processor for same-format baseline pipelines.

    Forwards upstream chunks directly, applying only shim body-level
    transforms.  Used for fidelity shadow-testing, not production.
    Has a dummy ``source_context`` so the gateway's terminal-event
    logic doesn't crash.
    """

    def __init__(
        self,
        *,
        response_body_transforms: tuple[Transform, ...] = (),
        pre_ir_transforms: tuple[Transform, ...] = (),
        post_ir_transforms: tuple[Transform, ...] = (),
    ) -> None:
        self._response_body_transforms = response_body_transforms
        self._pre_ir_transforms = pre_ir_transforms
        self._post_ir_transforms = post_ir_transforms
        self._ctx = self._BaselineCtx()
        self._usage: dict[str, int] | None = None

    @property
    def source_context(self) -> Any:
        """Minimal context for baseline mode — enough for gateway error handling."""
        return self._ctx

    class _BaselineCtx:
        def __init__(self) -> None:
            self.is_ended = False
            self.response_id = ""
            self.options: dict[str, Any] = {}

        def mark_ended(self) -> None:
            self.is_ended = True

        @property
        def next_sequence_number(self) -> None:
            return None

        @property
        def outbound_response_id(self) -> str:
            return self.response_id

    _TERMINAL_TYPES = frozenset(
        {"response.completed", "response.failed", "message_stop"}
    )

    def get_accumulated_usage(self) -> dict[str, int] | None:
        """Return token usage extracted from baseline chunks."""
        return self._usage

    @staticmethod
    def _extract_int(d: dict[str, Any], key: str) -> int | None:
        v = d.get(key)
        return v if isinstance(v, int) and v >= 0 else None

    @classmethod
    def _extract_nested_int(
        cls, parent: dict[str, Any], parent_key: str, child_key: str
    ) -> int | None:
        sub = parent.get(parent_key)
        return cls._extract_int(sub, child_key) if isinstance(sub, dict) else None

    @classmethod
    def _extract_extended_usage(
        cls, usage: dict[str, Any], result: dict[str, int]
    ) -> None:
        """Extract cache/reasoning tokens from a usage payload.

        Which fields carry them, and whether the cache ones add to the input or
        sit inside it, is declared in
        :mod:`llm_rosetta.contracts.usage.semantics` — the same declaration
        the total's derivation and the historical repair read, so the three
        cannot disagree.  Every shape is tried because a payload does not say
        which dialect produced it.
        """
        for shape in USAGE_SHAPES:
            result.update(shape.extended(usage))

    @classmethod
    def _extract_usage(cls, chunk: dict[str, Any]) -> dict[str, int] | None:
        """Extract usage from a raw provider chunk (any format)."""
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            result: dict[str, int] = {}
            for src, dst in (
                ("prompt_tokens", "prompt_tokens"),
                ("input_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("output_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                v = usage.get(src)
                if isinstance(v, int) and v >= 0:
                    result[dst] = v
            if "total_tokens" not in result and "prompt_tokens" in result:
                result["total_tokens"] = result.get("prompt_tokens", 0) + result.get(
                    "completion_tokens", 0
                )
            cls._extract_extended_usage(usage, result)
            return result if result else None
        # Google format
        um = chunk.get("usageMetadata") or chunk.get("usage_metadata")
        if isinstance(um, dict):
            result = {}
            for src, dst in (
                ("promptTokenCount", "prompt_tokens"),
                ("prompt_token_count", "prompt_tokens"),
                ("candidatesTokenCount", "completion_tokens"),
                ("candidates_token_count", "completion_tokens"),
                ("totalTokenCount", "total_tokens"),
                ("total_token_count", "total_tokens"),
            ):
                v = um.get(src)
                if isinstance(v, int) and v >= 0:
                    result[dst] = v
            result.update(GOOGLE_METADATA.extended(um))
            return result if result else None
        return None

    def process_chunk(self, chunk: dict[str, Any] | None) -> list[dict[str, Any]]:
        if chunk is None:
            return []
        if self._response_body_transforms:
            chunk = apply_transforms(self._response_body_transforms, chunk)
        if self._pre_ir_transforms:
            chunk = apply_transforms(self._pre_ir_transforms, chunk)
        if self._post_ir_transforms:
            chunk = apply_transforms(self._post_ir_transforms, chunk)
        # Detect terminal events so source_context.is_ended reflects reality
        ctype = chunk.get("type", "")
        choices = chunk.get("choices", [])
        if ctype in self._TERMINAL_TYPES or (
            choices and choices[0].get("finish_reason") is not None
        ):
            self._ctx.mark_ended()
        u = self._extract_usage(chunk)
        if u is not None:
            self._usage = u
        return [chunk]

    def iter_events(self, chunks: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Iterate over upstream chunks and yield source-format events.

        Equivalent to calling :meth:`process_chunk` for each chunk
        followed by a final ``process_chunk(None)`` to flush any
        deferred terminal events.  Baseline passthrough has no
        deferred state, but the final flush is included for
        interface consistency.

        Args:
            chunks: Iterable of parsed upstream response chunks.

        Yields:
            Source-format event dicts.
        """
        for chunk in chunks:
            yield from self.process_chunk(chunk)
        yield from self.process_chunk(None)


class StreamProcessor:
    """Stateful per-chunk converter for streaming responses.

    Created by :meth:`ConversionPipeline.create_stream_processor`.
    Converts upstream response chunks one at a time, maintaining
    :class:`~llm_rosetta.ir.context.StreamContext` state
    across calls.

    Each call to :meth:`process_chunk` returns a list of source-format
    event dicts (NOT formatted SSE strings — SSE formatting is the
    transport's responsibility).

    Args:
        target_converter: The upstream format converter.
        source_converter: The client format converter.
        from_ctx: StreamContext for upstream→IR conversion.
        to_ctx: StreamContext for IR→source conversion.
        pre_ir_transforms: Shim pre_ir_transforms to apply before conversion.
        custom_tool_names: Names of tools downgraded from custom to
            function by :func:`~llm_rosetta.ir.passes.enforce_custom_tools`.
        name_map: Bidirectional map for tools renamed on the request leg
            (namespace flattening, collision qualification).
        on_ir_event: Optional callback for each IR event.
    """

    def __init__(
        self,
        *,
        target_converter: Any,
        source_converter: Any,
        from_ctx: Any,
        to_ctx: Any,
        response_body_transforms: tuple[Transform, ...] = (),
        pre_ir_transforms: tuple[Transform, ...] = (),
        post_ir_transforms: tuple[Transform, ...] = (),
        custom_tool_names: frozenset[str] = frozenset(),
        name_map: ToolNameMap | None = None,
        on_ir_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._target_converter = target_converter
        self._source_converter = source_converter
        self._from_ctx = from_ctx
        self._to_ctx = to_ctx
        self._response_body_transforms = response_body_transforms
        self._pre_ir_transforms = pre_ir_transforms
        self._post_ir_transforms = post_ir_transforms
        self._custom_tools = CustomToolRestorer(custom_tool_names, context=from_ctx)
        self._name_map = name_map or ToolNameMap()
        self._on_ir_event = on_ir_event
        self._usage: dict[str, int] | None = None

    @property
    def source_context(self) -> Any:
        """The IR→source StreamContext.

        Exposed so transports can inspect stream state (e.g. whether the
        stream already ended, the assigned response ID) when synthesizing
        a terminal event after an upstream failure.
        """
        return self._to_ctx

    def get_accumulated_usage(self) -> dict[str, int] | None:
        """Return token usage accumulated during streaming.

        Captured from IR usage events as they flow through the
        processor.  Returns ``None`` if no usage event was received.
        """
        return self._usage

    def _merge_usage(self, u: dict[str, Any]) -> None:
        """Merge an IR usage dict into accumulated usage.

        The first usage event (typically from ``message_start``) seeds
        the accumulator; subsequent events (e.g. ``message_delta``)
        merge non-zero values so that fields like ``prompt_tokens``
        from earlier events are not overwritten by a later event that
        only carries ``completion_tokens``.
        """
        if self._usage is None:
            self._usage = dict(u)
        else:
            for k, v in u.items():
                if v is not None and v != 0:
                    self._usage[k] = v

    def _flush_deferred(self) -> list[dict[str, Any]]:
        """Flush deferred terminal events from the target converter.

        Called when the upstream stream ends (``process_chunk(None)``)
        to emit any ``StreamEndEvent`` that was held back while
        waiting for a usage chunk that never arrived.
        """
        with recording_to(self._from_ctx.warnings):
            ir_events = self._target_converter.stream_response_from_provider(
                None, context=self._from_ctx
            )
        result: list[dict[str, Any]] = []
        for ir_event in ir_events:
            if self._on_ir_event is not None:
                self._on_ir_event(ir_event)
            with recording_to(self._to_ctx.warnings):
                source_chunks = self._source_converter.stream_response_to_provider(
                    ir_event, context=self._to_ctx
                )
            if isinstance(source_chunks, list):
                result.extend(sc for sc in source_chunks if sc)
            elif source_chunks:
                result.append(source_chunks)
        if self._post_ir_transforms and result:
            result = [apply_transforms(self._post_ir_transforms, c) for c in result]
        return result

    def process_chunk(self, chunk: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Convert one upstream chunk to source-format events.

        Pass ``None`` to signal end-of-stream.  This flushes any
        deferred terminal events from the target converter (e.g. a
        ``StreamEndEvent`` held back while waiting for a usage chunk
        that never arrived).

        Args:
            chunk: A parsed upstream response chunk (e.g. from SSE),
                or ``None`` to signal EOF.

        Returns:
            List of source-format event dicts.  May be empty (some
            upstream chunks produce no source events), one, or multiple.
        """
        if chunk is None:
            return self._flush_deferred()

        # Apply response body transforms (e.g. harmony tool-call rewrite)
        if self._response_body_transforms:
            chunk = apply_transforms(self._response_body_transforms, chunk)
        # Apply shim pre_ir_transforms
        if self._pre_ir_transforms:
            chunk = apply_transforms(self._pre_ir_transforms, chunk)

        # Target → IR events
        with recording_to(self._from_ctx.warnings):
            ir_events = self._target_converter.stream_response_from_provider(
                chunk, context=self._from_ctx
            )

        # Bridge response extras
        if "_response_extras" in self._from_ctx.metadata:
            self._to_ctx.metadata["_response_extras"] = self._from_ctx.metadata[
                "_response_extras"
            ]

        # Restore custom tool types for downgraded tools
        ir_events = self._custom_tools.restore_events(ir_events)

        # Undo any request-leg tool renaming before events reach the client
        restore_client_tool_names(ir_events, name_map=self._name_map)

        # IR → Source events
        result: list[dict[str, Any]] = []
        for ir_event in ir_events:
            if ir_event.get("type") == "usage":
                u = ir_event.get("usage")
                if u:
                    self._merge_usage(u)

            if self._on_ir_event is not None:
                self._on_ir_event(ir_event)

            with recording_to(self._to_ctx.warnings):
                source_chunks = self._source_converter.stream_response_to_provider(
                    ir_event, context=self._to_ctx
                )
            if isinstance(source_chunks, list):
                result.extend(sc for sc in source_chunks if sc)
            elif source_chunks:
                result.append(source_chunks)

        # Apply source shim post_ir_transforms to outbound chunks
        if self._post_ir_transforms and result:
            result = [apply_transforms(self._post_ir_transforms, c) for c in result]

        return result

    def iter_events(self, chunks: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Iterate over upstream chunks and yield source-format events.

        Equivalent to calling :meth:`process_chunk` for each chunk
        followed by a final ``process_chunk(None)`` to flush any
        deferred terminal events (e.g. a ``StreamEndEvent`` held back
        while waiting for a usage chunk that never arrived).

        This is the recommended entry point for streaming conversion —
        it guarantees that the stream is properly finalized regardless
        of whether the upstream sent usage data.

        Args:
            chunks: Iterable of parsed upstream response chunks.

        Yields:
            Source-format event dicts.
        """
        for chunk in chunks:
            yield from self.process_chunk(chunk)
        yield from self.process_chunk(None)
