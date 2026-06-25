"""Conversion pipeline — orchestrates format conversion between LLM APIs.

This module provides two layers of API:

**High-level** — :class:`ConversionPipeline` class that encapsulates the
full conversion lifecycle (Phase 1→2→4).  Use this when you need
request conversion, response conversion, and/or streaming:

    pipeline = ConversionPipeline("openai_chat", "anthropic", shim="argo--anthropic")
    target_body = pipeline.convert_request(body)
    # ... transport sends target_body, receives upstream_response ...
    source_response = pipeline.convert_response(upstream_response)

**Low-level** — :func:`apply_ir_transforms` and the functions in
:mod:`llm_rosetta.capabilities` for finer control over individual stages.

The pipeline is part of the core library — **no network dependency**.
It produces a target request body and consumes a target response body;
the caller (gateway, argo-proxy, etc.) owns the transport.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from llm_rosetta.capabilities import enforce_reasoning, enforce_vision
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.shims.provider_shim import ProviderShim, resolve_shim
from llm_rosetta.shims.transforms import (
    Transform,
    TransformContext,
    apply_ir_transforms as _apply_ir_transforms_exec,
    apply_transforms,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_context(
    ctx: ConversionContext,
    shim: ProviderShim | str | None,
    *,
    model: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> None:
    """Deprecated: use :func:`llm_rosetta.capabilities.enforce_reasoning`."""
    import warnings

    warnings.warn(
        "configure_context is deprecated; use capabilities.enforce_reasoning()",
        DeprecationWarning,
        stacklevel=2,
    )
    enforce_reasoning(ctx, shim, model=model, config_override=config_override)


def apply_ir_transforms(
    ir_request: dict[str, Any],
    shim: ProviderShim | str | None,
    *,
    upstream_model: str | None = None,
    model_capabilities: list[str] | None = None,
    request_id: str = "-",
) -> dict[str, Any]:
    """Apply all shim-driven IR-level transforms.

    Builds a :class:`~llm_rosetta.shims.transforms.TransformContext` from
    the provided parameters and runs the shim's ``ir_transforms`` tuple
    through :func:`~llm_rosetta.shims.transforms.apply_ir_transforms`.

    Args:
        ir_request: The IR request dict.  Some operations mutate in-place,
            others return a new dict — **always use the return value**.
        shim: ProviderShim instance, registered name, or None (no-op).
        upstream_model: The upstream model ID (for pattern matching).
        model_capabilities: Model capability list (e.g. ``["text", "vision"]``).
            When ``None``, transforms that check capabilities treat the
            model as unknown and skip capability-dependent operations.
        request_id: Request identifier for logging.

    Returns:
        The IR request dict after all applicable transforms.  Always
        assign the return value: ``ir = apply_ir_transforms(ir, shim, ...)``.
    """
    resolved = resolve_shim(shim)
    if resolved is None or not resolved.ir_transforms:
        return ir_request

    ctx = TransformContext(
        model=upstream_model or "",
        model_capabilities=model_capabilities,
        request_id=request_id,
    )
    return _apply_ir_transforms_exec(resolved.ir_transforms, ir_request, ctx)


# ---------------------------------------------------------------------------
# Deprecated aliases (backward compatibility with v0.6.x)
# ---------------------------------------------------------------------------


def setup_shim_context(*args: Any, **kwargs: Any) -> None:
    """Deprecated: use :func:`configure_context`."""
    import warnings

    warnings.warn(
        "setup_shim_context is deprecated; use configure_context()",
        DeprecationWarning,
        stacklevel=2,
    )
    return configure_context(*args, **kwargs)


def apply_shim_to_ir(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Deprecated: use :func:`apply_ir_transforms`."""
    import warnings

    warnings.warn(
        "apply_shim_to_ir is deprecated; use apply_ir_transforms()",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_ir_transforms(*args, **kwargs)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConversionError(Exception):
    """Raised when a conversion phase fails.

    Attributes:
        phase: Which pipeline phase failed (``"source_to_ir"``,
            ``"ir_to_target"``, ``"response_to_ir"``,
            ``"ir_to_source"``).
    """

    def __init__(self, message: str, phase: str) -> None:
        self.phase = phase
        super().__init__(message)


# ---------------------------------------------------------------------------
# ConversionPipeline
# ---------------------------------------------------------------------------

_EMPTY_TRANSFORMS: tuple[Transform, ...] = ()


class ConversionPipeline:
    """Orchestrates format conversion between LLM API standards.

    Owns Phase 1 (Source→IR), Phase 2 (IR adapt + IR→Target), and
    Phase 4 (Response→Source).  Phase 3 (upstream forwarding) is the
    caller's responsibility.

    Usage::

        pipeline = ConversionPipeline("openai_chat", "anthropic",
                                      shim="argo--anthropic")

        # Phase 1+2: request conversion
        target_body = pipeline.convert_request(body)

        # Phase 3: caller forwards target_body to upstream

        # Phase 4: response conversion
        source_response = pipeline.convert_response(upstream_json)

    For streaming, call :meth:`create_stream_processor` after
    :meth:`convert_request` to get a stateful chunk converter.

    Args:
        source_provider: Client API format (e.g. ``"openai_chat"``).
        target_provider: Upstream API format (e.g. ``"anthropic"``).
        shim: Provider shim instance, registered name, or ``None``.
        upstream_model: The upstream model ID (for shim pattern matching).
        model_capabilities: Model capability list (e.g. ``["text", "vision"]``).
        reasoning_config_override: External reasoning override (e.g. admin UI).
    """

    def __init__(
        self,
        source_provider: str,
        target_provider: str,
        shim: ProviderShim | str | None = None,
        *,
        upstream_model: str | None = None,
        model_capabilities: list[str] | None = None,
        reasoning_config_override: dict[str, Any] | None = None,
    ) -> None:
        from llm_rosetta import get_converter_for_provider

        self._source_provider = source_provider
        self._target_provider = target_provider
        self._shim = shim
        self._upstream_model = upstream_model
        self._model_capabilities = model_capabilities
        self._reasoning_config_override = reasoning_config_override

        self._source_converter = get_converter_for_provider(source_provider)
        self._target_converter = get_converter_for_provider(target_provider)

        # Resolve body-level transforms from shim
        resolved = resolve_shim(shim)
        if resolved is not None:
            self._from_transforms = resolved.from_transforms
            self._to_transforms = resolved.to_transforms
        else:
            self._from_transforms = _EMPTY_TRANSFORMS
            self._to_transforms = _EMPTY_TRANSFORMS

        # Set after convert_request()
        self._ctx: ConversionContext | None = None
        self._ir_request: dict[str, Any] | None = None

    @property
    def context(self) -> ConversionContext:
        """The request-phase conversion context.

        Available after :meth:`convert_request` has been called.

        Raises:
            RuntimeError: If called before :meth:`convert_request`.
        """
        if self._ctx is None:
            raise RuntimeError(
                "context is not available until convert_request() is called"
            )
        return self._ctx

    @property
    def ir_request(self) -> dict[str, Any]:
        """The IR request produced by the last :meth:`convert_request` call.

        Useful for logging and metadata store injection.

        Raises:
            RuntimeError: If called before :meth:`convert_request`.
        """
        if self._ir_request is None:
            raise RuntimeError(
                "ir_request is not available until convert_request() is called"
            )
        return self._ir_request

    @property
    def warnings(self) -> list[str]:
        """Conversion warnings accumulated during the pipeline.

        Returns an empty list if :meth:`convert_request` hasn't been called.
        """
        if self._ctx is None:
            return []
        return self._ctx.warnings

    def convert_request(
        self,
        body: dict[str, Any],
        *,
        on_ir_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Convert a source-format request body to target format.

        Executes Phase 1 (Source→IR) and Phase 2 (IR adapt + IR→Target).

        Args:
            body: Source-format request body.
            on_ir_ready: Optional callback invoked after Source→IR
                conversion, before shim IR transforms.  Use this to
                inject cached metadata (e.g.
                ``store.inject_into_request``).

        Returns:
            Target-format request body ready for transport.

        Raises:
            ConversionError: If source→IR or IR→target conversion fails.
            RuntimeError: If called more than once on the same instance.
                Create a new ``ConversionPipeline`` per request.
        """
        if self._ctx is not None:
            raise RuntimeError(
                "convert_request() already called on this pipeline instance. "
                "ConversionPipeline is one-shot — create a new instance per request."
            )

        # Setup context
        ctx = ConversionContext()
        ctx.options["metadata_mode"] = "preserve"
        if self._target_provider == "google":
            ctx.options["output_format"] = "rest"

        # Capability enforcement: reasoning (pre-IR)
        enforce_reasoning(
            ctx,
            self._shim,
            model=self._upstream_model or body.get("model"),
            config_override=self._reasoning_config_override,
        )
        self._ctx = ctx

        # Phase 1: Source → IR
        try:
            ir_request = self._source_converter.request_from_provider(body, context=ctx)
        except Exception as exc:
            raise ConversionError(
                f"Failed to parse request: {exc}", phase="source_to_ir"
            ) from exc

        # Hook: let caller inject metadata before IR transforms
        if on_ir_ready is not None:
            on_ir_ready(ir_request)

        request_id = ctx.options.get("request_id", "-")

        # Capability enforcement: vision (post-IR)
        ir_request = enforce_vision(
            ir_request,
            model_capabilities=self._model_capabilities,
            model=self._upstream_model or body.get("model") or "",
            request_id=request_id,
        )

        # Phase 2a: Shim-driven IR transforms
        ir_request = apply_ir_transforms(
            ir_request,
            self._shim,
            upstream_model=self._upstream_model or body.get("model"),
            model_capabilities=self._model_capabilities,
            request_id=request_id,
        )
        self._ir_request = ir_request

        # Phase 2b: IR → Target
        try:
            target_body, _ = self._target_converter.request_to_provider(
                ir_request, context=ctx
            )
        except Exception as exc:
            raise ConversionError(
                f"Conversion error: {exc}", phase="ir_to_target"
            ) from exc

        # Phase 2c: Body-level shim to_transforms
        if self._to_transforms:
            target_body = apply_transforms(self._to_transforms, target_body)

        return target_body

    def convert_response(
        self,
        upstream_response: dict[str, Any],
        *,
        on_ir_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Convert a target-format response body back to source format.

        Executes Phase 4 (from_transforms → Target→IR → IR→Source).

        Must be called after :meth:`convert_request` (uses the same
        conversion context).

        Args:
            upstream_response: Target-format response body from upstream.
            on_ir_ready: Optional callback invoked after Target→IR
                conversion.  Use this to cache metadata (e.g.
                ``store.cache_from_response``).

        Returns:
            Source-format response body.

        Raises:
            ConversionError: If response conversion fails.
            RuntimeError: If called before :meth:`convert_request`.
        """
        ctx = self.context  # raises RuntimeError if not ready

        # Phase 4a: Body-level shim from_transforms
        response = upstream_response
        if self._from_transforms:
            response = apply_transforms(self._from_transforms, response)

        # Phase 4b: Target response → IR
        try:
            ir_response = self._target_converter.response_from_provider(
                response, context=ctx
            )
        except Exception as exc:
            raise ConversionError(
                f"Failed to parse upstream response: {exc}",
                phase="response_to_ir",
            ) from exc

        # Hook: let caller cache metadata from IR response
        if on_ir_ready is not None:
            on_ir_ready(ir_response)

        # Phase 4c: IR → Source response
        try:
            source_response = self._source_converter.response_to_provider(
                ir_response, context=ctx
            )
        except Exception as exc:
            raise ConversionError(
                f"Failed to convert response: {exc}", phase="ir_to_source"
            ) from exc

        return source_response

    def create_stream_processor(
        self,
        *,
        on_ir_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> StreamProcessor:
        """Create a stateful processor for streaming response chunks.

        Must be called after :meth:`convert_request`.  The returned
        :class:`StreamProcessor` converts upstream chunks one at a time,
        maintaining state (tool call tracking, usage accumulation, etc.)
        across calls.

        Args:
            on_ir_event: Optional callback invoked for each IR event
                produced from an upstream chunk.  Use this to cache
                streaming metadata (e.g. ``store.cache_from_stream_event``).

        Returns:
            A new StreamProcessor bound to this pipeline's converters
            and context.

        Raises:
            RuntimeError: If called before :meth:`convert_request`.
        """
        ctx = self.context  # raises RuntimeError if not ready

        from_ctx = self._target_converter.create_stream_context()
        to_ctx = self._source_converter.create_stream_context()

        # Bridge preserve-mode metadata from request phase
        to_ctx.options["metadata_mode"] = "preserve"
        from_ctx.options["metadata_mode"] = "preserve"
        if "_request_echo" in ctx.metadata:
            to_ctx.metadata["_request_echo"] = ctx.metadata["_request_echo"]

        return StreamProcessor(
            target_converter=self._target_converter,
            source_converter=self._source_converter,
            from_ctx=from_ctx,
            to_ctx=to_ctx,
            from_transforms=self._from_transforms,
            on_ir_event=on_ir_event,
        )


# ---------------------------------------------------------------------------
# StreamProcessor
# ---------------------------------------------------------------------------


class StreamProcessor:
    """Stateful per-chunk converter for streaming responses.

    Created by :meth:`ConversionPipeline.create_stream_processor`.
    Converts upstream response chunks one at a time, maintaining
    :class:`~llm_rosetta.converters.base.context.StreamContext` state
    across calls.

    Each call to :meth:`process_chunk` returns a list of source-format
    event dicts (NOT formatted SSE strings — SSE formatting is the
    transport's responsibility).

    Args:
        target_converter: The upstream format converter.
        source_converter: The client format converter.
        from_ctx: StreamContext for upstream→IR conversion.
        to_ctx: StreamContext for IR→source conversion.
        from_transforms: Shim from_transforms to apply before conversion.
        on_ir_event: Optional callback for each IR event.
    """

    def __init__(
        self,
        *,
        target_converter: Any,
        source_converter: Any,
        from_ctx: Any,
        to_ctx: Any,
        from_transforms: tuple[Transform, ...] = (),
        on_ir_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._target_converter = target_converter
        self._source_converter = source_converter
        self._from_ctx = from_ctx
        self._to_ctx = to_ctx
        self._from_transforms = from_transforms
        self._on_ir_event = on_ir_event

    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert one upstream chunk to source-format events.

        Args:
            chunk: A parsed upstream response chunk (e.g. from SSE).

        Returns:
            List of source-format event dicts.  May be empty (some
            upstream chunks produce no source events), one, or multiple.
        """
        # Apply shim from_transforms
        if self._from_transforms:
            chunk = apply_transforms(self._from_transforms, chunk)

        # Target → IR events
        ir_events = self._target_converter.stream_response_from_provider(
            chunk, context=self._from_ctx
        )

        # Bridge response extras
        if "_response_extras" in self._from_ctx.metadata:
            self._to_ctx.metadata["_response_extras"] = self._from_ctx.metadata[
                "_response_extras"
            ]

        # IR → Source events
        result: list[dict[str, Any]] = []
        for ir_event in ir_events:
            if self._on_ir_event is not None:
                self._on_ir_event(ir_event)

            source_chunks = self._source_converter.stream_response_to_provider(
                ir_event, context=self._to_ctx
            )
            if isinstance(source_chunks, list):
                result.extend(sc for sc in source_chunks if sc)
            elif source_chunks:
                result.append(source_chunks)

        return result
