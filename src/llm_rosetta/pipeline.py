"""Conversion pipeline — orchestrates format conversion between LLM APIs.

This module provides two layers of API:

**High-level** — :class:`ConversionPipeline` class that encapsulates the
full conversion lifecycle (Phase 1→2→4).  Use this when you need
request conversion, response conversion, and/or streaming:

    pipeline = ConversionPipeline("openai_chat", "anthropic", target_shim="argo--anthropic")
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
import time
import warnings
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from llm_rosetta.capabilities import (
    enforce_custom_tools,
    enforce_reasoning,
    enforce_vision,
    get_custom_tool_names,
    restore_custom_tool_calls,
    unwrap_custom_tool_input,
)
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
    hoist_system_messages: bool = True,
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
        hoist_system_messages=hoist_system_messages,
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


@runtime_checkable
class StreamProcessorProtocol(Protocol):
    """Shared interface for StreamProcessor and PassthroughStreamProcessor."""

    @property
    def source_context(self) -> Any: ...

    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]: ...


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

    Transform ordering (diamond flow)::

        Request path:
          source_shim.pre_ir → Source→IR → [enforce/ir_transforms] →
          IR→Target → target_shim.post_ir

        Response path (mirror):
          target_shim.pre_ir → Target→IR → IR→Source →
          source_shim.post_ir

    In passthrough mode (source == target, force_conversion=False),
    the IR round-trip is skipped but shim body-level transforms still
    apply: source pre_ir → target post_ir (request) and target pre_ir
    → source post_ir (response).

    Args:
        source_provider: Client API format (e.g. ``"openai_chat"``).
        target_provider: Upstream API format (e.g. ``"anthropic"``).
        target_shim: Provider shim for the upstream/target side.
        source_shim: Provider shim for the client/source side.
        upstream_model: The upstream model ID (for shim pattern matching).
        model_capabilities: Model capability list (e.g. ``["text", "vision"]``).
        reasoning_config_override: External reasoning override (e.g. admin UI).
    """

    def __init__(
        self,
        source_provider: str,
        target_provider: str,
        target_shim: ProviderShim | str | None = None,
        *,
        source_shim: ProviderShim | str | None = None,
        shim: ProviderShim | str | None = None,
        upstream_model: str | None = None,
        model_capabilities: list[str] | None = None,
        reasoning_config_override: dict[str, Any] | None = None,
        supports_custom_tools: bool = False,
        hoist_system_messages: bool = True,
        force_conversion: bool = True,
        fidelity_mode: Literal["critical", "full"] | None = None,
        metadata_mode: str = "preserve",
        google_output_format: str = "rest",
    ) -> None:
        from llm_rosetta import get_converter_for_provider

        # Backward compat: accept legacy ``shim=`` as alias for target_shim
        if shim is not None:
            if target_shim is not None:
                raise ValueError(
                    "Cannot pass both 'shim' and 'target_shim'. "
                    "Use 'target_shim' (shim is deprecated)."
                )
            warnings.warn(
                "ConversionPipeline(shim=...) is deprecated, use target_shim instead",
                DeprecationWarning,
                stacklevel=2,
            )
            target_shim = shim

        self._source_provider = source_provider
        self._target_provider = target_provider
        self._target_shim = target_shim
        self._upstream_model = upstream_model
        self._model_capabilities = model_capabilities
        self._reasoning_config_override = reasoning_config_override
        self._supports_custom_tools = supports_custom_tools
        self._hoist_system_messages = hoist_system_messages
        self._metadata_mode = metadata_mode
        self._google_output_format = google_output_format

        self._passthrough = source_provider == target_provider and not force_conversion
        self._fidelity: Any = None
        if self._passthrough and fidelity_mode is not None:
            from llm_rosetta.fidelity import FidelityChecker

            self._fidelity = FidelityChecker(
                mode=fidelity_mode, format_name=source_provider
            )

        self._source_converter = get_converter_for_provider(source_provider)
        self._target_converter = get_converter_for_provider(target_provider)

        # Resolve body-level transforms from source shim
        resolved_source = resolve_shim(source_shim)
        if resolved_source is not None:
            self._source_pre_ir_transforms = resolved_source.pre_ir_transforms
            self._source_post_ir_transforms = resolved_source.post_ir_transforms
        else:
            self._source_pre_ir_transforms = _EMPTY_TRANSFORMS
            self._source_post_ir_transforms = _EMPTY_TRANSFORMS

        # Resolve body-level transforms from target shim
        resolved_target = resolve_shim(target_shim)
        if resolved_target is not None:
            self._target_pre_ir_transforms = resolved_target.pre_ir_transforms
            self._target_post_ir_transforms = resolved_target.post_ir_transforms
        else:
            self._target_pre_ir_transforms = _EMPTY_TRANSFORMS
            self._target_post_ir_transforms = _EMPTY_TRANSFORMS

        # Resolve response_id_prefix for each converter from its shim.
        resolved_target_for_prefix = resolved_target or resolve_shim(target_provider)
        self._target_id_prefix = (
            resolved_target_for_prefix.response_id_prefix
            if resolved_target_for_prefix
            else ""
        )
        self._multimodal_tool_result = (
            resolved_target_for_prefix.multimodal_tool_result
            if resolved_target_for_prefix
            else None
        )
        resolved_source_for_prefix = resolved_source or resolve_shim(source_provider)
        self._source_id_prefix = (
            resolved_source_for_prefix.response_id_prefix
            if resolved_source_for_prefix
            else ""
        )

        # Set after convert_request()
        self._ctx: ConversionContext | None = None
        self._ir_request: dict[str, Any] | None = None

        # Per-phase timing (always-on, ~30ns per perf_counter call)
        self._profile: dict[str, float] = {}

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

    @property
    def profile(self) -> dict[str, float]:
        """Per-phase timing data collected during conversion.

        Contains millisecond durations for each conversion sub-phase.
        Populated incrementally by :meth:`convert_request` and
        :meth:`convert_response`.  Always available (returns ``{}``
        before any conversion).

        Keys after :meth:`convert_request`::

            source_to_ir_ms      — Source format → IR parsing
            ir_transforms_ms     — Vision enforcement + shim IR transforms
            ir_to_target_ms      — IR → target format serialization
            body_transforms_ms   — Shim body-level post_ir_transforms
            request_conversion_ms — Total request conversion time

        Keys added by :meth:`convert_response`::

            response_from_target_ms — Target response → IR parsing
            response_to_source_ms   — IR → source response serialization
            response_conversion_ms  — Total response conversion time
        """
        return self._profile

    def _run_fidelity_check(self, body: dict[str, Any], *, direction: str) -> None:
        """Shadow round-trip and compare for fidelity monitoring."""
        try:
            ctx = ConversionContext()
            ctx.options["metadata_mode"] = "preserve"
            if direction == "request":
                ir = self._source_converter.request_from_provider(body, context=ctx)
                rt, _ = self._target_converter.request_to_provider(ir, context=ctx)
            else:
                ir = self._target_converter.response_from_provider(body, context=ctx)
                rt = self._source_converter.response_to_provider(ir, context=ctx)
            diffs = self._fidelity.compare(body, rt, direction=direction)
            if diffs:
                logger.warning(
                    "Fidelity loss in %s round-trip (%s→%s): %s",
                    direction,
                    self._source_provider,
                    self._target_provider,
                    "; ".join(str(d) for d in diffs[:10]),
                )
        except Exception:
            logger.debug("Fidelity check failed", exc_info=True)

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
        self._ctx = ctx

        # Same-format short-circuit: skip IR round-trip, apply only shim
        # body-level transforms.  This avoids lossy round-trips when
        # source == target (e.g. Anthropic → gateway → Anthropic upstream).
        if self._passthrough:
            t0 = time.perf_counter()
            result = body
            if self._source_pre_ir_transforms:
                result = apply_transforms(self._source_pre_ir_transforms, dict(result))
            if self._target_post_ir_transforms:
                result = apply_transforms(
                    self._target_post_ir_transforms,
                    dict(result) if result is body else result,
                )
            # No IR produced in passthrough mode
            self._ir_request = {}
            # Shadow round-trip for fidelity monitoring
            if self._fidelity is not None:
                self._run_fidelity_check(body, direction="request")
            self._profile["request_conversion_ms"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )
            return result
        ctx.options["metadata_mode"] = self._metadata_mode
        if self._target_provider == "google":
            ctx.options["output_format"] = self._google_output_format

        if self._multimodal_tool_result is not None:
            ctx.options["multimodal_tool_result"] = self._multimodal_tool_result

        # Capability enforcement: reasoning (pre-IR)
        enforce_reasoning(
            ctx,
            self._target_shim,
            model=self._upstream_model or body.get("model"),
            config_override=self._reasoning_config_override,
        )

        t_total = time.perf_counter()

        # Phase 0: Source shim pre_ir_transforms (normalise source dialect)
        if self._source_pre_ir_transforms:
            body = apply_transforms(self._source_pre_ir_transforms, body)

        # Phase 1: Source → IR
        t0 = time.perf_counter()
        try:
            ir_request = self._source_converter.request_from_provider(body, context=ctx)
        except Exception as exc:
            raise ConversionError(
                f"Failed to parse request: {exc}", phase="source_to_ir"
            ) from exc
        self._profile["source_to_ir_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Hook: let caller inject metadata before IR transforms
        if on_ir_ready is not None:
            on_ir_ready(ir_request)

        request_id = ctx.options.get("request_id", "-")

        # Capability enforcement: vision (post-IR) + shim IR transforms
        t0 = time.perf_counter()
        ir_request = enforce_vision(
            ir_request,
            model_capabilities=self._model_capabilities,
            model=self._upstream_model or body.get("model") or "",
            request_id=request_id,
        )

        # Capability enforcement: custom tools (post-IR)
        ir_request = enforce_custom_tools(
            ir_request,
            shim=self._target_shim,
            config_override=self._supports_custom_tools,
        )

        # Phase 2a: Shim-driven IR transforms
        ir_request = apply_ir_transforms(
            ir_request,
            self._target_shim,
            upstream_model=self._upstream_model or body.get("model"),
            model_capabilities=self._model_capabilities,
            request_id=request_id,
            hoist_system_messages=self._hoist_system_messages,
        )
        self._profile["ir_transforms_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self._ir_request = ir_request

        # Phase 2b: IR → Target
        t0 = time.perf_counter()
        try:
            target_body, _ = self._target_converter.request_to_provider(
                ir_request, context=ctx
            )
        except Exception as exc:
            raise ConversionError(
                f"Conversion error: {exc}", phase="ir_to_target"
            ) from exc
        self._profile["ir_to_target_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Phase 2c: Body-level target shim post_ir_transforms
        t0 = time.perf_counter()
        if self._target_post_ir_transforms:
            target_body = apply_transforms(self._target_post_ir_transforms, target_body)
        self._profile["body_transforms_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        self._profile["request_conversion_ms"] = round(
            (time.perf_counter() - t_total) * 1000, 2
        )
        return target_body

    def convert_response(
        self,
        upstream_response: dict[str, Any],
        *,
        on_ir_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Convert a target-format response body back to source format.

        Executes Phase 4 (pre_ir_transforms → Target→IR → IR→Source).

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

        # Same-format short-circuit: skip IR round-trip for response too
        if self._passthrough:
            t0 = time.perf_counter()
            result = upstream_response
            if self._target_pre_ir_transforms:
                result = apply_transforms(self._target_pre_ir_transforms, dict(result))
            if self._source_post_ir_transforms:
                result = apply_transforms(
                    self._source_post_ir_transforms,
                    dict(result) if result is upstream_response else result,
                )
            if self._fidelity is not None:
                self._run_fidelity_check(upstream_response, direction="response")
            self._profile["response_conversion_ms"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )
            return result

        t_total = time.perf_counter()

        # Phase 4a: Body-level target shim pre_ir_transforms
        response = upstream_response
        if self._target_pre_ir_transforms:
            response = apply_transforms(self._target_pre_ir_transforms, response)

        # Phase 4b: Target response → IR
        t0 = time.perf_counter()
        ctx.options["response_id_prefix"] = self._target_id_prefix
        try:
            ir_response = self._target_converter.response_from_provider(
                response, context=ctx
            )
        except Exception as exc:
            raise ConversionError(
                f"Failed to parse upstream response: {exc}",
                phase="response_to_ir",
            ) from exc
        self._profile["response_from_target_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # Hook: let caller cache metadata from IR response
        if on_ir_ready is not None:
            on_ir_ready(ir_response)

        # Restore custom tool types downgraded by enforce_custom_tools
        if self._ir_request is not None:
            custom_names = get_custom_tool_names(self._ir_request)
            if custom_names:
                restore_custom_tool_calls(ir_response, custom_tool_names=custom_names)

        # Phase 4c: IR → Source response
        t0 = time.perf_counter()
        ctx.options["response_id_prefix"] = self._source_id_prefix
        try:
            source_response = self._source_converter.response_to_provider(
                ir_response, context=ctx
            )
        except Exception as exc:
            raise ConversionError(
                f"Failed to convert response: {exc}", phase="ir_to_source"
            ) from exc
        self._profile["response_to_source_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        # Phase 4d: Source shim post_ir_transforms (denormalise back)
        if self._source_post_ir_transforms:
            source_response = apply_transforms(
                self._source_post_ir_transforms, source_response
            )

        self._profile["response_conversion_ms"] = round(
            (time.perf_counter() - t_total) * 1000, 2
        )
        return source_response

    def create_stream_processor(
        self,
        *,
        on_ir_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> StreamProcessorProtocol:
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

        # Same-format short-circuit: return a passthrough processor
        if self._passthrough:
            return PassthroughStreamProcessor(
                pre_ir_transforms=self._target_pre_ir_transforms,
                post_ir_transforms=self._source_post_ir_transforms,
            )

        from_ctx = self._target_converter.create_stream_context()
        to_ctx = self._source_converter.create_stream_context()

        # Bridge preserve-mode metadata and shim-driven prefix
        to_ctx.options["metadata_mode"] = self._metadata_mode
        from_ctx.options["metadata_mode"] = self._metadata_mode
        from_ctx.options["response_id_prefix"] = self._target_id_prefix
        to_ctx.options["response_id_prefix"] = self._source_id_prefix
        if "_request_echo" in ctx.metadata:
            to_ctx.metadata["_request_echo"] = ctx.metadata["_request_echo"]

        # Custom tool names downgraded by enforce_custom_tools
        custom_names = (
            get_custom_tool_names(self._ir_request)
            if self._ir_request is not None
            else frozenset()
        )

        return StreamProcessor(
            target_converter=self._target_converter,
            source_converter=self._source_converter,
            from_ctx=from_ctx,
            to_ctx=to_ctx,
            pre_ir_transforms=self._target_pre_ir_transforms,
            post_ir_transforms=self._source_post_ir_transforms,
            custom_tool_names=custom_names,
            on_ir_event=on_ir_event,
        )


# ---------------------------------------------------------------------------
# StreamProcessor
# ---------------------------------------------------------------------------


class PassthroughStreamProcessor:
    """No-op stream processor for same-format pipelines.

    Forwards upstream chunks directly, applying only shim body-level
    transforms.  Has a dummy ``source_context`` so the gateway's
    terminal-event logic doesn't crash.
    """

    def __init__(
        self,
        *,
        pre_ir_transforms: tuple[Transform, ...] = (),
        post_ir_transforms: tuple[Transform, ...] = (),
    ) -> None:
        self._pre_ir_transforms = pre_ir_transforms
        self._post_ir_transforms = post_ir_transforms
        self._ctx = self._PassthroughCtx()

    @property
    def source_context(self) -> Any:
        """Minimal context for passthrough — enough for gateway error handling."""
        return self._ctx

    class _PassthroughCtx:
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

    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
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
        return [chunk]


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
        pre_ir_transforms: Shim pre_ir_transforms to apply before conversion.
        custom_tool_names: Names of tools downgraded from custom to
            function by :func:.
        on_ir_event: Optional callback for each IR event.
    """

    def __init__(
        self,
        *,
        target_converter: Any,
        source_converter: Any,
        from_ctx: Any,
        to_ctx: Any,
        pre_ir_transforms: tuple[Transform, ...] = (),
        post_ir_transforms: tuple[Transform, ...] = (),
        custom_tool_names: frozenset[str] = frozenset(),
        on_ir_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._target_converter = target_converter
        self._source_converter = source_converter
        self._from_ctx = from_ctx
        self._to_ctx = to_ctx
        self._pre_ir_transforms = pre_ir_transforms
        self._post_ir_transforms = post_ir_transforms
        self._custom_tool_names = custom_tool_names
        self._custom_arg_buffers: dict[str, str] = {}
        self._on_ir_event = on_ir_event

    @property
    def source_context(self) -> Any:
        """The IR→source StreamContext.

        Exposed so transports can inspect stream state (e.g. whether the
        stream already ended, the assigned response ID) when synthesizing
        a terminal event after an upstream failure.
        """
        return self._to_ctx

    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert one upstream chunk to source-format events.

        Args:
            chunk: A parsed upstream response chunk (e.g. from SSE).

        Returns:
            List of source-format event dicts.  May be empty (some
            upstream chunks produce no source events), one, or multiple.
        """
        # Apply shim pre_ir_transforms
        if self._pre_ir_transforms:
            chunk = apply_transforms(self._pre_ir_transforms, chunk)

        # Target → IR events
        ir_events = self._target_converter.stream_response_from_provider(
            chunk, context=self._from_ctx
        )

        # Bridge response extras
        if "_response_extras" in self._from_ctx.metadata:
            self._to_ctx.metadata["_response_extras"] = self._from_ctx.metadata[
                "_response_extras"
            ]

        # Bridge message phase for Responses streaming round-trip
        if "_responses_phase" in self._from_ctx.metadata:
            self._to_ctx.metadata["_responses_phase"] = self._from_ctx.metadata[
                "_responses_phase"
            ]

        # Restore custom tool types for downgraded tools
        if self._custom_tool_names:
            ir_events = self._restore_custom_tool_events(ir_events)

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

        # Apply source shim post_ir_transforms to outbound chunks
        if self._post_ir_transforms and result:
            result = [apply_transforms(self._post_ir_transforms, c) for c in result]

        return result

    def _restore_custom_tool_events(
        self, ir_events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Re-tag streamed tool calls that were downgraded from custom.

        - tool_call_start: set tool_type to "custom" and
          fix the from_ctx registration.
        - tool_call_delta: buffer JSON argument fragments for
          custom tool calls (they contain JSON wrapping that needs to
          be unwrapped before the client sees them).
        - finish: flush buffered arguments as a single unwrapped
          delta before emitting the finish event.
        """
        staged: list[dict[str, Any]] = []
        for event in ir_events:
            if not isinstance(event, dict):
                staged.append(event)
                continue
            etype = event.get("type")

            if (
                etype == "tool_call_start"
                and event.get("tool_name") in self._custom_tool_names
            ):
                event["tool_type"] = "custom"
                call_id = event.get("tool_call_id", "")
                if call_id:
                    self._custom_arg_buffers[call_id] = ""
                    self._from_ctx.set_tool_call_type(call_id, "custom")
                staged.append(event)
                continue

            if (
                etype == "tool_call_delta"
                and event.get("tool_call_id") in self._custom_arg_buffers
            ):
                call_id = event["tool_call_id"]
                self._custom_arg_buffers[call_id] += event.get("arguments_delta") or ""
                continue

            if etype == "finish" and self._custom_arg_buffers:
                for call_id, raw in list(self._custom_arg_buffers.items()):
                    staged.append(
                        {
                            "type": "tool_call_delta",
                            "tool_call_id": call_id,
                            "arguments_delta": unwrap_custom_tool_input(raw),
                        }
                    )
                self._custom_arg_buffers.clear()

            staged.append(event)
        return staged
