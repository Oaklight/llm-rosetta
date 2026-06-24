"""Conversion pipeline — orchestrates format conversion between LLM APIs.

This module provides two layers of API:

**High-level** — :class:`ConversionPipeline` class that encapsulates the
full conversion lifecycle (Phase 1→2→4).  Use this when you need
request conversion, response conversion, and/or streaming:

    pipeline = ConversionPipeline("openai_chat", "anthropic", shim="argo--anthropic")
    target_body = pipeline.convert_request(body)
    # ... transport sends target_body, receives upstream_response ...
    source_response = pipeline.convert_response(upstream_response)

**Low-level** — :func:`setup_shim_context` and :func:`apply_shim_to_ir`
for consumers that need finer control over individual pipeline stages.

The pipeline is part of the core library — **no network dependency**.
It produces a target request body and consumes a target response body;
the caller (gateway, argo-proxy, etc.) owns the transport.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.shims.provider_shim import ProviderShim, ReasoningCapability, get_shim
from llm_rosetta.shims.transforms import Transform, apply_transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_shim(shim: ProviderShim | str | None) -> ProviderShim | None:
    """Resolve a shim argument to a ProviderShim instance.

    Args:
        shim: A ProviderShim instance, a registered name, or None.

    Returns:
        The resolved ProviderShim, or None.
    """
    if shim is None:
        return None
    if isinstance(shim, ProviderShim):
        return shim
    return get_shim(shim)


def _apply_config_reasoning_override(
    base: ReasoningCapability,
    override: dict[str, Any],
) -> ReasoningCapability:
    """Merge config-level reasoning overrides onto a base capability.

    Only fields present in *override* are replaced; the rest inherit
    from *base*.

    Args:
        base: The base reasoning capability from the shim.
        override: A dict of field overrides (e.g. from admin UI).

    Returns:
        A new ReasoningCapability with merged values.
    """
    return ReasoningCapability(
        disabled=override.get("disabled", base.disabled),
        effort_field=override.get("effort_field", base.effort_field),
        max_effort=override.get("max_effort", base.max_effort),
        thinking_type=override.get("thinking_type", base.thinking_type),
        unsigned_reasoning_blocks=override.get(
            "unsigned_reasoning_blocks", base.unsigned_reasoning_blocks
        ),
        effort_map=override.get("effort_map", base.effort_map),
        budget_tokens_default_ratio=override.get(
            "budget_tokens_default_ratio", base.budget_tokens_default_ratio
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_shim_context(
    ctx: ConversionContext,
    shim: ProviderShim | str | None,
    *,
    model: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> None:
    """Inject shim-driven options into a ConversionContext.

    Currently injects ``reasoning_cap`` so converters produce the correct
    thinking/reasoning output for the target provider.

    Resolution priority (highest first):

    1. *config_override* — per-model override from external config
       (e.g. gateway admin UI).
    2. ``shim.model_reasoning[model]`` — per-model override from the
       provider YAML.
    3. ``shim.reasoning`` — provider-level default.

    Args:
        ctx: Conversion context to mutate.
        shim: ProviderShim instance, registered name, or None (no-op).
        model: Upstream model ID (for per-model reasoning overrides).
        config_override: External reasoning override (highest priority).
    """
    resolved = _resolve_shim(shim)
    if resolved is None:
        return

    cap = resolved.reasoning
    # Model-level override (keyed by upstream model ID)
    if model and resolved.model_reasoning and model in resolved.model_reasoning:
        cap = resolved.model_reasoning[model]
    # Config-level override (from admin UI, keyed by gateway model name)
    if cap is not None and config_override:
        cap = _apply_config_reasoning_override(cap, config_override)
    if cap is not None:
        ctx.options["reasoning_cap"] = cap


def apply_shim_to_ir(
    ir_request: dict[str, Any],
    shim: ProviderShim | str | None,
    *,
    upstream_model: str | None = None,
    model_capabilities: list[str] | None = None,
    request_id: str = "-",
) -> dict[str, Any]:
    """Apply all shim-driven IR-level transforms in canonical order.

    Operations applied (in order):

    1. **Strip non-vision images** — if *model_capabilities* is provided
       and does not include ``"vision"``, replace all images with text
       placeholders.  Driven by the caller, not by the shim.
    2. **Enforce image count limit** — if the shim declares
       ``max_images`` (and the upstream model matches
       ``max_images_pattern`` when set), truncate excess images.
    3. **Unwind parallel tool calls** — if the shim declares
       ``unwind_parallel_tool_calls`` (and the upstream model matches
       ``unwind_parallel_tool_calls_pattern`` when set), split parallel
       tool calls into sequential call-result pairs.

    All operations are no-ops when the corresponding shim field is unset
    or when the pattern guard doesn't match.

    Args:
        ir_request: The IR request dict.  Some operations mutate in-place,
            others return a new dict — **always use the return value**.
        shim: ProviderShim instance, registered name, or None (no-op).
        upstream_model: The upstream model ID (for pattern matching).
        model_capabilities: Model capability list (e.g. ``["text", "vision"]``).
            When ``None``, the non-vision image stripping step is skipped.
        request_id: Request identifier for logging.

    Returns:
        The IR request dict after all applicable transforms.  Always
        assign the return value: ``ir = apply_shim_to_ir(ir, shim, ...)``.
    """
    # 1. Strip images for non-vision models (caller-driven, not shim-driven)
    if model_capabilities is not None and "vision" not in model_capabilities:
        from llm_rosetta.converters.base.helpers.image_limit import (
            strip_images_for_non_vision,
        )

        ir_request = strip_images_for_non_vision(
            ir_request, model=upstream_model or "", request_id=request_id
        )

    resolved = _resolve_shim(shim)
    if resolved is None:
        return ir_request

    # 2. Enforce per-shim image count limit
    if resolved.max_images is not None:
        apply_limit = True
        if resolved.max_images_pattern is not None:
            apply_limit = bool(
                upstream_model
                and re.search(resolved.max_images_pattern, upstream_model)
            )
        if apply_limit:
            from llm_rosetta.converters.base.helpers.image_limit import truncate_images

            ir_request = truncate_images(
                ir_request, resolved.max_images, request_id=request_id
            )

    # 3. Unwind parallel tool calls
    if resolved.unwind_parallel_tool_calls:
        apply_unwind = True
        if resolved.unwind_parallel_tool_calls_pattern is not None:
            apply_unwind = bool(
                upstream_model
                and re.search(
                    resolved.unwind_parallel_tool_calls_pattern, upstream_model
                )
            )
        if apply_unwind:
            from llm_rosetta.converters.base.helpers.tool_call_unwind import (
                unwind_parallel_tool_calls_ir,
            )

            ir_request = unwind_parallel_tool_calls_ir(ir_request)

    return ir_request


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
        resolved = _resolve_shim(shim)
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

        setup_shim_context(
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

        # Phase 2a: Shim-driven IR transforms
        request_id = ctx.options.get("request_id", "-")
        ir_request = apply_shim_to_ir(
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
