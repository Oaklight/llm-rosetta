"""The chat conversion pipeline."""

import logging
import time
import warnings
from collections.abc import Callable
from typing import Any, Literal

from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
from llm_rosetta.ir.passes import (
    dedupe_tool_definitions,
    qualify_tool_names,
    degrade_intrinsic_history,
    relocate_oversized_tool_descriptions,
    ToolNameMap,
    apply_upstream_tool_names,
    build_tool_name_map,
    restore_client_tool_names,
    sanitize_tool_names,
)
from llm_rosetta.pipeline.enforcement import (
    backfill_reasoning_replay,
    drop_nameless_tools,
    enforce_custom_tools,
    resolve_intrinsic_tools,
    enforce_reasoning,
    enforce_vision,
    get_custom_tool_names,
    restore_custom_tool_calls,
    strip_reasoning_for_non_reasoning,
)
from llm_rosetta.ir.context import ConversionContext, ImageFetcher
from llm_rosetta.conversion_warnings import WarningSink, recording_to
from llm_rosetta import retired_names
from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE
from llm_rosetta.contracts.shim import ProviderShim
from llm_rosetta.contracts.resolver import (
    lookup_shim,
    resolve_format,
    resolve_shim,
)
from llm_rosetta.ir.primitives.body import Transform, apply_transforms
from .errors import ConversionError
from .introspection import PipelineIntrospection
from .ir_transforms import apply_ir_transforms
from .streaming import (
    BaselineStreamProcessor,
    StreamProcessor,
    StreamProcessorProtocol,
)

logger = logging.getLogger("llm_rosetta.pipeline.chat")

_EMPTY_TRANSFORMS: tuple[Transform, ...] = ()

# Base types that resolve to the ``google_generate`` converter and
# therefore use the Generative Language REST body shape by default.
# ``"google_generate"`` is the converter name.  Shim names (e.g.
# ``asksage--google_generate``) are normalised through ``resolve_format``
# before the membership test so they select REST too.  The SDK body
# shape (``config``) is rejected by the REST endpoint with HTTP 400.
_GOOGLE_GENERATE_PROVIDERS: frozenset[str] = frozenset(["google_generate"])


def _check_shim_name(shim_name: str) -> None:
    """Refuse a shim name that would otherwise resolve to no shim silently.

    A retired spelling names its replacement; a shim whose vendor moved out
    of core and is not installed names the repository it moved to (#1660).

    Raises:
        ValueError: For either case.
    """
    retired_names.check_not_retired(shim_name, retired_names.SHIM)
    if lookup_shim(shim_name) is None:
        moved = retired_names.moved_shim_message(shim_name)
        if moved is not None:
            raise ValueError(moved)


class ConversionPipeline(PipelineIntrospection):
    """Orchestrates format conversion between LLM API standards.

    Owns Phase 1 (Source→IR), Phase 2 (IR adapt + IR→Target), and
    Phase 4 (Response→Source).  Phase 3 (upstream forwarding) is the
    caller's responsibility.

    Usage::

        pipeline = ConversionPipeline("chat_completions", "messages",
                                      shim="anthropic--messages")

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

    In baseline mode (source == target, baseline=True), the IR
    round-trip is skipped but shim body-level transforms still apply:
    source pre_ir → target post_ir (request) and target pre_ir →
    source post_ir (response).  Baseline mode exists for fidelity
    shadow-testing — diffing unconverted output against IR-converted
    output to catch round-trip regressions — not as a production
    path.  The gateway always converts through IR.  See #577.

    Args:
        source_provider: Client API format (e.g. ``"chat_completions"``).
        target_provider: Upstream API format (e.g. ``"messages"``).
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
        supports_custom_tools: bool | None = None,
        max_tool_description_length: int | None = None,
        hoist_system_messages: bool = True,
        baseline: bool = False,
        fidelity_mode: Literal["critical", "full"] | None = None,
        metadata_mode: str = "preserve",
        google_output_format: Literal["rest", "sdk"] = "rest",
        image_fetcher: ImageFetcher | None = None,
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

        # A retired *shim* name supplied as a string would resolve to no shim
        # and be dropped silently, so reject it here with its replacement.  A
        # retired provider name is rejected by ``get_converter_for_provider``
        # below.  A bare chat format is not a shim name, whatever it is called
        # in the retired-names table.
        for shim_name in (source_shim, target_shim):
            if isinstance(shim_name, str) and shim_name not in FORMATS_BY_TYPE["llm"]:
                _check_shim_name(shim_name)

        # ``google_output_format`` is read only when the target is a Google
        # Generate provider (see ``convert_request``), so validate it under the
        # same condition rather than rejecting a value that would never be
        # used.  Where it *is* used, the converter selects the REST body by
        # exact match and otherwise emits the SDK shape, which the Generative
        # Language REST endpoint rejects with HTTP 400 ``Unknown name
        # "config"`` (#882) — so fail at construction, not on the upstream
        # call.
        if resolve_format(target_provider) in _GOOGLE_GENERATE_PROVIDERS:
            from llm_rosetta.converters.llm.google_generate import (
                validate_google_output_format,
            )

            validate_google_output_format(
                google_output_format, param="google_output_format"
            )

        self._source_provider = source_provider
        self._target_provider = target_provider
        self._target_shim_arg: ProviderShim | str | None = target_shim
        self._upstream_model = upstream_model
        self._model_capabilities = model_capabilities
        self._reasoning_config_override = reasoning_config_override
        self._supports_custom_tools = supports_custom_tools
        self._max_tool_description_length = max_tool_description_length
        self._hoist_system_messages = hoist_system_messages
        self._metadata_mode = metadata_mode
        self._google_output_format = google_output_format
        self._image_fetcher = image_fetcher

        self._source_converter = get_converter_for_provider(source_provider)
        self._target_converter = get_converter_for_provider(target_provider)

        self._baseline = self._same_format and baseline
        self._fidelity: Any = None
        if self._baseline and fidelity_mode is not None:
            from llm_rosetta.fidelity import FidelityChecker

            self._fidelity = FidelityChecker(
                mode=fidelity_mode, format_name=source_provider
            )

        # Resolve body-level transforms from source shim
        resolved_source = resolve_shim(source_shim)
        if resolved_source is not None:
            self._source_pre_ir_transforms = resolved_source.pre_ir_transforms
            self._source_post_ir_transforms = resolved_source.post_ir_transforms
        else:
            self._source_pre_ir_transforms = _EMPTY_TRANSFORMS
            self._source_post_ir_transforms = _EMPTY_TRANSFORMS

        # Resolve body-level transforms from target shim
        resolved_target = self._resolved_target_shim
        if resolved_target is not None:
            self._target_pre_ir_transforms = resolved_target.pre_ir_transforms
            self._target_post_ir_transforms = resolved_target.post_ir_transforms
            self._target_response_body_transforms = (
                resolved_target.response_body_transforms
            )
        else:
            self._target_pre_ir_transforms = _EMPTY_TRANSFORMS
            self._target_post_ir_transforms = _EMPTY_TRANSFORMS
            self._target_response_body_transforms = _EMPTY_TRANSFORMS

        # Resolve response_id_prefix for each converter from its shim.
        resolved_target_for_prefix = self._resolved_target_shim
        self._target_id_prefix = (
            resolved_target_for_prefix.response_id_prefix
            if resolved_target_for_prefix
            else ""
        )
        self._multimodal_tool_result = (
            resolved_target_for_prefix.tools.multimodal_result
            if resolved_target_for_prefix
            else None
        )
        self._target_tool_groups = (
            resolved_target_for_prefix.tools.tool_groups
            if resolved_target_for_prefix
            else False
        )
        resolved_source_for_prefix = resolved_source or resolve_shim(source_provider)
        self._source_id_prefix = (
            resolved_source_for_prefix.response_id_prefix
            if resolved_source_for_prefix
            else ""
        )

        # Shim-declared Responses profile traits, threaded to the converter
        # through ``context.options`` exactly like ``response_id_prefix``.
        # Default (no shim / neutral shim) = spec behavior.
        self._target_preserve_fields = (
            resolved_target_for_prefix.preserve_fields
            if resolved_target_for_prefix
            else ()
        )
        self._target_store_default = (
            resolved_target_for_prefix.store_default
            if resolved_target_for_prefix
            else None
        )
        # Request-bound fields the *target* accepts beyond its base format.
        # The reader forwards a client's value only when the target declares
        # it, so a neutral upstream never receives a field its spec omits.
        self._target_request_fields = (
            resolved_target_for_prefix.request_fields
            if resolved_target_for_prefix
            else ()
        )
        self._source_preserve_fields = (
            resolved_source_for_prefix.preserve_fields
            if resolved_source_for_prefix
            else ()
        )
        self._source_store_default = (
            resolved_source_for_prefix.store_default
            if resolved_source_for_prefix
            else None
        )

        # Set after convert_request()
        self._ctx: ConversionContext | None = None
        # The sink `warnings` hands out before a conversion has started.  It
        # is one object, not a fresh one per access, so a caller that keeps
        # the property's value is not looking at a list nothing writes to;
        # it is still not the conversion's sink — that one is created by
        # convert_request() — and it does not log.
        self._idle_warnings = WarningSink(log=False)
        self._ir_request: dict[str, Any] | None = None
        # Built once from the final IR request; the response and streaming
        # legs must translate names through the same map the request leg
        # used, so it is stored rather than rederived per leg.
        self._name_map: ToolNameMap = ToolNameMap()

        # Per-phase timing (always-on, ~30ns per perf_counter call)
        self._profile: dict[str, float] = {}

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
        ctx = ConversionContext(image_fetcher=self._image_fetcher)
        self._ctx = ctx
        # Provenance for everything this conversion loses: which leg noticed,
        # and which pair it was converting between.
        ctx.warnings.leg = "request"
        ctx.warnings.converter = self._conversion_label

        # Baseline mode: skip IR round-trip, apply only shim body-level
        # transforms.  Used for fidelity shadow-testing (diffing against
        # the converted output to catch round-trip regressions).
        if self._baseline:
            t0 = time.perf_counter()
            result = body
            if self._source_pre_ir_transforms:
                result = apply_transforms(self._source_pre_ir_transforms, dict(result))
            if self._target_post_ir_transforms:
                result = apply_transforms(
                    self._target_post_ir_transforms,
                    dict(result) if result is body else result,
                )
            # No IR produced in baseline mode
            self._ir_request = {}
            # Shadow round-trip for fidelity monitoring
            if self._fidelity is not None:
                self._run_fidelity_check(body, direction="request")
            self._profile["request_conversion_ms"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )
            return result
        ctx.options["metadata_mode"] = self._metadata_mode
        # Source-side profile traits: request echo is captured from the source
        # request (see the converter's ``request_from_provider``).
        ctx.options["preserve_fields"] = self._source_preserve_fields
        ctx.options["store_default"] = self._source_store_default
        ctx.options["request_fields"] = self._target_request_fields
        if resolve_format(self._target_provider) in _GOOGLE_GENERATE_PROVIDERS:
            ctx.options["output_format"] = self._google_output_format

        if self._multimodal_tool_result is not None:
            ctx.options["multimodal_tool_result"] = self._multimodal_tool_result

        # Whether the target can carry a tool group decides how the tool list is
        # rendered, so the writer has to be told.  The decision comes from the
        # shim, which the converter cannot see.
        ctx.options["tool_groups"] = self._target_tool_groups

        # Capability enforcement: reasoning (pre-IR)
        enforce_reasoning(
            ctx,
            self._resolved_target_shim,
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
            with recording_to(ctx.warnings):
                ir_request = self._source_converter.request_from_provider(
                    body, context=ctx
                )
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
            warnings=ctx.warnings,
        )

        # Capability enforcement: reasoning (post-IR)
        ir_request = strip_reasoning_for_non_reasoning(
            ir_request,
            model_capabilities=self._model_capabilities,
            model=self._upstream_model or body.get("model") or "",
            request_id=request_id,
        )

        # Capability enforcement: reasoning replay.  A thinking-mode target
        # that declares ``replay_missing: fill_empty`` needs the field present
        # on every assistant turn once tools are in play; an empty value is
        # accepted.  This is a property of the target model, not of a vendor
        # shim, so it runs here for every target format.  See epic #1143.
        ir_request = backfill_reasoning_replay(
            ir_request,
            reasoning_cap=ctx.options.get("reasoning_cap"),
            model=self._upstream_model or body.get("model") or "",
            request_id=request_id,
        )

        # Capability enforcement: intrinsic tool *definitions*, reconciled
        # against the target provider's shim (unsupported kinds dropped,
        # bare-name functions promoted).
        ir_request = resolve_intrinsic_tools(
            ir_request,
            shim=self._resolved_target_shim,
            same_format=self._same_format,
            # Promotion exists because the *client* format cannot name a server
            # tool natively, so it is a property of the source converter's
            # format, not of the literal provider name.
            allow_name_promotion=not self._source_converter._SUPPORTS_NATIVE_SERVER_TOOLS,
            request_id=request_id,
            warnings=ctx.warnings,
        )

        # Tool definitions must be addressable.  A nameless one can only go
        # out as an empty string, which every upstream rejects with a 400 that
        # names a position in a body the client never wrote; drop it here and
        # say why.  Runs after the intrinsic pass so a promoted definition
        # (which always carries a name) is not re-examined.
        ir_request = drop_nameless_tools(
            ir_request,
            request_id=request_id,
            warnings=ctx.warnings,
        )

        # Capability enforcement: intrinsic *history*.  A history part is
        # context, not a capability declaration, so it is not gated by the
        # target shim (the definitions above are).  What it *must* be is a
        # shape the target can replay: each converter asks its target
        # counterpart's own vocabulary
        # (``intrinsic_renderable(kind, input, leg="history")``), and a call it
        # cannot render — a sibling dialect's kind, a kind whose native part
        # needs a field the IR does not carry, a part with no kind at all —
        # is degraded here, together with the results answering it, into an
        # ordinary function pair.  This is the one place the native/degrade
        # decision for history is made; a converter used without a pipeline
        # still falls back to the parent client shape for itself.  See #839.
        ir_request = degrade_intrinsic_history(
            ir_request,
            self._target_converter.tool_ops,
            warnings=ctx.warnings,
            request_id=request_id,
        )

        # Capability enforcement: custom tools (post-IR)
        ir_request = enforce_custom_tools(
            ir_request,
            shim=self._resolved_target_shim,
            config_override=self._supports_custom_tools,
        )

        # Tool names: qualify the ones that collide.  This runs against the
        # target, which is why it is here rather than where the container
        # was flattened — whether a group name has to be folded into the
        # tool's own name depends on what the target can express, and only
        # this side knows that.  A target that carries the group on the wire
        # needs a child unique only among its own container's members, so the
        # pass is told; every other target gets the name made unique against
        # the whole list.  Before the dedupe below, so that pass judges the
        # names that will actually go upstream.
        ir_request = qualify_tool_names(
            ir_request,
            warnings=ctx.warnings,
            carries_groups=self._target_tool_groups,
        )

        # Capability enforcement: duplicate tool definitions (post-custom-tools).
        # Runs before description relocation so a dropped tool does not leave
        # its documentation duplicated in the relocated system message.
        # Exact duplicates are collapsed because several upstreams reject a
        # repeated name outright; differing definitions under one name are
        # reported but left alone — only the client can disambiguate them.
        # The conflicting names are also kept structured (name -> distinct
        # definition count) so the gateway can attribute an upstream
        # "tool names must be unique" rejection to the client's own tool list
        # and record it in the error dump.  See #907.
        _tool_name_collisions: dict[str, int] = {}
        ir_request = dedupe_tool_definitions(
            ir_request,
            warnings=ctx.warnings,
            collisions=_tool_name_collisions,
        )
        ctx.metadata["_tool_name_collisions"] = _tool_name_collisions

        # Capability enforcement: oversized tool descriptions
        ir_request = relocate_oversized_tool_descriptions(
            ir_request,
            max_description_length=self._max_tool_description_length,
            request_id=request_id,
        )

        # Phase 2a: Intrinsic + shim-driven IR transforms
        ir_request = apply_ir_transforms(
            ir_request,
            self._resolved_target_shim,
            upstream_model=self._upstream_model or body.get("model"),
            model_capabilities=self._model_capabilities,
            request_id=request_id,
            hoist_system_messages=self._hoist_system_messages,
            converter=self._target_converter,
            warnings=ctx.warnings,
        )
        self._profile["ir_transforms_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Tool definitions may have been renamed above (namespace flattening,
        # collision qualification).  History tool calls still carry the name
        # the client knows, so re-spell them to match before they go upstream.
        # Derived from the final IR so both legs agree on the same mapping.
        # Called unconditionally: it also checks that tool_choice and
        # allowed_tools name a declared tool, which is worth reporting even
        # when nothing was renamed.  It skips its own rewrite on an empty map.
        self._name_map = build_tool_name_map(ir_request)
        # A target that publishes a name shape it enforces gets every name
        # rewritten to it — an MCP tool exported as ``srv/fs__read`` is legal
        # everywhere but the OpenAI and Anthropic function-name rules — and
        # the pair is recorded in the map, so the history rewrite below and
        # the response leg both use the client's spelling again.
        target_name_pattern = self._target_converter._TOOL_NAME_PATTERN
        if target_name_pattern is not None:
            sanitize_tool_names(
                ir_request,
                pattern=target_name_pattern,
                name_map=self._name_map,
                warnings=ctx.warnings,
            )
        apply_upstream_tool_names(
            ir_request, name_map=self._name_map, warnings=ctx.warnings
        )

        # Request-field constraints: what the dialect requires of its own
        # fields, with the target shim's per-vendor declaration laid over it.
        # Only the merge happens here: the target converter applies the result
        # to the copy it writes, so the IR this pipeline holds keeps exactly
        # what the source request carried.  That matters beyond tidiness — the
        # original-request record and the round-trip guards both read this IR,
        # and a target's injected default appearing in it would report a value
        # the client never sent (#1457).
        ctx.options["request_constraints"] = (
            self._target_converter.request_constraints.merged_with(
                getattr(self._resolved_target_shim, "request", None)
            )
        )

        self._ir_request = ir_request

        # A target that cannot state an ``allowed_tools`` restriction warns
        # from its own tool-choice writer, where the widening happens, so the
        # warning names the format and travels with the loss.

        # Phase 2b: IR → Target
        t0 = time.perf_counter()
        try:
            with recording_to(ctx.warnings):
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
        if self._baseline:
            t0 = time.perf_counter()
            result = upstream_response
            if self._target_response_body_transforms:
                result = apply_transforms(
                    self._target_response_body_transforms, dict(result)
                )
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

        # The response leg reports into the same sink as the request leg, so
        # a loss while reading the upstream is not a second list to inspect.
        ctx.warnings.leg = "response"

        # Phase 4a: Response body transforms (e.g. harmony tool-call rewrite)
        response = upstream_response
        if self._target_response_body_transforms:
            response = apply_transforms(self._target_response_body_transforms, response)

        # Phase 4b: Body-level target shim pre_ir_transforms
        if self._target_pre_ir_transforms:
            response = apply_transforms(self._target_pre_ir_transforms, response)

        # Phase 4b: Target response → IR
        t0 = time.perf_counter()
        ctx.options["response_id_prefix"] = self._target_id_prefix
        ctx.options["preserve_fields"] = self._target_preserve_fields
        ctx.options["store_default"] = self._target_store_default
        try:
            with recording_to(ctx.warnings):
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

        return self._ir_response_to_source(ir_response, ctx, started_at=t_total)

    def convert_stream_events(
        self,
        events: list[dict[str, Any]],
        *,
        on_ir_ready: Callable[[dict[str, Any]], None] | None = None,
        response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fold IR stream events into one source-format response body.

        The path for a client that asked for a single body while the upstream
        answered with a stream (a shim's ``connection.stream_required``).  The
        events the stream processor produced are folded back into the response
        they describe, and that response then takes the ordinary response leg
        — the same custom-tool restore, tool-name restore, source shim
        transforms and source writer :meth:`convert_response` uses, so the two
        cannot answer differently.

        Must be called after :meth:`convert_request`.

        Args:
            events: IR stream events from this pipeline's stream processor.
            on_ir_ready: Optional callback invoked with the folded response.
            response: Envelope facts the caller already knows (``id``,
                ``model``, ``created``), for a stream whose ``stream_start``
                never reached the fold.

        Returns:
            The source-format response body.

        Raises:
            ConversionError: If the source writer refuses the folded turn.
            RuntimeError: If called before :meth:`convert_request`.
        """
        ctx = self.context  # raises RuntimeError if not ready

        if self._baseline:
            # A same-format baseline forwards chunks untouched; there is no
            # IR leg to fold, so the caller's events are what it already had.
            raise RuntimeError(
                "convert_stream_events() needs an IR conversion; a baseline "
                "(same-format passthrough) pipeline has none"
            )

        t_total = time.perf_counter()
        ctx.warnings.leg = "response"
        from llm_rosetta.ir.primitives import fold_response_events

        with recording_to(ctx.warnings):
            folded = fold_response_events(events, response=response)
        # The turn's own parts decide a ``stop`` that owes the client a tool
        # answer — the same rule every body reader applies, and the reason the
        # fold does not trust the terminal event's reason alone: a terminal
        # event may carry an empty ``output`` array.
        self._target_converter._finalize_completed_turn(folded)
        self._profile["response_from_target_ms"] = round(
            (time.perf_counter() - t_total) * 1000, 2
        )
        # The response leg takes a plain body: ``IRResponse`` is a TypedDict,
        # which is not assignable to ``dict`` (a reader of either may mutate).
        ir_response: dict[str, Any] = dict(folded)

        if on_ir_ready is not None:
            on_ir_ready(ir_response)

        return self._ir_response_to_source(ir_response, ctx, started_at=t_total)

    def _ir_response_to_source(
        self,
        ir_response: dict[str, Any],
        ctx: ConversionContext,
        *,
        started_at: float | None = None,
    ) -> dict[str, Any]:
        """Write a read-back IR response in the source dialect.

        The tail both response legs share: restore what the request leg
        changed, hand the source writer its own shim traits, and apply the
        source shim's post-IR transforms to the body it produced.

        Args:
            ir_response: The turn, as the target reader (or the fold) built it.
            ctx: The conversion context, whose options pick the source traits.
            started_at: When the caller's response leg began, for
                ``response_conversion_ms``.  The leg starts before the target
                read, so the total has to span that too; ``None`` starts the
                clock here, which is what a caller with nothing earlier gets.

        Returns:
            The source-format response body.

        Raises:
            ConversionError: If the source writer refuses the turn.
        """
        t_total = time.perf_counter() if started_at is None else started_at
        # Restore custom tool types downgraded by enforce_custom_tools
        if self._ir_request is not None:
            custom_names = get_custom_tool_names(self._ir_request)
            if custom_names:
                restore_custom_tool_calls(ir_response, custom_tool_names=custom_names)

        # Undo any request-leg tool renaming (namespace flattening,
        # collision qualification) before handing calls to the client.
        # Unguarded, like the streaming leg: the function early-returns on an
        # empty map, and a guard here would be a second place deciding when
        # the restore runs — which is how the tool_choice check ended up
        # unreachable.
        restore_client_tool_names(ir_response, name_map=self._name_map)

        # Phase 4c: IR → Source response
        t0 = time.perf_counter()
        ctx.options["response_id_prefix"] = self._source_id_prefix
        ctx.options["preserve_fields"] = self._source_preserve_fields
        ctx.options["store_default"] = self._source_store_default
        try:
            with recording_to(ctx.warnings):
                source_response = self._source_converter.response_to_provider(
                    ir_response, context=ctx
                )
        except Exception as exc:
            payload: dict[str, Any] = {}
            if isinstance(exc, UnrepresentableFinishReason):
                payload = {
                    "reason": exc.reason,
                    "response_id": exc.response_id,
                    "provider_metadata": exc.provider_metadata,
                }
            raise ConversionError(
                f"Failed to convert response: {exc}",
                phase="ir_to_source",
                **payload,
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

        # Baseline mode: return a baseline stream processor
        if self._baseline:
            return BaselineStreamProcessor(
                response_body_transforms=self._target_response_body_transforms,
                pre_ir_transforms=self._target_pre_ir_transforms,
                post_ir_transforms=self._source_post_ir_transforms,
            )

        from_ctx = self._target_converter.create_stream_context()
        to_ctx = self._source_converter.create_stream_context()

        # One sink for the whole conversion, including both halves of the
        # stream: a drop noticed while converting a chunk used to die with
        # that chunk's context, and the caller's ``pipeline.warnings`` never
        # heard about it.  The leg moves to ``stream`` for the processor's
        # lifetime — a streamed response never runs ``convert_response``.
        ctx.warnings.leg = "stream"
        from_ctx.warnings = ctx.warnings
        to_ctx.warnings = ctx.warnings

        # Bridge preserve-mode metadata and shim-driven prefix
        to_ctx.image_fetcher = self._image_fetcher
        from_ctx.image_fetcher = self._image_fetcher
        to_ctx.options["metadata_mode"] = self._metadata_mode
        from_ctx.options["metadata_mode"] = self._metadata_mode
        from_ctx.options["response_id_prefix"] = self._target_id_prefix
        to_ctx.options["response_id_prefix"] = self._source_id_prefix
        from_ctx.options["preserve_fields"] = self._target_preserve_fields
        to_ctx.options["preserve_fields"] = self._source_preserve_fields
        from_ctx.options["store_default"] = self._target_store_default
        to_ctx.options["store_default"] = self._source_store_default
        if "_request_echo" in ctx.metadata:
            to_ctx.metadata["_request_echo"] = ctx.metadata["_request_echo"]
        # What the request enabled upstream is what lets the stream reader tell
        # a hosted tool's call from a client function of the same name; the
        # request vocabulary belongs to the conversion, not to either
        # direction, so it is handed to the context that reads the stream.
        ctx.bridge_request_tool_vocabulary(from_ctx)

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
            response_body_transforms=self._target_response_body_transforms,
            pre_ir_transforms=self._target_pre_ir_transforms,
            post_ir_transforms=self._source_post_ir_transforms,
            custom_tool_names=custom_names,
            name_map=self._name_map,
            on_ir_event=on_ir_event,
        )
