"""Read-only views and diagnostics of a ConversionPipeline."""

import logging
from typing import Any

from llm_rosetta.capabilities import (
    ToolNameMap,
)
from llm_rosetta.converters.base.context import ConversionContext, ImageFetcher
from llm_rosetta.conversion_warnings import WarningSink, recording_to
from llm_rosetta.contracts.shim import ProviderShim
from llm_rosetta.shims.provider_shim import resolve_shim

logger = logging.getLogger("llm_rosetta.pipeline.chat")


class PipelineIntrospection:
    """Properties and fidelity helper shared by :class:`ConversionPipeline`.

    The instance attributes below are assigned by
    ``ConversionPipeline.__init__``.
    """

    _source_provider: str
    _target_provider: str
    _target_shim_arg: ProviderShim | str | None
    _source_converter: Any
    _target_converter: Any
    _image_fetcher: ImageFetcher | None
    _fidelity: Any
    _ctx: ConversionContext | None
    _ir_request: dict[str, Any] | None
    _idle_warnings: WarningSink
    _name_map: ToolNameMap
    _profile: dict[str, float]

    @property
    def _same_format(self) -> bool:
        """Whether source and target are the *same* provider spec.

        Compares the two specs the caller passed, already canonicalized, rather
        than comparing converters.  Converter identity is coarser than the
        notion this gates: a bare ``chat_completions`` source and a
        ``deepseek--chat_completions`` target share a converter class, yet they
        are deliberately distinct here — a same-format passthrough would skip the
        target shim's IR-level traits (its reasoning config, its transforms),
        which is exactly what a caller targeting a shim expects to get.

        Computed on access; it is one string comparison.
        """
        return self._source_provider == self._target_provider

    @property
    def _resolved_target_shim(self) -> ProviderShim | None:
        """The target shim as an object, resolved on access.

        The caller may hand over the shim itself, name it, or omit it and name
        the shim as *target_provider* instead — the gateway does the second,
        ``convert()`` the first.  Resolving here rather than storing a second
        attribute means every reader gets the same object from one place.

        The raw spelling lives in ``_target_shim_arg`` and is deliberately read
        nowhere: a gate that reached for it would get ``None`` or a string and
        silently behave as "no shim", which is the defect this replaced.  Reading
        the wrong name is now an ``AttributeError`` instead.
        """
        return resolve_shim(self._target_shim_arg) or resolve_shim(
            self._target_provider
        )

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
    def warnings(self) -> WarningSink:
        """Conversion warnings accumulated during the pipeline.

        One sink for the whole conversion: the request and response legs
        append to it, and a stream processor created from this pipeline
        reports the losses it sees into it too — so a caller that inspects
        only this misses nothing.  Each entry is a
        :class:`~llm_rosetta.conversion_warnings.ConversionWarning`, which is
        still the message string it always was.

        Returns an empty sink if :meth:`convert_request` hasn't been called.
        That sink is read-only in effect: it is not the conversion's, so
        nothing appends to it — record through :meth:`convert_request` instead.
        """
        if self._ctx is None:
            return self._idle_warnings
        return self._ctx.warnings

    @property
    def _conversion_label(self) -> str:
        """The conversion a warning belongs to, as ``<source>-><target>``."""
        return f"{self._source_provider}->{self._target_provider}"

    @property
    def tool_name_collisions(self) -> dict[str, int]:
        """Tool names declared more than once with differing definitions.

        Maps each conflicting name to the number of distinct definitions seen
        for it.  ``dedupe_tool_definitions`` leaves these tools in place (only
        the client can disambiguate them), so an upstream that requires unique
        tool names may reject the request — this map is what lets the gateway
        attribute that rejection to the client's own tool list.  The
        byte-identical duplicates the pass collapses are not listed.

        Returns an empty dict if :meth:`convert_request` hasn't been called or
        no name conflicts.  A copy, so callers cannot mutate pipeline state.
        """
        if self._ctx is None:
            return {}
        return dict(self._ctx.metadata.get("_tool_name_collisions", {}))

    @property
    def tool_name_map(self) -> ToolNameMap:
        """The client↔upstream tool-name pairing the request leg produced.

        The response and streaming legs translate names through this same map,
        and so does anything that has to explain an upstream rejection in the
        client's own terms: an upstream naming ``tools[16]`` or a sanitized
        spelling names a tool the client knows by another one.

        Empty before :meth:`convert_request` and when nothing was renamed, in
        which case :meth:`ToolNameMap.to_client` falls back to the input name.
        Live state — read it, do not mutate it.
        """
        return self._name_map

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
            ctx = ConversionContext(image_fetcher=self._image_fetcher)
            ctx.options["metadata_mode"] = "preserve"
            # A shadow round trip is a diagnostic: what it loses is not what
            # the request lost, so it neither logs nor joins the sink.
            ctx.warnings.log = False
            # Convert under the shadow's own silent sink: a report made by a
            # helper with no context in hand must not fall through to the
            # request's sink, which would log a loss the request never had.
            with recording_to(ctx.warnings):
                if direction == "request":
                    ir = self._source_converter.request_from_provider(body, context=ctx)
                    rt, _ = self._target_converter.request_to_provider(ir, context=ctx)
                else:
                    ir = self._target_converter.response_from_provider(
                        body, context=ctx
                    )
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
