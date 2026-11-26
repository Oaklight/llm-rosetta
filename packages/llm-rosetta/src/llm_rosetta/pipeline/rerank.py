"""The rerank conversion pipeline: a request or response through the IR.

Lightweight orchestrator over the rerank converter family in
:mod:`llm_rosetta.converters.rerank`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.ir.context import ConversionContext
from llm_rosetta.converters.base.defaults import apply_ir_defaults
from llm_rosetta.converters.rerank.lookup import get_rerank_converter

__all__ = ["RerankConversionPipeline"]


class RerankConversionPipeline:
    """Convert rerank requests/responses to and from the IR.

    Every request and response goes through the IR, even when source and
    target are the same format — the rule the chat pipeline follows (#577).
    The round-trip canonicalises the body and keeps one conversion path in
    production and in tests.
    """

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = source_format
        self.target_format = target_format
        self._source_converter = get_rerank_converter(source_format)
        self._target_converter = get_rerank_converter(target_format)
        self._ctx = ConversionContext()

    @property
    def warnings(self) -> list[str]:
        return self._ctx.warnings

    def convert_request(
        self,
        body: dict[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Source format request → IR → target format request.

        *defaults* are the shim's declared parameter defaults.  They fill a
        field the client did not supply and never override one it did.
        """
        ir = self._source_converter.request_from_provider(body, context=self._ctx)
        ir = apply_ir_defaults(ir, defaults)
        target_body, _ = self._target_converter.request_to_provider(
            ir, context=self._ctx
        )
        return target_body

    def convert_response(self, body: dict[str, Any]) -> dict[str, Any]:
        """Target format response → IR → source format response."""
        ir = self._target_converter.response_from_provider(body, context=self._ctx)
        return self._source_converter.response_to_provider(ir, context=self._ctx)
