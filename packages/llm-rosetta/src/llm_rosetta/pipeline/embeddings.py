"""The embedding conversion pipeline: a request or response through the IR.

Lightweight orchestrator over the embedding converter family in
:mod:`llm_rosetta.converters.embeddings`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.ir.context import ConversionContext
from llm_rosetta.converters.base.defaults import apply_ir_defaults
from llm_rosetta.converters.embeddings.lookup import get_embedding_converter

__all__ = ["EmbeddingConversionPipeline"]


class EmbeddingConversionPipeline:
    """Convert embedding requests/responses to and from the IR.

    Every request and response goes through the IR, even when source and
    target are the same format — the rule the chat pipeline follows (#577).
    The round-trip canonicalises the body (an OpenAI ``input`` string becomes
    a list, unknown fields are dropped) and keeps one conversion path in
    production and in tests.

    The pipeline keeps no per-request state beyond the shared warning
    context: request and response conversion are independent, pure
    conversions, so either leg may be called without the other (unlike a
    chat pipeline, whose response leg can depend on request-side context).
    """

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = source_format
        self.target_format = target_format
        self._source_converter = get_embedding_converter(self.source_format)
        self._target_converter = get_embedding_converter(self.target_format)
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

    def convert_response(
        self, body: dict[str, Any], *, model: str | None = None
    ) -> dict[str, Any]:
        """Target format response → IR → source format response.

        *model* is the request's model.  It is stored on the context because
        some upstreams (Google) do not echo it in the response, while the IR
        response requires it.
        """
        if model is not None:
            self._ctx.metadata["model"] = model
        ir = self._target_converter.response_from_provider(body, context=self._ctx)
        return self._source_converter.response_to_provider(ir, context=self._ctx)
