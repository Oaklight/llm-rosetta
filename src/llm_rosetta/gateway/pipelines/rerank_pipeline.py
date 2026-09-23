"""Rerank conversion pipeline.

Lightweight pipeline that converts rerank requests/responses between
provider formats via IR, using the rerank converter family.  No
streaming, no shims — much simpler than the chat
:class:`~llm_rosetta.pipeline.ConversionPipeline`.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.rerank_converter import BaseRerankConverter
from llm_rosetta.converters.rerank import (
    CohereRerankConverter,
    JinaRerankConverter,
    VoyageRerankConverter,
)

_RERANK_CONVERTERS: dict[str, type[BaseRerankConverter]] = {
    "jina": JinaRerankConverter,
    "cohere": CohereRerankConverter,
    "voyage": VoyageRerankConverter,
}

RERANK_FORMATS = frozenset(_RERANK_CONVERTERS.keys())


def get_rerank_converter(format_name: str) -> BaseRerankConverter:
    """Return a converter instance for *format_name*.

    Raises:
        ValueError: If *format_name* is not a known rerank format.
    """
    cls = _RERANK_CONVERTERS.get(format_name)
    if cls is None:
        raise ValueError(
            f"Unknown rerank format: '{format_name}'. "
            f"Available: {', '.join(sorted(RERANK_FORMATS))}"
        )
    return cls()


class RerankConversionPipeline:
    """Convert rerank requests/responses between two provider formats.

    When *source_format* == *target_format*, conversion is skipped
    (only the model alias is applied).
    """

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = source_format
        self.target_format = target_format
        self._needs_conversion = source_format != target_format
        self._source_converter = get_rerank_converter(source_format)
        self._target_converter = get_rerank_converter(target_format)
        self._ctx = ConversionContext()

    @property
    def warnings(self) -> list[str]:
        return self._ctx.warnings

    def convert_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Source format request → target format request."""
        if not self._needs_conversion:
            return body
        ir = self._source_converter.request_from_provider(body, context=self._ctx)
        target_body, _ = self._target_converter.request_to_provider(
            ir, context=self._ctx
        )
        return target_body

    def convert_response(self, body: dict[str, Any]) -> dict[str, Any]:
        """Target format response → source format response."""
        if not self._needs_conversion:
            return body
        ir = self._target_converter.response_from_provider(body, context=self._ctx)
        return self._source_converter.response_to_provider(ir, context=self._ctx)
