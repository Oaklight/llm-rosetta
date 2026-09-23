"""Embedding conversion pipeline.

Lightweight pipeline that converts embedding requests/responses between
provider formats via IR, using the embedding converter family.
Mirrors :class:`~gateway.rerank_pipeline.RerankConversionPipeline`.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.embedding_converter import BaseEmbeddingConverter
from llm_rosetta.converters.embedding import (
    CohereEmbeddingConverter,
    JinaEmbeddingConverter,
    OpenAIEmbeddingConverter,
    VoyageEmbeddingConverter,
)

_EMBEDDING_CONVERTERS: dict[str, type[BaseEmbeddingConverter]] = {
    "openai": OpenAIEmbeddingConverter,
    "jina": JinaEmbeddingConverter,
    "voyage": VoyageEmbeddingConverter,
    "cohere": CohereEmbeddingConverter,
}

EMBEDDING_FORMATS = frozenset(_EMBEDDING_CONVERTERS.keys())


def get_embedding_converter(format_name: str) -> BaseEmbeddingConverter:
    """Return a converter instance for *format_name*.

    Raises:
        ValueError: If *format_name* is not a known embedding format.
    """
    cls = _EMBEDDING_CONVERTERS.get(format_name)
    if cls is None:
        raise ValueError(
            f"Unknown embedding format: '{format_name}'. "
            f"Available: {', '.join(sorted(EMBEDDING_FORMATS))}"
        )
    return cls()


class EmbeddingConversionPipeline:
    """Convert embedding requests/responses between two provider formats.

    When *source_format* == *target_format*, conversion is skipped.
    """

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = source_format
        self.target_format = target_format
        self._needs_conversion = source_format != target_format
        self._source_converter = get_embedding_converter(source_format)
        self._target_converter = get_embedding_converter(target_format)
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
