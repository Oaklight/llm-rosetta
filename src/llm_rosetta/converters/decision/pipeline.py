"""Decision conversion pipeline.

Lightweight pipeline that converts decision requests/responses between
provider formats via IR, using the decision converter family.
Mirrors :class:`~llm_rosetta.converters.embedding.pipeline.EmbeddingConversionPipeline`
and :class:`~llm_rosetta.converters.rerank.pipeline.RerankConversionPipeline`.

Currently only the ``typesafe`` native format is supported through this
pipeline.  Cross-paradigm decision converters (``llm_chat``, ``embedding``,
``reranker``) convert decision IR to other paradigm IRs (chat, embedding,
rerank) and require a second-hop converter to reach a provider wire format.
They are available as standalone converter classes but are not yet wired
into this pipeline.
"""

from __future__ import annotations

from typing import Any

# Import converter submodules directly (not through __init__) to avoid
# circular imports — __init__.py re-exports from this module.
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.converters.decision.typesafe import TypeSafeDecisionConverter

__all__ = [
    "DECISION_FORMATS",
    "DecisionConversionPipeline",
    "get_decision_converter",
]

_DECISION_CONVERTERS: dict[str, type[BaseDecisionConverter]] = {
    "typesafe": TypeSafeDecisionConverter,
}

DECISION_FORMATS = frozenset(_DECISION_CONVERTERS.keys())


def get_decision_converter(format_name: str) -> BaseDecisionConverter:
    """Return a converter instance for *format_name*.

    Raises:
        ValueError: If *format_name* is not a known decision format.
    """
    cls = _DECISION_CONVERTERS.get(format_name)
    if cls is None:
        raise ValueError(
            f"Unknown decision format: '{format_name}'. "
            f"Available: {', '.join(sorted(DECISION_FORMATS))}"
        )
    return cls()


class DecisionConversionPipeline:
    """Convert decision requests/responses between two provider formats.

    When *source_format* == *target_format*, conversion is skipped.
    """

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = source_format
        self.target_format = target_format
        self._needs_conversion = source_format != target_format
        self._source_converter = get_decision_converter(source_format)
        self._target_converter = get_decision_converter(target_format)
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
