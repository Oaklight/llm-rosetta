"""Decision conversion pipeline.

Lightweight pipeline that converts decision requests/responses between
provider formats via IR, using the decision converter family.
Mirrors :class:`~llm_rosetta.converters.embeddings.pipeline.EmbeddingConversionPipeline`
and :class:`~llm_rosetta.converters.rerank.pipeline.RerankConversionPipeline`.

Currently the ``systemone`` and ``openai_decisions`` native formats are
supported through this pipeline.  Cross-paradigm decision converters (``llm_chat``, ``embedding``,
``reranker``) convert decision IR to other paradigm IRs (chat, embedding,
rerank) and require a second-hop converter to reach a provider wire format.
They are available as standalone converter classes but are not yet wired
into this pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Import converter submodules directly (not through __init__) to avoid
# circular imports — __init__.py re-exports from this module.
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.converters.base.defaults import apply_ir_defaults
from llm_rosetta.converters.decisions.openai import OpenAIDecisionsConverter
from llm_rosetta.converters.decisions.systemone import SystemOneDecisionConverter

__all__ = [
    "DECISION_FORMATS",
    "DecisionConversionPipeline",
    "get_decision_converter",
    "response_usage",
]

_DECISION_CONVERTERS: dict[str, type[BaseDecisionConverter]] = {
    "systemone": SystemOneDecisionConverter,
    "openai_decisions": OpenAIDecisionsConverter,
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


def response_usage(
    format_name: str, response_body: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The IR usage of an upstream response, read by *format_name*'s converter.

    The gateway's decision handler never holds the IR — the conversion pipeline
    goes from the provider's format back to the source format in one step,
    including when both sides are the same format — so the ledger's figures
    are read here, by the converter that would have produced them.

    A body the converter cannot read, or none at all, reports no usage rather
    than raising: a metering gap is not worth failing a response the caller has
    already been given.
    """
    if not response_body:
        return {}
    try:
        converter = get_decision_converter(format_name)
        ir = converter.response_from_provider(dict(response_body))
    except Exception:
        return {}
    return dict(ir.get("usage") or {})


class DecisionConversionPipeline:
    """Convert decision requests/responses to and from the IR.

    Every request and response goes through the IR, even when source and
    target are the same format — the rule the chat pipeline follows (#577).
    The round-trip canonicalises the body and keeps one conversion path in
    production and in tests.
    """

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = source_format
        self.target_format = target_format
        self._source_converter = get_decision_converter(source_format)
        self._target_converter = get_decision_converter(target_format)
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
