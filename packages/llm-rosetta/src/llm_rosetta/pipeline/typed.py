"""The non-chat pipelines, re-exported for one import site.

Each family owns its pipeline in ``converters/<family>/pipeline.py``; this
module only makes all four paradigms reachable as ``llm_rosetta.pipeline``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.converters.decisions.pipeline import (
    DecisionConversionPipeline,
    get_decision_converter,
    response_usage as _decision_usage,
)
from llm_rosetta.converters.embeddings.pipeline import (
    EmbeddingConversionPipeline,
    get_embedding_converter,
    response_usage as _embedding_usage,
)
from llm_rosetta.converters.rerank.pipeline import (
    RerankConversionPipeline,
    get_rerank_converter,
    response_usage as _rerank_usage,
)
from llm_rosetta.formats import canonical_model_type

_USAGE_READERS = {
    "embeddings": _embedding_usage,
    "rerank": _rerank_usage,
    "decisions": _decision_usage,
}


def typed_response_usage(
    model_type: str,
    format_name: str,
    response_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The IR usage of a non-chat upstream response.

    One entry point for the three typed families: the family's converter for
    *format_name* reads *response_body* and its IR ``usage`` is returned.  A
    body the converter cannot read, or none at all, yields ``{}`` rather than
    raising, because a metering gap must not fail a response already given.

    Args:
        model_type: ``embeddings``, ``rerank`` or ``decisions`` (the legacy
            ``embedding`` / ``decision`` spellings are accepted).
        format_name: The upstream format that produced *response_body*.
        response_body: The decoded upstream response.

    Returns:
        The IR usage mapping, possibly empty.

    Raises:
        ValueError: *model_type* is not a typed family (``llm`` has no
            separate usage reader).
    """
    canonical = canonical_model_type(model_type)
    reader = _USAGE_READERS.get(canonical)
    if reader is None:
        raise ValueError(
            f"no typed usage reader for model type {model_type!r}; "
            f"expected one of {sorted(_USAGE_READERS)}"
        )
    return reader(format_name, response_body)


__all__ = [
    "DecisionConversionPipeline",
    "EmbeddingConversionPipeline",
    "RerankConversionPipeline",
    "get_decision_converter",
    "get_embedding_converter",
    "get_rerank_converter",
    "typed_response_usage",
]
