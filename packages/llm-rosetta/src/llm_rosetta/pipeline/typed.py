"""The non-chat pipelines and their shared usage entry point.

The embeddings, rerank and decisions orchestrators live beside the chat one in
this package (``pipeline/embeddings.py`` and so on); this module gathers them
and reads the usage of a typed upstream response.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.converters.decisions.lookup import (
    get_decision_converter,
    response_usage as _decision_usage,
)
from llm_rosetta.pipeline.decisions import DecisionConversionPipeline
from llm_rosetta.converters.embeddings.lookup import (
    get_embedding_converter,
    response_usage as _embedding_usage,
)
from llm_rosetta.pipeline.embeddings import EmbeddingConversionPipeline
from llm_rosetta.converters.rerank.lookup import (
    get_rerank_converter,
    response_usage as _rerank_usage,
)
from llm_rosetta.pipeline.rerank import RerankConversionPipeline
from llm_rosetta import retired_names

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
    reader = _USAGE_READERS.get(model_type)
    if reader is None:
        retired_names.check_not_retired(model_type, retired_names.MODEL_TYPE)
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
