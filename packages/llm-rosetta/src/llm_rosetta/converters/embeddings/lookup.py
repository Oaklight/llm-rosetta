"""The embedding converters by format, and the usage their responses carry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta import retired_names

from llm_rosetta.converters.base.embedding_converter import BaseEmbeddingConverter
from llm_rosetta.converters.embeddings.cohere import CohereEmbeddingConverter
from llm_rosetta.converters.embeddings.google import GoogleEmbeddingConverter
from llm_rosetta.converters.embeddings.jina import JinaEmbeddingConverter
from llm_rosetta.converters.embeddings.minimax import MiniMaxEmbeddingConverter
from llm_rosetta.converters.embeddings.openai import OpenAIEmbeddingConverter
from llm_rosetta.converters.embeddings.voyage import VoyageEmbeddingConverter

__all__ = [
    "EMBEDDING_FORMATS",
    "get_embedding_converter",
    "response_usage",
]

_EMBEDDING_CONVERTERS: dict[str, type[BaseEmbeddingConverter]] = {
    "openai": OpenAIEmbeddingConverter,
    "cohere": CohereEmbeddingConverter,
    "jina": JinaEmbeddingConverter,
    "voyage": VoyageEmbeddingConverter,
    "google_generate": GoogleEmbeddingConverter,
    "minimax": MiniMaxEmbeddingConverter,
}

EMBEDDING_FORMATS = frozenset(_EMBEDDING_CONVERTERS.keys())


def get_embedding_converter(format_name: str) -> BaseEmbeddingConverter:
    """Return a converter instance for *format_name*.

    Raises:
        ValueError: If *format_name* is not a known embedding format.  A
            retired spelling (``google``) gets the retired-names message,
            which names its replacement.
    """
    cls = _EMBEDDING_CONVERTERS.get(format_name)
    if cls is None:
        retired_names.check_not_retired(format_name, retired_names.PROVIDER_TYPE)
        raise ValueError(
            f"Unknown embedding format: '{format_name}'. "
            f"Available: {', '.join(sorted(EMBEDDING_FORMATS))}"
        )
    return cls()


def response_usage(
    format_name: str, response_body: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The IR usage of an upstream response, read by *format_name*'s converter.

    The gateway's embedding handler never holds the IR — the conversion
    pipeline goes from the provider's format back to the source format in one
    step, including when both sides are the same format — so the ledger's
    figures are read here, by the converter that would have produced them.

    A body the converter cannot read, or none at all, reports no usage rather
    than raising: a metering gap is not worth failing a response the caller has
    already been given.
    """
    if not response_body:
        return {}
    try:
        converter = get_embedding_converter(format_name)
        ir = converter.response_from_provider(dict(response_body))
    except Exception:
        return {}
    return dict(ir.get("usage") or {})
