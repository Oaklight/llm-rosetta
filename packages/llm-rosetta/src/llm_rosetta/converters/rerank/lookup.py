"""The rerank converters by format, and the usage their responses carry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.converters.base.rerank_converter import BaseRerankConverter
from llm_rosetta.converters.rerank.cohere import CohereRerankConverter
from llm_rosetta.converters.rerank.jina import JinaRerankConverter
from llm_rosetta.converters.rerank.voyage import VoyageRerankConverter

__all__ = [
    "RERANK_FORMATS",
    "get_rerank_converter",
    "response_usage",
]

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


def response_usage(
    format_name: str, response_body: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The IR usage of an upstream response, read by *format_name*'s converter.

    The gateway's rerank handler never holds the IR — the conversion pipeline
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
        converter = get_rerank_converter(format_name)
        ir = converter.response_from_provider(dict(response_body))
    except Exception:
        return {}
    return dict(ir.get("usage") or {})
