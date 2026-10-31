"""Embedding conversion pipeline.

Lightweight pipeline that converts embedding requests/responses between
provider formats via IR, using the embedding converter family.
Mirrors :class:`~llm_rosetta.converters.rerank.pipeline.RerankConversionPipeline`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.provider_names import canonical_provider_name

# Import converter submodules directly (not through __init__) to avoid
# circular imports — __init__.py re-exports from this module.
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.defaults import apply_ir_defaults
from llm_rosetta.converters.base.embedding_converter import BaseEmbeddingConverter
from llm_rosetta.converters.embeddings.cohere import CohereEmbeddingConverter
from llm_rosetta.converters.embeddings.google import GoogleEmbeddingConverter
from llm_rosetta.converters.embeddings.jina import JinaEmbeddingConverter
from llm_rosetta.converters.embeddings.minimax import MiniMaxEmbeddingConverter
from llm_rosetta.converters.embeddings.openai import OpenAIEmbeddingConverter
from llm_rosetta.converters.embeddings.voyage import VoyageEmbeddingConverter

__all__ = [
    "EMBEDDING_FORMATS",
    "EmbeddingConversionPipeline",
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


def canonical_embedding_format(format_name: str) -> str:
    """Resolve a legacy format spelling to its canonical registry key.

    ``google`` and ``google-genai`` both name the ``google_generate``
    embedding family; every other embedding format passes through unchanged.
    """
    return canonical_provider_name(format_name)


def get_embedding_converter(format_name: str) -> BaseEmbeddingConverter:
    """Return a converter instance for *format_name*.

    Raises:
        ValueError: If *format_name* is not a known embedding format.
    """
    cls = _EMBEDDING_CONVERTERS.get(canonical_embedding_format(format_name))
    if cls is None:
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


class EmbeddingConversionPipeline:
    """Convert embedding requests/responses to and from the IR.

    Every request and response goes through the IR, even when source and
    target are the same format — the rule the chat pipeline follows (#577).
    The round-trip canonicalises the body (an OpenAI ``input`` string becomes
    a list, unknown fields are dropped) and keeps one conversion path in
    production and in tests.  Both spellings are canonicalised first, so
    ``google`` and ``google_generate`` are the same format.

    The pipeline keeps no per-request state beyond the shared warning
    context: request and response conversion are independent, pure
    conversions, so either leg may be called without the other (unlike a
    chat pipeline, whose response leg can depend on request-side context).
    """

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = canonical_embedding_format(source_format)
        self.target_format = canonical_embedding_format(target_format)
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
