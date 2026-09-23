"""Backward-compat shim — real module is at converters.embedding.pipeline."""

from llm_rosetta.converters.embedding.pipeline import (  # noqa: F401
    EMBEDDING_FORMATS,
    EmbeddingConversionPipeline,
    get_embedding_converter,
)

__all__ = [
    "EMBEDDING_FORMATS",
    "EmbeddingConversionPipeline",
    "get_embedding_converter",
]
