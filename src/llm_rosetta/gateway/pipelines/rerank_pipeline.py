"""Backward-compat shim — real module is at converters.rerank.pipeline."""

from llm_rosetta.converters.rerank.pipeline import (  # noqa: F401
    RERANK_FORMATS,
    RerankConversionPipeline,
    get_rerank_converter,
)

__all__ = [
    "RERANK_FORMATS",
    "RerankConversionPipeline",
    "get_rerank_converter",
]
