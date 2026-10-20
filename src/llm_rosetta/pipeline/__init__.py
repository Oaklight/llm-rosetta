"""Conversion pipelines — one import site for every paradigm.

The chat pipeline lives in :mod:`.chat`; the embedding, rerank and decision
pipelines are re-exported from their converter families in :mod:`.typed`.
Existing imports from ``llm_rosetta.pipeline`` keep working: the package
re-exports the same public names the module had.
"""

from __future__ import annotations

from .chat import (
    BaselineStreamProcessor,
    ConversionError,
    ConversionPipeline,
    StreamProcessor,
    StreamProcessorProtocol,
    apply_ir_transforms,
    apply_shim_to_ir,
    configure_context,
    setup_shim_context,
)
from .typed import (
    DecisionConversionPipeline,
    EmbeddingConversionPipeline,
    RerankConversionPipeline,
    get_decision_converter,
    get_embedding_converter,
    get_rerank_converter,
)

__all__ = [
    "BaselineStreamProcessor",
    "ConversionError",
    "ConversionPipeline",
    "DecisionConversionPipeline",
    "EmbeddingConversionPipeline",
    "RerankConversionPipeline",
    "StreamProcessor",
    "StreamProcessorProtocol",
    "apply_ir_transforms",
    "apply_shim_to_ir",
    "configure_context",
    "get_decision_converter",
    "get_embedding_converter",
    "get_rerank_converter",
    "setup_shim_context",
]
