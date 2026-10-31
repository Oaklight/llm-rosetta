"""The non-chat pipelines, re-exported for one import site.

Each family owns its pipeline in ``converters/<family>/pipeline.py``; this
module only makes all four paradigms reachable as ``llm_rosetta.pipeline``.
"""

from __future__ import annotations

from llm_rosetta.converters.decisions.pipeline import (
    DecisionConversionPipeline,
    get_decision_converter,
)
from llm_rosetta.converters.embeddings.pipeline import (
    EmbeddingConversionPipeline,
    get_embedding_converter,
)
from llm_rosetta.converters.rerank.pipeline import (
    RerankConversionPipeline,
    get_rerank_converter,
)

__all__ = [
    "DecisionConversionPipeline",
    "EmbeddingConversionPipeline",
    "RerankConversionPipeline",
    "get_decision_converter",
    "get_embedding_converter",
    "get_rerank_converter",
]
