"""Deprecated: moved to :mod:`llm_rosetta.ir.model.embedding`.

``llm_rosetta.types.ir.embedding`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.embedding import (
    EmbeddingEncodingFormat,
    EmbeddingItem,
    EmbeddingTaskType,
    EmbeddingUsageInfo,
    IREmbeddingRequest,
    IREmbeddingResponse,
)

__all__ = [
    "EmbeddingEncodingFormat",
    "EmbeddingItem",
    "EmbeddingTaskType",
    "EmbeddingUsageInfo",
    "IREmbeddingRequest",
    "IREmbeddingResponse",
]
