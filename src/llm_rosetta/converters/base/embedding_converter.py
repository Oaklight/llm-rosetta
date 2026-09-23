"""
LLM-Rosetta - Base Embedding Converter

Embedding 转换器抽象基类
Abstract base class for embedding converters

Thin specialization of ``BaseSimpleConverter`` that binds the generic
type parameters to the embedding IR types (IREmbeddingRequest,
IREmbeddingResponse, EmbeddingUsageInfo).
"""

from __future__ import annotations

from llm_rosetta.types.ir.embedding import (
    EmbeddingUsageInfo,
    IREmbeddingRequest,
    IREmbeddingResponse,
)

from .simple_converter import BaseSimpleConverter


class BaseEmbeddingConverter(
    BaseSimpleConverter[IREmbeddingRequest, IREmbeddingResponse, EmbeddingUsageInfo]
):
    """Abstract base class for embedding format converters.

    Subclasses MUST:
    - Set ``_CONVERTER_TAG`` to a unique string identifier
    - Implement all ``_do_*`` hooks and usage conversion methods
    """
