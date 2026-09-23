"""
LLM-Rosetta - Base Rerank Converter

Rerank 转换器抽象基类
Abstract base class for rerank converters

Thin specialization of ``BaseSimpleConverter`` that binds the generic
type parameters to the rerank IR types (IRRerankRequest,
IRRerankResponse, RerankUsageInfo).
"""

from __future__ import annotations

from llm_rosetta.types.ir.rerank import (
    IRRerankRequest,
    IRRerankResponse,
    RerankUsageInfo,
)

from .simple_converter import BaseSimpleConverter


class BaseRerankConverter(
    BaseSimpleConverter[IRRerankRequest, IRRerankResponse, RerankUsageInfo]
):
    """Abstract base class for rerank format converters.

    Each concrete converter implements bidirectional conversion between
    a provider's rerank API format and the IR rerank types.

    Subclasses MUST:
    - Set ``_CONVERTER_TAG`` to a unique string identifier
    - Implement all ``_do_*`` hooks and usage conversion methods

    Public methods follow the template-method pattern: they create a
    fallback ``ConversionContext``, delegate to the abstract ``_do_*``
    hook, and return the result along with any accumulated warnings.
    """
