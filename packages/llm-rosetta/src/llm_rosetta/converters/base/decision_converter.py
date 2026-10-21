"""
LLM-Rosetta - Base Decision Converter

Decision 转换器抽象基类
Abstract base class for decision converters

Thin specialization of ``BaseSimpleConverter`` that binds the generic
type parameters to the decision IR types (IRDecisionRequest,
IRDecisionResponse, DecisionUsageInfo).
"""

from __future__ import annotations

from llm_rosetta.types.ir.decision import (
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)

from .simple_converter import BaseSimpleConverter


class BaseDecisionConverter(
    BaseSimpleConverter[IRDecisionRequest, IRDecisionResponse, DecisionUsageInfo]
):
    """Abstract base class for decision format converters.

    Subclasses MUST:
    - Set ``_CONVERTER_TAG`` to a unique string identifier
    - Implement all ``_do_*`` hooks and usage conversion methods
    """
