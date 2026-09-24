"""Backward-compat shim — real module is at converters.decision.pipeline."""

from llm_rosetta.converters.decision.pipeline import (  # noqa: F401
    DECISION_FORMATS,
    DecisionConversionPipeline,
    get_decision_converter,
)

__all__ = [
    "DECISION_FORMATS",
    "DecisionConversionPipeline",
    "get_decision_converter",
]
