"""The decision converters by format, and the usage their responses carry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.converters.decisions.openai import OpenAIDecisionsConverter
from llm_rosetta.converters.decisions.systemone import SystemOneDecisionConverter

__all__ = [
    "DECISION_FORMATS",
    "get_decision_converter",
    "response_usage",
]

_DECISION_CONVERTERS: dict[str, type[BaseDecisionConverter]] = {
    "systemone": SystemOneDecisionConverter,
    "openai_decisions": OpenAIDecisionsConverter,
}

DECISION_FORMATS = frozenset(_DECISION_CONVERTERS.keys())


def get_decision_converter(format_name: str) -> BaseDecisionConverter:
    """Return a converter instance for *format_name*.

    Raises:
        ValueError: If *format_name* is not a known decision format.
    """
    cls = _DECISION_CONVERTERS.get(format_name)
    if cls is None:
        raise ValueError(
            f"Unknown decision format: '{format_name}'. "
            f"Available: {', '.join(sorted(DECISION_FORMATS))}"
        )
    return cls()


def response_usage(
    format_name: str, response_body: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The IR usage of an upstream response, read by *format_name*'s converter.

    The gateway's decision handler never holds the IR — the conversion pipeline
    goes from the provider's format back to the source format in one step,
    including when both sides are the same format — so the ledger's figures
    are read here, by the converter that would have produced them.

    A body the converter cannot read, or none at all, reports no usage rather
    than raising: a metering gap is not worth failing a response the caller has
    already been given.
    """
    if not response_body:
        return {}
    try:
        converter = get_decision_converter(format_name)
        ir = converter.response_from_provider(dict(response_body))
    except Exception:
        return {}
    return dict(ir.get("usage") or {})
