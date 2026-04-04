"""
LLM-Rosetta - IR Validation Utilities

Runtime validation for IR types using zerodep validate.
These functions validate dicts against IR TypedDict definitions,
catching malformed structures at IR boundaries rather than
downstream in converters.
"""

from __future__ import annotations

from typing import Any, Union

from ..._vendor.validate import ValidationError, validate
from .extensions import ExtensionItem
from .messages import Message
from .request import IRRequest
from .response import IRResponse


def validate_ir_request(data: dict[str, Any]) -> IRRequest:
    """Validate a dict against IRRequest TypedDict.

    Args:
        data: Dict to validate.

    Returns:
        The validated data (same object, typed as IRRequest).

    Raises:
        ValidationError: If the data doesn't match IRRequest structure.
    """
    return validate(data, IRRequest)


def validate_ir_response(data: dict[str, Any]) -> IRResponse:
    """Validate a dict against IRResponse TypedDict.

    Args:
        data: Dict to validate.

    Returns:
        The validated data (same object, typed as IRResponse).

    Raises:
        ValidationError: If the data doesn't match IRResponse structure.
    """
    return validate(data, IRResponse)


def validate_messages(
    messages: list[Any],
) -> list[Message | ExtensionItem]:
    """Validate a message list against IR Message/ExtensionItem types.

    Args:
        messages: List of message dicts to validate.

    Returns:
        The validated list.

    Raises:
        ValidationError: If any message doesn't match expected structure.
    """
    return validate(messages, list[Union[Message, ExtensionItem]])


__all__ = [
    "ValidationError",
    "validate_ir_request",
    "validate_ir_response",
    "validate_messages",
]
