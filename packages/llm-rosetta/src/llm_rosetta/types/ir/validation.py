"""Deprecated: moved to :mod:`llm_rosetta.ir.model.validation`.

``llm_rosetta.types.ir.validation`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.validation import (
    ValidationError,
    validate_ir_request,
    validate_ir_response,
    validate_messages,
    validate_tools,
)

__all__ = [
    "ValidationError",
    "validate_ir_request",
    "validate_ir_response",
    "validate_messages",
    "validate_tools",
]
