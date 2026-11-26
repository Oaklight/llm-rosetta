"""Deprecated: moved to :mod:`llm_rosetta.ir.model.response`.

``llm_rosetta.types.ir.response`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.response import (
    ChoiceInfo,
    ExtensionItem,
    FinishReason,
    IRResponse,
    UsageInfo,
    is_extension_item,
)

__all__ = [
    "ChoiceInfo",
    "ExtensionItem",
    "FinishReason",
    "IRResponse",
    "UsageInfo",
    "is_extension_item",
]
