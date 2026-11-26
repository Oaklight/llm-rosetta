"""Deprecated: moved to :mod:`llm_rosetta.ir.model.stream`.

``llm_rosetta.types.ir.stream`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.stream import (
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    IRStreamEvent,
    ProviderPassthroughEvent,
    ReasoningDeltaEvent,
    RefusalDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    UsageEvent,
)

__all__ = [
    "ContentBlockEndEvent",
    "ContentBlockStartEvent",
    "FinishEvent",
    "IRStreamEvent",
    "ProviderPassthroughEvent",
    "ReasoningDeltaEvent",
    "RefusalDeltaEvent",
    "StreamEndEvent",
    "StreamStartEvent",
    "TextDeltaEvent",
    "ToolCallDeltaEvent",
    "ToolCallStartEvent",
    "UsageEvent",
]
