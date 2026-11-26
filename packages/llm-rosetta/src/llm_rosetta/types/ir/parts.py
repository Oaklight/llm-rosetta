"""Deprecated: moved to :mod:`llm_rosetta.ir.model.parts`.

``llm_rosetta.types.ir.parts`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.parts import (
    AssistantContentPart,
    AudioData,
    AudioPart,
    CitationPart,
    ContentPart,
    FileData,
    FilePart,
    ImageData,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    SystemContentPart,
    TextCitation,
    TextPart,
    ToolCallPart,
    ToolContentPart,
    ToolResultPart,
    UrlCitation,
    UserContentPart,
)

__all__ = [
    "AssistantContentPart",
    "AudioData",
    "AudioPart",
    "CitationPart",
    "ContentPart",
    "FileData",
    "FilePart",
    "ImageData",
    "ImagePart",
    "ReasoningPart",
    "RefusalPart",
    "SystemContentPart",
    "TextCitation",
    "TextPart",
    "ToolCallPart",
    "ToolContentPart",
    "ToolResultPart",
    "UrlCitation",
    "UserContentPart",
]
