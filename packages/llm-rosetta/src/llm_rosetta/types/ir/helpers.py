"""Deprecated: moved to :mod:`llm_rosetta.ir.model.helpers`.

``llm_rosetta.types.ir.helpers`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.helpers import (
    create_tool_result_message,
    extract_all_text,
    extract_text_content,
    extract_tool_calls,
)

__all__ = [
    "create_tool_result_message",
    "extract_all_text",
    "extract_text_content",
    "extract_tool_calls",
]
