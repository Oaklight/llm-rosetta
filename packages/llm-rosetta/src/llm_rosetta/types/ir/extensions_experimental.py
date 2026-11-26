"""Deprecated: moved to :mod:`llm_rosetta.ir.model.extensions_experimental`.

``llm_rosetta.types.ir.extensions_experimental`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.extensions_experimental import (
    BatchMarker,
    EXTENSION_TYPE_MAP,
    ExtensionItem,
    SessionControl,
    SystemEvent,
    ToolChainNode,
    get_extension_type,
    is_extension_item,
    is_extension_type,
    isinstance_extension,
)

__all__ = [
    "BatchMarker",
    "EXTENSION_TYPE_MAP",
    "ExtensionItem",
    "SessionControl",
    "SystemEvent",
    "ToolChainNode",
    "get_extension_type",
    "is_extension_item",
    "is_extension_type",
    "isinstance_extension",
]
