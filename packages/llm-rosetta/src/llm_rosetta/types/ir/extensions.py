"""Deprecated: the 0.13.0 IR extension module; import ``extensions_experimental``.

``llm_rosetta.types.ir.extensions`` shipped in 0.13.0, already as a deprecated
forwarder.  The extension types live in
:mod:`llm_rosetta.ir.model.extensions_experimental`; this module forwards the
ten names 0.13.0 exported, as the same objects, for one minor release.
Importing it warns once.  It will be removed in 0.15.0.

These types are speculative and no provider currently implements them.
"""

from __future__ import annotations

import warnings

from llm_rosetta.ir.model.extensions_experimental import (
    EXTENSION_TYPE_MAP,
    BatchMarker,
    ExtensionItem,
    SessionControl,
    SystemEvent,
    ToolChainNode,
    get_extension_type,
    is_extension_item,
    is_extension_type,
    isinstance_extension,
)

# Deprecated 0.13.0 path, removed in 0.15.0; the replacement is
# llm_rosetta.ir.model.extensions_experimental.
warnings.warn(
    "llm_rosetta.types.ir.extensions is deprecated and will be removed in "
    "0.15.0; import from llm_rosetta.ir.model.extensions_experimental",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EXTENSION_TYPE_MAP",
    "BatchMarker",
    "ExtensionItem",
    "SessionControl",
    "SystemEvent",
    "ToolChainNode",
    "get_extension_type",
    "is_extension_item",
    "is_extension_type",
    "isinstance_extension",
]
