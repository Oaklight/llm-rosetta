"""
LLM-Rosetta - IR Extension Types (Deprecated Module)

.. deprecated::
    This module has been superseded by :mod:`extensions_experimental`.
    Import directly from ``llm_rosetta.types.ir.extensions_experimental``
    or via ``llm_rosetta.types.ir`` (experimental exports are available
    under the ``experimental`` namespace).

    These types are speculative and no provider currently implements them.
"""

import warnings as _warnings

_warnings.warn(
    "llm_rosetta.types.ir.extensions is deprecated; "
    "use llm_rosetta.types.ir.extensions_experimental instead.",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.types.ir.extensions_experimental import (  # noqa: F401, E402
    BatchMarker,
    ExtensionItem,
    EXTENSION_TYPE_MAP,
    SessionControl,
    SystemEvent,
    ToolChainNode,
    get_extension_type,
    is_extension_item,
    is_extension_type,
    isinstance_extension,
)

__all__ = [
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    "ExtensionItem",
    "is_extension_item",
    "is_extension_type",
    "get_extension_type",
    "isinstance_extension",
    "EXTENSION_TYPE_MAP",
]
