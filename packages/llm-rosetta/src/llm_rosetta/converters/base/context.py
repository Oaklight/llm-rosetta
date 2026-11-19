"""Compatibility re-export of the conversion context.

The context hierarchy now lives in :mod:`llm_rosetta.ir.context` so the IR
layer does not depend on ``converters``.  Every name is re-exported here, so
existing ``llm_rosetta.converters.base.context`` imports keep working.
"""

from __future__ import annotations

from llm_rosetta.ir.context import (
    _REQUEST_TOOL_VOCABULARY_KEY,
    ConversionContext,
    ImageFetcher,
    MetadataMode,
    StreamContext,
    carry_registered_call_identity,
)

__all__ = [
    "_REQUEST_TOOL_VOCABULARY_KEY",
    "ConversionContext",
    "ImageFetcher",
    "MetadataMode",
    "StreamContext",
    "carry_registered_call_identity",
]
