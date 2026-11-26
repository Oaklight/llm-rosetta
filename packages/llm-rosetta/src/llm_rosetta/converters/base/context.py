"""Deprecated: the conversion context moved to :mod:`llm_rosetta.ir.context`.

``llm_rosetta.converters.base.context`` shipped in 0.13.0.  The context
hierarchy lives in :mod:`llm_rosetta.ir.context` so the IR layer does not depend
on ``converters``.  This module forwards the three names 0.13.0 exported, as the
same objects, for one minor release.  Importing it warns once.  It will be
removed in 0.15.0.  Names first shipped in 0.14.0 (``ImageFetcher`` and the
others) are not forwarded: import them from :mod:`llm_rosetta.ir.context`.
"""

from __future__ import annotations

import warnings

from llm_rosetta.ir.context import ConversionContext, MetadataMode, StreamContext

# Deprecated 0.13.0 path, removed in 0.15.0; the replacement is
# llm_rosetta.ir.context.
warnings.warn(
    "llm_rosetta.converters.base.context is deprecated and will be removed in "
    "0.15.0; import from llm_rosetta.ir.context",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ConversionContext", "MetadataMode", "StreamContext"]
