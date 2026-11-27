"""Deprecated: the 0.13.0 auto-detection module.  Import from :mod:`llm_rosetta`.

``llm_rosetta.auto_detect`` shipped in 0.13.0 holding provider detection, the
converter lookup and the one-call ``convert`` helpers.  They now live where
they belong: :func:`~llm_rosetta.converters.registry.detect_provider` and
:func:`~llm_rosetta.converters.registry.get_converter_for_provider` in
:mod:`llm_rosetta.converters.registry`, and
:func:`~llm_rosetta.pipeline.convert` and
:func:`~llm_rosetta.pipeline.convert_response` in :mod:`llm_rosetta.pipeline`;
all five names are also importable from :mod:`llm_rosetta`.  This module
forwards the five names 0.13.0 exported, as the same objects, for one minor
release.  Importing it warns once.  It will be removed in 0.15.0.
"""

from __future__ import annotations

import warnings

from llm_rosetta.contracts.routing import ProviderType
from llm_rosetta.converters.registry import (
    detect_provider,
    get_converter_for_provider,
)
from llm_rosetta.pipeline.facade import convert, convert_response

# Deprecated 0.13.0 path, removed in 0.15.0; the replacement is
# ``from llm_rosetta import convert, convert_response, detect_provider,
# get_converter_for_provider, ...``.
warnings.warn(
    "llm_rosetta.auto_detect is deprecated and will be removed in 0.15.0; "
    "import its names from llm_rosetta",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ProviderType",
    "convert",
    "convert_response",
    "detect_provider",
    "get_converter_for_provider",
]
