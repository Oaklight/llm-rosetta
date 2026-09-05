"""
Deprecated — use ``llm_rosetta.converters.google_generate`` instead.

This module exists only for backward compatibility and will be removed
in a future release.
"""

import warnings as _warnings

_warnings.warn(
    "llm_rosetta.converters.google_genai is deprecated, "
    "use llm_rosetta.converters.google_generate instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..google_generate import (  # noqa: F401, E402
    GoogleConverter,
    GoogleGenerateConfigOps as GoogleGenAIConfigOps,
    GoogleGenerateContentOps as GoogleGenAIContentOps,
    GoogleGenerateConverter as GoogleGenAIConverter,
    GoogleGenerateMessageOps as GoogleGenAIMessageOps,
    GoogleGenerateToolOps as GoogleGenAIToolOps,
)

__all__ = [
    "GoogleGenAIConverter",
    "GoogleConverter",
    "GoogleGenAIContentOps",
    "GoogleGenAIToolOps",
    "GoogleGenAIMessageOps",
    "GoogleGenAIConfigOps",
]
