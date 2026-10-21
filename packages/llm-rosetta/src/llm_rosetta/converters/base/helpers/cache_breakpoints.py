"""Compatibility shim: moved to :mod:`llm_rosetta.ir.primitives.cache_breakpoints`."""

from __future__ import annotations

import warnings

from llm_rosetta.ir.primitives.cache_breakpoints import *  # noqa: F401,F403

warnings.warn(
    "llm_rosetta.converters.base.helpers.cache_breakpoints moved to "
    "llm_rosetta.ir.primitives.cache_breakpoints; the old path is removed in 0.16.0",
    DeprecationWarning,
    stacklevel=2,
)
