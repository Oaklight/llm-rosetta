"""Compatibility shim: moved to :mod:`llm_rosetta.ir.primitives.image_limit`."""

from __future__ import annotations

import warnings

from llm_rosetta.ir.primitives.image_limit import *  # noqa: F401,F403

warnings.warn(
    "llm_rosetta.converters.base.helpers.image_limit moved to "
    "llm_rosetta.ir.primitives.image_limit; the old path is removed in 0.16.0",
    DeprecationWarning,
    stacklevel=2,
)
