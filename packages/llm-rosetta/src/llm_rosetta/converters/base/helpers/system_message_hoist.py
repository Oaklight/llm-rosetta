"""Compatibility shim: moved to :mod:`llm_rosetta.ir.primitives.system_message_hoist`."""

from __future__ import annotations

import warnings

from llm_rosetta.ir.primitives.system_message_hoist import *  # noqa: F401,F403

warnings.warn(
    "llm_rosetta.converters.base.helpers.system_message_hoist moved to "
    "llm_rosetta.ir.primitives.system_message_hoist; the old path is removed in 0.16.0",
    DeprecationWarning,
    stacklevel=2,
)
