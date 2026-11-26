"""Deprecated: the 0.13.0 capabilities module.

``llm_rosetta.capabilities`` shipped in 0.13.0 as a flat module holding the
capability passes.  The passes are now internal (``llm_rosetta.ir.passes``)
and the pipeline runs them; the shim types it re-exported live in
:mod:`llm_rosetta.contracts.shim`, and ``ConversionContext`` in
:mod:`llm_rosetta.converters.base`.  This module forwards the twelve names
0.13.0 exported, as the same objects, for one minor release.  Importing it
warns once.  It will be removed in 0.15.0.
"""

from __future__ import annotations

import warnings

from llm_rosetta.contracts.shim import ProviderShim, ReasoningCapability
from llm_rosetta.converters.base import ConversionContext
from llm_rosetta.ir.passes.description_limit import (
    relocate_oversized_tool_descriptions,
)
from llm_rosetta.ir.passes.enforcement import (
    enforce_custom_tools,
    enforce_reasoning,
    enforce_vision,
    get_custom_tool_names,
    restore_custom_tool_calls,
    strip_reasoning_for_non_reasoning,
    unwrap_custom_tool_input,
)
from llm_rosetta.contracts.resolver import resolve_shim

# Deprecated 0.13.0 path, removed in 0.15.0; the replacements are
# llm_rosetta.contracts.shim, llm_rosetta.converters.base and the pipeline
# (which runs the passes).
warnings.warn(
    "llm_rosetta.capabilities is deprecated and will be removed in 0.15.0; "
    "the capability passes run inside the conversion pipeline, ProviderShim and "
    "ReasoningCapability are in llm_rosetta.contracts.shim, ConversionContext is "
    "in llm_rosetta.converters.base",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ConversionContext",
    "ProviderShim",
    "ReasoningCapability",
    "enforce_custom_tools",
    "enforce_reasoning",
    "enforce_vision",
    "get_custom_tool_names",
    "relocate_oversized_tool_descriptions",
    "resolve_shim",
    "restore_custom_tool_calls",
    "strip_reasoning_for_non_reasoning",
    "unwrap_custom_tool_input",
]
