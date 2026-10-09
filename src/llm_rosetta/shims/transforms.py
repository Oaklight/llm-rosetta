"""Backward-compatible re-exports — use ``llm_rosetta.transforms`` instead.

.. deprecated::
    This module is a compatibility shim.  Import from
    :mod:`llm_rosetta.transforms` (or its submodules ``.ir`` / ``.body``)
    directly.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "llm_rosetta.shims.transforms is deprecated; "
    "import from llm_rosetta.transforms instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.transforms import *  # noqa: E402,F401,F403
from llm_rosetta.transforms.body import _NamedTransform as _NamedTransform  # noqa: E402,F401
from llm_rosetta.transforms.ir import _NamedIRTransform as _NamedIRTransform  # noqa: E402,F401
