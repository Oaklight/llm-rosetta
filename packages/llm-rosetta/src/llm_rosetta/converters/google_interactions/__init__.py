"""
Deprecated — use ``llm_rosetta.converters.llm.google_interactions`` instead.

``converters.google_interactions`` was an importable package in 0.13.0.  The
implementation now lives under ``converters.llm``; this package warns on
import and then registers the new submodules under the old dotted paths, so
``converters.google_interactions.converter`` and friends keep resolving to the
same module objects the registry holds.  Removed in 0.16.0 (#919).
"""

import sys as _sys
import warnings as _warnings
from types import ModuleType as _ModuleType

_warnings.warn(
    "llm_rosetta.converters.google_interactions is deprecated, "
    "use llm_rosetta.converters.llm.google_interactions instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..llm import google_interactions as _package  # noqa: E402
from ..llm.google_interactions import *  # noqa: F401, F403, E402
from ..llm.google_interactions import __all__  # noqa: F401, E402

for _attr, _value in vars(_package).items():
    if isinstance(_value, _ModuleType) and _value.__name__.startswith(
        _package.__name__ + "."
    ):
        _sys.modules[f"{__name__}.{_attr}"] = _value
        # Registering in ``sys.modules`` only covers the dotted import form;
        # bind the attribute too so ``import ... as gg;
        # gg.converter.X`` keeps working, as it did in 0.13.0.
        globals()[_attr] = _value
