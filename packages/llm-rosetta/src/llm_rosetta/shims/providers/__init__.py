"""The bundled vendor data, and the deprecated 0.13.0 loader path.

Each ``<vendor>/`` directory below holds ``provider.yaml`` files and their
sidecars (``transforms.py``, ``quota.py``, ``auth.py``).  The code that reads
them is :mod:`llm_rosetta.shims._loader`; this package exists so the sidecars
load under a real package name and so the loader can find this directory.

Deprecated: in 0.13.0 this module held the shim loader itself, and shim
packages imported ``load_providers``, ``get_model_list_transform`` and the
rest from here.  Those names still resolve, as the same objects, and each
access warns with its replacement: :mod:`llm_rosetta.api` for hosts,
:mod:`llm_rosetta.shims` for ``load_providers_from_dir``,
:mod:`llm_rosetta.contracts.shim` for the data classes.  The forwarding is
removed in 0.15.0.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any

#: The built-in providers root.  :mod:`llm_rosetta.shims._loader` imports it;
#: rebind it there to redirect the scan.
_PROVIDERS_DIR = Path(__file__).parent

# Deprecated 0.13.0 path, removed in 0.15.0.  Each name 0.13.0's
# llm_rosetta.shims.providers exported -> (the module that holds it now, the
# import a caller should use instead).
_FORWARDED: dict[str, tuple[str, str]] = {
    "ProviderShim": ("llm_rosetta.contracts.shim", "llm_rosetta.contracts.shim"),
    "ReasoningCapability": (
        "llm_rosetta.contracts.shim",
        "llm_rosetta.contracts.shim",
    ),
    "register_shim": ("llm_rosetta.shims.provider_shim", "llm_rosetta.api"),
    "load_providers": ("llm_rosetta.shims._loader", "llm_rosetta.api"),
    "load_providers_from_dir": ("llm_rosetta.shims._loader", "llm_rosetta.shims"),
    "get_model_list_transform": ("llm_rosetta.shims._loader", "llm_rosetta.api"),
    "register_model_list_transform": (
        "llm_rosetta.shims._loader",
        "a model_list_transform exported by the shim's transforms.py",
    ),
    "ModelListTransform": (
        "llm_rosetta.shims._loader",
        "a model_list_transform exported by the shim's transforms.py",
    ),
}

__all__ = sorted(_FORWARDED)


def __getattr__(name: str) -> Any:
    """Resolve a 0.13.0 name to the same object at its current home, warning once."""
    entry = _FORWARDED.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    home, instead = entry
    warnings.warn(
        f"llm_rosetta.shims.providers.{name} is deprecated and will be removed "
        f"in 0.15.0; use {instead}",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(importlib.import_module(home), name)
    # Bind it here so the next lookup, including the second one a
    # ``from ... import`` makes (import machinery probes the name, then the
    # bytecode reads it), finds a plain attribute: one warning per name.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FORWARDED))
