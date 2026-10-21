"""Deprecated alias for :mod:`llm_rosetta_gateway` (removed in 0.16.0).

The gateway moved to its own distribution (epic #1127, issue #1157).  The
``llm_rosetta.gateway`` import root keeps working until the 0.16.0 removal
window (#919): importing it warns, and every submodule resolves to the same
module object as its ``llm_rosetta_gateway`` counterpart, so state (the
config, the registries) is never duplicated.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from typing import Any

_ALIAS = "llm_rosetta.gateway"
_TARGET = "llm_rosetta_gateway"

try:
    _real = importlib.import_module(_TARGET)
except ModuleNotFoundError as exc:
    # The core distribution ships this shim but not the gateway.  Say which
    # package to install instead of leaking the alias target's module name.
    raise ImportError(
        "llm_rosetta.gateway is a deprecated alias for llm_rosetta_gateway, "
        "which is not installed. Install the gateway distribution: "
        "pip install llm-rosetta-gateway"
    ) from exc

warnings.warn(
    "importing llm_rosetta.gateway is deprecated; use llm_rosetta_gateway "
    "(llm_rosetta.gateway will be removed in 0.16.0)",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``llm_rosetta.gateway.*`` to the same objects as the real package."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == _ALIAS or fullname.startswith(_ALIAS + "."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        return importlib.import_module(_TARGET + spec.name[len(_ALIAS) :])

    def exec_module(self, module: Any) -> None:
        """The aliased module is already executed."""


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str) -> Any:
    """Forward attribute access (including submodules) to the real package."""
    return getattr(_real, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_real)))


__all__ = list(getattr(_real, "__all__", []))
