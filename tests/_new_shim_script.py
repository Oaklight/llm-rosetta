"""Load ``scripts/dev/new_shim.py`` once, as the one module the tests share.

The scaffolder's prompts, base-url probe and icon lookup live in the script (they
are not in the wheel), so a test that patches one of them patches this module and
runs ``main`` of the same object.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dev/new_shim.py"

_MODULE: ModuleType | None = None


def load() -> ModuleType:
    """Return the script as a module, importing it on the first call."""
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location("new_shim", SCRIPT)
        assert spec is not None and spec.loader is not None, SCRIPT
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE
