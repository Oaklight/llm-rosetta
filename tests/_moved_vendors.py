"""Frozen copies of the shim vendors that moved from core to contrib (#1660 S7).

Nothing ships these directories.  Core's and the gateway's own tests load them
to exercise paths that only these vendors declare today — reasoning effort
ladders, the tool-group matrix, request-range clamping, the ACP transport, a
credential exchange, a model-list transform.  The live copies are in
Oaklight/llm-rosetta-shim-contrib; a change there does not need a change here,
because these tests pin core behaviour, not the vendor's.

A test file outside ``tests/`` itself imports this module after
``sys.path.insert(0, <tests dir>)``, the way ``_import_scan`` is imported.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

#: Where the frozen vendor directories live.
MOVED_VENDORS_DIR = Path(__file__).parent / "fixtures" / "moved_vendors"


def load_moved_vendors() -> list[Any]:
    """Register every frozen moved vendor's shims; return them."""
    from llm_rosetta.api import load_providers_from_dir

    return load_providers_from_dir(MOVED_VENDORS_DIR)


def load_providers_with_moved() -> list[Any]:
    """Load core's built-ins, then the frozen moved vendors; return both lists."""
    from llm_rosetta.shims.providers import load_providers

    return [*load_providers(), *load_moved_vendors()]


def moved_vendor_module(relpath: str) -> ModuleType:
    """Import one sidecar of a frozen moved vendor by its path.

    Args:
        relpath: The path under ``tests/fixtures/moved_vendors``
            (``"opencode/quota.py"``).
    """
    path = MOVED_VENDORS_DIR / relpath
    name = "_moved_vendor_" + relpath.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
